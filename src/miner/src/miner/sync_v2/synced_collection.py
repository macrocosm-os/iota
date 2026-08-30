"""Bridge variable batch operations for SyncedDictionary.

Module-level functions take an HTTP client and ``run_id`` explicitly rather
than storing them on an instance.  Callers own the *variables* dict that tracks
registered :class:`~miner.sync_v2.synced_variable.SyncedVariableV2` objects.

Usage::

    variables: dict[str, SyncedVariableV2] = {}
    await register_many(client, run_id, entries, variables=variables)
    raw = await wildcard_fetch(client, run_id, "prefix/*", variables=variables)
    await set_many(client, run_id, variables, {"name": value})
"""

from __future__ import annotations

import asyncio
import fnmatch
import random
from typing import Any, Literal

import httpx
from loguru import logger

from miner.sync_v2.synced_variable import Lock, SyncedVariableV2, _bridge_request


def apply_get_response(
    data: dict,
    *,
    run_id: str,
    variables: dict[str, SyncedVariableV2] | None = None,
) -> dict[str, Any]:
    """Parse a bridge POST /get response and optionally update local caches.

    Returns ``{rel_name: value}`` for all entries in the response.
    """
    result: dict[str, Any] = {}
    run_prefix = f"{run_id}/"
    for entry in data.get("variables", []):
        try:
            full_id: str = entry["var_id"]
            rel = full_id.removeprefix(run_prefix) if full_id.startswith(run_prefix) else full_id
            value = entry["value"]
            result[rel] = value
            if variables is not None and rel in variables:
                var = variables[rel]
                var._cached_value = value
                var.version = entry.get("version", var.version)
                var.updated_at = entry.get("metadata", {}).get("updated_at", var.updated_at)
                var._has_fetched = True
        except Exception as exc:
            logger.warning(f"apply_get_response: skipping malformed entry: {exc}")
    return result


# ── Registration ──────────────────────────────────────────────────────────


async def add(
    client: httpx.AsyncClient,
    run_id: str,
    name: str,
    default_value: Any,
    *,
    var_type: str = "dict",
    rule: Literal["LOCK", "CAS", "LWW"] = "LWW",
    require_fetch: bool = False,
    variables: dict[str, SyncedVariableV2],
) -> SyncedVariableV2:
    """Register one variable on the bridge and track it in *variables*.

    Idempotent: if the variable already exists the bridge returns
    ``already_registered`` and the local entry is updated.
    """
    var = SyncedVariableV2(
        run_id=run_id,
        name=name,
        var_type=var_type,
        write_rule=rule,
        require_fetch=require_fetch,
        default=default_value,
    )
    var._client = client
    await var.register(default_value)
    variables[name] = var
    return var


async def register_many(
    client: httpx.AsyncClient,
    run_id: str,
    entries: list[dict[str, Any]],
    *,
    rule: str = "LWW",
    require_fetch: bool = False,
    variables: dict[str, SyncedVariableV2],
) -> None:
    """Batch-register variables and track them in *variables*.

    Each entry: ``{name, default_value, var_type, write_rule (optional)}``.
    On success each variable's local cache is seeded with ``default_value``.
    """
    if not entries:
        return
    payload: list[dict[str, Any]] = []
    ordered: list[tuple[str, SyncedVariableV2]] = []
    for entry in entries:
        name = entry["name"]
        var = SyncedVariableV2(
            run_id=run_id,
            name=name,
            var_type=entry["var_type"],
            write_rule=entry.get("write_rule") or rule,
            require_fetch=require_fetch,
            default=entry["default_value"],
        )
        var._client = client
        payload.append(
            {
                "var_id": var.var_id,
                "default_value": entry["default_value"],
                "var_type": var.var_type,
                "write_rule": var.write_rule,
            }
        )
        variables[name] = var
        ordered.append((name, var))

    resp = await _bridge_request(client, "POST", "/register", json={"variables": payload})
    resp.raise_for_status()
    for i, result in enumerate(resp.json().get("results", [])):
        if result.get("status") == "error":
            logger.warning(f"register_many {payload[i]['var_id']!r}: {result.get('error')}")
        else:
            _, var = ordered[i]
            var.version = result.get("version", 1)
            var._cached_value = entries[i]["default_value"]
            var._bridge_registered = True


# ── Reads ─────────────────────────────────────────────────────────────────


def get_cached(name: str, variables: dict[str, SyncedVariableV2]) -> Any:
    """Return cached value of a tracked variable (synchronous, no network call)."""
    if name not in variables:
        raise KeyError(f"{name!r} is not tracked; call register_many() first")
    return variables[name].get_cached_value()


async def get(
    name: str,
    *,
    variables: dict[str, SyncedVariableV2],
) -> Any:
    """Fetch the current value of one variable from the bridge."""
    if name not in variables:
        raise KeyError(f"{name!r} is not tracked; call register_many() first")
    return await variables[name].fetch_value()


async def fetch_all(
    client: httpx.AsyncClient,
    run_id: str,
    variables: dict[str, SyncedVariableV2],
) -> dict[str, Any]:
    """Fetch all tracked variables in a single batch request."""
    if not variables:
        return {}
    var_ids = [var.var_id for var in variables.values()]
    resp = await _bridge_request(client, "POST", "/get", json={"var_ids": var_ids})
    resp.raise_for_status()
    apply_get_response(resp.json(), run_id=run_id, variables=variables)
    return {name: var.get_cached_value() for name, var in variables.items()}


async def wildcard_fetch(
    client: httpx.AsyncClient,
    run_id: str,
    path: str,
    *,
    variables: dict[str, SyncedVariableV2] | None = None,
) -> dict[str, Any]:
    """Fetch all bridge variables whose name matches the wildcard *path*.

    *path* is relative to *run_id*; ``*`` matches any single path segment.
    Example: ``await wildcard_fetch(client, run_id, "node_registry/*/iroh_nodes")``.

    If *variables* is provided, cached values for tracked vars are updated.
    Returns ``{rel_name: value}`` for all matched variables.
    """
    full_pattern = f"{run_id}/{path}"
    star = full_pattern.find("*")
    prefix = full_pattern[:star] if star != -1 else full_pattern

    resp = await _bridge_request(client, "GET", "/vars", params={"prefix": prefix})
    resp.raise_for_status()
    all_entries = resp.json()

    matching_ids = [e["var_id"] for e in all_entries if fnmatch.fnmatch(e["var_id"], full_pattern)]
    if not matching_ids:
        return {}

    resp = await _bridge_request(client, "POST", "/get", json={"var_ids": matching_ids})
    resp.raise_for_status()

    return apply_get_response(resp.json(), run_id=run_id, variables=variables)


# ── Writes ────────────────────────────────────────────────────────────────


async def set_value(
    name: str,
    value: Any,
    *,
    variables: dict[str, SyncedVariableV2],
    lock: Lock | None = None,
    current_version: int | None = None,
) -> None:
    """Write a new value for a tracked variable."""
    if name not in variables:
        raise KeyError(f"{name!r} is not tracked; call register_many() first")
    await variables[name].set_value(value, lock=lock, current_version=current_version)


async def set_many(
    client: httpx.AsyncClient,
    run_id: str,
    variables: dict[str, SyncedVariableV2],
    values: dict[str, Any],
    *,
    lock: Lock | None = None,
    current_versions: dict[str, int] | None = None,
) -> tuple[list[str], list[str]]:
    """Batch-set tracked variables in a single request.

    Returns ``(not_found, mismatched)`` variable names: bridge entries that
    were missing (VariableNotFound — re-register and retry) and CAS writes
    rejected for a stale version (refresh the version and retry).
    """
    if not values:
        return [], []
    updates: list[dict[str, Any]] = []
    for name, value in values.items():
        if name not in variables:
            raise KeyError(f"{name!r} is not tracked; call register_many() first")
        var = variables[name]
        update: dict[str, Any] = {"var_id": var.var_id, "value": value}
        if lock is not None:
            tok = lock.token_for(var.var_id)
            if tok is not None:
                update["lock_token"] = tok
            elif var.write_rule == "LOCK":
                raise AssertionError(f"LOCK variable {name!r} has no token in the provided lock")
        if var.write_rule == "CAS":
            cv = (current_versions or {}).get(name)
            update["current_version"] = cv if cv is not None else var.version
        updates.append(update)

    resp = await _bridge_request(client, "POST", "/set", json={"updates": updates})
    resp.raise_for_status()

    not_found: list[str] = []
    mismatched: list[str] = []
    run_prefix = f"{run_id}/"
    for result in resp.json().get("results", []):
        if result.get("status") == "error":
            logger.warning(f"set_many {result['var_id']!r}: {result.get('error')}")
            error = result.get("error") or ""
            full_id = result["var_id"]
            rel = full_id.removeprefix(run_prefix) if full_id.startswith(run_prefix) else full_id
            if "VariableNotFound" in error:
                not_found.append(rel)
            elif "CASVersionMismatch" in error:
                mismatched.append(rel)
            continue
        full_id = result["var_id"]
        rel = full_id.removeprefix(run_prefix) if full_id.startswith(run_prefix) else full_id
        if rel in variables:
            variables[rel].version = result.get("version", variables[rel].version + 1)
            if rel in values:
                variables[rel]._cached_value = values[rel]
            variables[rel]._needs_push = False
    return not_found, mismatched


async def _refresh_versions(
    client: httpx.AsyncClient,
    run_id: str,
    variables: dict[str, SyncedVariableV2],
    names: list[str],
) -> list[str]:
    """Refresh local CAS versions for *names* with a keyed /get.

    Fetches by var_id (not by wildcard listing, which can under-report — the
    listing missing a key while CAS sets still see it is exactly how stale
    versions get stuck). Updates only the version, keeping the local dirty
    value intact for the retry. Returns names the bridge reported missing.
    """
    tracked = [n for n in names if n in variables]
    if not tracked:
        return []
    resp = await _bridge_request(client, "POST", "/get", json={"var_ids": [variables[n].var_id for n in tracked]})
    resp.raise_for_status()
    run_prefix = f"{run_id}/"
    data = resp.json()
    for entry in data.get("variables", []):
        full_id = entry["var_id"]
        rel = full_id.removeprefix(run_prefix) if full_id.startswith(run_prefix) else full_id
        if rel in variables:
            variables[rel].version = entry.get("version", variables[rel].version)
    missing: list[str] = []
    for err in data.get("errors", []):
        full_id = err["var_id"]
        rel = full_id.removeprefix(run_prefix) if full_id.startswith(run_prefix) else full_id
        missing.append(rel)
    return missing


async def _reregister_all(
    client: httpx.AsyncClient,
    variables: dict[str, SyncedVariableV2],
) -> None:
    """Re-register the caller's *entire* entry (all tracked leaves).

    Unlike re-registering only the missing leaves, this rebuilds an entry that
    was wiped in bulk (e.g. a Redis-cluster node reschedule dropping hash slots),
    which drops leaves this process isn't currently pushing as dirty. Registration
    is idempotent, so re-registering already-present leaves is harmless.
    """
    vars_list = list(variables.values())
    if not vars_list:
        return
    payload = [
        {
            "var_id": var.var_id,
            "default_value": var._cached_value,
            "var_type": var.var_type,
            "write_rule": var.write_rule,
        }
        for var in vars_list
    ]
    resp = await _bridge_request(client, "POST", "/register", json={"variables": payload})
    resp.raise_for_status()
    for i, result in enumerate(resp.json().get("results", [])):
        if result.get("status") != "error":
            var = vars_list[i]
            var.version = result.get("version", 1)
            var._bridge_registered = True


async def push_dirty(
    client: httpx.AsyncClient,
    run_id: str,
    variables: dict[str, SyncedVariableV2],
    *,
    max_retries: int = 3,
) -> None:
    """Push all dirty (LWW/CAS) variables in one batch.

    Skips ``write_rule="LOCK"`` variables — callers must acquire a lock
    and call :meth:`set_value` explicitly.  If any variables are missing
    from the bridge (VariableNotFound), the entry's registry state was lost;
    re-register the *whole* entry and retry with bounded backoff until the
    leaves persist. A persistent failure escalates rather than being swallowed.
    """
    dirty: dict[str, Any] = {}
    for name, var in variables.items():
        if not var._needs_push:
            continue
        if var.write_rule == "LOCK":
            logger.warning(
                f"push_dirty: skipping LOCK variable {name!r} " "(acquire a lock and call set_value() explicitly)"
            )
            continue
        dirty[name] = var._cached_value
    if not dirty:
        return

    not_found, mismatched = await set_many(client, run_id, variables, dirty)

    if mismatched:
        # Stale local CAS versions — e.g. background pulls missed these keys
        # (bridge listing under-reported) so versions were never refreshed.
        # Refresh by keyed /get and retry; a key missing on refresh joins the
        # re-register path below.
        missing = await _refresh_versions(client, run_id, variables, mismatched)
        not_found.extend(missing)
        retry_mismatched = {n: dirty[n] for n in mismatched if n in dirty and n not in missing}
        if retry_mismatched:
            nf, still_mismatched = await set_many(client, run_id, variables, retry_mismatched)
            not_found.extend(nf)
            if still_mismatched:
                # Lost another race; leaves stay dirty and the next push cycle
                # retries with the versions refreshed above.
                logger.warning(f"push_dirty: {len(still_mismatched)} CAS retries still mismatched (run={run_id})")

    if not not_found:
        return

    # Missing leaves => the bridge lost (part of) this entry's registry state.
    # Rebuild the full entry, not just the dirty misses, and retry until the
    # leaves stick or we hit the cap (the next push cycle retries regardless).
    delay = 0.5
    for attempt in range(1, max_retries + 1):
        try:
            await _reregister_all(client, variables)
        except Exception as exc:
            if attempt >= max_retries:
                logger.error(
                    f"push_dirty: re-register failed after {attempt} attempts "
                    f"(run={run_id}, {len(not_found)} missing leaves): {exc}"
                )
                return
            await asyncio.sleep(delay + random.uniform(0, delay))
            delay *= 2
            continue

        retry = {name: dirty[name] for name in not_found if name in dirty}
        not_found = (await set_many(client, run_id, variables, retry))[0] if retry else []
        if not not_found:
            return
        if attempt < max_retries:
            await asyncio.sleep(delay + random.uniform(0, delay))
            delay *= 2

    logger.error(
        f"push_dirty: {len(not_found)} leaves still missing after {max_retries} "
        f"re-register attempts (run={run_id}); bridge registry state may be lost"
    )


# ── Deletion ──────────────────────────────────────────────────────────────


async def delete(
    client: httpx.AsyncClient,
    run_id: str,
    name: str,
    *,
    variables: dict[str, SyncedVariableV2],
) -> None:
    """Delete one variable from the bridge and remove it from *variables*."""
    var_id = variables[name].var_id if name in variables else f"{run_id}/{name}"
    resp = await _bridge_request(client, "POST", "/delete", json={"var_ids": [var_id]})
    resp.raise_for_status()
    variables.pop(name, None)


async def wildcard_delete(
    client: httpx.AsyncClient,
    run_id: str,
    path: str,
    *,
    variables: dict[str, SyncedVariableV2],
) -> list[str]:
    """Delete all bridge variables matching *path* and remove them from *variables*.

    Returns the list of relative names that were deleted.
    """
    full_pattern = f"{run_id}/{path}"
    star = full_pattern.find("*")
    prefix = full_pattern[:star] if star != -1 else full_pattern

    resp = await _bridge_request(client, "GET", "/vars", params={"prefix": prefix})
    resp.raise_for_status()
    matching_ids = [e["var_id"] for e in resp.json() if fnmatch.fnmatch(e["var_id"], full_pattern)]
    if not matching_ids:
        return []

    resp = await _bridge_request(client, "POST", "/delete", json={"var_ids": matching_ids})
    resp.raise_for_status()

    run_prefix = f"{run_id}/"
    deleted: list[str] = []
    for entry in resp.json().get("results", []):
        if entry.get("status") == "deleted":
            full_id = entry["var_id"]
            rel = full_id.removeprefix(run_prefix) if full_id.startswith(run_prefix) else full_id
            variables.pop(rel, None)
            deleted.append(rel)
    return deleted
