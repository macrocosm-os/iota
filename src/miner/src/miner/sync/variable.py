"""SyncedVariable — polling-loop-driven synced variable."""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Callable, Generic, TypeVar
from common import settings as common_settings
from common.models.run_flags import RUN_FLAGS
from miner import settings as miner_settings
import httpx
import msgpack
from loguru import logger

from miner.sync.collections import (
    SyncedDict,
    SyncedList,
    apply_patch as _apply_patch,
)
from miner.sync.registry import NodeRegistry

T = TypeVar("T")

_COMPLEX_TYPES = (SyncedDict, SyncedList)


def sync_run_sync_prefix(run_id: str | None) -> str:
    """Return a sanitised run-scoped prefix for bridge keys.

    Every :class:`SyncedVariable` or :class:`DistributedCounter` that belongs
    to a specific training run should use this as the first path segment of its
    ``variable_id``, e.g. ``f"{sync_run_sync_prefix(run_id)}/node_registry"``.

    Pre-registration uses ``run-pending`` until a real ``run_id`` is assigned.
    """
    if not run_id:
        return "run-pending"
    safe = "".join((c if c.isalnum() or c in "-_" else "_") for c in str(run_id))
    return f"run-{safe}"


class SyncError(Exception):
    """Raised when an explicit ``SyncedVariable.push()`` / ``pull()`` fails."""

    pass


def _is_version_mismatch_error(msg: str) -> bool:
    return "version mismatch" in msg.lower()


# Retries after pull when another writer advanced the server ahead of our client_version.
_PUSH_MAX_ATTEMPTS = 5


try:
    from pydantic import BaseModel as _PydanticBase
except ImportError:
    _PydanticBase = None  # type: ignore[assignment,misc]


def _is_pydantic(value: Any) -> bool:
    return _PydanticBase is not None and isinstance(value, _PydanticBase)


def _jsonify(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonify(value.model_dump())
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _write_spec_for_sv(sv: "SyncedVariable", *, expected_version: int) -> tuple[dict, Callable[..., Any] | None]:
    """Build batch-write payload fragment and optional commit callback for ``sv``.

    For ``SyncedDict`` / ``SyncedList`` with patches enabled and a non-empty patch,
    only ``patch`` + ``client_version`` are sent (no full ``value``). The bridge
    rejects the write if ``client_version`` does not match the stored version.
    """
    value = sv._value
    write_spec: dict = {
        "service": sv.sync_service,
        "client_version": expected_version,
    }
    commit_fn: Callable[..., Any] | None = None

    if isinstance(value, _COMPLEX_TYPES):
        if RUN_FLAGS.sync_patches.isOn():
            patch = value.get_patch()
            if patch:
                safe = [{**op, "value": _jsonify(op["value"])} if "value" in op else op for op in patch]
                write_spec["patch"] = safe
                commit_fn = value.commit
                return write_spec, commit_fn
        raw = dict(value) if isinstance(value, SyncedDict) else list(value)
        write_spec["value"] = _jsonify(raw)
        commit_fn = value.commit
    else:
        write_spec["value"] = _jsonify(value)

    return write_spec, commit_fn


# ── Polling loop ───────────────────────────────────────────────────────────────


class PollingLoop:
    """Single background loop driving all SyncedVariable instances."""

    def __init__(self, server_url: str, tick: float = 2.0) -> None:
        self._server_url = server_url
        self._client = httpx.AsyncClient(base_url=self._server_url, timeout=30)
        self._svs: list[SyncedVariable] = []
        self._task: asyncio.Task | None = None
        # id(sv) → last server version seen (0 = never polled)
        self._versions: dict[int, int] = {}
        self._tick = tick
        # Serialize push + pull + poll-apply per variable to avoid races between
        # background batch push, explicit push, and incoming fetch updates.
        self._push_locks: dict[int, asyncio.Lock] = {}

    def _push_lock(self, sv: SyncedVariable) -> asyncio.Lock:
        k = id(sv)
        if k not in self._push_locks:
            self._push_locks[k] = asyncio.Lock()
        return self._push_locks[k]

    def register(self, sv: SyncedVariable) -> None:
        logger.info(f"Registering SyncedVariable: {sv._name}")
        if sv not in self._svs:
            self._svs.append(sv)
        self.start()

    def unregister(self, sv: SyncedVariable) -> None:
        self._svs = [s for s in self._svs if s is not sv]
        self._versions.pop(id(sv), None)
        self._push_locks.pop(id(sv), None)

    def unregister_all(self) -> None:
        """Drop every registered SyncedVariable and all associated loop state.

        The background task keeps running (use :meth:`stop` to cancel it), but
        has nothing to push or pull until callers re-register variables via
        :meth:`register`.
        """
        self._svs.clear()
        self._versions.clear()
        self._push_locks.clear()

    def start(self) -> None:
        """Ensure the background polling task is running. Safe to call repeatedly."""
        if self._task is None or self._task.done():
            logger.info("PollingLoop starting")
            self._task = asyncio.get_event_loop().create_task(self._run())

    async def stop(self) -> None:
        """Cancel the background polling task and wait for it to exit. Safe if not running."""
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        logger.info("PollingLoop stopping")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("PollingLoop task raised during stop")

    async def _run(self) -> None:
        logger.info("PollingLoop started")
        while True:
            now = time.monotonic()
            to_push = [sv for sv in list(self._svs) if sv._needs_push(now)]
            if to_push:
                asyncio.get_event_loop().create_task(self._batch_push(now, to_push))
            asyncio.get_event_loop().create_task(self._batch_fetch(now))
            await asyncio.sleep(self._tick)

    async def push_sv(self, sv: SyncedVariable) -> None:
        """Push ``sv`` to the server immediately; raise :class:`SyncError` on failure."""
        async with self._push_lock(sv):
            for attempt in range(_PUSH_MAX_ATTEMPTS):
                # Same as batch: clear pending so concurrent sets during the request are picked up next.
                sv._pending_push = False
                write_spec, commit_fn = _write_spec_for_sv(sv, expected_version=self._versions.get(id(sv), 0))
                vars_payload = {sv._name: write_spec}

                try:
                    logger.debug(f"PollingLoop explicit push: {vars_payload}")
                    resp = await self._client.post(
                        "/batch-write",
                        content=msgpack.packb({"vars": vars_payload}),
                        headers={"Content-Type": "application/msgpack"},
                    )
                except Exception as exc:
                    raise SyncError(f"SyncedVariable[{sv._name}] push failed: {exc}") from exc

                if resp.status_code != 200:
                    raise SyncError(f"SyncedVariable[{sv._name}] push failed: HTTP {resp.status_code}")

                try:
                    results = msgpack.unpackb(resp.content, raw=False).get("results", {})
                except Exception as exc:
                    raise SyncError(f"SyncedVariable[{sv._name}] push failed: invalid response") from exc

                entry = results.get(sv._name) or {}
                if entry.get("status") == "ok":
                    new_ver = entry.get("version")
                    if new_ver is not None:
                        self._versions[id(sv)] = int(new_ver)
                    else:
                        self._versions[id(sv)] = self._versions.get(id(sv), 0) + 1
                    push_time = time.monotonic()
                    if commit_fn:
                        commit_fn()
                    sv._last_push = push_time
                    return

                err = entry.get("error") or ""
                if _is_version_mismatch_error(err) and attempt < _PUSH_MAX_ATTEMPTS - 1:
                    logger.debug(
                        f"SyncedVariable[{sv._name}] push version mismatch, pull+retry "
                        f"({attempt + 1}/{_PUSH_MAX_ATTEMPTS}): {err}"
                    )
                    try:
                        await self._pull_sv_impl(sv)
                    except Exception as exc:
                        raise SyncError(
                            f"SyncedVariable[{sv._name}] push failed after version mismatch; pull failed: {exc}"
                        ) from exc
                    continue

                detail = f": {err}" if err else ""
                raise SyncError(f"SyncedVariable[{sv._name}] push failed{detail}")

    async def pull_sv(self, sv: SyncedVariable) -> None:
        """Pull server state for ``sv`` immediately; raise :class:`SyncError` on failure."""
        async with self._push_lock(sv):
            await self._pull_sv_impl(sv)

    async def _pull_sv_impl(self, sv: SyncedVariable) -> None:
        """POST /poll and apply.

        Call only while holding ``_push_lock(sv)`` (or from ``push_sv`` retry, same lock).
        """
        now = time.monotonic()
        vars_spec = {
            sv._name: {
                "service": sv.sync_service,
                "version": self._versions.get(id(sv), 0),
            }
        }

        try:
            resp = await self._client.post(
                "/poll",
                content=msgpack.packb({"vars": vars_spec}),
                headers={"Content-Type": "application/msgpack"},
            )
        except Exception as exc:
            raise SyncError(f"SyncedVariable[{sv._name}] pull failed: {exc}") from exc

        if resp.status_code != 200:
            raise SyncError(f"SyncedVariable[{sv._name}] pull failed: HTTP {resp.status_code}")

        try:
            data = msgpack.unpackb(resp.content, raw=False)
        except Exception as exc:
            raise SyncError(f"SyncedVariable[{sv._name}] pull failed: invalid response") from exc

        sv._next_poll_at = now + sv.pull_frequency
        changed: dict[str, Any] = data.get("changed", {})
        update = changed.get(sv._name)
        self._apply_poll_change(sv, update, strict=True)

    def _apply_poll_change(self, sv: SyncedVariable, update: dict[str, Any] | None, *, strict: bool) -> None:
        """Apply one poll entry for ``sv``. In non-strict mode, patch conflicts log and resync."""
        if update is None:
            return

        new_version: int = int(update.get("version", 0))

        patch = update.get("patch")
        raw_value = update.get("value")

        if patch is not None:
            if RUN_FLAGS.sync_patches.isOn():
                base = (
                    dict(sv._value)
                    if isinstance(sv._value, SyncedDict)
                    else (list(sv._value) if isinstance(sv._value, SyncedList) else sv._value)
                )
                try:
                    sv._apply(_apply_patch(base, patch))
                    self._versions[id(sv)] = new_version
                except Exception as exc:
                    if strict:
                        raise SyncError(f"SyncedVariable[{sv._name}] pull failed applying patch: {exc}") from exc
                    # If the update also carries a full value, apply it instead
                    # of waiting for the next poll cycle.
                    if raw_value is not None:
                        logger.debug(
                            f"SyncedVariable[{sv._name}] patch conflict — "
                            f"falling back to full value in same response: {exc}"
                        )
                        sv._apply(raw_value)
                        self._versions[id(sv)] = new_version
                    else:
                        logger.warning(
                            f"SyncedVariable[{sv._name}] patch conflict — "
                            f"resetting version to force full resync: {exc}"
                        )
                        self._versions[id(sv)] = 0
            else:
                self._versions[id(sv)] = 0
        elif raw_value is not None:
            sv._apply(raw_value)
            self._versions[id(sv)] = new_version

    async def _batch_push(self, now: float, svs: list[SyncedVariable]) -> None:
        """Issue a single POST /batch-write with all pending variables."""
        if not svs:
            return
        seen: set[int] = set()
        unique_svs: list[SyncedVariable] = []
        for sv in svs:
            k = id(sv)
            if k not in seen:
                seen.add(k)
                unique_svs.append(sv)
        unique_svs.sort(key=id)
        locks = [self._push_lock(sv) for sv in unique_svs]
        for lock in locks:
            await lock.acquire()
        try:
            vars_payload: dict[str, dict] = {}
            commit_fns: dict[str, Any] = {}

            for sv in svs:
                # Clear the pending flag upfront so any set() that arrives during
                # the HTTP request is captured by the next tick.
                sv._pending_push = False

                write_spec, commit_fn = _write_spec_for_sv(sv, expected_version=self._versions.get(id(sv), 0))
                vars_payload[sv._name] = write_spec
                commit_fns[sv._name] = (sv, commit_fn)

            try:
                logger.debug(f"PollingLoop batch push: {vars_payload}")
                resp = await self._client.post(
                    "/batch-write",
                    content=msgpack.packb({"vars": vars_payload}),
                    headers={"Content-Type": "application/msgpack"},
                )
                if resp.status_code != 200:
                    logger.error(f"PollingLoop batch push failed: HTTP {resp.status_code}")
                    return
                results = msgpack.unpackb(resp.content, raw=False).get("results", {})
            except Exception as exc:
                logger.exception("PollingLoop batch push error:")
                return

            push_time = time.monotonic()
            for name, (sv, commit_fn) in commit_fns.items():
                res = results.get(name, {})
                if res.get("status") != "ok":
                    continue
                new_ver = res.get("version")
                if new_ver is not None:
                    self._versions[id(sv)] = int(new_ver)
                else:
                    self._versions[id(sv)] = self._versions.get(id(sv), 0) + 1
                if commit_fn:
                    commit_fn()
                sv._last_push = push_time
        finally:
            for lock in reversed(locks):
                lock.release()

    async def _batch_fetch(self, now: float) -> None:
        """Issue a single POST /poll with all variables due for a fetch."""
        due = [sv for sv in list(self._svs) if now >= sv._next_poll_at]
        if not due:
            return

        vars_spec = {
            sv._name: {
                "service": sv.sync_service,
                "version": self._versions.get(id(sv), 0),
            }
            for sv in due
        }

        try:
            resp = await self._client.post(
                "/poll",
                content=msgpack.packb({"vars": vars_spec}),
                headers={"Content-Type": "application/msgpack"},
            )
            if resp.status_code != 200:
                logger.error(f"PollingLoop batch fetch failed: HTTP {resp.status_code}")
                return
            data = msgpack.unpackb(resp.content, raw=False)
        except Exception as exc:
            logger.exception("PollingLoop batch fetch error:")
            return

        changed: dict[str, Any] = data.get("changed", {})

        for sv in sorted(due, key=id):
            async with self._push_lock(sv):
                sv._next_poll_at = now + sv.pull_frequency
                update = changed.get(sv._name)
                self._apply_poll_change(sv, update, strict=False)


# ── SyncedVariable ─────────────────────────────────────────────────────────────


class SyncedVariable(Generic[T]):
    """Variable kept in sync with the remote server via the PollingLoop.

    The ``variable_id`` is used *directly* as the Redis key on the bridge
    (after a ``sync:`` prefix added by the backend).  To scope a variable
    to a training run, include the run prefix in the id::

        f"{sync_run_sync_prefix(run_id)}/node_registry"

    Args:
        variable_id:    Full bridge key for this variable, e.g.
                        ``"run-abc/node_registry"`` or ``"global/phase"``.
        default:        Initial local value. Pydantic BaseModel instances are
                        stored as-is and reconstructed via ``model_validate``
                        on incoming updates.
        sync_service:   Backend: ``"redis"``.
        push_on_set:    Push to server when ``.value`` is assigned.
        pull_frequency: Seconds between background pulls. Default 2.0.
        push_frequency: Minimum seconds between pushes (coalesces rapid writes).
                        ``None`` = no rate limiting.
        polling_loop:   Custom PollingLoop; defaults to the shared class-level loop.
        on_update:      Optional callback invoked whenever the variable receives
                        an update from the server.  Signature: ``(new_value: T) -> None``.
                        Can be a plain function or a coroutine function; coroutines
                        are scheduled as tasks on the running event loop.
    """

    # Shared polling loop for all instances that don't specify their own.
    # Must be set (e.g. SyncedVariable.polling_loop = PollingLoop(server_url))
    # before creating SyncedVariable instances without an explicit polling_loop.
    polling_loop: PollingLoop | None = PollingLoop(
        server_url=common_settings.BRIDGE_URL, tick=miner_settings.SYNC_POLL_TICK
    )

    def __init__(
        self,
        variable_id: str,
        default: T = None,  # type: ignore[assignment]
        sync_service: str = "redis",
        push_on_set: bool = True,
        pull_frequency: float = 2.0,
        push_frequency: float | None = None,
        polling_loop: PollingLoop | None = None,
        on_update: Callable[[T], Any] | None = None,
    ) -> None:
        self._name = variable_id
        self.default = default
        self.sync_service = sync_service
        self.push_on_set = push_on_set
        self.pull_frequency = pull_frequency
        self.push_frequency = push_frequency
        self._model_cls: type | None = type(default) if _is_pydantic(default) else None
        self._on_update = on_update

        self._value: Any = copy.deepcopy(default) if isinstance(default, _COMPLEX_TYPES) else default
        self._next_poll_at: float = 0.0
        self._last_push: float = 0.0
        self._pending_push: bool = False

        loop = polling_loop or SyncedVariable.polling_loop
        if loop is None:
            raise RuntimeError(
                "No PollingLoop configured. Either pass polling_loop= or set "
                "SyncedVariable.polling_loop = PollingLoop(server_url) before creating instances."
            )
        self._polling_loop = loop
        try:
            asyncio.get_running_loop()
            loop.register(self)
        except RuntimeError:
            logger.error("No running event loop found, can't register SyncedVariable")
            pass

    def rebind_namespace(self, new_prefix: str, bare_name: str | None = None) -> None:
        """Switch this variable to a new run namespace and reset local state.

        The bridge key becomes ``"{new_prefix}/{bare_name}"``.  If *bare_name*
        is ``None`` the last ``/``-segment of the current ``_name`` is reused.

        .. note::

           The local value is cleared but ``_pending_push`` is deliberately
           **not** set.  Callers are expected to populate the local value
           (e.g. re-register their own node, stamp fields) and then
           ``await push()`` explicitly.  This prevents the background
           PollingLoop tick from racing ahead and pushing an empty/partial
           value before the caller has finished updating it.
        """
        if bare_name is None:
            bare_name = self._name.rsplit("/", 1)[-1]
        new_name = f"{new_prefix}/{bare_name}"
        if self._name == new_name:
            return
        self._name = new_name
        self._polling_loop._versions.pop(id(self), None)
        default = self.default
        if isinstance(default, NodeRegistry):
            self._value = NodeRegistry()
        elif isinstance(default, SyncedDict):
            self._value = type(default)({})
        elif isinstance(default, SyncedList):
            self._value = type(default)([])
        elif isinstance(default, _COMPLEX_TYPES):
            self._value = copy.deepcopy(default)
        else:
            self._value = copy.deepcopy(default) if default is not None else None
        # Do NOT set _pending_push — caller must push() after populating.
        self._next_poll_at = 0.0

    # Backward-compat alias for callers that haven't been updated yet.
    def rebind_sync_groups(self, new_groups: list[str]) -> None:
        """Deprecated — prefer :meth:`rebind_namespace`."""
        self.rebind_namespace(new_groups[0])

    # ── Value access ───────────────────────────────────────────────────────────

    @property
    def value(self) -> T:
        return self._value  # type: ignore[return-value]

    @value.setter
    def value(self, v: T) -> None:
        self._value = v
        if self.push_on_set:
            logger.debug(f"SyncedVariable[{self._name}] pushing value: {v}")
            self._pending_push = True

    async def push(self) -> None:
        """Push the current value to the server and wait until it succeeds.

        Raises:
            SyncError: If the HTTP request fails or the server reports an error.
        """
        await self._polling_loop.push_sv(self)

    async def pull(self) -> None:
        """Fetch the latest value from the server and wait until the round-trip completes.

        Raises:
            SyncError: If the HTTP request fails, the response is invalid, or a
                patch cannot be applied (when using strict explicit pull).
        """
        await self._polling_loop.pull_sv(self)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def reconstruct(self, raw: Any) -> Any:
        if self._model_cls is not None and isinstance(raw, dict):
            return self._model_cls.model_validate(raw)
        return raw

    def _needs_push(self, now: float) -> bool:
        can_push = self.push_frequency is None or now - self._last_push >= self.push_frequency
        if not can_push:
            return False
        if self._pending_push:
            return True
        return isinstance(self._value, _COMPLEX_TYPES) and self._value.is_dirty()

    def _apply(self, raw: Any) -> None:
        current = self._value
        if isinstance(current, SyncedDict) and isinstance(raw, dict):
            current.apply_full_value(raw)
        elif isinstance(current, SyncedList) and isinstance(raw, list):
            current.apply_full_value(raw)
        else:
            self._value = self.reconstruct(raw)

        if self._on_update is not None:
            try:
                result = self._on_update(self._value)
                if asyncio.iscoroutine(result):
                    asyncio.get_event_loop().create_task(result)
            except Exception as exc:
                logger.exception(f"SyncedVariable[{self._name}] on_update error:")
