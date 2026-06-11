from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import httpx
from loguru import logger

from miner.sync_v2.synced_variable import Lock, SyncedVariableV2, _bridge_request

if TYPE_CHECKING:
    from miner.sync_v2.synced_dictionary import SyncedDictionary


class _SyncCollection(Protocol):
    async def wildcard_fetch(self, path: str) -> dict:
        ...

    async def push_dirty(self) -> None:
        ...

    async def fetch_all(self) -> dict:
        ...


@dataclass
class _CollectionPollEntry:
    """One synced collection registered for background sync."""

    key: str
    collection: _SyncCollection
    # If not ``None``, pull via ``wildcard_fetch(wildcard_path)``; if ``None``, pull via ``fetch_all()``.
    wildcard_path: str | None
    after_pull: Any
    pull_frequency: float
    push_frequency: float | None
    last_pull: float = field(default=-1.0)
    last_push: float = field(default=-1.0)


class VariableManager:
    """Drives background push/pull for :class:`SyncedVariableV2` and synced collections."""

    def __init__(self, url: str, *, timeout: float = 30.0) -> None:
        self._http = httpx.AsyncClient(base_url=url, timeout=timeout)
        self.registered_vars: dict[str, SyncedVariableV2] = {}
        self._collection_entries: dict[str, _CollectionPollEntry] = {}
        self._task: asyncio.Task | None = None

    def _ensure_started(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._poll_loop())

    async def create_var(
        self,
        run_id: str,
        name: str,
        *,
        var_type: str,
        default: Any,
        write_rule: str = "LOCK",
        require_fetch: bool = True,
        pull_frequency: float | None = None,
        push_frequency: float | None = None,
    ) -> SyncedVariableV2:
        """Create and register a variable on the bridge, optionally enrolling it in background sync."""
        var = await SyncedVariableV2.create(
            client=self._http,
            run_id=run_id,
            name=name,
            var_type=var_type,
            default=default,
            write_rule=write_rule,
            require_fetch=require_fetch,
        )
        if pull_frequency is not None or push_frequency is not None:
            var._pull_frequency = pull_frequency if pull_frequency is not None else 1e10
            var._push_frequency = push_frequency if push_frequency is not None else 1e10
            self.registered_vars[var.var_id] = var
            self._ensure_started()
        return var

    def register_collection(
        self,
        key: str,
        collection: _SyncCollection,
        *,
        wildcard_path: str | None = None,
        after_pull: Any = None,
        pull_frequency: float = 1e10,
        push_frequency: float | None = None,
    ) -> None:
        """Register an object for background pull/push in the poll loop.

        *collection* must expose ``wildcard_fetch(path)`` and (if
        *push_frequency* is set) ``push_dirty()`` async instance methods.
        """
        self._collection_entries[key] = _CollectionPollEntry(
            key=key,
            collection=collection,
            wildcard_path=wildcard_path,
            after_pull=after_pull,
            pull_frequency=pull_frequency,
            push_frequency=push_frequency,
        )
        self._ensure_started()

    def synced_dict(
        self,
        run_id: str,
        name: str,
        *,
        rule: str = "LWW",
        pull_frequency: float | None = None,
        push_frequency: float | None = None,
        on_pull: Any = None,
    ) -> "SyncedDictionary":
        """Create a SyncedDictionary, optionally enrolled in background sync.

        *on_pull* is called with the :class:`SyncedDictionary` instance after
        each background or explicit pull updates the local cache.
        """
        from miner.sync_v2.synced_dictionary import SyncedDictionary

        return SyncedDictionary(
            run_id,
            name,
            manager=self,
            rule=rule,
            pull_frequency=pull_frequency,
            push_frequency=push_frequency,
            on_pull=on_pull,
        )

    def unwatch_collection(self, key: str) -> None:
        """Stop background sync for a collection registered under key."""
        self._collection_entries.pop(key, None)

    def lock(self, *vars: SyncedVariableV2, ttl_ms: int = 30_000, wait_ms: int = 5_000) -> Lock:
        """Create a Lock for one or more variables using this manager's HTTP client."""
        return Lock(list(vars), client=self._http, ttl_ms=ttl_ms, wait_ms=wait_ms)

    async def stop(self) -> None:
        """Cancel the background polling loop and wait for it to exit."""
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._collection_entries.clear()
        await self._http.aclose()

    async def _poll_loop(self) -> None:
        """Background loop: pull/push variables and poll registered collections."""
        while True:
            try:
                vars_to_push, vars_to_pull = await self.get_vars_to_sync()

                if vars_to_pull:
                    await self._batch_pull(vars_to_pull)

                if vars_to_push:
                    await self._batch_push(vars_to_push)

                await self._poll_registered_collections()

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(f"VariableManager poll loop error: {exc}")

            await asyncio.sleep(self.get_min_interval())

    async def _poll_registered_collections(self) -> None:
        now = time.time()
        for entry in list(self._collection_entries.values()):
            try:
                if now - entry.last_pull > entry.pull_frequency:
                    if entry.wildcard_path is not None:
                        raw = await entry.collection.wildcard_fetch(entry.wildcard_path)
                    else:
                        raw = await entry.collection.fetch_all()
                    if entry.after_pull is not None:
                        res = entry.after_pull(raw)
                        if inspect.isawaitable(res):
                            await res
                    entry.last_pull = now

                if entry.push_frequency is not None and now - entry.last_push > entry.push_frequency:
                    await entry.collection.push_dirty()
                    entry.last_push = now
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"VariableManager collection {entry.key!r} sync failed: {exc}")

    async def _batch_pull(self, vars_to_pull: list[SyncedVariableV2]) -> None:
        """POST /get for all variables that are due for a pull."""
        if not vars_to_pull:
            return
        var_ids = [var.var_id for var in vars_to_pull]
        try:
            resp = await _bridge_request(self._http, "POST", "/get", json={"var_ids": var_ids})
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"VariableManager batch pull failed: {exc}")
            return

        # Intentionally separate from apply_get_response: looks up by var_id (not rel name)
        # and sets _last_pulled on each variable.
        var_map = {var.var_id: var for var in vars_to_pull}
        now = time.time()
        for entry in data.get("variables", []):
            var = var_map.get(entry["var_id"])
            if var is None:
                continue
            var._cached_value = entry.get("value")
            var.version = entry.get("version", var.version)
            var.updated_at = entry.get("metadata", {}).get("updated_at", var.updated_at)
            var._last_pulled = now
            var._has_fetched = True

        for err in data.get("errors", []):
            logger.warning(f"VariableManager pull error for {err['var_id']!r}: {err.get('error')}")

    async def _batch_push(self, vars_to_push: list[SyncedVariableV2]) -> None:
        """POST /set for LWW and CAS variables that are dirty.

        LOCK variables are skipped — callers must acquire a lock explicitly
        and call set_value() directly.
        """
        updates = []
        ready: list[SyncedVariableV2] = []
        for var in vars_to_push:
            if var.write_rule == "LOCK":
                logger.warning(
                    f"VariableManager: skipping LOCK variable {var.name!r} in background push "
                    "(acquire a lock and call set_value() explicitly)"
                )
                continue
            update: dict[str, Any] = {"var_id": var.var_id, "value": var._cached_value}
            if var.write_rule == "CAS":
                update["current_version"] = var.version
            updates.append(update)
            ready.append(var)

        if not updates:
            return

        try:
            resp = await _bridge_request(self._http, "POST", "/set", json={"updates": updates})
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as exc:
            logger.warning(f"VariableManager batch push failed: {exc}")
            return

        result_map = {r["var_id"]: r for r in results}
        now = time.time()
        for var in ready:
            result = result_map.get(var.var_id, {})
            if result.get("status") == "error":
                logger.warning(f"VariableManager push error for {var.name!r}: {result.get('error')}")
                continue
            var.version = result.get("version", var.version)  # don't guess; keep old version if absent
            var._needs_push = False
            var._last_pushed = now

    def get_min_interval(self) -> float:
        """Return the smallest poll/push interval across variables and collections."""
        intervals: list[float] = []
        if self.registered_vars:
            intervals.append(min(var._pull_frequency for var in self.registered_vars.values()))
            intervals.append(min(var._push_frequency for var in self.registered_vars.values()))
        for ent in self._collection_entries.values():
            intervals.append(ent.pull_frequency)
            if ent.push_frequency is not None:
                intervals.append(ent.push_frequency)
        return min(intervals) if intervals else 1.0

    async def get_vars_to_sync(self) -> tuple[list[SyncedVariableV2], list[SyncedVariableV2]]:
        """Return (vars_to_push, vars_to_pull) based on timestamps and dirty flags."""
        now = time.time()
        vars_to_pull: list[SyncedVariableV2] = []
        vars_to_push: list[SyncedVariableV2] = []
        for var in self.registered_vars.values():
            if now - var._last_pulled > var._pull_frequency:
                vars_to_pull.append(var)
            if var._needs_push and now - var._last_pushed > var._push_frequency:
                vars_to_push.append(var)
        return vars_to_push, vars_to_pull
