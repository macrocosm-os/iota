from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

import httpx
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr


async def _bridge_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    retries: int = 3,
    **kwargs: Any,
) -> httpx.Response:
    """Make a bridge HTTP request with automatic retry on transient errors.

    Retries on network/transport errors and 5xx responses with exponential backoff.
    Returns 2xx and 4xx responses to the caller; raises after exhausting retries.
    """
    last_exc: Exception | None = None
    caller = getattr(client, method.lower())
    for attempt in range(retries):
        try:
            resp = await caller(path, **kwargs)
            if resp.status_code < 500:
                return resp  # 2xx or 4xx — caller handles with raise_for_status()
            last_exc = httpx.HTTPStatusError(f"Server error {resp.status_code}", request=resp.request, response=resp)
        except httpx.TransportError as exc:
            last_exc = exc
        if attempt < retries - 1:
            await asyncio.sleep(0.5 * (2**attempt))
    assert last_exc is not None
    raise last_exc


class SyncedVariableV2(BaseModel):
    run_id: str
    name: str
    # Bridge type string: "str" | "int" | "float" | "bool" | "dict" | "list" | "none"
    var_type: str
    version: int = Field(default=0)
    updated_at: str = Field(default="")
    write_rule: Literal["LOCK", "CAS", "LWW"] = Field(default="LOCK")
    # When True, get_cached_value() raises until fetch_value() has been called at least once.
    require_fetch: bool = Field(default=True)
    # Value sent on first POST /register; used by :meth:`create` and :meth:`register`.
    default: Any = Field(exclude=True, repr=False)

    _cached_value: Any = PrivateAttr(default=None)
    _has_fetched: bool = PrivateAttr(default=False)
    _pull_frequency: float = PrivateAttr(default=1e10)
    _push_frequency: float = PrivateAttr(default=1e10)
    _last_pulled: float = PrivateAttr(default=-1.0)
    _last_pushed: float = PrivateAttr(default=-1.0)
    _needs_push: bool = PrivateAttr(default=False)
    _client: Any = PrivateAttr(default=None)  # set by VariableManager or create()
    _bridge_registered: bool = PrivateAttr(default=False)

    async def aclose(self) -> None:
        """Close the underlying HTTP client and release connections."""
        await self._client.aclose()

    @property
    def var_id(self) -> str:
        """Full bridge key: ``{run_id}/{name}``."""
        return f"{self.run_id}/{self.name}"

    @classmethod
    async def create(cls, *, client: httpx.AsyncClient, **data: Any) -> SyncedVariableV2:
        """Construct a variable and register it on the bridge. Raises on registration failure."""
        var = cls(**data)
        var._client = client
        await var.register(var.default)
        return var

    async def register(self, default_value: Any) -> None:
        """Register this variable on the bridge with a default value. Idempotent."""
        resp = await _bridge_request(
            self._client,
            "POST",
            "/register",
            json={
                "variables": [
                    {
                        "var_id": self.var_id,
                        "default_value": default_value,
                        "var_type": self.var_type,
                        "write_rule": self.write_rule,
                    }
                ]
            },
        )
        resp.raise_for_status()
        result = resp.json().get("results", [{}])[0]
        if result.get("status") == "error":
            raise RuntimeError(f"register {self.var_id!r} failed: {result.get('error')}")
        self.version = result.get("version", 1)
        status = result.get("status")
        if status == "created":
            self._cached_value = default_value
        self._bridge_registered = True

    async def fetch_value(self) -> Any:
        """Pull the current value from the bridge and update the local cache."""
        resp = await _bridge_request(self._client, "POST", "/get", json={"var_ids": [self.var_id]})
        resp.raise_for_status()
        data = resp.json()
        errors = data.get("errors", [])
        if errors:
            raise RuntimeError(f"fetch {self.var_id!r} failed: {errors[0].get('error')}")
        variables = data.get("variables", [])
        if not variables:
            logger.warning(f"fetch {self.var_id!r}: bridge returned no variables; returning cached value")
            self._has_fetched = True
            return self._cached_value
        entry = variables[0]
        self._cached_value = entry.get("value")
        self.version = entry.get("version", self.version)
        self.updated_at = entry.get("metadata", {}).get("updated_at", self.updated_at)
        self._last_pulled = time.time()
        self._has_fetched = True
        return self._cached_value

    async def set_value(
        self,
        value: Any,
        lock: Lock | None = None,
        current_version: int | None = None,
    ) -> None:
        """Write a new value to the bridge, enforcing write_rule constraints.

        - LOCK: ``lock`` must be an acquired :class:`Lock` context manager.
        - CAS:  ``current_version`` must equal the server's current version.
        - LWW:  no concurrency requirements.
        """
        if self.write_rule == "LOCK":
            if lock is None or lock.token_for(self.var_id) is None:
                raise ValueError(f"An acquired Lock must be passed for write_rule=LOCK (variable {self.name!r})")
        if self.write_rule == "CAS":
            if current_version is None:
                raise ValueError(f"current_version must be provided for write_rule=CAS (variable {self.name!r})")

        update: dict[str, Any] = {"var_id": self.var_id, "value": value}
        if lock is not None:
            update["lock_token"] = lock.token_for(self.var_id)
        if current_version is not None:
            update["current_version"] = current_version

        resp = await _bridge_request(self._client, "POST", "/set", json={"updates": [update]})
        resp.raise_for_status()
        result = resp.json().get("results", [{}])[0]
        if result.get("status") == "error":
            raise RuntimeError(f"set {self.var_id!r} failed: {result.get('error')}")

        self.version = result.get("version", self.version)  # don't guess; keep old version if absent
        self._cached_value = value
        self._needs_push = False
        self._last_pushed = time.time()

    def get_cached_value(self) -> Any:
        """Return the locally cached value (synchronous; no network).

        Raises if ``require_fetch=True`` and :meth:`fetch_value` has never been called.
        """
        if self.require_fetch and not self._has_fetched:
            raise RuntimeError(
                f"Must call fetch_value() before get_cached_value() on {self.name!r} (require_fetch=True)"
            )
        return self._cached_value

    def set(self, value: Any) -> None:
        """Update the local cache and flag this variable for the next push."""
        if self.write_rule == "LOCK":
            raise ValueError(
                f"set() is not allowed for write_rule=LOCK (variable {self.name!r}); "
                "acquire a lock and call set_and_push() or set_value() explicitly"
            )
        self._cached_value = value
        self._needs_push = True

    async def set_and_push(
        self,
        value: Any,
        *,
        lock: Lock | None = None,
        current_version: int | None = None,
    ) -> None:
        """Update the local cache and immediately push the value to the bridge.

        For CAS variables, the current cached version is used when
        ``current_version`` is not provided.
        """
        if self.write_rule == "LOCK":
            await self.set_value(value, lock=lock, current_version=current_version)
            return
        if self.write_rule == "CAS" and current_version is None:
            current_version = self.version
        self.set(value)
        await self.set_value(value, lock=lock, current_version=current_version)


LockTarget = str | SyncedVariableV2 | list[str | SyncedVariableV2]


def _resolve_lock_targets(
    targets: LockTarget,
    client: httpx.AsyncClient | None,
) -> tuple[list[str], httpx.AsyncClient]:
    items: list[str | SyncedVariableV2]
    if isinstance(targets, (str, SyncedVariableV2)):
        items = [targets]
    else:
        if not targets:
            raise ValueError("Lock targets list must not be empty")
        items = targets

    var_ids: list[str] = []
    resolved_client = client

    for item in items:
        if isinstance(item, SyncedVariableV2):
            var_ids.append(item.var_id)
            if resolved_client is None:
                resolved_client = item._client
            elif resolved_client is not item._client:
                raise ValueError("All SyncedVariableV2 instances must share the same HTTP client")
        elif isinstance(item, str):
            var_ids.append(item)

    if resolved_client is None:
        raise ValueError("client is required when locking by var_id string")

    return var_ids, resolved_client


class Lock:
    """Async context manager that acquires and releases bridge v2 distributed lock(s).

    Pass bridge keys, :class:`SyncedVariableV2` instances, or a mix::

        async with Lock(config) as lock: ...
        async with Lock([step_var, lr_var]) as lock: ...
        async with Lock(["run/a", "run/b"], client) as lock: ...

    Each variable gets its own token; use :meth:`token_for` when passing tokens to
    :meth:`SyncedVariableV2.set_value` (required when this lock covers more than one id).
    """

    def __init__(
        self,
        targets: LockTarget,
        client: httpx.AsyncClient | None = None,
        ttl_ms: int = 30_000,
        wait_ms: int = 5_000,
    ):
        self.var_ids, self._client = _resolve_lock_targets(targets, client)
        self.ttl_ms = ttl_ms
        self.wait_ms = wait_ms
        self._tokens: dict[str, str] = {}

    @property
    def var_id(self) -> str:
        """First locked variable id (same as ``var_ids[0]``)."""
        return self.var_ids[0]

    def token_for(self, var_id: str) -> str | None:
        """Lock token for *var_id*, if that id was locked in this context."""
        return self._tokens.get(var_id)

    @property
    def lock_token(self) -> str | None:
        """Set only when exactly one variable is locked; otherwise use :meth:`token_for`."""
        if len(self._tokens) == 1:
            return next(iter(self._tokens.values()))
        return None

    async def _release_acquired(self) -> None:
        if not self._tokens:
            return
        payload = [{"var_id": vid, "lock_token": tok} for vid, tok in self._tokens.items()]
        try:
            await _bridge_request(self._client, "POST", "/unlock", json={"locks": payload})
        except Exception as exc:
            ids = ", ".join(repr(v) for v in self._tokens)
            logger.warning(f"Failed to release lock(s) for [{ids}]: {exc}")
        self._tokens.clear()

    async def __aenter__(self) -> Lock:
        resp = await _bridge_request(
            self._client,
            "POST",
            "/lock",
            json={"var_ids": self.var_ids, "ttl_ms": self.ttl_ms, "wait_ms": self.wait_ms},
        )
        resp.raise_for_status()
        locks_list = resp.json().get("locks", [])
        acquired: dict[str, str] = {}
        for i, var_id in enumerate(self.var_ids):
            lock = locks_list[i] if i < len(locks_list) else {}
            if lock.get("status") == "locked" and lock.get("lock_token"):
                acquired[var_id] = lock["lock_token"]
                continue
            err = lock.get("error") or lock.get("status") or "timeout"
            self._tokens = acquired
            await self._release_acquired()
            raise RuntimeError(f"Could not acquire lock for {var_id!r}: {err}")
        self._tokens = acquired
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        await self._release_acquired()
