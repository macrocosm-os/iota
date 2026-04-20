"""DistributedCounter — atomic integer counter backed by the sync server."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
from loguru import logger


class DistributedCounter:
    """A distributed integer counter with server-side atomic increment/decrement.

    Uses ``POST /counter/{service}/{key}`` for all mutations.  The server holds
    a per-key lock for the entire read-add-write cycle so concurrent increments
    from many nodes are never lost.

    The Redis key is ``{namespace}/{name}``.  Pass the run-scoped prefix
    (from :func:`sync_run_sync_prefix`) as *namespace* to isolate counters
    per training run.

    Args:
        name:          Counter name within the namespace.
        server_url:    Base URL of the sync server.
        namespace:     Key namespace prefix.  Default ``"global"``.
        sync_service:  Backend (``"redis"``).  Default ``"redis"``.
        initial:       Value to write if the counter does not yet exist.
        node_id:       Human-readable label for log messages.
        http_timeout:  Per-request HTTP timeout in seconds.  Default ``5.0``.
        poll_interval: Seconds between background polls.  Default ``2.0``.
    """

    def __init__(
        self,
        name: str,
        server_url: str,
        namespace: str = "global",
        sync_service: str = "redis",
        initial: int = 0,
        node_id: str | None = None,
        http_timeout: float = 5.0,
        poll_interval: float = 2.0,
    ) -> None:
        self._name = name
        self._server_url = server_url.rstrip("/")
        self._namespace = namespace
        self._service = sync_service
        self._initial = initial
        self._node_id = node_id or f"counter-{uuid.uuid4().hex[:8]}"
        self._http_timeout = http_timeout
        self._poll_interval = poll_interval

        self._value: int = 0
        self._version: int = 0

        self._client: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task | None = None

    @property
    def value(self) -> int:
        """Last polled value (eventually consistent snapshot)."""
        return self._value

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._server_url,
            timeout=self._http_timeout,
        )
        key_exists = await self._sync_once()
        if not key_exists and self._initial != 0:
            await self._put_value(self._initial)
        self._poll_task = asyncio.create_task(
            self._background_loop(),
            name=f"DistributedCounter[{self._node_id}].poll",
        )
        logger.info(f"[{self._node_id}] counter {self._namespace}/{self._name!r} started, value={self._value}")

    async def stop(self) -> None:
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info(f"[{self._node_id}] counter stopped, final value={self._value}")

    async def __aenter__(self) -> "DistributedCounter":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    async def increment(self, delta: int = 1) -> int:
        """Atomically add *delta* to the counter; returns the new value."""
        return await self._do_delta(delta)

    async def decrement(self, delta: int = 1) -> int:
        """Atomically subtract *delta* from the counter; returns the new value."""
        return await self._do_delta(-delta)

    async def reset(self, value: int = 0) -> int:
        """Overwrite the counter (last-write-wins); returns the written value."""
        await self._put_value(value)
        return self._value

    def _server_key(self) -> str:
        return f"{self._namespace}/{self._name}"

    async def _do_delta(self, delta: int) -> int:
        """Atomically add *delta* to the counter; returns the new value.

        Retries only on connection-setup failures (TCP connect / TLS handshake)
        because those happen before the request is sent, so retrying cannot
        double-apply the delta. Post-send errors (ReadError, ReadTimeout,
        RemoteProtocolError, HTTPStatusError) are propagated so the caller's
        error handling can decide without risking duplicated increments.
        """
        if self._client is None:
            raise RuntimeError("DistributedCounter is not started — use it as an async context manager")

        max_attempts = 3
        backoff = 0.5
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await self._client.post(
                    f"/counter/{self._service}/{self._server_key()}",
                    json={"delta": delta},
                )
                resp.raise_for_status()
                data = resp.json()
                self._value = int(data["value"])
                self._version = int(data["version"])
                return self._value
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt >= max_attempts:
                    logger.error(
                        f"[{self._node_id}] counter delta exhausted {max_attempts} attempts: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise
                logger.warning(
                    f"[{self._node_id}] counter delta attempt {attempt}/{max_attempts} failed "
                    f"({type(exc).__name__}: {exc}); retrying in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
                backoff *= 2

        raise RuntimeError("unreachable: _do_delta retry loop exited without returning or raising")

    async def _put_value(self, value: int) -> None:
        if self._client is None:
            return
        try:
            resp = await self._client.put(
                f"/vars/{self._service}/{self._server_key()}",
                json={"value": value, "is_patch": False},
            )
            if resp.status_code == 200:
                self._value = value
                self._version = int(resp.json().get("version", self._version))
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.debug(f"[{self._node_id}] put_value error: {exc}")

    async def _sync_once(self) -> bool:
        if self._client is None:
            return False
        try:
            resp = await self._client.get(f"/vars/{self._service}/{self._server_key()}")
            if resp.status_code == 200:
                entry = resp.json()
                self._value = int(entry.get("value", 0))
                self._version = int(entry.get("version", 0))
                return True
            if resp.status_code == 404:
                self._value = 0
                self._version = 0
                return False
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            logger.debug(f"[{self._node_id}] sync_once error: {exc}")
        return False

    async def _background_loop(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._sync_once()
            except Exception as exc:
                logger.debug(f"[{self._node_id}] poll error: {exc}")
