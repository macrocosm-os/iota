"""
PeerConnection: manages a single QUIC connection to a remote Iroh peer.

Handles health detection (via the async ``closed()`` watcher), lazy
connection establishment, transparent reconnection, and stream opening.
"""

from __future__ import annotations

import asyncio

from iroh import BiStream, Connection, Endpoint, NodeAddr, PublicKey, SendStream
from loguru import logger


class PeerConnection:
    """
    Wraps a single iroh Connection to a remote peer, handling health
    checks and transparent reconnection.

    A PeerConnection is identified by (node_id, protocol_id).  It lazily
    establishes the underlying QUIC connection on first use and will
    automatically reconnect if the connection is found to be closed.
    Callers obtain streams via `open_uni()` / `open_bi()` without
    needing to manage the raw Connection lifecycle.

    Health detection uses the async ``Connection.closed()`` future which
    resolves when QUIC detects the connection is gone (including idle
    timeouts and transport errors) — unlike ``close_reason()`` which only
    reports clean CLOSE frames and misses silently-dead connections.
    """

    def __init__(
        self,
        node_id: str,
        protocol_id: bytes,
        endpoint: Endpoint,
        relay_url: str | None = None,
        direct_addresses: list[str] | None = None,
    ):
        self._node_id = node_id
        self._protocol_id = protocol_id
        self._endpoint = endpoint
        self._relay_url = relay_url
        self._direct_addresses = list(direct_addresses or [])
        self._conn: Connection | None = None
        self._closed_reason: str | None = None
        self._closed_watcher: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    # ── properties ───────────────────────────────────────────────────

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def protocol_id(self) -> bytes:
        return self._protocol_id

    @property
    def cache_key(self) -> str:
        """Unique key suitable for use in connection caches."""
        return self._node_id + self._protocol_id.decode()

    # ── health ───────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        """Return True if the underlying connection is open.

        Uses the result of the background ``closed()`` watcher so that
        silently-dead connections (no CLOSE frame) are detected as soon
        as QUIC notices the failure.
        """
        if self._conn is None:
            return False
        if self._closed_reason is not None:
            return False
        # Fallback: also check close_reason() for immediate CLOSE frames
        # that may arrive before the watcher task runs.
        return self._conn.close_reason() is None

    def close_reason(self) -> str | None:
        """Return the close reason, or None if the connection is still open."""
        if self._conn is None:
            return "not connected"
        if self._closed_reason is not None:
            return self._closed_reason
        return self._conn.close_reason()

    # ── connection lifecycle ─────────────────────────────────────────

    def _start_closed_watcher(self) -> None:
        """Spawn a background task that awaits ``conn.closed()`` and marks
        this PeerConnection dead the instant QUIC detects the failure."""
        conn = self._conn
        if conn is None:
            return

        async def _watch() -> None:
            try:
                reason = await conn.closed()
                self._closed_reason = reason or "connection closed"
                logger.debug(
                    f"Connection to {self._node_id[:16]}... closed (detected by watcher): {self._closed_reason}"
                )
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self._closed_reason = f"watcher error: {exc}"
                logger.debug(f"Closed-watcher for {self._node_id[:16]}... failed: {exc}")

        self._closed_watcher = asyncio.create_task(_watch(), name=f"closed-watcher-{self._node_id[:16]}")

    async def connect(self, timeout: float = 5.0) -> Connection:
        """Return the live connection, reconnecting if necessary.

        Args:
            timeout: Maximum seconds to wait for the QUIC connection to
                     establish.  Defaults to 5s to avoid hanging on stale
                     routes or unreachable peers.

        Raises:
            asyncio.TimeoutError: If the connection attempt exceeds *timeout*.
        """
        if self.is_alive():
            return self._conn

        async with self._lock:
            # Re-check under lock
            if self.is_alive():
                return self._conn

            # Cancel previous watcher if any
            if self._closed_watcher is not None:
                self._closed_watcher.cancel()
                self._closed_watcher = None

            if self._conn is not None:
                logger.debug(f"Connection to {self._node_id[:16]}... closed: {self.close_reason()}")

            self._closed_reason = None
            receiver_key = PublicKey.from_string(self._node_id)
            connect_addr = NodeAddr(
                node_id=receiver_key,
                derp_url=self._relay_url,
                addresses=list(self._direct_addresses),
            )
            self._conn = await asyncio.wait_for(
                self._endpoint.connect(connect_addr, self._protocol_id),
                timeout=timeout,
            )
            self._start_closed_watcher()
            logger.debug(f"Opened connection to {self._node_id[:16]}...")
            return self._conn

    async def open_uni(self, timeout: float = 5.0) -> SendStream:
        """Open a unidirectional send stream, connecting first if needed."""
        conn = await self.connect(timeout=timeout)
        return await asyncio.wait_for(conn.open_uni(), timeout=timeout)

    async def open_bi(self, timeout: float = 5.0) -> BiStream:
        """Open a bidirectional stream, connecting first if needed."""
        conn = await self.connect(timeout=timeout)
        return await asyncio.wait_for(conn.open_bi(), timeout=timeout)

    def close(self) -> None:
        """Discard the underlying connection (does not call conn.close())."""
        if self._closed_watcher is not None:
            self._closed_watcher.cancel()
            self._closed_watcher = None
        self._conn = None
        self._closed_reason = None

    def __repr__(self) -> str:
        status = "alive" if self.is_alive() else "closed"
        return f"<PeerConnection node={self._node_id[:16]}... proto={self._protocol_id!r} {status}>"
