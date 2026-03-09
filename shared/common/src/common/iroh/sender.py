from __future__ import annotations

import asyncio
import time as _time
from collections import OrderedDict
from typing import Callable, TypeVar

from iroh import (
    Iroh,
    NodeOptions,
    iroh_ffi,
)
from loguru import logger
from pydantic import BaseModel

from common.iroh.cleanup import _force_free_iroh_node
from common.iroh.connection import PeerConnection
from common.iroh.monitored_node import HealthCheckResult, MonitoredNode
from common.iroh.protocol import PROTOCOL_ID_BI, PROTOCOL_ID_UNI
from common.iroh.retry import P2PRetry, P2PRetryPolicy, P2PTimeouts
from common.iroh.timings import P2POperationTimings, TimingsPhaseField
from common.iroh.serializer import Serializer, unwrap_envelope, wrap_envelope

ModelT = TypeVar("ModelT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timed(timings: P2POperationTimings | None, attr: TimingsPhaseField) -> float | None:
    """Context-manager-like helper that records elapsed time into *timings.attr*.

    Usage::

        t0 = _timed(timings, "send_duration")
        await do_work()
        _timed_end(timings, "send_duration", t0)

    We use a start/end pair rather than a real context manager because the
    inner calls are ``await`` expressions that don't nest cleanly with
    ``async with``.
    """
    if timings is not None:
        return _time.time()
    return None


def _timed_end(timings: P2POperationTimings | None, attr: TimingsPhaseField, t0: float | None) -> None:
    if timings is not None and t0 is not None:
        setattr(timings, attr, _time.time() - t0)


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------


class Sender:
    """
    P2P sender that reuses a single Iroh node and caches connections to peers.
    QUIC connections support multiplexing, so multiple streams can share one
    connection — avoiding the costly per-request DERP relay discovery and
    NAT hole-punching (~3s) on every call.

    Retry and per-phase timeout logic is handled by the embedded ``P2PRetry``
    executor, configured via *retry_policy* and *timeouts*.
    """

    def __init__(
        self,
        retry_policy: P2PRetryPolicy | None = None,
        timeouts: P2PTimeouts | None = None,
        health_check_interval: float = 30.0,
    ):
        self._node: Iroh | None = None
        self._node_lock = asyncio.Lock()
        # cache_key (node_id + protocol_id) -> PeerConnection
        self._connections: dict[str, PeerConnection] = {}
        self._conn_lock = asyncio.Lock()

        self._timeouts = timeouts or P2PTimeouts()
        self._retry_policy = retry_policy or P2PRetryPolicy()
        self._retry = P2PRetry(self._retry_policy, self._timeouts)

        self._monitored_node = MonitoredNode(
            on_unhealthy=self._on_node_unhealthy,
            check_interval=health_check_interval,
            label="sender",
        )

    # ── properties ───────────────────────────────────────────────────

    @property
    def timeouts(self) -> P2PTimeouts:
        return self._timeouts

    @property
    def retry_policy(self) -> P2PRetryPolicy:
        return self._retry_policy

    # ── node management ──────────────────────────────────────────────

    async def _get_node(self) -> Iroh:
        """Lazily create and reuse a single Iroh node."""
        if self._node is None:
            async with self._node_lock:
                if self._node is None:
                    iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
                    self._node = await Iroh.memory_with_options(NodeOptions(protocols={}))
                    self._monitored_node.set_node(self._node)
                    self._monitored_node.start_monitoring()
        return self._node

    async def _on_node_unhealthy(self, monitored: MonitoredNode, result: HealthCheckResult) -> None:
        """Reset the node when health checks detect it is unhealthy."""
        logger.warning(f"Sender node unhealthy ({result.health.value}), resetting")
        await self._reset_node()
        self._monitored_node.clear_node()

    async def _reset_node(self) -> None:
        """Shut down the iroh node and clear all connections so the next
        ``_get_node()`` call creates a fresh node."""
        async with self._node_lock:
            if self._node is not None:
                try:
                    await asyncio.wait_for(self._node.node().shutdown(), timeout=1.0)
                except Exception:
                    pass
                self._node = None
        async with self._conn_lock:
            for peer_conn in self._connections.values():
                peer_conn.close()
            self._connections.clear()

    async def _get_connection(
        self, node_id: str, protocol_id: bytes, timings: P2POperationTimings | None = None
    ) -> PeerConnection:
        """Get a cached PeerConnection or create a new one."""
        cache_key = node_id + protocol_id.decode()

        # Fast path: check without lock
        peer_conn = self._connections.get(cache_key)
        if peer_conn is not None and peer_conn.is_alive():
            return peer_conn

        async with self._conn_lock:
            # Re-check under lock
            peer_conn = self._connections.get(cache_key)
            if peer_conn is not None and peer_conn.is_alive():
                return peer_conn

            # Stale connection — remove it
            if peer_conn is not None:
                logger.debug(f"Removing stale connection: {peer_conn}")
                peer_conn.close()
                del self._connections[cache_key]

            # Create new PeerConnection
            node = await self._get_node()
            endpoint = node.node().endpoint()
            peer_conn = PeerConnection(node_id, protocol_id, endpoint)
            # Eagerly connect so callers get a warm connection
            t0 = _timed(timings, "connection_duration")
            await peer_conn.connect(timeout=self._timeouts.connection)
            _timed_end(timings, "connection_duration", t0)
            self._connections[cache_key] = peer_conn
            return peer_conn

    async def invalidate_connection(self, node_id: str, protocol_id: bytes | None = None) -> None:
        """Discard the cached connection for *node_id* so the next call creates a fresh one.

        Args:
            node_id: The remote peer whose connection should be invalidated.
            protocol_id: Which protocol connection to invalidate.  When
                         ``None`` (the default), both UNI and BI connections
                         for the peer are invalidated.
        """
        protocol_ids = [protocol_id] if protocol_id is not None else [PROTOCOL_ID_UNI, PROTOCOL_ID_BI]
        async with self._conn_lock:
            for pid in protocol_ids:
                cache_key = node_id + pid.decode()
                peer_conn = self._connections.pop(cache_key, None)
                if peer_conn is not None:
                    peer_conn.close()
                    logger.debug(f"Invalidated cached connection to {node_id[:16]}... (proto={pid!r})")

    # ── send (unidirectional) ────────────────────────────────────────

    async def send_message(
        self,
        node_id: str,
        message: bytes,
        timings: P2POperationTimings | None = None,
    ) -> None:
        """Send a unidirectional message (fire-and-forget) with retry."""

        async def _do_send() -> None:
            # Reset per-phase timings on each attempt (retries overwrite)
            peer_conn = await self._get_connection(node_id, PROTOCOL_ID_UNI, timings)

            t0 = _timed(timings, "stream_open_duration")
            send_stream = await peer_conn.open_uni(timeout=self._timeouts.stream_open)
            _timed_end(timings, "stream_open_duration", t0)

            t0 = _timed(timings, "send_duration")
            await asyncio.wait_for(send_stream.write_all(message), timeout=self._timeouts.send)
            await asyncio.wait_for(send_stream.finish(), timeout=self._timeouts.send)
            _timed_end(timings, "send_duration", t0)

            if timings is not None:
                timings.bytes_sent = len(message)

        if timings is not None:
            timings.total_start = _time.time()
        try:
            await self._retry.execute(
                _do_send,
                on_invalidate=lambda: self.invalidate_connection(node_id, PROTOCOL_ID_UNI),
                on_node_reset=self._reset_node,
                timings=timings,
            )
        finally:
            if timings is not None:
                timings.total_end = _time.time()
                timings.total_duration = timings.total_end - timings.total_start

    async def send_model(self, node_id: str, model: BaseModel, serializer: Serializer) -> None:
        """Send a pydantic model as a unidirectional message (fire-and-forget)."""
        await self.send_message(node_id, wrap_envelope(model, serializer))

    # ── send + receive (bidirectional) ───────────────────────────────

    async def send_message_bi(
        self,
        node_id: str,
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = None,
        timings: P2POperationTimings | None = None,
    ) -> bytes:
        """Send a message and wait for response (bidirectional) with retry."""

        async def _do_send_bi() -> bytes:
            peer_conn = await self._get_connection(node_id, PROTOCOL_ID_BI, timings)

            # ── stream open ──────────────────────────────────────
            t0 = _timed(timings, "stream_open_duration")
            stream = await peer_conn.open_bi(timeout=self._timeouts.stream_open)
            _timed_end(timings, "stream_open_duration", t0)

            # ── send phase ───────────────────────────────────────
            t0 = _timed(timings, "send_duration")
            await asyncio.wait_for(stream.send().write_all(message), timeout=self._timeouts.send)
            await asyncio.wait_for(stream.send().finish(), timeout=self._timeouts.send)
            await asyncio.wait_for(stream.send().stopped(), timeout=self._timeouts.send)
            _timed_end(timings, "send_duration", t0)

            if timings is not None:
                timings.bytes_sent = len(message)

            # ── receive phase ────────────────────────────────────
            t0 = _timed(timings, "receive_duration")
            out = await asyncio.wait_for(
                stream.recv().read_to_end(max_message_size),
                timeout=self._timeouts.receive,
            )
            _timed_end(timings, "receive_duration", t0)

            if timings is not None:
                timings.bytes_received = len(out)

            if callback:
                return callback(out)
            return out

        if timings is not None:
            timings.total_start = _time.time()
        try:
            return await self._retry.execute(
                _do_send_bi,
                on_invalidate=lambda: self.invalidate_connection(node_id, PROTOCOL_ID_BI),
                on_node_reset=self._reset_node,
                timings=timings,
            )
        finally:
            if timings is not None:
                timings.total_end = _time.time()
                timings.total_duration = timings.total_end - timings.total_start

    async def send_model_bi(
        self,
        node_id: str,
        model: BaseModel,
        serializer: Serializer,
        response_model_cls: type[ModelT],
        max_message_size: int,
    ) -> ModelT:
        """Send a pydantic model and receive a pydantic model response (bidirectional)."""
        wire_bytes = wrap_envelope(model, serializer)
        response_bytes = await self.send_message_bi(node_id, wire_bytes, max_message_size)
        return unwrap_envelope(response_bytes, response_model_cls)

    # ── lifecycle ────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Shutdown the shared Iroh node and all cached connections."""
        await self._monitored_node.stop_monitoring()
        async with self._conn_lock:
            for peer_conn in self._connections.values():
                peer_conn.close()
            self._connections.clear()
        if self._node is not None:
            await self._node.node().shutdown()
            self._node = None
            logger.debug("Sender shutdown complete")


# ---------------------------------------------------------------------------
# PooledSender
# ---------------------------------------------------------------------------


class PooledSender:
    """
    P2P sender with connection pooling, LRU eviction, and health checks.
    Shares a single Iroh node across all connections. Reuses connections
    when possible, evicting the oldest when at capacity.

    Retry and per-phase timeout logic is handled by the embedded ``P2PRetry``
    executor, configured via *retry_policy* and *timeouts*.
    """

    def __init__(
        self,
        max_connections: int = 5,
        retry_policy: P2PRetryPolicy | None = None,
        timeouts: P2PTimeouts | None = None,
        health_check_interval: float = 30.0,
    ):
        self.max_connections = max_connections
        self._node: Iroh | None = None
        self._node_lock = asyncio.Lock()
        # cache_key -> PeerConnection
        self._connections: OrderedDict[str, PeerConnection] = OrderedDict()
        self._conn_lock = asyncio.Lock()

        self._timeouts = timeouts or P2PTimeouts()
        self._retry_policy = retry_policy or P2PRetryPolicy()
        self._retry = P2PRetry(self._retry_policy, self._timeouts)

        self._monitored_node = MonitoredNode(
            on_unhealthy=self._on_node_unhealthy,
            check_interval=health_check_interval,
            label="pooled-sender",
        )

    # ── properties ───────────────────────────────────────────────────

    @property
    def timeouts(self) -> P2PTimeouts:
        return self._timeouts

    @property
    def retry_policy(self) -> P2PRetryPolicy:
        return self._retry_policy

    # ── node management ──────────────────────────────────────────────

    async def _get_node(self) -> Iroh:
        """Lazily create and reuse a single Iroh node."""
        if self._node is None:
            async with self._node_lock:
                if self._node is None:
                    iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
                    self._node = await Iroh.memory_with_options(NodeOptions(protocols={}))
                    self._monitored_node.set_node(self._node)
                    self._monitored_node.start_monitoring()
        return self._node

    async def _on_node_unhealthy(self, monitored: MonitoredNode, result: HealthCheckResult) -> None:
        """Reset the node when health checks detect it is unhealthy."""
        logger.warning(f"PooledSender node unhealthy ({result.health.value}), resetting")
        await self._reset_node()
        self._monitored_node.clear_node()

    async def _reset_node(self) -> None:
        """Shut down the iroh node and clear all connections so the next
        ``_get_node()`` call creates a fresh node."""
        async with self._node_lock:
            if self._node is not None:
                try:
                    await asyncio.wait_for(self._node.node().shutdown(), timeout=1.0)
                except Exception:
                    pass
                self._node = None
        async with self._conn_lock:
            for peer_conn in self._connections.values():
                peer_conn.close()
            self._connections.clear()

    async def _get_connection(
        self, node_id: str, protocol_id: bytes, timings: P2POperationTimings | None = None
    ) -> PeerConnection:
        """Get or create a PeerConnection to the given node."""
        cache_key = node_id + protocol_id.decode()
        async with self._conn_lock:
            # Check if we have an existing live connection
            if cache_key in self._connections:
                peer_conn = self._connections[cache_key]
                if peer_conn.is_alive():
                    self._connections.move_to_end(cache_key)
                    return peer_conn
                # Connection closed, clean up
                logger.debug(f"Removing stale connection: {peer_conn}")
                peer_conn.close()
                del self._connections[cache_key]

            # Evict oldest if at capacity
            if len(self._connections) >= self.max_connections:
                evicted_key, evicted_conn = self._connections.popitem(last=False)
                evicted_conn.close()
                logger.debug(f"Evicting connection to {evicted_key[:16]}... (LRU)")

            # Create new PeerConnection using shared node
            node = await self._get_node()
            endpoint = node.node().endpoint()
            peer_conn = PeerConnection(node_id, protocol_id, endpoint)
            t0 = _timed(timings, "connection_duration")
            await peer_conn.connect(timeout=self._timeouts.connection)
            _timed_end(timings, "connection_duration", t0)

            self._connections[cache_key] = peer_conn
            logger.debug(f"Created new connection to {node_id[:16]}...")
            return peer_conn

    async def invalidate_connection(self, node_id: str, protocol_id: bytes | None = None) -> None:
        """Discard the cached connection for *node_id* so the next call creates a fresh one."""
        protocol_ids = [protocol_id] if protocol_id is not None else [PROTOCOL_ID_UNI, PROTOCOL_ID_BI]
        async with self._conn_lock:
            for pid in protocol_ids:
                cache_key = node_id + pid.decode()
                peer_conn = self._connections.pop(cache_key, None)
                if peer_conn is not None:
                    peer_conn.close()
                    logger.debug(f"Invalidated cached connection to {node_id[:16]}... (proto={pid!r})")

    # ── send (unidirectional) ────────────────────────────────────────

    async def send_message(
        self,
        node_id: str,
        message: bytes,
        timings: P2POperationTimings | None = None,
    ) -> None:
        """Send a unidirectional message (fire-and-forget) with retry."""

        async def _do_send() -> None:
            peer_conn = await self._get_connection(node_id, PROTOCOL_ID_UNI, timings)

            t0 = _timed(timings, "stream_open_duration")
            send_stream = await peer_conn.open_uni(timeout=self._timeouts.stream_open)
            _timed_end(timings, "stream_open_duration", t0)

            t0 = _timed(timings, "send_duration")
            await asyncio.wait_for(send_stream.write_all(message), timeout=self._timeouts.send)
            await asyncio.wait_for(send_stream.finish(), timeout=self._timeouts.send)
            _timed_end(timings, "send_duration", t0)

            if timings is not None:
                timings.bytes_sent = len(message)

        if timings is not None:
            timings.total_start = _time.time()
        try:
            await self._retry.execute(
                _do_send,
                on_invalidate=lambda: self.invalidate_connection(node_id, PROTOCOL_ID_UNI),
                on_node_reset=self._reset_node,
                timings=timings,
            )
        finally:
            if timings is not None:
                timings.total_end = _time.time()
                timings.total_duration = timings.total_end - timings.total_start

    async def send_model(self, node_id: str, model: BaseModel, serializer: Serializer) -> None:
        """Send a pydantic model as a unidirectional message (fire-and-forget)."""
        await self.send_message(node_id, wrap_envelope(model, serializer))

    # ── send + receive (bidirectional) ───────────────────────────────

    async def send_message_bi(
        self,
        node_id: str,
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = None,
        timings: P2POperationTimings | None = None,
    ) -> bytes:
        """Send a message and wait for response (bidirectional) with retry."""

        async def _do_send_bi() -> bytes:
            peer_conn = await self._get_connection(node_id, PROTOCOL_ID_BI, timings)

            # ── stream open ──────────────────────────────────────
            t0 = _timed(timings, "stream_open_duration")
            stream = await peer_conn.open_bi(timeout=self._timeouts.stream_open)
            _timed_end(timings, "stream_open_duration", t0)

            # ── send phase ───────────────────────────────────────
            t0 = _timed(timings, "send_duration")
            await asyncio.wait_for(stream.send().write_all(message), timeout=self._timeouts.send)
            await asyncio.wait_for(stream.send().finish(), timeout=self._timeouts.send)
            await asyncio.wait_for(stream.send().stopped(), timeout=self._timeouts.send)
            _timed_end(timings, "send_duration", t0)

            if timings is not None:
                timings.bytes_sent = len(message)

            # ── receive phase ────────────────────────────────────
            t0 = _timed(timings, "receive_duration")
            out = await asyncio.wait_for(
                stream.recv().read_to_end(max_message_size),
                timeout=self._timeouts.receive,
            )
            _timed_end(timings, "receive_duration", t0)

            if timings is not None:
                timings.bytes_received = len(out)

            if callback:
                return callback(out)
            return out

        if timings is not None:
            timings.total_start = _time.time()
        try:
            return await self._retry.execute(
                _do_send_bi,
                on_invalidate=lambda: self.invalidate_connection(node_id, PROTOCOL_ID_BI),
                on_node_reset=self._reset_node,
                timings=timings,
            )
        finally:
            if timings is not None:
                timings.total_end = _time.time()
                timings.total_duration = timings.total_end - timings.total_start

    async def send_model_bi(
        self,
        node_id: str,
        model: BaseModel,
        serializer: Serializer,
        response_model_cls: type[ModelT],
        max_message_size: int,
    ) -> ModelT:
        """Send a pydantic model and receive a pydantic model response (bidirectional)."""
        wire_bytes = wrap_envelope(model, serializer)
        response_bytes = await self.send_message_bi(node_id, wire_bytes, max_message_size)
        return unwrap_envelope(response_bytes, response_model_cls)

    # ── lifecycle ────────────────────────────────────────────────────

    async def force_destroy_node(self, timeout: float = 5.0) -> None:
        """Force-free the Rust node when async shutdown hangs."""
        self._monitored_node._on_unhealthy = None
        if self._monitored_node._monitor_task and not self._monitored_node._monitor_task.done():
            self._monitored_node._monitor_task.cancel()
        self._monitored_node._node = None

        node_obj = self._node
        self._node = None

        # Close Python-side connections (sync, no FFI)
        for peer_conn in self._connections.values():
            peer_conn.close()
        self._connections.clear()

        if node_obj is None:
            return

        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, _force_free_iroh_node, node_obj),
            timeout=timeout,
        )

    async def shutdown(self) -> None:
        """Shutdown shared node and all cached connections."""
        await self._monitored_node.stop_monitoring()
        async with self._conn_lock:
            for peer_conn in self._connections.values():
                peer_conn.close()
            self._connections.clear()
        if self._node is not None:
            await self._node.node().shutdown()
            self._node = None
        logger.debug("PooledSender shutdown complete")
