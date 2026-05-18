from __future__ import annotations

import asyncio
import time as _time
from collections import OrderedDict
from typing import Awaitable, Callable, TypeVar, overload

from iroh import (
    Iroh,
    NodeDiscoveryConfig,
    NodeOptions,
    iroh_ffi,
)
from loguru import logger
from pydantic import BaseModel

from common.iroh.cleanup import _force_free_iroh_node
from common.iroh.connection import PeerConnection
from common.iroh.monitored_node import HealthCheckResult, MonitoredNode
from common.iroh.protocol import PROTOCOL_ID_BI, PROTOCOL_ID_UNI
from common.iroh.retry import P2PRetry, P2PRetryPolicy, P2PSendCancelledError, P2PTimeouts
from common.iroh.timings import P2POperationTimings, TimingsPhaseField
from common.iroh.serializer import Serializer, unwrap_envelope, wrap_envelope

ModelT = TypeVar("ModelT", bound=BaseModel)
T_Ret = TypeVar("T_Ret")

_SEND_CANCELLED_MSG = "P2P send cancelled (parent timeout)"


class PeerAddressUnknownError(Exception):
    """Raised when a dial is attempted to a peer that has no registered address hints.

    Iroh's default discovery is disabled, so the sender's per-peer address book
    is the sole source of dialable addresses.  Callers should call
    :meth:`PooledSender.register_peer` (or wait for the registry sync that does
    so automatically) before attempting to send.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_send_cancelled(cancel_event: asyncio.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise P2PSendCancelledError(_SEND_CANCELLED_MSG)


async def _await_func_or_cancel(
    aw: Awaitable[T_Ret],
    timeout: float | None,
    cancel_event: asyncio.Event | None,
) -> T_Ret:
    """Wait for *aw*, optionally capped by *timeout*, while *cancel_event* can abort early.

    ``timeout is None`` means no extra cap (beyond whatever the awaitable does internally);
    cancellation still wins when ``cancel_event`` is set.
    """
    if cancel_event is None:
        return await asyncio.wait_for(aw, timeout=timeout)
    step_task = asyncio.create_task(asyncio.wait_for(aw, timeout=timeout))
    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        done, _ = await asyncio.wait(
            {step_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancel_task in done:
            step_task.cancel()
            try:
                await step_task
            except (asyncio.CancelledError, Exception):
                pass
            raise P2PSendCancelledError(_SEND_CANCELLED_MSG)
        cancel_task.cancel()
        try:
            await cancel_task
        except asyncio.CancelledError:
            pass
        return step_task.result()
    finally:
        if not cancel_task.done():
            cancel_task.cancel()
            try:
                await cancel_task
            except asyncio.CancelledError:
                pass
        if not step_task.done():
            step_task.cancel()
            try:
                await step_task
            except (asyncio.CancelledError, Exception):
                pass


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


def _phase_ms(timings: P2POperationTimings | None, attr: TimingsPhaseField) -> str:
    """Format ``timings.<attr>`` as ``Xms`` (or ``?ms``) for boundary log lines."""
    val = getattr(timings, attr, None) if timings is not None else None
    return f"{val * 1000:.0f}ms" if val is not None else "?ms"


def _peer_label(node_id: str, hotkey: str | None) -> str:
    """Render ``<iroh:16> hk=<hotkey:8>`` for log lines so each send line
    shows both the iroh node identifier (used by the transport) and the
    SS58 hotkey (the human-meaningful identity used elsewhere)."""
    hk = (hotkey or "?")[:8] if hotkey else "?"
    return f"{node_id[:16]}.. hk={hk}"


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

    _NODE_CREATE_TIMEOUT: float = 10.0

    async def _get_node(self) -> Iroh:
        """Lazily create and reuse a single Iroh node."""
        if self._node is None:
            async with self._node_lock:
                if self._node is None:
                    iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
                    self._node = await asyncio.wait_for(
                        Iroh.memory_with_options(
                            NodeOptions(protocols={}, node_discovery=NodeDiscoveryConfig.NONE),
                        ),
                        timeout=self._NODE_CREATE_TIMEOUT,
                    )
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
        node_id: str | list[str],
        message: bytes,
        timings: P2POperationTimings | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> None:
        """Send a unidirectional message (fire-and-forget) with retry.

        Pass a list of node IDs to fan out to multiple peers concurrently.
        Timings are not tracked for multi-send.
        """
        if isinstance(node_id, list):
            await asyncio.gather(
                *[
                    self.send_message(nid, message, timings=timings, cancellation_event=cancellation_event)
                    for nid in node_id
                ]
            )
            return

        peer_lbl = _peer_label(node_id, getattr(self, "_peer_hotkeys", {}).get(node_id))

        async def _do_send() -> None:
            # Reset per-phase timings on each attempt (retries overwrite)
            _check_send_cancelled(cancellation_event)
            peer_conn = await _await_func_or_cancel(
                self._get_connection(node_id, PROTOCOL_ID_UNI, timings),
                None,
                cancellation_event,
            )
            logger.debug(f"[sender] uni {peer_lbl} conn ready ({_phase_ms(timings, 'connection_duration')})")

            _check_send_cancelled(cancellation_event)
            t0 = _timed(timings, "stream_open_duration")
            send_stream = await _await_func_or_cancel(
                peer_conn.open_uni(timeout=self._timeouts.stream_open),
                None,
                cancellation_event,
            )
            _timed_end(timings, "stream_open_duration", t0)
            logger.debug(f"[sender] uni {peer_lbl} stream_open done ({_phase_ms(timings, 'stream_open_duration')})")

            _check_send_cancelled(cancellation_event)
            t0 = _timed(timings, "send_duration")
            await _await_func_or_cancel(
                send_stream.write_all(message),
                self._timeouts.send,
                cancellation_event,
            )
            await _await_func_or_cancel(
                send_stream.finish(),
                self._timeouts.send,
                cancellation_event,
            )
            _timed_end(timings, "send_duration", t0)
            logger.debug(f"[sender] uni {peer_lbl} send done ({_phase_ms(timings, 'send_duration')}, {len(message)}B)")

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

    async def send_model(self, node_id: str | list[str], model: BaseModel, serializer: Serializer) -> None:
        """Send a pydantic model as a unidirectional message (fire-and-forget).

        Pass a list of node IDs to fan out to multiple peers concurrently.
        """
        await self.send_message(node_id, wrap_envelope(model, serializer))

    async def send_routed(
        self,
        route: str,
        node_id: str | list[str],
        model: BaseModel,
        serializer: Serializer | None = None,
        timings: P2POperationTimings | None = None,
    ) -> None:
        """Send a typed UNI message with routed envelope.

        Pass a list of node IDs to fan out to multiple peers concurrently.
        Timings are not tracked for multi-send.
        """
        from common.iroh.router import wrap_routed_envelope

        payload = wrap_routed_envelope(route, model, serializer)
        await self.send_message(node_id, payload, timings=timings)

    # ── send + receive (bidirectional) ───────────────────────────────

    @overload
    async def send_message_bi(
        self,
        node_id: str,
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = ...,
        timings: P2POperationTimings | None = ...,
        cancellation_event: asyncio.Event | None = ...,
    ) -> bytes:
        ...

    @overload
    async def send_message_bi(
        self,
        node_id: list[str],
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = ...,
        timings: P2POperationTimings | None = ...,
        cancellation_event: asyncio.Event | None = ...,
    ) -> list[bytes]:
        ...

    async def send_message_bi(
        self,
        node_id: str | list[str],
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = None,
        timings: P2POperationTimings | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> bytes | list[bytes]:
        """Send a message and wait for response (bidirectional) with retry.

        Pass a list of node IDs to fan out to multiple peers concurrently,
        returning a list of responses in the same order. Timings are not
        tracked for multi-send.

        ``cancellation_event`` is set by :class:`SenderProxy` when the parent
        wait times out so the subprocess can abort between QUIC phases.
        """
        if isinstance(node_id, list):
            return list(
                await asyncio.gather(
                    *[
                        self.send_message_bi(
                            nid,
                            message,
                            max_message_size,
                            callback,
                            timings=timings,
                            cancellation_event=cancellation_event,
                        )
                        for nid in node_id
                    ]
                )
            )

        peer_lbl = _peer_label(node_id, getattr(self, "_peer_hotkeys", {}).get(node_id))

        async def _do_send_bi() -> bytes:
            _check_send_cancelled(cancellation_event)
            peer_conn = await _await_func_or_cancel(
                self._get_connection(node_id, PROTOCOL_ID_BI, timings),
                None,
                cancellation_event,
            )
            logger.debug(f"[sender] bi {peer_lbl} conn ready ({_phase_ms(timings, 'connection_duration')})")

            # ── stream open ──────────────────────────────────────
            _check_send_cancelled(cancellation_event)
            t0 = _timed(timings, "stream_open_duration")
            stream = await _await_func_or_cancel(
                peer_conn.open_bi(timeout=self._timeouts.stream_open),
                None,
                cancellation_event,
            )
            _timed_end(timings, "stream_open_duration", t0)
            logger.debug(f"[sender] bi {peer_lbl} stream_open done ({_phase_ms(timings, 'stream_open_duration')})")

            # ── send phase ───────────────────────────────────────
            _check_send_cancelled(cancellation_event)
            t0 = _timed(timings, "send_duration")
            await _await_func_or_cancel(
                stream.send().write_all(message),
                self._timeouts.send,
                cancellation_event,
            )
            await _await_func_or_cancel(
                stream.send().finish(),
                self._timeouts.send,
                cancellation_event,
            )
            await _await_func_or_cancel(
                stream.send().stopped(),
                self._timeouts.send,
                cancellation_event,
            )
            _timed_end(timings, "send_duration", t0)
            logger.debug(f"[sender] bi {peer_lbl} send done ({_phase_ms(timings, 'send_duration')}, {len(message)}B)")

            if timings is not None:
                timings.bytes_sent = len(message)

            # ── receive phase ────────────────────────────────────
            _check_send_cancelled(cancellation_event)
            t0 = _timed(timings, "receive_duration")
            out = await _await_func_or_cancel(
                stream.recv().read_to_end(max_message_size),
                self._timeouts.receive,
                cancellation_event,
            )
            _timed_end(timings, "receive_duration", t0)
            logger.debug(f"[sender] bi {peer_lbl} recv done ({_phase_ms(timings, 'receive_duration')}, {len(out)}B)")

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

    @overload
    async def send_model_bi(
        self,
        node_id: str,
        model: BaseModel,
        serializer: Serializer,
        response_model_cls: type[ModelT],
        max_message_size: int,
        cancellation_event: asyncio.Event | None = ...,
    ) -> ModelT:
        ...

    @overload
    async def send_model_bi(
        self,
        node_id: list[str],
        model: BaseModel,
        serializer: Serializer,
        response_model_cls: type[ModelT],
        max_message_size: int,
        cancellation_event: asyncio.Event | None = ...,
    ) -> list[ModelT]:
        ...

    async def send_model_bi(
        self,
        node_id: str | list[str],
        model: BaseModel,
        serializer: Serializer,
        response_model_cls: type[ModelT],
        max_message_size: int,
        cancellation_event: asyncio.Event | None = None,
    ) -> ModelT | list[ModelT]:
        """Send a pydantic model and receive a pydantic model response (bidirectional).

        Pass a list of node IDs to fan out to multiple peers concurrently,
        returning a list of responses in the same order.
        """
        wire_bytes = wrap_envelope(model, serializer)
        if isinstance(node_id, list):
            responses = await self.send_message_bi(
                node_id, wire_bytes, max_message_size, cancellation_event=cancellation_event
            )
            return [unwrap_envelope(r, response_model_cls) for r in responses]
        response_bytes = await self.send_message_bi(
            node_id, wire_bytes, max_message_size, cancellation_event=cancellation_event
        )
        return unwrap_envelope(response_bytes, response_model_cls)

    @overload
    async def send_routed_bi(
        self,
        route: str,
        node_id: str,
        model: BaseModel,
        response_model_cls: type[ModelT],
        max_message_size: int,
        serializer: Serializer | None = ...,
        timings: P2POperationTimings | None = ...,
        cancellation_event: asyncio.Event | None = ...,
    ) -> ModelT:
        ...

    @overload
    async def send_routed_bi(
        self,
        route: str,
        node_id: list[str],
        model: BaseModel,
        response_model_cls: type[ModelT],
        max_message_size: int,
        serializer: Serializer | None = ...,
        timings: P2POperationTimings | None = ...,
        cancellation_event: asyncio.Event | None = ...,
    ) -> list[ModelT]:
        ...

    async def send_routed_bi(
        self,
        route: str,
        node_id: str | list[str],
        model: BaseModel,
        response_model_cls: type[ModelT],
        max_message_size: int,
        serializer: Serializer | None = None,
        timings: P2POperationTimings | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> ModelT | list[ModelT]:
        """Send a typed BI request with routed envelope, receive a typed response.

        Pass a list of node IDs to fan out to multiple peers concurrently,
        returning a list of responses in the same order. Timings are not
        tracked for multi-send.
        """
        from common.iroh.router import unwrap_routed_envelope, wrap_routed_envelope

        payload = wrap_routed_envelope(route, model, serializer)
        if isinstance(node_id, list):
            responses = await self.send_message_bi(
                node_id,
                payload,
                max_message_size,
                timings=timings,
                cancellation_event=cancellation_event,
            )
            return [
                ser.deserialize(body, response_model_cls)
                for _, body, ser in (unwrap_routed_envelope(r) for r in responses)
            ]
        response_bytes = await self.send_message_bi(
            node_id,
            payload,
            max_message_size,
            timings=timings,
            cancellation_event=cancellation_event,
        )
        _, body, ser = unwrap_routed_envelope(response_bytes)
        return ser.deserialize(body, response_model_cls)

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
        max_connections: int = 32,
        retry_policy: P2PRetryPolicy | None = None,
        timeouts: P2PTimeouts | None = None,
        health_check_interval: float = 30.0,
    ):
        self.max_connections = max_connections
        self._node: Iroh | None = None
        self._node_lock = asyncio.Lock()
        # cache_key -> PeerConnection
        self._connections: OrderedDict[str, PeerConnection] = OrderedDict()
        # Protects _connections dict mutation only — NOT held across connect().
        self._conn_lock = asyncio.Lock()
        # Per-peer (cache_key) lock so concurrent dials to the same peer don't
        # waste handshakes. Only one connect to a given peer at a time; dials
        # to *different* peers run fully in parallel.
        self._peer_connect_locks: dict[str, asyncio.Lock] = {}
        # node_id -> (relay_url, direct_addresses) — populated via register_peer
        # so dials skip n0 DNS discovery.
        self._peer_addrs: dict[str, tuple[str | None, list[str]]] = {}
        # iroh node_id -> SS58 hotkey, populated via register_peer.  Purely
        # for log labelling so ``[sender]`` lines pair the iroh hex with the
        # human-meaningful hotkey of the target.
        self._peer_hotkeys: dict[str, str] = {}

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

    # ── peer address book ────────────────────────────────────────────

    def register_peer(
        self,
        node_id: str,
        relay_url: str | None,
        direct_addresses: list[str],
        hotkey: str | None = None,
    ) -> None:
        """Cache a peer's relay + direct addresses so the next dial skips discovery.

        New entries take effect on the next connect; existing cached connections
        to *node_id* are invalidated so a stale (discovery-found) connection is
        not reused.

        ``hotkey`` (the peer's SS58) is recorded purely for log labelling.
        """
        if hotkey:
            self._peer_hotkeys[node_id] = hotkey
        prev = self._peer_addrs.get(node_id)
        new = (relay_url, list(direct_addresses or []))
        if prev == new:
            return
        self._peer_addrs[node_id] = new
        # Drop any cached PeerConnection so the next connect uses the new hints
        for proto in (PROTOCOL_ID_UNI, PROTOCOL_ID_BI):
            cache_key = node_id + proto.decode()
            existing = self._connections.pop(cache_key, None)
            if existing is not None:
                existing.close()

    # ── node management ──────────────────────────────────────────────

    _NODE_CREATE_TIMEOUT: float = 10.0

    async def _get_node(self) -> Iroh:
        """Lazily create and reuse a single Iroh node."""
        if self._node is None:
            async with self._node_lock:
                if self._node is None:
                    iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
                    self._node = await asyncio.wait_for(
                        Iroh.memory_with_options(
                            NodeOptions(protocols={}, node_discovery=NodeDiscoveryConfig.NONE),
                        ),
                        timeout=self._NODE_CREATE_TIMEOUT,
                    )
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
        """Get or create a PeerConnection to the given node.

        ``_conn_lock`` is held only across cache reads/writes and LRU
        eviction — never across ``peer_conn.connect()``. A per-peer
        ``_peer_connect_locks`` entry serializes concurrent dials to the
        same peer so we don't waste handshakes; dials to *different*
        peers run fully in parallel.
        """
        cache_key = node_id + protocol_id.decode()

        # ── fast path: cache hit under tight global lock ─────────────
        async with self._conn_lock:
            existing = self._connections.get(cache_key)
            if existing is not None:
                if existing.is_alive():
                    self._connections.move_to_end(cache_key)
                    return existing
                logger.debug(f"Removing stale connection: {existing}")
                existing.close()
                del self._connections[cache_key]

        # ── slow path: serialize per-peer dials ──────────────────────
        peer_lock = self._peer_connect_locks.get(cache_key)
        if peer_lock is None:
            peer_lock = self._peer_connect_locks.setdefault(cache_key, asyncio.Lock())

        async with peer_lock:
            # Re-check under peer lock — another task may have dialed while we waited.
            async with self._conn_lock:
                existing = self._connections.get(cache_key)
                if existing is not None and existing.is_alive():
                    self._connections.move_to_end(cache_key)
                    return existing

            # Iroh's default discovery is disabled, so we MUST have at least
            # one address hint (relay URL or a direct sockaddr) for this peer
            # — otherwise iroh will fail with the cryptic "No addressing
            # information for NodeId(...)" error from inside the Rust runtime.
            # Surface a clear error here instead so callers can attribute it.
            relay_url, direct_addresses = self._peer_addrs.get(node_id, (None, []))
            if relay_url is None and not direct_addresses:
                raise PeerAddressUnknownError(
                    f"No address hints registered for peer {node_id[:16]}... — "
                    f"call PooledSender.register_peer() before dialing "
                    f"(iroh discovery is disabled, so the address book is the only source)."
                )
            node = await self._get_node()
            endpoint = node.node().endpoint()
            peer_conn = PeerConnection(
                node_id,
                protocol_id,
                endpoint,
                relay_url=relay_url,
                direct_addresses=direct_addresses,
            )
            t0 = _timed(timings, "connection_duration")
            await peer_conn.connect(timeout=self._timeouts.connection)
            _timed_end(timings, "connection_duration", t0)

            # Insert under tight global lock + apply LRU eviction.
            async with self._conn_lock:
                if len(self._connections) >= self.max_connections:
                    evicted_key, evicted_conn = self._connections.popitem(last=False)
                    evicted_conn.close()
                    logger.debug(f"Evicting connection to {evicted_key[:16]}... (LRU)")
                self._connections[cache_key] = peer_conn

            logger.debug(
                f"Created new connection to {node_id[:16]}... "
                f"(hints: relay={'yes' if relay_url else 'no'}, "
                f"direct={len(direct_addresses)})"
            )
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
        node_id: str | list[str],
        message: bytes,
        timings: P2POperationTimings | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> None:
        """Send a unidirectional message (fire-and-forget) with retry.

        Pass a list of node IDs to fan out to multiple peers concurrently.
        Timings are not tracked for multi-send.
        """
        if isinstance(node_id, list):
            await asyncio.gather(
                *[
                    self.send_message(nid, message, timings=timings, cancellation_event=cancellation_event)
                    for nid in node_id
                ]
            )
            return

        peer_lbl = _peer_label(node_id, getattr(self, "_peer_hotkeys", {}).get(node_id))

        async def _do_send() -> None:
            _check_send_cancelled(cancellation_event)
            peer_conn = await _await_func_or_cancel(
                self._get_connection(node_id, PROTOCOL_ID_UNI, timings),
                None,
                cancellation_event,
            )
            logger.debug(f"[sender] uni {peer_lbl} conn ready ({_phase_ms(timings, 'connection_duration')})")

            _check_send_cancelled(cancellation_event)
            t0 = _timed(timings, "stream_open_duration")
            send_stream = await _await_func_or_cancel(
                peer_conn.open_uni(timeout=self._timeouts.stream_open),
                None,
                cancellation_event,
            )
            _timed_end(timings, "stream_open_duration", t0)
            logger.debug(f"[sender] uni {peer_lbl} stream_open done ({_phase_ms(timings, 'stream_open_duration')})")

            _check_send_cancelled(cancellation_event)
            t0 = _timed(timings, "send_duration")
            await _await_func_or_cancel(
                send_stream.write_all(message),
                self._timeouts.send,
                cancellation_event,
            )
            await _await_func_or_cancel(
                send_stream.finish(),
                self._timeouts.send,
                cancellation_event,
            )
            _timed_end(timings, "send_duration", t0)
            logger.debug(f"[sender] uni {peer_lbl} send done ({_phase_ms(timings, 'send_duration')}, {len(message)}B)")

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

    async def send_model(self, node_id: str | list[str], model: BaseModel, serializer: Serializer) -> None:
        """Send a pydantic model as a unidirectional message (fire-and-forget).

        Pass a list of node IDs to fan out to multiple peers concurrently.
        """
        await self.send_message(node_id, wrap_envelope(model, serializer))

    async def send_routed(
        self,
        route: str,
        node_id: str | list[str],
        model: BaseModel,
        serializer: Serializer | None = None,
        timings: P2POperationTimings | None = None,
    ) -> None:
        """Send a typed UNI message with routed envelope.

        Pass a list of node IDs to fan out to multiple peers concurrently.
        Timings are not tracked for multi-send.
        """
        from common.iroh.router import wrap_routed_envelope

        payload = wrap_routed_envelope(route, model, serializer)
        await self.send_message(node_id, payload, timings=timings)

    # ── send + receive (bidirectional) ───────────────────────────────

    @overload
    async def send_message_bi(
        self,
        node_id: str,
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = ...,
        timings: P2POperationTimings | None = ...,
        cancellation_event: asyncio.Event | None = ...,
    ) -> bytes:
        ...

    @overload
    async def send_message_bi(
        self,
        node_id: list[str],
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = ...,
        timings: P2POperationTimings | None = ...,
        cancellation_event: asyncio.Event | None = ...,
    ) -> list[bytes]:
        ...

    async def send_message_bi(
        self,
        node_id: str | list[str],
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = None,
        timings: P2POperationTimings | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> bytes | list[bytes]:
        """Send a message and wait for response (bidirectional) with retry.

        Pass a list of node IDs to fan out to multiple peers concurrently,
        returning a list of responses in the same order. Timings are not
        tracked for multi-send.

        ``cancellation_event`` is set by :class:`SenderProxy` when the parent
        wait times out so the subprocess can abort between QUIC phases.
        """
        if isinstance(node_id, list):
            return list(
                await asyncio.gather(
                    *[
                        self.send_message_bi(
                            nid,
                            message,
                            max_message_size,
                            callback,
                            timings=timings,
                            cancellation_event=cancellation_event,
                        )
                        for nid in node_id
                    ]
                )
            )

        peer_lbl = _peer_label(node_id, getattr(self, "_peer_hotkeys", {}).get(node_id))

        async def _do_send_bi() -> bytes:
            _check_send_cancelled(cancellation_event)
            peer_conn = await _await_func_or_cancel(
                self._get_connection(node_id, PROTOCOL_ID_BI, timings),
                None,
                cancellation_event,
            )
            logger.debug(f"[sender] bi {peer_lbl} conn ready ({_phase_ms(timings, 'connection_duration')})")

            # ── stream open ──────────────────────────────────────
            _check_send_cancelled(cancellation_event)
            t0 = _timed(timings, "stream_open_duration")
            stream = await _await_func_or_cancel(
                peer_conn.open_bi(timeout=self._timeouts.stream_open),
                None,
                cancellation_event,
            )
            _timed_end(timings, "stream_open_duration", t0)
            logger.debug(f"[sender] bi {peer_lbl} stream_open done ({_phase_ms(timings, 'stream_open_duration')})")

            # ── send phase ───────────────────────────────────────
            _check_send_cancelled(cancellation_event)
            t0 = _timed(timings, "send_duration")
            await _await_func_or_cancel(
                stream.send().write_all(message),
                self._timeouts.send,
                cancellation_event,
            )
            await _await_func_or_cancel(
                stream.send().finish(),
                self._timeouts.send,
                cancellation_event,
            )
            await _await_func_or_cancel(
                stream.send().stopped(),
                self._timeouts.send,
                cancellation_event,
            )
            _timed_end(timings, "send_duration", t0)
            logger.debug(f"[sender] bi {peer_lbl} send done ({_phase_ms(timings, 'send_duration')}, {len(message)}B)")

            if timings is not None:
                timings.bytes_sent = len(message)

            # ── receive phase ────────────────────────────────────
            _check_send_cancelled(cancellation_event)
            t0 = _timed(timings, "receive_duration")
            out = await _await_func_or_cancel(
                stream.recv().read_to_end(max_message_size),
                self._timeouts.receive,
                cancellation_event,
            )
            _timed_end(timings, "receive_duration", t0)
            logger.debug(f"[sender] bi {peer_lbl} recv done ({_phase_ms(timings, 'receive_duration')}, {len(out)}B)")

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

    @overload
    async def send_model_bi(
        self,
        node_id: str,
        model: BaseModel,
        serializer: Serializer,
        response_model_cls: type[ModelT],
        max_message_size: int,
        cancellation_event: asyncio.Event | None = ...,
    ) -> ModelT:
        ...

    @overload
    async def send_model_bi(
        self,
        node_id: list[str],
        model: BaseModel,
        serializer: Serializer,
        response_model_cls: type[ModelT],
        max_message_size: int,
        cancellation_event: asyncio.Event | None = ...,
    ) -> list[ModelT]:
        ...

    async def send_model_bi(
        self,
        node_id: str | list[str],
        model: BaseModel,
        serializer: Serializer,
        response_model_cls: type[ModelT],
        max_message_size: int,
        cancellation_event: asyncio.Event | None = None,
    ) -> ModelT | list[ModelT]:
        """Send a pydantic model and receive a pydantic model response (bidirectional).

        Pass a list of node IDs to fan out to multiple peers concurrently,
        returning a list of responses in the same order.
        """
        wire_bytes = wrap_envelope(model, serializer)
        if isinstance(node_id, list):
            responses = await self.send_message_bi(
                node_id, wire_bytes, max_message_size, cancellation_event=cancellation_event
            )
            return [unwrap_envelope(r, response_model_cls) for r in responses]
        response_bytes = await self.send_message_bi(
            node_id, wire_bytes, max_message_size, cancellation_event=cancellation_event
        )
        return unwrap_envelope(response_bytes, response_model_cls)

    @overload
    async def send_routed_bi(
        self,
        route: str,
        node_id: str,
        model: BaseModel,
        response_model_cls: type[ModelT],
        max_message_size: int,
        serializer: Serializer | None = ...,
        timings: P2POperationTimings | None = ...,
        cancellation_event: asyncio.Event | None = ...,
    ) -> ModelT:
        ...

    @overload
    async def send_routed_bi(
        self,
        route: str,
        node_id: list[str],
        model: BaseModel,
        response_model_cls: type[ModelT],
        max_message_size: int,
        serializer: Serializer | None = ...,
        timings: P2POperationTimings | None = ...,
        cancellation_event: asyncio.Event | None = ...,
    ) -> list[ModelT]:
        ...

    async def send_routed_bi(
        self,
        route: str,
        node_id: str | list[str],
        model: BaseModel,
        response_model_cls: type[ModelT],
        max_message_size: int,
        serializer: Serializer | None = None,
        timings: P2POperationTimings | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> ModelT | list[ModelT]:
        """Send a typed BI request with routed envelope, receive a typed response.

        Pass a list of node IDs to fan out to multiple peers concurrently,
        returning a list of responses in the same order. Timings are not
        tracked for multi-send.
        """
        from common.iroh.router import unwrap_routed_envelope, wrap_routed_envelope

        payload = wrap_routed_envelope(route, model, serializer)
        if isinstance(node_id, list):
            responses = await self.send_message_bi(
                node_id,
                payload,
                max_message_size,
                timings=timings,
                cancellation_event=cancellation_event,
            )
            return [
                ser.deserialize(body, response_model_cls)
                for _, body, ser in (unwrap_routed_envelope(r) for r in responses)
            ]
        response_bytes = await self.send_message_bi(
            node_id,
            payload,
            max_message_size,
            timings=timings,
            cancellation_event=cancellation_event,
        )
        _, body, ser = unwrap_routed_envelope(response_bytes)
        return ser.deserialize(body, response_model_cls)

    async def send_routed_bi_raw(
        self,
        route: str,
        node_id: str,
        model: BaseModel,
        max_message_size: int,
        serializer: Serializer | None = None,
        timings: P2POperationTimings | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> bytes:
        """Send a typed BI request with routed envelope, return raw response bytes.

        Unlike ``send_routed_bi``, this does **not** unwrap the response as a
        routed envelope — it returns the raw bytes from the peer.  Useful when
        the response is a simple status byte (e.g. activation push ack).
        """
        from common.iroh.router import wrap_routed_envelope

        payload = wrap_routed_envelope(route, model, serializer)
        return await self.send_message_bi(
            node_id,
            payload,
            max_message_size,
            timings=timings,
            cancellation_event=cancellation_event,
        )

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
