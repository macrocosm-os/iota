"""In-process P2PStack.

Composition over inheritance: this class doesn't reimplement transport,
it wires up iota-sdk's `Sender` + `Receiver` + `P2PRouter` and adds the
miner-specific orchestration that used to live in the subprocess wrappers:

  * In-process activation cache (replaces SharedMemory blocks).
  * `asyncio.Queue` for inbound activation pushes (replaces
    `multiprocessing.Queue`).
  * Plain `dict` for peer-status broadcasts (replaces
    `multiprocessing.Manager` proxy).

Public API matches the old `common.iroh.p2p_stack.P2PStack` so call
sites in `new_miner.py` and friends don't need to change shape.
Subprocess-only methods (`restart`, `restart_sender`) are dropped — in
the in-process model an unhealthy sender/receiver means the iroh stack
is wedged, which is something a process-manager restart should handle.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict

from loguru import logger

from common.models.activation_push import ActivationPushMessage
from common.models.peer_status import PeerStatusBroadcast
from common.utils.epistula import verify_p2p_request
from iota_sdk.p2p import (
    DEFAULT_MAX_MESSAGE_SIZE,
    PROTOCOL_ID_BI,
    PROTOCOL_ID_UNI,
    P2PResponseStatus,
    P2PRetryPolicy,
    P2PRouter,
    P2PTimeouts,
    Receiver,
    Sender,
    decode_activation_request,
    encode_activation_response,
    encode_push_ack,
    p2p_auth_timeout_ms,
    spawn_endpoint,
)


class SenderUnavailableError(RuntimeError):
    """Raised when a send is attempted while the sender is gone.

    Replaces the old IPC-era exception of the same name in
    `common.iroh.sender_subprocess`. Same name kept so call sites that
    `except SenderUnavailableError` need only swap the import.
    """


class P2PStack:
    """In-process P2P stack: endpoint + sender + receiver + router + cache.

    The cache stores `(tensor_bytes, cached_at)` keyed by activation_id.
    Eviction is lazy (on insert) — TTL'd entries plus LRU above
    `max_cache_size`.
    """

    def __init__(
        self,
        retry_policy: P2PRetryPolicy | None = None,
        timeouts: P2PTimeouts | None = None,
        cache_ttl: float = 3000.0,
        max_cache_size: int = 100,
        p2p_auth_timeout_ms_override: int | None = None,
        max_sender_connections: int | None = None,
        push_queue_max_size: int = 0,
    ) -> None:
        self._retry_policy = retry_policy or P2PRetryPolicy(max_retries=2)
        self._timeouts = timeouts or P2PTimeouts(connection=5.0, send=30.0, receive=15.0)
        self._cache_ttl = cache_ttl
        self._max_cache_size = max_cache_size
        # Override the SDK default if the caller asked; otherwise read the
        # env-var-driven default at start time.
        self._auth_timeout_ms = p2p_auth_timeout_ms_override
        self._max_sender_connections = max_sender_connections

        self._endpoint = None  # iota_sdk.p2p.Endpoint
        self._sender: Sender | None = None
        self._receiver: Receiver | None = None
        self._router: P2PRouter | None = None
        self._seed: str | None = None

        # Lifetime: created in start(), drained/cleared at epoch boundaries.
        # `OrderedDict` so we can evict in insertion order (≈ LRU).
        self._activation_cache: OrderedDict[str, tuple[bytes, float]] = OrderedDict()

        # `dict` is fine since everything is in-process now; no Manager proxy.
        self._peer_status_dict: dict[str, tuple[dict, float]] = {}

        # Run-scope hotkey filter for /peer/status broadcasts. Populated by the
        # miner's `_sync_valid_hotkeys` loop from the node registry; consulted
        # by `on_peer_status` to drop cross-run noise. Empty dict = filter
        # disabled (accept all hotkeys); the miner fills it on the first
        # registry-sync tick after start().
        self._valid_hotkeys_dict: dict[str, bool] = {}

        # `asyncio.Queue` because the consumer (`new_miner._handle_inbound_pushes`)
        # is an async task. Bounded if the caller asked, else unbounded.
        self._push_queue: asyncio.Queue[ActivationPushMessage] = asyncio.Queue(maxsize=push_queue_max_size)

    # ── properties ────────────────────────────────────────────────────

    @property
    def node_ids(self) -> list[str]:
        return [self._endpoint.id()] if self._endpoint is not None else []

    @property
    def node_id(self) -> str | None:
        return self._endpoint.id() if self._endpoint is not None else None

    @property
    def relay_url(self) -> str | None:
        return self._endpoint.relay_url() if self._endpoint is not None else None

    @property
    def direct_addresses(self) -> list[str]:
        if self._endpoint is None:
            return []
        return list(self._endpoint.direct_addresses())

    @property
    def sender(self) -> Sender | None:
        return self._sender

    @property
    def peer_status_dict(self) -> dict[str, tuple[dict, float]]:
        return self._peer_status_dict

    @property
    def valid_hotkeys_dict(self) -> dict[str, bool]:
        return self._valid_hotkeys_dict

    @property
    def push_queue(self) -> asyncio.Queue[ActivationPushMessage]:
        return self._push_queue

    # ── lifecycle ─────────────────────────────────────────────────────

    async def start(self, seed: str, timeout: float = 5.0) -> None:
        """Spawn the endpoint, build the router, start sender + receiver.

        ``timeout`` is accepted for API compat with the old subprocess
        version; in the in-process model startup is much faster and the
        param is currently unused.
        """
        del timeout  # in-process startup is fast; param kept for compat
        if self._endpoint is not None:
            raise RuntimeError("P2PStack already started")
        start_ts = time.monotonic()
        logger.info("P2P start: begin")

        self._seed = seed

        # Bind both ALPNs so a single endpoint can serve UNI (peer status)
        # + BI (activation push, activation requests) traffic.
        self._endpoint = await spawn_endpoint(
            seed.encode("utf-8"),
            [PROTOCOL_ID_UNI, PROTOCOL_ID_BI],
        )
        logger.info(f"P2P start: endpoint up (node_id={self._endpoint.id()[:16]}...)")

        # Wire handlers BEFORE the receiver starts so no incoming streams
        # are dispatched to a half-built router.
        self._router = P2PRouter()
        self._wire_router_handlers()

        self._receiver = Receiver(self._endpoint, router=self._router)
        await self._receiver.start()

        self._sender = Sender(
            self._endpoint,
            timeouts=self._timeouts,
            retry_policy=self._retry_policy,
            max_connections=self._max_sender_connections,
        )

        elapsed = time.monotonic() - start_ts
        logger.info(f"P2P started with node_ids: {self.node_ids} (elapsed={elapsed:.2f}s)")

    async def stop(self) -> None:
        """Shut down the receiver + sender + endpoint."""
        if self._sender is not None:
            try:
                await self._sender.shutdown()
            except Exception as exc:
                logger.warning(f"P2P stop: sender shutdown failed: {exc}")
            self._sender = None

        if self._receiver is not None:
            try:
                await self._receiver.shutdown()
            except Exception as exc:
                logger.warning(f"P2P stop: receiver shutdown failed: {exc}")
            self._receiver = None

        if self._endpoint is not None:
            try:
                await self._endpoint.close()
            except Exception as exc:
                logger.warning(f"P2P stop: endpoint close failed: {exc}")
            self._endpoint = None

        self._router = None
        self._activation_cache.clear()
        logger.info("P2P shutdown complete")

    # ── epoch boundary cleanup ────────────────────────────────────────

    def drain_push_queue(self) -> int:
        """Discard all pending activation pushes. Returns how many were dropped."""
        dropped = 0
        while not self._push_queue.empty():
            try:
                self._push_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            dropped += 1
        return dropped

    def clear_activation_cache(self) -> int:
        """Drop every cached activation. Returns how many entries were cleared."""
        count = len(self._activation_cache)
        self._activation_cache.clear()
        return count

    # ── activation cache ──────────────────────────────────────────────

    def cache_activation(self, activation_id: str, tensor_bytes: bytes) -> None:
        """Insert a tensor into the local cache for peers to fetch."""
        if self._endpoint is None:
            raise RuntimeError("P2P stack not started")
        self._evict_expired()
        # LRU eviction: refresh on re-insert by removing first.
        if activation_id in self._activation_cache:
            del self._activation_cache[activation_id]
        while len(self._activation_cache) >= self._max_cache_size:
            old_id, _ = self._activation_cache.popitem(last=False)
            logger.debug(f"P2P cache: evicted activation {old_id}")
        self._activation_cache[activation_id] = (tensor_bytes, time.time())

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [aid for aid, (_, ts) in self._activation_cache.items() if now - ts > self._cache_ttl]
        for aid in expired:
            del self._activation_cache[aid]

    # ── router wiring ─────────────────────────────────────────────────

    def _wire_router_handlers(self) -> None:
        """Register the three miner-side handlers on the router.

        - ``/peer/status``  (UNI): dropped into peer_status_dict
        - ``/activation/push`` (BI): enqueued onto push_queue, ack returned
        - default          (BI): activation-request bytes — auth-checked,
                                  served from activation_cache
        """
        assert self._router is not None
        router = self._router

        @router.handler("/peer/status")
        async def on_peer_status(msg: PeerStatusBroadcast, peer_id: str) -> None:
            # Drop unsigned / hotkey-less broadcasts.
            if not msg.source_hotkey:
                return
            # Run-scope filter: when the registry has populated the dict,
            # drop broadcasts from hotkeys outside the current run. Empty
            # dict = filter disabled (e.g. just after start, before the
            # registry-sync loop has run).
            if self._valid_hotkeys_dict and msg.source_hotkey not in self._valid_hotkeys_dict:
                logger.debug(
                    f"P2P /peer/status from {msg.source_hotkey[:8]}... dropped — " f"hotkey not in current run"
                )
                return
            self._peer_status_dict[msg.source_hotkey] = (msg.model_dump(), time.time())
            logger.debug(f"P2P /peer/status from {msg.source_hotkey[:8]}... " f"(peer={peer_id[:16]}...)")

        @router.handler("/activation/push")
        async def on_activation_push(msg: ActivationPushMessage, peer_id: str) -> tuple[bytes, object]:
            """Defer the enqueue until after the ACK is delivered.

            Returning ``(ack_bytes, _enqueue_after_ack)`` tells the SDK to write
            the ACK first, wait for ``send.stopped()`` to confirm the peer
            received it, and only then run the enqueue closure. This breaks the
            duplicate-submission race that the old ``enqueue → ack`` ordering
            had: if ACK delivery failed, the sender retried to a different peer
            and both peers ended up enqueuing the same activation.
            """

            async def _enqueue_after_ack() -> None:
                try:
                    self._push_queue.put_nowait(msg)
                    logger.debug(f"P2P /activation/push {msg.activation_id} enqueued " f"(peer={peer_id[:16]}...)")
                except asyncio.QueueFull:
                    # Rare: queue filled between ACK and post-ACK. Peer
                    # already got SUCCESS; consistent with the system's
                    # at-most-once delivery semantic, drop here.
                    logger.warning(f"P2P /activation/push queue full at post-ACK — " f"dropping {msg.activation_id}")
                except Exception as exc:
                    logger.warning(f"P2P /activation/push enqueue error: {exc}")

            return encode_push_ack(P2PResponseStatus.SUCCESS), _enqueue_after_ack

        # Raw-bytes default handler: this is where activation request
        # (encode_activation_request) traffic lands, since those bytes
        # don't carry a routed envelope.
        @router.default
        def on_activation_request(message: bytes, peer_id: str) -> bytes:
            return self._serve_activation_request(message, peer_id)

    def _serve_activation_request(self, message: bytes, peer_id: str) -> bytes:
        """Decode the v1/v2 request, verify auth, look up cache, return response."""
        try:
            activation_id, auth_fields = decode_activation_request(message)
        except Exception as exc:
            logger.warning(f"P2P request decode failed from {peer_id[:16]}...: {exc}")
            return encode_activation_response(None, P2PResponseStatus.ERROR)

        if auth_fields is None:
            logger.warning(f"Unsigned P2P request from {peer_id[:16]}... — rejecting")
            return encode_activation_response(None, P2PResponseStatus.UNAUTHORIZED)

        # Verify the v2 signature. Body is the activation_id bytes
        # (matches the encode side — see iota_sdk.p2p.encode_activation_request_v2).
        timeout_ms = self._auth_timeout_ms or p2p_auth_timeout_ms()
        is_valid, reason = verify_p2p_request(
            body=activation_id.encode("utf-8"),
            timestamp_ms=auth_fields.timestamp_ms,
            ss58_address=auth_fields.ss58_address,
            signature=auth_fields.signature,
            timeout_ms=timeout_ms,
        )
        if not is_valid:
            logger.warning(f"P2P auth failed from {peer_id[:16]}...: {reason}")
            return encode_activation_response(None, P2PResponseStatus.UNAUTHORIZED)

        cached = self._activation_cache.get(activation_id)
        if cached is None:
            return encode_activation_response(None, P2PResponseStatus.NOT_FOUND)

        tensor_bytes, cached_at = cached
        if time.time() - cached_at > self._cache_ttl:
            # Lazy TTL check: on lookup, expire stale entries.
            self._activation_cache.pop(activation_id, None)
            return encode_activation_response(None, P2PResponseStatus.EXPIRED)

        return encode_activation_response(tensor_bytes, P2PResponseStatus.SUCCESS)


__all__ = ["P2PStack", "SenderUnavailableError", "DEFAULT_MAX_MESSAGE_SIZE"]
