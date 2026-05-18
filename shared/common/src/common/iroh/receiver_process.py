"""ReceiverProcess: runs an Iroh Receiver in a child subprocess for fault isolation.

When the Iroh Receiver's Rust QUIC/DERP stack becomes poisoned (DERP relay
loss on Ubuntu), ``Iroh.memory_with_options()`` blocks indefinitely in the
same process.  By running the Receiver in a child subprocess we can ``kill()``
it cleanly (OS guarantees full resource cleanup) and spawn a fresh one —
without restarting the entire miner.

IPC design:
  - ``multiprocessing.Manager().dict()`` for metadata (activation_id -> shm info)
  - ``multiprocessing.shared_memory.SharedMemory`` per cached activation (zero-copy reads)
  - ``multiprocessing.Queue`` for child -> parent status messages
"""

import atexit
import asyncio
import hashlib
import multiprocessing
import os
import signal
import time
from multiprocessing.managers import DictProxy, SyncManager
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable

from loguru import logger

from common.iroh.iroh_subprocess import IrohSubprocess, _mp_ctx
from common.utils.gpu_process_utils import remove_shm_manifest, write_shm_manifest

from common.iroh.p2p_protocol import (
    P2PResponseStatus,
    decode_activation_request,
    encode_activation_response,
)


def _shm_name(activation_id: str) -> str:
    """Deterministic SharedMemory name from activation_id.

    macOS ``shm_open`` has a 31-char limit (including leading slash added by
    the OS).  We use ``iota_`` prefix + 16-char hex digest = 21 chars total.
    """
    digest = hashlib.md5(activation_id.encode()).hexdigest()[:16]
    return f"iota_{digest}"


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------


def _receiver_worker(
    seed: str,
    max_message_size: int,
    metadata_dict: DictProxy,
    status_queue: "multiprocessing.Queue",
    cache_ttl: float,
    p2p_auth_timeout_ms: int = 30000,
    peer_status_dict: DictProxy | None = None,
    push_queue: "multiprocessing.Queue | None" = None,
    valid_hotkeys_dict: DictProxy | None = None,
) -> None:
    """Entry point for the receiver subprocess.

    A fresh process means a completely clean Rust runtime — no poisoned
    QUIC/DERP state carried over from a previous incarnation.
    """
    # Allow SIGTERM to trigger KeyboardInterrupt so asyncio.run can clean up
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    try:
        asyncio.run(
            _run_receiver(
                seed,
                max_message_size,
                metadata_dict,
                status_queue,
                cache_ttl,
                p2p_auth_timeout_ms,
                peer_status_dict,
                push_queue,
                valid_hotkeys_dict,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        try:
            status_queue.put(("error", str(exc)))
        except Exception:
            pass


async def _run_receiver(
    seed: str,
    max_message_size: int,
    metadata_dict: DictProxy,
    status_queue: "multiprocessing.Queue",
    cache_ttl: float,
    p2p_auth_timeout_ms: int = 30000,
    peer_status_dict: DictProxy | None = None,
    push_queue: "multiprocessing.Queue | None" = None,
    valid_hotkeys_dict: DictProxy | None = None,
) -> None:
    """Async core of the receiver subprocess."""
    from common.iroh.receiver import Receiver
    from common.iroh.router import P2PRouter
    from common.iroh.activation_push import ActivationPushMessage
    from common.models.peer_status import PeerStatusBroadcast

    receiver = Receiver(seed=seed, max_message_size=max_message_size)
    router = P2PRouter()

    @router.handler("/activation/push")
    def handle_activation_push(msg: ActivationPushMessage, node_id: str) -> tuple[bytes, Callable[[], None]]:
        """Forward incoming push activation to the parent process via queue.

        Returns ``(ack_bytes, post_ack_action)``: the protocol layer writes the
        ACK and only then runs the enqueue. If the ACK write fails, the sender
        will retry to a different peer — committing this peer to processing
        before that point would cause both peers to process the same activation
        and produce duplicate ``miner_submissions`` rows.
        """
        from common.iroh.activation_push import encode_push_ack

        if push_queue is None:
            logger.warning(f"Received /activation/push but no push_queue configured (from {node_id[:16]}...)")
            return encode_push_ack(P2PResponseStatus.ERROR)

        tb = len(msg.tensor_bytes)
        logger.info(
            f"Activation push RECV | receiver subprocess | handoff to parent queue | "
            f"id={msg.activation_id} dir={msg.direction} from_peer={node_id[:16]}… "
            f"tensor_bytes={tb} src_layer={msg.source_layer} tgt_layer={msg.target_layer}"
        )

        def _enqueue_after_ack() -> None:
            try:
                push_queue.put_nowait(msg)
                logger.info(
                    f"Activation push RECV | receiver subprocess | enqueued OK | id={msg.activation_id} "
                    f"dir={msg.direction}"
                )
                logger.debug(f"Received /activation/push {msg.activation_id} from {node_id[:16]}...")
            except Exception as exc:
                logger.warning(f"Failed to enqueue activation push {msg.activation_id} after ack: {exc}")

        return encode_push_ack(P2PResponseStatus.SUCCESS), _enqueue_after_ack

    @router.handler("/peer/status")
    def handle_peer_status(msg: PeerStatusBroadcast, node_id: str) -> None:
        """Store incoming peer status broadcast in the shared dict.

        Drops messages from hotkeys not present in *valid_hotkeys_dict* (when
        configured) so peers from other runs cannot accumulate entries.
        """
        if not msg.source_hotkey:
            return
        if valid_hotkeys_dict is not None and msg.source_hotkey not in valid_hotkeys_dict:
            logger.debug(
                f"Dropping /peer/status from out-of-run hotkey {msg.source_hotkey[:8]}... " f"(node={node_id[:16]}...)"
            )
            return
        logger.debug(f"Received /peer/status from {msg.source_hotkey[:8]}... (node={node_id[:16]}...)")
        if peer_status_dict is not None:
            peer_status_dict[msg.source_hotkey] = (msg.model_dump(), time.time())

    @router.default
    def handle_request(message: bytes, node_id: str) -> bytes:
        """Callback invoked by the Iroh protocol handler for each incoming request."""
        try:
            activation_id, auth_fields = decode_activation_request(message)

            # Reject unsigned requests
            if auth_fields is None:
                logger.warning(f"Unsigned P2P request from {node_id[:16]}... — rejecting")
                return encode_activation_response(None, P2PResponseStatus.UNAUTHORIZED)

            # Verify signature
            from common.utils.epistula import verify_p2p_request

            # The signed body is the raw activation_id bytes (same bytes used in encode)
            id_bytes = activation_id.encode("utf-8")
            is_valid, reason = verify_p2p_request(
                body=id_bytes,
                timestamp_ms=auth_fields.timestamp_ms,
                ss58_address=auth_fields.ss58_address,
                signature=auth_fields.signature,
                timeout_ms=p2p_auth_timeout_ms,
            )
            if not is_valid:
                logger.warning(f"P2P auth failed from {node_id[:16]}...: {reason}")
                return encode_activation_response(None, P2PResponseStatus.UNAUTHORIZED)
            meta = metadata_dict.get(activation_id)

            if not meta:
                return encode_activation_response(None, P2PResponseStatus.NOT_FOUND)

            shm_name, size, cached_at = meta

            if time.time() - cached_at > cache_ttl:
                return encode_activation_response(None, P2PResponseStatus.EXPIRED)

            shm = SharedMemory(name=shm_name, create=False)
            try:
                tensor_bytes = bytes(shm.buf[:size])
            finally:
                shm.close()

            return encode_activation_response(tensor_bytes, P2PResponseStatus.SUCCESS)

        except Exception as exc:
            logger.exception(f"Error handling P2P request from {node_id[:16]}...: {exc}")
            return encode_activation_response(None, P2PResponseStatus.ERROR)

    async def on_unhealthy(monitored: Any, result: Any) -> None:
        """Forward unhealthy status to parent process via queue."""
        try:
            status_queue.put(("unhealthy", result.health.value))
        except Exception:
            pass

    await receiver.start(router=router, on_unhealthy=on_unhealthy)
    status_queue.put(("started", receiver.node_id))
    status_queue.put(
        (
            "node_addr",
            {
                "relay_url": receiver.relay_url,
                "direct_addresses": list(receiver.direct_addresses),
            },
        )
    )

    async def _refresh_node_addr() -> None:
        # Initial relay_url is often None until the relay handshake completes
        # a few seconds after start; re-publish once it's known so peers
        # can dial through the relay without consulting n0 DNS.
        last: tuple[str | None, tuple[str, ...]] = (
            receiver.relay_url,
            tuple(receiver.direct_addresses),
        )
        for _ in range(20):
            await asyncio.sleep(1.0)
            if receiver.node is None:
                return
            try:
                addr = await receiver.node.net().node_addr()
                relay_url = addr.relay_url()
                direct = tuple(addr.direct_addresses() or [])
            except Exception:
                continue
            current = (relay_url, direct)
            if current != last and (relay_url is not None or direct):
                receiver.relay_url = relay_url
                receiver.direct_addresses = list(direct)
                try:
                    status_queue.put(
                        (
                            "node_addr",
                            {"relay_url": relay_url, "direct_addresses": list(direct)},
                        )
                    )
                except Exception:
                    pass
                last = current
                if relay_url is not None and direct:
                    return

    asyncio.create_task(_refresh_node_addr(), name="refresh-node-addr")
    await receiver.serve_forever()


# ---------------------------------------------------------------------------
# Parent-process manager
# ---------------------------------------------------------------------------


class ReceiverProcess(IrohSubprocess):
    """Runs an Iroh Receiver in a child subprocess for fault isolation.

    The parent process owns the activation cache (SharedMemory blocks +
    metadata dict).  The subprocess only *reads* from them to serve
    incoming P2P requests.  When the subprocess becomes unhealthy we
    ``os.kill()`` it and spawn a fresh one — the cache is preserved.

    When *external_metadata_dict* is provided the ReceiverProcess does **not**
    create its own Manager or SharedMemory — it shares the caller-owned dict.
    In that mode ``cache_activation`` must be called on the owner (e.g.
    ``P2PStack``) rather than on this instance.
    """

    def __init__(
        self,
        seed: str,
        max_message_size: int,
        cache_ttl: float,
        max_cache_size: int = 100,
        p2p_auth_timeout_ms: int = 30000,
        external_metadata_dict: DictProxy | None = None,
        external_peer_status_dict: DictProxy | None = None,
        push_queue: "multiprocessing.Queue | None" = None,
        external_valid_hotkeys_dict: DictProxy | None = None,
    ):
        super().__init__(process_name="P2PReceiver")
        self._seed = seed
        self._max_message_size = max_message_size
        self._cache_ttl = cache_ttl
        self._max_cache_size = max_cache_size
        self._p2p_auth_timeout_ms = p2p_auth_timeout_ms
        self._push_queue = push_queue

        # When external_metadata_dict is supplied we are NOT the owner: skip
        # Manager creation and SharedMemory cleanup on stop().
        self._owns_cache: bool = external_metadata_dict is None
        self._manager: SyncManager | None = None
        self._metadata_dict: DictProxy | None = external_metadata_dict
        self._peer_status_dict: DictProxy | None = external_peer_status_dict
        self._valid_hotkeys_dict: DictProxy | None = external_valid_hotkeys_dict
        self._shm_blocks: dict[str, SharedMemory] = {}
        self._atexit_registered: bool = False

    # ── abstract method implementations ──────────────────────────

    def _worker_target(self) -> Callable[..., Any]:
        return _receiver_worker

    def _build_process_args(self) -> tuple:
        return (
            self._seed,
            self._max_message_size,
            self._metadata_dict,
            self._status_queue,
            self._cache_ttl,
            self._p2p_auth_timeout_ms,
            self._peer_status_dict,
            self._push_queue,
            self._valid_hotkeys_dict,
        )

    # ── lifecycle (overrides) ────────────────────────────────────

    async def start(self, timeout: float = 15.0) -> str:
        """Start the Manager, spawn the subprocess, and return the node_id."""
        if self._owns_cache and self._manager is None:
            self._manager = _mp_ctx.Manager()
            self._metadata_dict = self._manager.dict()

        node_id = await super().start(timeout=timeout)

        # Register atexit handler so shared memory is cleaned up even if
        # the process exits without a graceful stop() (e.g. crash, SIGTERM).
        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

        return node_id

    async def stop(self) -> None:
        """Terminate the subprocess and clean up all SharedMemory."""
        await super().stop()

        if self._owns_cache:
            self._cleanup_all_shm()
            if self._manager is not None:
                try:
                    self._manager.shutdown()
                except Exception:
                    pass
                self._manager = None
                self._metadata_dict = None
            remove_shm_manifest()
            if self._atexit_registered:
                atexit.unregister(self._atexit_cleanup)
                self._atexit_registered = False

        logger.info("ReceiverProcess stopped and all SharedMemory cleaned up")

    async def restart(self) -> str:
        """Kill the subprocess and spawn a new one.  Cache is preserved.

        Same seed -> same node_id, so no orchestrator notification needed.
        """
        return await super().restart(timeout=15.0)

    # ── activation cache (main process side) ──────────────────────

    def cache_activation(self, activation_id: str, tensor_bytes: bytes) -> None:
        """Create a SharedMemory block, copy tensor data, update metadata."""
        if self._metadata_dict is None:
            raise RuntimeError("ReceiverProcess not started")

        self._evict_expired()

        # Evict oldest if at capacity
        while len(self._shm_blocks) >= self._max_cache_size:
            oldest_key = next(iter(self._shm_blocks))
            self._remove_entry(oldest_key)
            logger.debug(f"Evicted activation {oldest_key} from SharedMemory cache")

        name = _shm_name(activation_id)

        # If this activation is already cached, remove the old block first
        if activation_id in self._shm_blocks:
            self._remove_entry(activation_id)

        shm = SharedMemory(name=name, create=True, size=len(tensor_bytes))
        shm.buf[: len(tensor_bytes)] = tensor_bytes
        self._shm_blocks[activation_id] = shm
        self._metadata_dict[activation_id] = (name, len(tensor_bytes), time.time())

        # Update manifest so orphaned segments can be found after a crash
        self._update_manifest()

    def _evict_expired(self) -> None:
        """Remove entries whose TTL has been exceeded."""
        if self._metadata_dict is None:
            return

        now = time.time()
        expired = []
        # Iterate over our local shm_blocks keys (which mirrors metadata_dict)
        for aid in list(self._shm_blocks.keys()):
            meta = self._metadata_dict.get(aid)
            if meta is None:
                expired.append(aid)
                continue
            _, _, cached_at = meta
            if now - cached_at > self._cache_ttl:
                expired.append(aid)

        for aid in expired:
            self._remove_entry(aid)

    def _remove_entry(self, activation_id: str) -> None:
        """Unlink a single SharedMemory block and remove its metadata."""
        shm = self._shm_blocks.pop(activation_id, None)
        if shm is not None:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(f"SharedMemory cleanup error for {activation_id}: {exc}")

        if self._metadata_dict is not None:
            try:
                del self._metadata_dict[activation_id]
            except KeyError:
                pass

    def _cleanup_all_shm(self) -> None:
        """Unlink all SharedMemory blocks (called on full stop)."""
        for aid in list(self._shm_blocks.keys()):
            self._remove_entry(aid)

    def _atexit_cleanup(self) -> None:
        """Best-effort shared memory cleanup on process exit."""
        try:
            self._cleanup_all_shm()
            remove_shm_manifest()
        except Exception:
            pass

    def _update_manifest(self) -> None:
        """Write current shared memory names to manifest for crash recovery."""
        names = [_shm_name(aid) for aid in self._shm_blocks]
        write_shm_manifest(names)
