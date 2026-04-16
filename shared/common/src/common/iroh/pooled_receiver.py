"""PooledReceiver: a pool of ReceiverProcess instances sharing one activation cache.

Each receiver subprocess has its own Iroh node (distinct seed → distinct node_id).
All subprocesses share a single metadata dict so any receiver can serve any cached
activation regardless of which endpoint the requester connects to.

The owner (PooledReceiver) holds all SharedMemory blocks and exposes the union of
node_ids via :attr:`node_ids` so callers can advertise multiple receiver endpoints
in the node registry.
"""

from __future__ import annotations

import atexit
import time
from multiprocessing.managers import DictProxy, SyncManager
from multiprocessing.shared_memory import SharedMemory

from loguru import logger

from common.iroh.receiver_process import ReceiverProcess, _mp_ctx, _shm_name
from common.utils.gpu_process_utils import remove_shm_manifest, write_shm_manifest


class PooledReceiver:
    """Manages a pool of :class:`ReceiverProcess` instances that share one cache.

    Args:
        seed:               Base seed string; each receiver gets ``seed-{i}`` so
                            they produce distinct Iroh node IDs.
        pool_size:          Number of receiver subprocesses to start.
        max_message_size:   Maximum P2P message size in bytes.
        cache_ttl:          Activation cache TTL in seconds.
        max_cache_size:     Maximum number of activations to hold in cache.
        p2p_auth_timeout_ms: Timeout for P2P request authentication.
    """

    def __init__(
        self,
        seed: str,
        pool_size: int,
        max_message_size: int,
        cache_ttl: float,
        max_cache_size: int = 100,
        p2p_auth_timeout_ms: int = 30000,
    ) -> None:
        self._base_seed = seed
        self._pool_size = pool_size
        self._max_message_size = max_message_size
        self._cache_ttl = cache_ttl
        self._max_cache_size = max_cache_size
        self._p2p_auth_timeout_ms = p2p_auth_timeout_ms

        self._manager: SyncManager | None = None
        self._metadata_dict: DictProxy | None = None
        self._peer_status_dict: DictProxy | None = None
        self._shm_blocks: dict[str, SharedMemory] = {}
        self._receivers: list[ReceiverProcess] = []
        self._atexit_registered: bool = False

    # ── properties ────────────────────────────────────────────────

    @property
    def node_ids(self) -> list[str]:
        """All receiver node IDs (one per subprocess)."""
        return [r.node_id for r in self._receivers if r.node_id is not None]

    @property
    def peer_status_dict(self) -> DictProxy | None:
        """Shared dict populated by ``/peer/status`` handlers in receiver subprocesses."""
        return self._peer_status_dict

    # ── lifecycle ─────────────────────────────────────────────────

    async def start(self) -> list[str]:
        """Create shared cache, spawn all receiver subprocesses, return node_ids."""
        if self._manager is None:
            self._manager = _mp_ctx.Manager()
            self._metadata_dict = self._manager.dict()
            self._peer_status_dict = self._manager.dict()

        self._receivers = []
        for i in range(self._pool_size):
            seed = f"{self._base_seed}-{i}"
            receiver = ReceiverProcess(
                seed=seed,
                max_message_size=self._max_message_size,
                cache_ttl=self._cache_ttl,
                max_cache_size=self._max_cache_size,
                p2p_auth_timeout_ms=self._p2p_auth_timeout_ms,
                external_metadata_dict=self._metadata_dict,
                external_peer_status_dict=self._peer_status_dict,
            )
            node_id = await receiver.start()
            self._receivers.append(receiver)
            logger.info(f"PooledReceiver: receiver {i} ready (node_id={node_id[:16]}...)")

        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

        return self.node_ids

    async def stop(self) -> None:
        """Terminate all subprocesses and clean up all SharedMemory."""
        for receiver in self._receivers:
            try:
                await receiver.stop()
            except Exception as exc:
                logger.warning(f"PooledReceiver: error stopping receiver: {exc}")
        self._receivers = []

        self._cleanup_all_shm()

        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None
            self._metadata_dict = None
            self._peer_status_dict = None

        remove_shm_manifest()
        if self._atexit_registered:
            atexit.unregister(self._atexit_cleanup)
            self._atexit_registered = False

    async def restart(self) -> list[str]:
        """Restart all receiver subprocesses. Shared cache is preserved."""
        node_ids = []
        for i, receiver in enumerate(self._receivers):
            try:
                node_id = await receiver.restart()
                node_ids.append(node_id)
                logger.info(f"PooledReceiver: receiver {i} restarted (node_id={node_id[:16]}...)")
            except Exception as exc:
                logger.error(f"PooledReceiver: failed to restart receiver {i}: {exc}")
        return node_ids

    def is_alive(self) -> bool:
        """True if at least one receiver subprocess is alive."""
        return any(r.is_alive() for r in self._receivers)

    # ── activation cache ──────────────────────────────────────────

    def cache_activation(self, activation_id: str, tensor_bytes: bytes) -> None:
        """Write activation into SharedMemory and update the shared metadata dict."""
        if self._metadata_dict is None:
            raise RuntimeError("PooledReceiver not started")

        self._evict_expired()

        while len(self._shm_blocks) >= self._max_cache_size:
            oldest_key = next(iter(self._shm_blocks))
            self._remove_entry(oldest_key)
            logger.debug(f"PooledReceiver: evicted activation {oldest_key} from cache")

        name = _shm_name(activation_id)

        if activation_id in self._shm_blocks:
            self._remove_entry(activation_id)

        try:
            shm = SharedMemory(name=name, create=True, size=len(tensor_bytes))
        except FileExistsError:
            # Orphaned segment from a previous crash — unlink and recreate.
            try:
                stale = SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
            except Exception:
                pass
            shm = SharedMemory(name=name, create=True, size=len(tensor_bytes))
        shm.buf[: len(tensor_bytes)] = tensor_bytes
        self._shm_blocks[activation_id] = shm
        self._metadata_dict[activation_id] = (name, len(tensor_bytes), time.time())

        self._update_manifest()

    # ── status queue ──────────────────────────────────────────────

    def check_status_queue(self) -> list[tuple[str, str]]:
        """Non-blocking drain of status messages from all receiver subprocesses."""
        messages: list[tuple[str, str]] = []
        for receiver in self._receivers:
            messages.extend(receiver.check_status_queue())
        return messages

    # ── SharedMemory helpers ───────────────────────────────────────

    def _evict_expired(self) -> None:
        if self._metadata_dict is None:
            return
        now = time.time()
        expired = [
            aid
            for aid in list(self._shm_blocks.keys())
            if now - (self._metadata_dict.get(aid) or (None, None, 0))[2] > self._cache_ttl
        ]
        for aid in expired:
            self._remove_entry(aid)

    def _remove_entry(self, activation_id: str) -> None:
        shm = self._shm_blocks.pop(activation_id, None)
        if shm is not None:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(f"PooledReceiver: SharedMemory cleanup error for {activation_id}: {exc}")
        if self._metadata_dict is not None:
            try:
                del self._metadata_dict[activation_id]
            except KeyError:
                pass

    def _cleanup_all_shm(self) -> None:
        for aid in list(self._shm_blocks.keys()):
            self._remove_entry(aid)

    def _update_manifest(self) -> None:
        names = [_shm_name(aid) for aid in self._shm_blocks]
        write_shm_manifest(names)

    def _atexit_cleanup(self) -> None:
        try:
            self._cleanup_all_shm()
            remove_shm_manifest()
        except Exception:
            pass
