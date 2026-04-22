"""P2PStack: lifecycle manager for a ReceiverProcess + SenderSubprocess pair.

Both the Receiver and Sender run in child subprocesses for fault isolation —
when the Iroh Rust QUIC/DERP stack becomes poisoned, we ``os.kill()`` the
subprocess and spawn a fresh one without restarting the miner.
"""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing
import os
import time
from multiprocessing.managers import DictProxy, SyncManager
from multiprocessing.shared_memory import SharedMemory

from loguru import logger

from common.iroh.iroh_subprocess import _mp_ctx
from common.iroh.receiver_process import ReceiverProcess, _shm_name
from common.iroh.sender_subprocess import SenderSubprocess, SenderProxy
from common.iroh.retry import P2PRetryPolicy, P2PTimeouts
from common.iroh.settings import DEFAULT_MAX_MESSAGE_SIZE, P2P_AUTH_TIMEOUT_MS
from common.utils.gpu_process_utils import remove_shm_manifest, write_shm_manifest


class P2PStack:
    """Owns a :class:`ReceiverProcess` + :class:`SenderSubprocess` and exposes a
    clean start/stop/restart API.

    Both Receiver and Sender run in subprocesses.  The activation cache lives in
    the main process (SharedMemory blocks) and is shared with the receiver
    subprocess via a Manager dict.
    """

    def __init__(
        self,
        retry_policy: P2PRetryPolicy | None = None,
        timeouts: P2PTimeouts | None = None,
        cache_ttl: float = 3000.0,
        max_cache_size: int = 100,
        p2p_auth_timeout_ms: int | None = None,
    ):
        self._receiver: ReceiverProcess | None = None
        self._sender_subprocess: SenderSubprocess | None = None
        self._restarting_receiver: bool = False
        self._restarting_sender: bool = False
        self._retry_policy = retry_policy or P2PRetryPolicy(max_retries=2)
        self._timeouts = timeouts or P2PTimeouts(connection=5.0, send=30.0, receive=15.0)
        self._cache_ttl = cache_ttl
        self._max_cache_size = max_cache_size
        self._p2p_auth_timeout_ms = p2p_auth_timeout_ms if p2p_auth_timeout_ms is not None else P2P_AUTH_TIMEOUT_MS

        # Shared state for receiver subprocess
        self._manager: SyncManager | None = None
        self._metadata_dict: DictProxy | None = None
        self._peer_status_dict: DictProxy | None = None
        self._shm_blocks: dict[str, SharedMemory] = {}
        self._atexit_registered: bool = False

        # Queue for activation push messages received by the receiver subprocess
        self._push_queue: multiprocessing.Queue = _mp_ctx.Queue()

        # Remembered for restart
        self._seed: str | None = None
        self._status_monitor_task: asyncio.Task | None = None

    # ── properties ───────────────────────────────────────────────────

    @property
    def node_ids(self) -> list[str]:
        """All receiver node IDs advertised by this stack."""
        if self._receiver is not None and self._receiver.node_id is not None:
            return [self._receiver.node_id]
        return []

    @property
    def node_id(self) -> str | None:
        """Receiver node ID."""
        if self._receiver is not None:
            return self._receiver.node_id
        return None

    @property
    def sender(self) -> SenderProxy | None:
        """Parent-side sender proxy (same API as PooledSender)."""
        if self._sender_subprocess is not None:
            try:
                return self._sender_subprocess.proxy
            except RuntimeError:
                return None
        return None

    @property
    def peer_status_dict(self) -> DictProxy | None:
        """Shared dict populated by ``/peer/status`` handlers in receiver subprocess."""
        return self._peer_status_dict

    @property
    def push_queue(self) -> multiprocessing.Queue:
        """Queue populated by ``/activation/push`` handlers in receiver subprocess."""
        return self._push_queue

    # ── start / stop / restart ───────────────────────────────────────

    async def start(self, seed: str, timeout: float = 5.0) -> None:
        """Create ReceiverProcess + SenderSubprocess, start both subprocesses."""
        start_ts = time.monotonic()
        logger.info("P2P start: begin")

        self._seed = seed

        # Create shared state for IPC with receiver subprocess
        self._manager = _mp_ctx.Manager()
        self._metadata_dict = self._manager.dict()
        self._peer_status_dict = self._manager.dict()

        # Create sender subprocess
        logger.info("P2P start: creating sender subprocess")
        self._sender_subprocess = SenderSubprocess(
            retry_policy=self._retry_policy,
            timeouts=self._timeouts,
        )

        logger.info("P2P start: starting sender subprocess")
        try:
            await asyncio.wait_for(
                self._sender_subprocess.start(timeout=timeout),
                timeout=20.0,
            )
        except asyncio.TimeoutError:
            logger.error("P2P start: sender start timed out")
            raise
        logger.info("P2P start: sender ready")

        # Create receiver subprocess
        logger.info("P2P start: creating receiver")
        self._receiver = ReceiverProcess(
            seed=seed,
            max_message_size=DEFAULT_MAX_MESSAGE_SIZE,
            cache_ttl=self._cache_ttl,
            max_cache_size=self._max_cache_size,
            p2p_auth_timeout_ms=self._p2p_auth_timeout_ms,
            external_metadata_dict=self._metadata_dict,
            external_peer_status_dict=self._peer_status_dict,
            push_queue=self._push_queue,
        )

        logger.info("P2P start: starting receiver subprocess")
        try:
            node_id = await asyncio.wait_for(
                self._receiver.start(timeout=timeout),
                timeout=max(timeout + 5.0, 20.0),
            )
        except asyncio.TimeoutError:
            logger.error("P2P start: receiver start timed out")
            raise

        logger.info(f"P2P start: receiver ready (node_id={node_id[:16]}...)")

        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

        # Start background task to monitor subprocess status queues
        self._status_monitor_task = asyncio.create_task(
            self._monitor_status_queue(),
            name="P2PStatusMonitor",
        )

        elapsed = time.monotonic() - start_ts
        logger.info(f"P2P started with node_ids: {self.node_ids} (elapsed={elapsed:.2f}s)")

    async def stop(self) -> None:
        """Terminate subprocesses.  Cleans up all SharedMemory."""
        # Cancel status monitor
        if self._status_monitor_task is not None:
            self._status_monitor_task.cancel()
            try:
                await asyncio.wait_for(self._status_monitor_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._status_monitor_task = None

        if self._sender_subprocess:
            try:
                await self._sender_subprocess.stop()
            except Exception as exc:
                logger.warning(f"P2P stop: sender stop failed: {exc}")
            self._sender_subprocess = None

        if self._receiver:
            try:
                await self._receiver.stop()
            except Exception as exc:
                logger.warning(f"P2P stop: receiver stop failed: {exc}")
            self._receiver = None

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

        logger.info("P2P shutdown complete")

    async def restart(self) -> None:
        """Restart the receiver subprocess.

        The SharedMemory cache is preserved — the new subprocess re-attaches
        and can immediately serve cached activations.

        If restart fails twice, raises ``RuntimeError`` to crash the process
        for a process-manager restart.
        """
        if self._seed is None or self._receiver is None:
            raise RuntimeError("Cannot restart P2P stack that was never started")

        timeout = 20.0
        logger.warning("P2P restart: starting receiver restart")

        try:
            node_id = await asyncio.wait_for(
                self._receiver.restart(),
                timeout=timeout,
            )
            logger.info(f"P2P restart: completed successfully (node_id={node_id[:16]}...)")
            self._restarting_receiver = False
            return
        except asyncio.TimeoutError:
            logger.error("P2P restart: timed out")
        except Exception as exc:
            logger.error(f"P2P restart: failed: {exc}")

        # Retry once
        logger.warning("P2P restart: retrying")
        try:
            node_id = await asyncio.wait_for(
                self._receiver.restart(),
                timeout=timeout,
            )
            logger.info(f"P2P restart: retry succeeded (node_id={node_id[:16]}...)")
            self._restarting_receiver = False
            return
        except asyncio.TimeoutError:
            logger.error("P2P restart: retry timed out")
        except Exception as exc:
            logger.error(f"P2P restart: retry failed: {exc}")
        finally:
            self._restarting_receiver = False

        raise RuntimeError("P2P restart failed after all retries — crashing for process manager restart")

    async def restart_sender(self) -> None:
        """Restart the sender subprocess.

        If restart fails twice, raises ``RuntimeError``.
        """
        if self._sender_subprocess is None:
            raise RuntimeError("Cannot restart sender that was never started")

        timeout = 20.0
        logger.warning("P2P sender restart: starting")

        try:
            await asyncio.wait_for(
                self._sender_subprocess.restart(),
                timeout=timeout,
            )
            logger.info("P2P sender restart: completed successfully")
            self._restarting_sender = False
            return
        except asyncio.TimeoutError:
            logger.error("P2P sender restart: timed out")
        except Exception as exc:
            logger.error(f"P2P sender restart: failed: {exc}")

        # Retry once
        logger.warning("P2P sender restart: retrying")
        try:
            await asyncio.wait_for(
                self._sender_subprocess.restart(),
                timeout=timeout,
            )
            logger.info("P2P sender restart: retry succeeded")
            self._restarting_sender = False
            return
        except asyncio.TimeoutError:
            logger.error("P2P sender restart: retry timed out")
        except Exception as exc:
            logger.error(f"P2P sender restart: retry failed: {exc}")
        finally:
            self._restarting_sender = False

        raise RuntimeError("P2P sender restart failed after all retries — crashing for process manager restart")

    # ── epoch boundary cleanup ────────────────────────────────────────

    def drain_push_queue(self) -> int:
        """Drain all pending messages from the push queue.

        Returns the number of messages discarded.  Called at epoch boundaries
        so stale backward-activations from the previous epoch are never
        processed with the new epoch's model weights.
        """
        dropped = 0
        while not self._push_queue.empty():
            try:
                self._push_queue.get_nowait()
                dropped += 1
            except Exception:
                break
        return dropped

    def clear_activation_cache(self) -> int:
        """Remove all entries from the SharedMemory activation cache.

        Returns the number of entries removed.  Called at epoch boundaries so
        stale activations cannot be served to peers after a merge.
        """
        count = len(self._shm_blocks)
        self._cleanup_all_shm()
        return count

    # ── activation cache ─────────────────────────────────────────────

    def cache_activation(self, activation_id: str, tensor_bytes: bytes) -> None:
        """Write activation data to SharedMemory for the receiver subprocess to read."""
        if self._metadata_dict is None:
            raise RuntimeError("P2P stack not started")

        self._evict_expired()

        while len(self._shm_blocks) >= self._max_cache_size:
            oldest_key = next(iter(self._shm_blocks))
            self._remove_entry(oldest_key)
            logger.debug(f"P2P cache: evicted activation {oldest_key}")

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

    # ── SharedMemory helpers ─────────────────────────────────────────

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
                logger.warning(f"P2P cache: SharedMemory cleanup error for {activation_id}: {exc}")
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

    # ── internal ─────────────────────────────────────────────────────

    async def _monitor_status_queue(self) -> None:
        """Poll both subprocess status queues for unhealthy events."""
        while True:
            await asyncio.sleep(1.0)

            # Monitor receiver
            if self._receiver is not None:
                messages = self._receiver.check_status_queue()
                for kind, value in messages:
                    if kind == "unhealthy":
                        await self._on_receiver_unhealthy(value)

                # Check if receiver subprocess died unexpectedly
                if (
                    not self._receiver.is_alive()
                    and not self._restarting_receiver
                    and self._receiver.node_id is not None
                ):
                    logger.error("P2P receiver subprocess died unexpectedly, scheduling restart")
                    await self._on_receiver_unhealthy("subprocess_died")

            # Monitor sender
            if self._sender_subprocess is not None:
                messages = self._sender_subprocess.check_status_queue()
                for kind, value in messages:
                    if kind == "unhealthy":
                        await self._on_sender_unhealthy(value)

                # Check if sender subprocess died unexpectedly
                if (
                    not self._sender_subprocess.is_alive()
                    and not self._restarting_sender
                    and self._sender_subprocess.node_id is not None
                ):
                    logger.error("P2P sender subprocess died unexpectedly, scheduling restart")
                    await self._on_sender_unhealthy("subprocess_died")

    async def _on_receiver_unhealthy(self, health_value: str) -> None:
        """Schedule a restart when the receiver subprocess reports unhealthy."""
        if self._restarting_receiver:
            logger.debug("P2P receiver restart already in progress, skipping")
            return
        self._restarting_receiver = True
        logger.warning(f"P2P receiver unhealthy ({health_value}), scheduling restart")
        task = asyncio.create_task(self.restart(), name="P2PReceiverRestart")
        task.add_done_callback(self._on_restart_done)

    async def _on_sender_unhealthy(self, health_value: str) -> None:
        """Schedule a restart when the sender subprocess reports unhealthy."""
        if self._restarting_sender:
            logger.debug("P2P sender restart already in progress, skipping")
            return
        self._restarting_sender = True
        logger.warning(f"P2P sender unhealthy ({health_value}), scheduling restart")
        task = asyncio.create_task(self.restart_sender(), name="P2PSenderRestart")
        task.add_done_callback(self._on_restart_done)

    def _on_restart_done(self, task: asyncio.Task) -> None:
        """Terminate the process on permanent restart failure."""
        exc = task.exception()
        if exc is not None:
            logger.critical(f"P2P restart failed permanently: {exc} — terminating process")
            os._exit(1)
