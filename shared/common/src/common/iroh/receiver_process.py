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

from __future__ import annotations

import atexit
import asyncio
import hashlib
import multiprocessing
import os
import signal
import time
from multiprocessing.managers import DictProxy, SyncManager
from multiprocessing.shared_memory import SharedMemory
from typing import Any

from loguru import logger

from common.utils.gpu_process_utils import remove_shm_manifest, write_shm_manifest

# Force 'spawn' to create a fresh Python interpreter for the child process.
# The default 'fork' on Linux copies the parent's Rust tokio runtime and
# iroh_ffi global state (already initialised by the PooledSender), which
# causes Iroh.memory_with_options() to deadlock in the child.
# freeze_support() in main_pool.py handles the frozen-binary case.
_mp_ctx = multiprocessing.get_context("spawn")

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
    status_queue: multiprocessing.Queue,
    cache_ttl: float,
    p2p_auth_timeout_ms: int = 30000,
) -> None:
    """Entry point for the receiver subprocess.

    A fresh process means a completely clean Rust runtime — no poisoned
    QUIC/DERP state carried over from a previous incarnation.
    """
    # Allow SIGTERM to trigger KeyboardInterrupt so asyncio.run can clean up
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    try:
        asyncio.run(_run_receiver(seed, max_message_size, metadata_dict, status_queue, cache_ttl, p2p_auth_timeout_ms))
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
    status_queue: multiprocessing.Queue,
    cache_ttl: float,
    p2p_auth_timeout_ms: int = 30000,
) -> None:
    """Async core of the receiver subprocess."""
    from common.iroh.receiver import Receiver

    receiver = Receiver(seed=seed, max_message_size=max_message_size)

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

    await receiver.start(callback_function=handle_request, on_unhealthy=on_unhealthy)
    status_queue.put(("started", receiver.node_id))
    await receiver.serve_forever()


# ---------------------------------------------------------------------------
# Parent-process manager
# ---------------------------------------------------------------------------


class ReceiverProcess:
    """Runs an Iroh Receiver in a child subprocess for fault isolation.

    The parent process owns the activation cache (SharedMemory blocks +
    metadata dict).  The subprocess only *reads* from them to serve
    incoming P2P requests.  When the subprocess becomes unhealthy we
    ``os.kill()`` it and spawn a fresh one — the cache is preserved.
    """

    def __init__(
        self,
        seed: str,
        max_message_size: int,
        cache_ttl: float,
        max_cache_size: int = 100,
        p2p_auth_timeout_ms: int = 30000,
    ):
        self._seed = seed
        self._max_message_size = max_message_size
        self._cache_ttl = cache_ttl
        self._max_cache_size = max_cache_size
        self._p2p_auth_timeout_ms = p2p_auth_timeout_ms

        self._process: multiprocessing.Process | None = None
        self._status_queue: multiprocessing.Queue = _mp_ctx.Queue()
        self._manager: SyncManager | None = None
        self._metadata_dict: DictProxy | None = None
        self._shm_blocks: dict[str, SharedMemory] = {}
        self._node_id: str | None = None
        self._atexit_registered: bool = False

    # ── properties ────────────────────────────────────────────────

    @property
    def node_id(self) -> str | None:
        return self._node_id

    # ── lifecycle ─────────────────────────────────────────────────

    async def start(self) -> str:
        """Start the Manager, spawn the subprocess, and return the node_id."""
        if self._manager is None:
            self._manager = _mp_ctx.Manager()
            self._metadata_dict = self._manager.dict()

        # Drain any leftover messages from a previous incarnation
        self._drain_queue()

        self._process = _mp_ctx.Process(
            target=_receiver_worker,
            args=(
                self._seed,
                self._max_message_size,
                self._metadata_dict,
                self._status_queue,
                self._cache_ttl,
                self._p2p_auth_timeout_ms,
            ),
            daemon=True,
            name="P2PReceiver",
        )
        self._process.start()
        logger.info(f"ReceiverProcess spawned (pid={self._process.pid})")

        # Register atexit handler so shared memory is cleaned up even if
        # the process exits without a graceful stop() (e.g. crash, SIGTERM).
        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

        # Wait for the subprocess to report its node_id
        node_id = await self._wait_for_started(timeout=15.0)
        self._node_id = node_id
        logger.info(f"ReceiverProcess ready (node_id={node_id[:16]}...)")
        return node_id

    async def stop(self) -> None:
        """Terminate the subprocess and clean up all SharedMemory."""
        self._kill_subprocess()
        self._cleanup_all_shm()

        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None
            self._metadata_dict = None

        self._node_id = None

        # Clean up manifest and atexit handler
        remove_shm_manifest()
        if self._atexit_registered:
            atexit.unregister(self._atexit_cleanup)
            self._atexit_registered = False

        logger.info("ReceiverProcess stopped and all SharedMemory cleaned up")

    async def restart(self) -> str:
        """Kill the subprocess and spawn a new one.  Cache is preserved.

        Same seed -> same node_id, so no orchestrator notification needed.
        """
        logger.warning("ReceiverProcess restart: killing subprocess")
        self._kill_subprocess()

        # Drain stale messages
        self._drain_queue()

        # Spawn a new subprocess — it re-attaches to the existing
        # metadata_dict and can immediately serve cached activations.
        self._process = _mp_ctx.Process(
            target=_receiver_worker,
            args=(
                self._seed,
                self._max_message_size,
                self._metadata_dict,
                self._status_queue,
                self._cache_ttl,
                self._p2p_auth_timeout_ms,
            ),
            daemon=True,
            name="P2PReceiver",
        )
        self._process.start()
        logger.info(f"ReceiverProcess respawned (pid={self._process.pid})")

        node_id = await self._wait_for_started(timeout=15.0)
        self._node_id = node_id
        logger.info(f"ReceiverProcess restarted (node_id={node_id[:16]}...)")
        return node_id

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

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

    # ── internal helpers ──────────────────────────────────────────

    def _kill_subprocess(self) -> None:
        """Stop the child process: SIGTERM first, SIGKILL if it doesn't exit."""
        if self._process is None:
            return

        pid = self._process.pid
        if self._process.is_alive() and pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to receiver subprocess (pid={pid})")
            except ProcessLookupError:
                self._process = None
                return
            except Exception as exc:
                logger.warning(f"Failed to SIGTERM receiver subprocess (pid={pid}): {exc}")

            # Give it a moment to exit gracefully
            try:
                self._process.join(timeout=3.0)
            except Exception:
                pass

            if self._process.is_alive():
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.warning(f"Sent SIGKILL to receiver subprocess (pid={pid}) after SIGTERM timeout")
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    logger.warning(f"Failed to SIGKILL receiver subprocess (pid={pid}): {exc}")

        # Reap the zombie
        try:
            self._process.join(timeout=5.0)
        except Exception:
            pass

        self._process = None

    def _drain_queue(self) -> None:
        """Empty the status queue of stale messages."""
        while True:
            try:
                self._status_queue.get_nowait()
            except Exception:
                break

    async def _wait_for_started(self, timeout: float) -> str:
        """Wait for the subprocess to send its (\"started\", node_id) message."""
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(
                    loop.run_in_executor(None, self._status_queue.get, True, min(remaining, 1.0)),
                    timeout=min(remaining, 2.0),
                )
            except (asyncio.TimeoutError, Exception):
                # Check if process died
                if self._process is not None and not self._process.is_alive():
                    raise RuntimeError(f"Receiver subprocess died during startup (exit code={self._process.exitcode})")
                continue

            kind, value = msg
            if kind == "started":
                return value
            elif kind == "error":
                raise RuntimeError(f"Receiver subprocess error: {value}")
            # Ignore other message types during startup

        raise TimeoutError(f"Receiver subprocess did not start within {timeout}s")

    def check_status_queue(self) -> list[tuple[str, str]]:
        """Non-blocking drain of status messages from the subprocess.

        Returns a list of (kind, value) tuples.  Used by P2PStack to
        detect unhealthy events without blocking.
        """
        messages: list[tuple[str, str]] = []
        while True:
            try:
                messages.append(self._status_queue.get_nowait())
            except Exception:
                break
        return messages
