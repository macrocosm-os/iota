"""P2PStack: lifecycle manager for a ReceiverProcess + PooledSender pair.

The Receiver runs in a child subprocess for fault isolation — when the Iroh
Rust QUIC/DERP stack becomes poisoned, we ``os.kill()`` the subprocess and
spawn a fresh one without restarting the miner.  The Sender stays in the
main process (bare nodes don't exhibit the hang).
"""

from __future__ import annotations

import asyncio
import os
import time

from loguru import logger

from common.iroh.receiver_process import ReceiverProcess
from common.iroh.sender import PooledSender
from common.iroh.retry import P2PRetryPolicy, P2PTimeouts
from common.iroh.settings import DEFAULT_MAX_MESSAGE_SIZE, P2P_AUTH_TIMEOUT_MS


class P2PStack:
    """Owns a :class:`ReceiverProcess` + :class:`PooledSender` and exposes a
    clean start/stop/restart API.

    The Receiver runs in a subprocess.  The activation cache lives in the
    main process (SharedMemory blocks) and is shared with the subprocess via
    a Manager dict.
    """

    def __init__(
        self,
        retry_policy: P2PRetryPolicy | None = None,
        timeouts: P2PTimeouts | None = None,
        cache_ttl: float = 3000.0,
        max_cache_size: int = 100,
        p2p_auth_timeout_ms: int | None = None,
    ):
        self._receiver_proc: ReceiverProcess | None = None
        self._sender: PooledSender | None = None
        self._restarting: bool = False
        self._retry_policy = retry_policy or P2PRetryPolicy(max_retries=2)
        self._timeouts = timeouts or P2PTimeouts(connection=5.0, send=30.0, receive=15.0)
        self._cache_ttl = cache_ttl
        self._max_cache_size = max_cache_size
        self._p2p_auth_timeout_ms = p2p_auth_timeout_ms if p2p_auth_timeout_ms is not None else P2P_AUTH_TIMEOUT_MS

        # Remembered for restart
        self._seed: str | None = None
        self._status_monitor_task: asyncio.Task | None = None

    # ── properties ───────────────────────────────────────────────────

    @property
    def node_id(self) -> str | None:
        if self._receiver_proc is not None:
            return self._receiver_proc.node_id
        return None

    @property
    def sender(self) -> PooledSender | None:
        return self._sender

    # ── start / stop / restart ───────────────────────────────────────

    async def start(self, seed: str) -> None:
        """Create ReceiverProcess + Sender, start receiver subprocess."""
        start_ts = time.monotonic()
        logger.info("P2P start: begin")

        self._seed = seed

        logger.info("P2P start: creating receiver subprocess")
        self._receiver_proc = ReceiverProcess(
            seed=seed,
            max_message_size=DEFAULT_MAX_MESSAGE_SIZE,
            cache_ttl=self._cache_ttl,
            max_cache_size=self._max_cache_size,
            p2p_auth_timeout_ms=self._p2p_auth_timeout_ms,
        )

        logger.info("P2P start: creating sender")
        self._sender = PooledSender(
            retry_policy=self._retry_policy,
            timeouts=self._timeouts,
        )

        logger.info("P2P start: starting receiver subprocess")
        try:
            node_id = await asyncio.wait_for(
                self._receiver_proc.start(),
                timeout=20.0,
            )
        except asyncio.TimeoutError:
            logger.error("P2P start: receiver subprocess start timed out after 20s")
            raise

        logger.info(f"P2P start: receiver ready (node_id={node_id[:16]}...)")

        # Start background task to monitor subprocess status queue
        self._status_monitor_task = asyncio.create_task(
            self._monitor_status_queue(),
            name="P2PStatusMonitor",
        )

        elapsed = time.monotonic() - start_ts
        logger.info(f"P2P started with node_id: {self.node_id} (elapsed={elapsed:.2f}s)")

    async def stop(self) -> None:
        """Terminate subprocess + sender.  Cleans up all SharedMemory."""
        # Cancel status monitor
        if self._status_monitor_task is not None:
            self._status_monitor_task.cancel()
            try:
                await asyncio.wait_for(self._status_monitor_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._status_monitor_task = None

        if self._sender:
            try:
                await self._sender.force_destroy_node()
            except Exception as exc:
                logger.warning(f"P2P stop: sender force-destroy failed: {exc}")
            self._sender = None

        if self._receiver_proc:
            try:
                await self._receiver_proc.stop()
            except Exception as exc:
                logger.warning(f"P2P stop: receiver process stop failed: {exc}")
            self._receiver_proc = None

        logger.info("P2P shutdown complete")

    async def restart(self) -> None:
        """Kill the receiver subprocess and spawn a new one.

        The SharedMemory cache is preserved — the new subprocess re-attaches
        and can immediately serve cached activations.  Same seed -> same
        node_id, so no orchestrator notification needed.

        If restart fails twice, raises ``RuntimeError`` to crash the process
        for a process-manager restart.
        """
        if self._seed is None or self._receiver_proc is None:
            raise RuntimeError("Cannot restart P2P stack that was never started")

        logger.warning("P2P restart: starting")

        try:
            node_id = await asyncio.wait_for(
                self._receiver_proc.restart(),
                timeout=20.0,
            )
            logger.info(f"P2P restart: completed successfully (node_id={node_id[:16]}...)")
            self._restarting = False
            return
        except asyncio.TimeoutError:
            logger.error("P2P restart: timed out after 20s")
        except Exception as exc:
            logger.error(f"P2P restart: failed: {exc}")

        # Retry once
        logger.warning("P2P restart: retrying")
        try:
            node_id = await asyncio.wait_for(
                self._receiver_proc.restart(),
                timeout=20.0,
            )
            logger.info(f"P2P restart: retry succeeded (node_id={node_id[:16]}...)")
            self._restarting = False
            return
        except asyncio.TimeoutError:
            logger.error("P2P restart: retry timed out after 20s")
        except Exception as exc:
            logger.error(f"P2P restart: retry failed: {exc}")
        finally:
            self._restarting = False

        raise RuntimeError("P2P restart failed after all retries — crashing for process manager restart")

    # ── cache delegation ─────────────────────────────────────────────

    def cache_activation(self, activation_id: str, tensor_bytes: bytes) -> None:
        """Write activation data to SharedMemory for the subprocess to read."""
        if self._receiver_proc is None:
            raise RuntimeError("P2P stack not started")
        self._receiver_proc.cache_activation(activation_id, tensor_bytes)

    # ── internal ─────────────────────────────────────────────────────

    async def _monitor_status_queue(self) -> None:
        """Poll the subprocess status queue for unhealthy events."""
        while True:
            await asyncio.sleep(1.0)

            if self._receiver_proc is None:
                continue

            messages = self._receiver_proc.check_status_queue()
            for kind, value in messages:
                if kind == "unhealthy":
                    await self._on_receiver_unhealthy(value)

    async def _on_receiver_unhealthy(self, health_value: str) -> None:
        """Schedule a restart when the receiver subprocess reports unhealthy."""
        if self._restarting:
            logger.debug("P2P restart already in progress, skipping")
            return
        self._restarting = True
        logger.warning(f"P2P receiver unhealthy ({health_value}), scheduling restart")
        task = asyncio.create_task(self.restart(), name="P2PRestart")
        task.add_done_callback(self._on_restart_done)

    def _on_restart_done(self, task: asyncio.Task) -> None:
        """Terminate the process on permanent restart failure."""
        exc = task.exception()
        if exc is not None:
            logger.critical(f"P2P restart failed permanently: {exc} — terminating process")
            os._exit(1)
