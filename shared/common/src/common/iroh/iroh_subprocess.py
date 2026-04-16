"""IrohSubprocess: base class for running Iroh components in child subprocesses.

Extracted from ReceiverProcess — provides the common subprocess lifecycle:
spawn, wait-for-started, health monitoring, kill (SIGTERM -> SIGKILL), restart.

Subclasses implement ``_worker_target()`` and ``_build_process_args()`` to
define what runs inside the subprocess.

Uses ``multiprocessing.get_context("spawn")`` — must be spawn, not fork.
Fork copies the parent's Rust tokio runtime causing deadlock in
``Iroh.memory_with_options()``.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from loguru import logger

# Force 'spawn' to create a fresh Python interpreter for the child process.
_mp_ctx = multiprocessing.get_context("spawn")


class IrohSubprocess(ABC):
    """Base class for Iroh components that run in a child subprocess.

    Manages the full lifecycle: spawn, wait-for-started, health check polling,
    SIGTERM/SIGKILL escalation, and restart.
    """

    def __init__(self, *, process_name: str = "IrohSubprocess"):
        self._process: multiprocessing.Process | None = None
        self._status_queue: multiprocessing.Queue = _mp_ctx.Queue()
        self._node_id: str | None = None
        self._process_name = process_name

    # ── properties ────────────────────────────────────────────────

    @property
    def node_id(self) -> str | None:
        return self._node_id

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    # ── abstract methods (subclasses must implement) ─────────────

    @abstractmethod
    def _worker_target(self) -> Callable[..., Any]:
        """Return the callable that will be the subprocess entry point."""
        ...

    @abstractmethod
    def _build_process_args(self) -> tuple:
        """Return the args tuple passed to ``Process(target=..., args=...)``."""
        ...

    # ── lifecycle ─────────────────────────────────────────────────

    async def start(self, timeout: float = 15.0) -> str:
        """Spawn the subprocess and wait for it to report its node_id.

        Returns the node_id string sent by the subprocess via
        ``status_queue.put(("started", node_id))``.
        """
        self._drain_queue()

        self._process = _mp_ctx.Process(
            target=self._worker_target(),
            args=self._build_process_args(),
            daemon=True,
            name=self._process_name,
        )
        self._process.start()
        logger.info(f"{self._process_name} spawned (pid={self._process.pid})")

        node_id = await self._wait_for_started(timeout=timeout)
        self._node_id = node_id
        logger.info(f"{self._process_name} ready (node_id={node_id[:16]}...)")
        return node_id

    async def stop(self) -> None:
        """Kill the subprocess."""
        self._kill_subprocess()
        self._node_id = None
        logger.info(f"{self._process_name} stopped")

    async def restart(self, timeout: float = 15.0) -> str:
        """Kill the subprocess and spawn a fresh one.

        Returns the new node_id.
        """
        logger.warning(f"{self._process_name} restart: killing subprocess")
        self._kill_subprocess()
        self._drain_queue()

        self._process = _mp_ctx.Process(
            target=self._worker_target(),
            args=self._build_process_args(),
            daemon=True,
            name=self._process_name,
        )
        self._process.start()
        logger.info(f"{self._process_name} respawned (pid={self._process.pid})")

        node_id = await self._wait_for_started(timeout=timeout)
        self._node_id = node_id
        logger.info(f"{self._process_name} restarted (node_id={node_id[:16]}...)")
        return node_id

    # ── status queue ─────────────────────────────────────────────

    def check_status_queue(self) -> list[tuple[str, str]]:
        """Non-blocking drain of status messages from the subprocess.

        Returns a list of (kind, value) tuples.
        """
        messages: list[tuple[str, str]] = []
        while True:
            try:
                messages.append(self._status_queue.get_nowait())
            except Exception:
                break
        return messages

    # ── internal helpers ─────────────────────────────────────────

    def _kill_subprocess(self) -> None:
        """Stop the child process: SIGTERM first, SIGKILL if it doesn't exit."""
        if self._process is None:
            return

        pid = self._process.pid
        if self._process.is_alive() and pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to {self._process_name} subprocess (pid={pid})")
            except ProcessLookupError:
                self._process = None
                return
            except Exception as exc:
                logger.warning(f"Failed to SIGTERM {self._process_name} subprocess (pid={pid}): {exc}")

            # Give it a moment to exit gracefully
            try:
                self._process.join(timeout=3.0)
            except Exception:
                pass

            if self._process.is_alive():
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.warning(f"Sent SIGKILL to {self._process_name} subprocess (pid={pid}) after SIGTERM timeout")
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    logger.warning(f"Failed to SIGKILL {self._process_name} subprocess (pid={pid}): {exc}")

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
        """Wait for the subprocess to send its ``("started", node_id)`` message."""
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
                    raise RuntimeError(
                        f"{self._process_name} subprocess died during startup " f"(exit code={self._process.exitcode})"
                    )
                continue

            kind, value = msg
            if kind == "started":
                return value
            elif kind == "error":
                raise RuntimeError(f"{self._process_name} subprocess error: {value}")
            # Ignore other message types during startup

        raise TimeoutError(f"{self._process_name} subprocess did not start within {timeout}s")
