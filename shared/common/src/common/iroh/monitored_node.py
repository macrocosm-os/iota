"""Background health monitoring for Iroh nodes."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Callable

from loguru import logger

if TYPE_CHECKING:
    from iroh import Iroh


class NodeHealth(Enum):
    """Health state of a monitored Iroh node."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # status() OK but home_relay() is None
    DEAD = "dead"  # status() hangs/throws


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    health: NodeHealth
    checked_at: float
    status_ok: bool = False
    home_relay: str | None = None
    relay_connected: bool = False
    error: str | None = None


# Callback type: async fn(monitored_node, result) -> None
OnUnhealthy = Callable[["MonitoredNode", HealthCheckResult], Awaitable[None]]


class MonitoredNode:
    """Wrapper around an Iroh node that runs periodic background health checks.

    When the node is detected as unhealthy, the *on_unhealthy* callback is
    invoked so the owner (Sender/Receiver) can reset the node.
    """

    def __init__(
        self,
        node: Iroh | None = None,
        on_unhealthy: OnUnhealthy | None = None,
        check_interval: float = 30.0,
        check_timeout: float = 5.0,
        label: str = "node",
    ):
        self._node: Iroh | None = node
        self._on_unhealthy = on_unhealthy
        self._check_interval = check_interval
        self._check_timeout = check_timeout
        self._label = label

        self._last_check: HealthCheckResult | None = None
        self._consecutive_failures: int = 0
        self._is_healthy: bool = True
        self._monitor_task: asyncio.Task | None = None
        self._stopping: bool = False

        callback_status = "registered" if on_unhealthy is not None else "NOT registered (no recovery)"
        logger.info(f"[{label}] MonitoredNode created, on_unhealthy callback {callback_status}")

    # ── properties ────────────────────────────────────────────────────

    @property
    def node(self) -> Iroh | None:
        return self._node

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    @property
    def last_check(self) -> HealthCheckResult | None:
        return self._last_check

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def is_monitoring(self) -> bool:
        return self._monitor_task is not None and not self._monitor_task.done()

    # ── node management ──────────────────────────────────────────────

    def set_node(self, node: Iroh) -> None:
        """Set or replace the monitored node, resetting health state."""
        self._node = node
        self._consecutive_failures = 0
        self._is_healthy = True
        self._last_check = None

    def clear_node(self) -> None:
        """Clear the node reference (after shutdown)."""
        self._node = None

    # ── monitoring lifecycle ─────────────────────────────────────────

    def start_monitoring(self) -> None:
        """Spawn the background monitor task (idempotent)."""
        if self.is_monitoring:
            return
        self._stopping = False
        self._monitor_task = asyncio.get_event_loop().create_task(self._monitor_loop(), name=f"monitor-{self._label}")

    async def stop_monitoring(self) -> None:
        """Cancel and await the background monitor task."""
        if self._monitor_task is not None:
            self._stopping = True
            # Prevent further recovery attempts while shutting down.
            self._on_unhealthy = None
            self._node = None
            self._monitor_task.cancel()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=self._check_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"[{self._label}] Monitor task did not stop within {self._check_timeout}s; detaching")
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    # ── health check ─────────────────────────────────────────────────

    async def check_health(self) -> HealthCheckResult:
        """Run a single health check against the node."""
        now = time.time()

        if self._node is None:
            result = HealthCheckResult(
                health=NodeHealth.DEAD,
                checked_at=now,
                error="no node set",
            )
            self._record_result(result)
            return result

        # Check node status
        try:
            await asyncio.wait_for(
                self._node.node().status(),
                timeout=self._check_timeout,
            )
            status_ok = True
        except Exception as exc:
            result = HealthCheckResult(
                health=NodeHealth.DEAD,
                checked_at=now,
                error=f"status() failed: {exc}",
            )
            self._record_result(result)
            return result

        # Check relay connectivity
        try:
            relay = await asyncio.wait_for(
                self._node.net().home_relay(),
                timeout=self._check_timeout,
            )
        except Exception:
            # Relay check timed out or failed — node alive but degraded
            result = HealthCheckResult(
                health=NodeHealth.DEGRADED,
                checked_at=now,
                status_ok=True,
                relay_connected=False,
                error="home_relay() timed out",
            )
            self._record_result(result)
            return result

        relay_url = str(relay) if relay else None
        relay_connected = relay_url is not None

        if relay_connected:
            health = NodeHealth.HEALTHY
        else:
            health = NodeHealth.DEGRADED

        result = HealthCheckResult(
            health=health,
            checked_at=now,
            status_ok=status_ok,
            home_relay=relay_url,
            relay_connected=relay_connected,
        )
        self._record_result(result)
        return result

    # ── internals ────────────────────────────────────────────────────

    def _record_result(self, result: HealthCheckResult) -> None:
        """Track state transitions and update counters."""
        was_healthy = self._is_healthy
        self._last_check = result

        if result.health == NodeHealth.HEALTHY:
            if not was_healthy and self._consecutive_failures > 0:
                logger.info(
                    f"[{self._label}] Node recovered after " f"{self._consecutive_failures} consecutive failures"
                )
            self._consecutive_failures = 0
            self._is_healthy = True
        else:
            self._consecutive_failures += 1
            self._is_healthy = False
            logger.warning(
                f"[{self._label}] Node unhealthy: {result.health.value} "
                f"(failures={self._consecutive_failures}, error={result.error})"
            )

    async def _monitor_loop(self) -> None:
        """Background loop that periodically checks node health."""
        while True:
            await asyncio.sleep(self._check_interval)

            if self._stopping:
                return

            if self._node is None:
                continue

            result = await self.check_health()

            if result.health != NodeHealth.HEALTHY:
                if self._on_unhealthy is not None:
                    try:
                        await self._on_unhealthy(self, result)
                    except Exception as exc:
                        logger.error(f"[{self._label}] on_unhealthy callback failed: {exc}")
                else:
                    logger.warning(
                        f"[{self._label}] Node unhealthy but no on_unhealthy callback registered — no recovery possible"
                    )
