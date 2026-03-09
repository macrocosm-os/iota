"""Miner-side telemetry buffer that flushes to the orchestrator."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger
from prometheus_client import CollectorRegistry

from common.models.telemetry_models import (
    FleetTelemetryPayload,
)
from common.snowpipe.base import SnowpipeMessage
from common.snowpipe.messages.lifecycle_events import LifecycleEvent
from common.snowpipe.messages.training_metrics import TrainingMetric
from miner.telemetry.metric_registry import MINER_REGISTRY, TELEMETRY_BUFFER_SIZE, TELEMETRY_FLUSHES_TOTAL
from miner.telemetry.snapshot import snapshot_registry

if TYPE_CHECKING:
    from substrateinterface.keypair import Keypair


class TelemetryBufferService:
    """Buffers SnowpipeMessages, auto-flushing to the orchestrator's
    /fleet-telemetry endpoint.

    Flush triggers:
        - Time since last flush >= flush_interval_sec (default 15s)
        - Explicit flush() call

    Prometheus metrics are snapshotted once per flush (not buffered separately).
    """

    def __init__(
        self,
        hotkey: Keypair,
        *,
        max_buffer_size: int = 1000,
        flush_interval_sec: float = 15.0,
        registry: CollectorRegistry = MINER_REGISTRY,
        is_mounted: bool = False,
        electron_version: str | None = None,
    ):
        self._hotkey = hotkey
        self._max_buffer_size = max_buffer_size
        self._flush_interval_sec = flush_interval_sec
        self._registry = registry
        self._is_mounted = is_mounted
        self._electron_version = electron_version

        # Typed buffers
        self._training_metrics: list[TrainingMetric] = []
        self._lifecycle_events: list[LifecycleEvent] = []

        # Tasks
        self._flush_task: asyncio.Task | None = None
        self._running = False

    @property
    def buffer_size(self) -> int:
        return len(self._training_metrics) + len(self._lifecycle_events)

    def log(self, message: SnowpipeMessage) -> None:
        """Add a SnowpipeMessage to the buffer. Non-blocking."""
        if isinstance(message, TrainingMetric):
            self._training_metrics.append(message)
        elif isinstance(message, LifecycleEvent):
            self._lifecycle_events.append(message)
        else:
            logger.warning(f"TelemetryBuffer: unsupported message type '{type(message).__name__}', dropping")
            return
        TELEMETRY_BUFFER_SIZE.set(self.buffer_size)

    def log_many(self, messages: list[SnowpipeMessage]) -> None:
        """Add multiple SnowpipeMessages to the buffer. Non-blocking."""
        for msg in messages:
            self.log(msg)

    async def start(self) -> None:
        """Start background flush task."""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop(), name="telemetry-flush")
        logger.info(
            f"TelemetryBufferService started (flush_interval={self._flush_interval_sec}s, "
            f"max_buffer={self._max_buffer_size})"
        )

    async def stop(self) -> None:
        """Stop background task and perform final flush."""
        self._running = False
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._do_flush()
        logger.info("TelemetryBufferService stopped")

    async def flush(self) -> None:
        """Explicit flush trigger."""
        await self._do_flush()

    # --- Background loop ---

    async def _flush_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval_sec)
                await self._do_flush()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in telemetry flush loop")
                await asyncio.sleep(5)

    async def _do_flush(self) -> None:
        # Snapshot prometheus metrics right before flush
        prometheus_snapshots = snapshot_registry(self._registry)

        if self.buffer_size == 0 and not prometheus_snapshots:
            return

        # Swap buffers (atomic in single-threaded async)
        training_batch = self._training_metrics
        lifecycle_batch = self._lifecycle_events
        self._training_metrics = []
        self._lifecycle_events = []
        TELEMETRY_BUFFER_SIZE.set(0)

        payload = FleetTelemetryPayload(
            training_metrics=training_batch,
            lifecycle_events=lifecycle_batch,
            prometheus_snapshots=prometheus_snapshots,
        )

        try:
            from subnet.common_api_client import CommonAPIClient

            await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/fleet-telemetry",
                hotkey=self._hotkey,
                body=payload.model_dump(mode="json"),
                is_mounted=self._is_mounted,
                electron_version=self._electron_version,
            )
            TELEMETRY_FLUSHES_TOTAL.labels(status="success").inc()
            total = len(training_batch) + len(lifecycle_batch) + len(prometheus_snapshots)
            logger.debug(
                f"Telemetry flush: {len(training_batch)} training, "
                f"{len(lifecycle_batch)} lifecycle, "
                f"{len(prometheus_snapshots)} prometheus (total={total})"
            )
        except Exception:
            # Re-buffer snowpipe messages on failure up to 2x max to avoid unbounded growth
            max_rebuffer = self._max_buffer_size * 2
            snowpipe_count = len(training_batch) + len(lifecycle_batch)
            if self.buffer_size + snowpipe_count <= max_rebuffer:
                self._training_metrics = training_batch + self._training_metrics
                self._lifecycle_events = lifecycle_batch + self._lifecycle_events
            else:
                logger.warning(f"Dropping {snowpipe_count} telemetry messages (buffer overflow after flush failure)")
            TELEMETRY_FLUSHES_TOTAL.labels(status="error").inc()
            TELEMETRY_BUFFER_SIZE.set(self.buffer_size)
            logger.warning("Telemetry flush failed, messages re-buffered", exc_info=True)
