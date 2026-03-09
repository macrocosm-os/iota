"""
Statistics tracker for the miner pool dashboard.

The tracker is designed to be lightweight and safe to call from async code paths
without imposing significant locking overhead. Callers simply invoke the record
methods whenever an event occurs (activation processed, bytes transferred,
loss computed). Aggregated views are exposed via properties and helper methods
that the dashboard can query on each refresh cycle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING, Deque, Iterable

import torch
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from common.iroh.timings import P2POperationTimings


@dataclass(slots=True)
class ActivationSample:
    """Represents a single activation processing event."""

    timestamp: float
    direction: str  # "forward" or "backward"


@dataclass(slots=True)
class LossSample:
    """Stores a scalar loss value with the time it was recorded."""

    timestamp: float
    value: float


class ActivationTimingStage(BaseModel):
    start: float | None = None
    end: float | None = None
    duration: float | None = None
    cache_len: int | None = None
    forward_queue_len: int | None = None
    backward_queue_len: int | None = None


class P2PTimingDetail(BaseModel):
    """Per-phase timing breakdown for a single P2P operation.

    Populated from ``P2POperationTimings`` (which lives in the iroh
    package and is transport-layer aware).  This Pydantic mirror is what
    gets persisted as JSON inside ``activation_stats``.
    """

    # per-phase durations (seconds)
    connection_duration: float | None = None
    stream_open_duration: float | None = None
    send_duration: float | None = None
    receive_duration: float | None = None

    # overall span
    total_start: float | None = None
    total_end: float | None = None
    total_duration: float | None = None

    # retry metadata
    attempt_count: int = 0
    retry_count: int = 0
    total_backoff_time: float = 0.0
    errors: list[str] = Field(default_factory=list)

    # payload metadata
    bytes_sent: int | None = None
    bytes_received: int | None = None

    @classmethod
    def from_operation_timings(cls, timings: "P2POperationTimings") -> "P2PTimingDetail":
        """Create from the mutable dataclass produced by the iroh Sender."""
        return cls(
            connection_duration=timings.connection_duration,
            stream_open_duration=timings.stream_open_duration,
            send_duration=timings.send_duration,
            receive_duration=timings.receive_duration,
            total_start=timings.total_start,
            total_end=timings.total_end,
            total_duration=timings.total_duration,
            attempt_count=timings.attempt_count,
            retry_count=timings.retry_count,
            total_backoff_time=timings.total_backoff_time,
            errors=list(timings.errors),
            bytes_sent=timings.bytes_sent,
            bytes_received=timings.bytes_received,
        )


class ActivationTiming(BaseModel):
    queue: ActivationTimingStage = Field(default_factory=ActivationTimingStage)
    download: ActivationTimingStage = Field(default_factory=ActivationTimingStage)
    forward: ActivationTimingStage = Field(default_factory=ActivationTimingStage)
    backward: ActivationTimingStage = Field(default_factory=ActivationTimingStage)
    epistula: ActivationTimingStage = Field(default_factory=ActivationTimingStage)
    p2p: P2PTimingDetail = Field(default_factory=P2PTimingDetail)
    publish: ActivationTimingStage = Field(default_factory=ActivationTimingStage)

    # Sub-phases of backward() for last-layer bottleneck analysis
    backward_gpu_setup: ActivationTimingStage = Field(default_factory=ActivationTimingStage)
    backward_forward: ActivationTimingStage = Field(default_factory=ActivationTimingStage)
    backward_loss: ActivationTimingStage = Field(default_factory=ActivationTimingStage)
    backward_pass: ActivationTimingStage = Field(default_factory=ActivationTimingStage)
    backward_grad_extract: ActivationTimingStage = Field(default_factory=ActivationTimingStage)

    # Sub-phase of download: S3 sample/target download (last layer only)
    sample_download: ActivationTimingStage = Field(default_factory=ActivationTimingStage)


class ActivationStats(BaseModel):
    time_received: float | None = None
    direction: str | None = None
    timing: ActivationTiming = Field(default_factory=ActivationTiming)


@dataclass(slots=True)
class StatsTracker:
    """Accumulates miner runtime statistics for dashboard consumption."""

    activation_history_window: float = 300.0  # seconds
    loss_history_size: int = 50
    current_layer: int | None = None
    current_phase: str | None = None
    remote_epoch: int | None = None
    local_epoch: int | None = None
    run_id: str | None = None
    activation_stats: dict[str, ActivationStats] = field(default_factory=dict)
    _forward_count: int = 0
    _backward_count: int = 0
    _download_bytes: int = 0
    _p2p_bytes: int = 0
    _activations: Deque[ActivationSample] = field(default_factory=deque)
    _loss_history: Deque[LossSample] = field(default_factory=lambda: deque(maxlen=50))

    def __post_init__(self) -> None:
        # Ensure deque maxlen respects configuration
        if self._loss_history.maxlen != self.loss_history_size:
            self._loss_history = deque(self._loss_history, maxlen=self.loss_history_size)

    # --- Recording helpers -------------------------------------------------

    def record_forward(self, *, timestamp: float | None = None) -> None:
        """Record a forward activation being processed."""
        self._forward_count += 1
        self._record_activation(direction="forward", timestamp=timestamp)

    def record_backward(self, *, timestamp: float | None = None) -> None:
        """Record a backward activation being processed."""
        self._backward_count += 1
        self._record_activation(direction="backward", timestamp=timestamp)

    def record_download(self, byte_count: int) -> None:
        """Add bytes downloaded."""
        if byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        self._download_bytes += byte_count

    def record_p2p_transfer(self, byte_count: int) -> None:
        """Add bytes transferred via P2P."""
        if byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        self._p2p_bytes += byte_count

    def record_p2p_operation(
        self,
        activation_id: str,
        timings: "P2POperationTimings",
        *,
        direction: str | None = None,
    ) -> None:
        """Record detailed P2P timing for an activation.

        Populates the ``timing.p2p`` field on the activation's
        ``ActivationStats`` with a full per-phase breakdown and also
        updates the aggregate ``_p2p_bytes`` counter.

        Args:
            activation_id: The activation this operation belongs to.
            timings: The mutable timing record filled by the Sender.
            direction: Optional direction hint forwarded to
                ``ensure_activation_stats``.
        """
        stats = self.ensure_activation_stats(activation_id, direction=direction)
        stats.timing.p2p = P2PTimingDetail.from_operation_timings(timings)
        if timings.bytes_received is not None:
            self.record_p2p_transfer(timings.bytes_received)

    def record_loss(self, loss_value: float, *, timestamp: float | None = None) -> None:
        """Record a loss metric for last-layer miners."""
        if timestamp is None:
            timestamp = monotonic()
        self._loss_history.append(LossSample(timestamp=timestamp, value=loss_value))

    def set_layer(self, layer: int | None) -> None:
        self.current_layer = layer

    def set_phase(self, phase: object | None) -> None:
        self.current_phase = str(phase) if phase is not None else None

    def set_remote_epoch(self, epoch: int | None) -> None:
        self.remote_epoch = epoch

    def set_local_epoch(self, epoch: int | None) -> None:
        self.local_epoch = epoch

    def set_run_id(self, run_id: object | None) -> None:
        self.run_id = str(run_id) if run_id is not None else None

    def reset(self) -> None:
        """
        Reset the StatsTracker to its initial state.

        Resets all counters, state fields, and clears history collections.
        Configuration fields (activation_history_window, loss_history_size) are preserved.
        """
        # TODO: Add config settings for these?
        self.activation_history_window = 300.0
        self.loss_history_size = 50

        # Reset state fields
        self.current_layer = None
        self.current_phase = None
        self.remote_epoch = None
        self.local_epoch = None
        self.run_id = None

        # Reset counters
        self._forward_count = 0
        self._backward_count = 0
        self._download_bytes = 0
        self._p2p_bytes = 0

        # Clear history collections (preserving maxlen for _loss_history)
        self._activations.clear()
        self._loss_history.clear()
        self.activation_stats.clear()

    # --- Aggregated views --------------------------------------------------

    @property
    def forward_count(self) -> int:
        return self._forward_count

    @property
    def backward_count(self) -> int:
        return self._backward_count

    @property
    def total_activations(self) -> int:
        return self._forward_count + self._backward_count

    @property
    def download_bytes(self) -> int:
        return self._download_bytes

    @property
    def p2p_bytes(self) -> int:
        return self._p2p_bytes

    def activation_rate(self, *, window_seconds: float | None = None) -> float:
        """
        Calculate activations processed per minute over the requested window.

        Args:
            window_seconds: Override for the lookback window; defaults to the
                tracker configuration.
        """
        if window_seconds is None:
            window_seconds = self.activation_history_window

        now = monotonic()
        self._prune_activations(now, window_seconds)

        if not self._activations:
            return 0.0

        count = sum(1 for sample in self._activations if now - sample.timestamp <= window_seconds)
        if count == 0 or window_seconds <= 0:
            return 0.0

        activations_per_second = count / min(window_seconds, now - self._activations[0].timestamp + 1e-6)
        return activations_per_second * 60.0

    def loss_history(self) -> list[LossSample]:
        """Return a copy of the recorded loss history."""
        return list(self._loss_history)

    def loss_summary(self) -> dict[str, float] | None:
        """Return min/avg/max loss statistics if available."""
        if not self._loss_history:
            return None
        losses = [sample.value for sample in self._loss_history]
        count = len(losses)
        return {
            "count": float(count),
            "min": min(losses),
            "max": max(losses),
            "avg": sum(losses) / count,
            "latest": losses[-1],
        }

    # --- Internal helpers --------------------------------------------------

    def _record_activation(self, *, direction: str, timestamp: float | None) -> None:
        if timestamp is None:
            timestamp = monotonic()
        self._activations.append(ActivationSample(timestamp=timestamp, direction=direction))
        self._prune_activations(timestamp, self.activation_history_window)

    def ensure_activation_stats(
        self,
        activation_id: str,
        *,
        direction: str | None = None,
        time_received: float | None = None,
    ) -> ActivationStats:
        stats = self.activation_stats.get(activation_id)
        if stats is None:
            stats = ActivationStats(time_received=time_received, direction=direction)
            self.activation_stats[activation_id] = stats
        else:
            if direction is not None:
                stats.direction = direction
            if time_received is not None:
                stats.time_received = time_received
        return stats

    def get_activation_stats_payload(self, activation_id: str) -> dict | None:
        stats = self.activation_stats.get(activation_id)
        if stats is None:
            return None
        return stats.model_dump()

    def _prune_activations(self, current_time: float, window_seconds: float) -> None:
        """Drop activation samples older than the requested window."""
        cutoff = current_time - window_seconds
        while self._activations and self._activations[0].timestamp < cutoff:
            self._activations.popleft()


def total_bytes(samples: Iterable[memoryview | bytes | bytearray]) -> int:
    """
    Utility helper to sum byte lengths from iterable payloads.

    Not yet used, but exposed for future integration when we aggregate sizes
    from multiple activation tensors or serialized uploads.
    """
    total = 0
    for sample in samples:
        total += len(sample)
    return total


def tensor_num_bytes(tensor: torch.Tensor | None) -> int:
    """Return the number of bytes occupied by a tensor."""
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.nelement()
