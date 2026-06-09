from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from common.snowpipe.messages.lifecycle_events import LifecycleEvent
from common.snowpipe.messages.training_metrics import TrainingMetric


class PrometheusMetricSnapshot(BaseModel):
    """A single metric sample from a miner's prometheus registry."""

    name: str
    type: str  # "counter", "gauge", "histogram", "summary"
    labels: dict[str, str] = {}
    value: float
    timestamp: datetime


class FleetTelemetryPayload(BaseModel):
    """Typed batched telemetry payload from miner to orchestrator."""

    training_metrics: list[TrainingMetric] = []
    lifecycle_events: list[LifecycleEvent] = []
    prometheus_snapshots: list[PrometheusMetricSnapshot] = []
    # Bittensor hotkey name reported by the miner (e.g. "miner-52"), distinct
    # from the wallet/coldkey name. Operator-chosen, treated as display
    # metadata only — the authenticated ss58 hotkey remains the trust
    # boundary. Used as a Pushgateway grouping_key dimension so dashboards
    # can label series by the human-readable per-miner identifier.
    hotkey_name: str | None = None


class FleetTelemetryResponse(BaseModel):
    """Acknowledgement from orchestrator."""

    accepted_snowpipe: int = 0
    accepted_prometheus: int = 0
    errors: list[str] = []
