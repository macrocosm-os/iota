from __future__ import annotations

from datetime import datetime

from common.snowpipe.base import SnowpipeIndexedMessage


class MinerTelemetry(SnowpipeIndexedMessage):
    """Prometheus/OTEL metric snapshot from a miner, persisted to Snowflake.

    Each row represents one metric sample at a point in time.

    Example::

        MinerTelemetry(
            type="prometheus_snapshot",
            source_service="miner",
            miner_hotkey="5F...",
            run_id="run-abc",
            metric_name="miner_training_loss",
            metric_type="gauge",
            metric_labels={"layer_idx": "3"},
            metric_value=0.42,
        )
    """

    metric_name: str
    metric_type: str  # "counter" | "gauge" | "histogram" | "summary"
    metric_labels: dict | None = None
    metric_value: float
    metric_timestamp: datetime | None = None
