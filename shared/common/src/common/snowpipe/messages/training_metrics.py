from __future__ import annotations

from common.snowpipe.base import SnowpipeIndexedMessage


class TrainingMetric(SnowpipeIndexedMessage):
    """Wide-table message for numeric training metrics (wandb.log equivalent).

    Callers set ``type`` to categorize the metric (e.g. ``"loss_report"``,
    ``"epoch_loss_summary"``, ``"miner_score"``, ``"activation_count"``).

    Example::

        TrainingMetric(
            type="miner_score", source_service="scheduler",
            run_id="run-abc", epoch=5, miner_hotkey="5F...",
            metric_name="score", metric_value={"value": 0.85},
        )
    """

    metric_name: str
    metric_value: dict | None = None
    details: dict | None = None
