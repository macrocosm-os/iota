"""Collect a snapshot of all metrics from a prometheus CollectorRegistry."""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger
from prometheus_client import CollectorRegistry

from common.models.telemetry_models import PrometheusMetricSnapshot


def snapshot_registry(registry: CollectorRegistry) -> list[PrometheusMetricSnapshot]:
    """Iterate MetricFamily objects and produce one snapshot per sample.

    Never raises — malformed metrics are skipped with a warning.
    """
    now = datetime.now(timezone.utc)
    snapshots: list[PrometheusMetricSnapshot] = []

    try:
        families = list(registry.collect())
    except Exception:
        logger.warning("Failed to collect metrics from registry", exc_info=True)
        return snapshots

    for metric_family in families:
        try:
            metric_type = metric_family.type
            for sample in metric_family.samples:
                snapshots.append(
                    PrometheusMetricSnapshot(
                        name=sample.name,
                        type=metric_type,
                        labels=dict(sample.labels) if sample.labels else {},
                        value=float(sample.value),
                        timestamp=now,
                    )
                )
        except Exception:
            logger.warning(f"Failed to snapshot metric family '{getattr(metric_family, 'name', '?')}'", exc_info=True)

    return snapshots
