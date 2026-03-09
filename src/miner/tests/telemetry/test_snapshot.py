"""Tests for miner telemetry prometheus registry snapshot collection."""

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge

from common.models.telemetry_models import PrometheusMetricSnapshot
from miner.telemetry.snapshot import snapshot_registry


@pytest.fixture
def fresh_registry():
    return CollectorRegistry()


def test_snapshot_empty_registry(fresh_registry):
    """Empty registry returns empty list."""
    result = snapshot_registry(fresh_registry)
    assert result == []


def test_snapshot_counter_and_gauge(fresh_registry):
    """Counter inc and gauge set produce correct snapshot values."""
    c = Counter("req_total", "Total requests", registry=fresh_registry)
    g = Gauge("temp", "Temperature", registry=fresh_registry)

    c.inc(5)
    g.set(42.5)

    snaps = snapshot_registry(fresh_registry)
    by_name = {s.name: s for s in snaps}

    # Counter exposes req_total and req_created samples
    assert "req_total" in by_name
    assert by_name["req_total"].value == 5.0
    assert by_name["req_total"].type == "counter"

    assert "temp" in by_name
    assert by_name["temp"].value == 42.5
    assert by_name["temp"].type == "gauge"

    # Every snapshot has a timestamp
    for snap in snaps:
        assert isinstance(snap, PrometheusMetricSnapshot)
        assert snap.timestamp is not None


def test_snapshot_with_labels(fresh_registry):
    """Metrics with labels produce correct label dicts in snapshots."""
    g = Gauge("speed", "Speed", labelnames=["direction", "layer_idx"], registry=fresh_registry)
    g.labels(direction="upload", layer_idx="3").set(100.0)
    g.labels(direction="download", layer_idx="3").set(200.0)

    snaps = snapshot_registry(fresh_registry)
    # Filter to only the "speed" samples (exclude _created etc.)
    speed_snaps = [s for s in snaps if s.name == "speed"]

    assert len(speed_snaps) == 2
    labels_set = {frozenset(s.labels.items()) for s in speed_snaps}
    assert frozenset({("direction", "upload"), ("layer_idx", "3")}) in labels_set
    assert frozenset({("direction", "download"), ("layer_idx", "3")}) in labels_set

    upload = next(s for s in speed_snaps if s.labels.get("direction") == "upload")
    assert upload.value == 100.0
