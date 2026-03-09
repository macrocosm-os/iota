"""Miner-side Prometheus metrics.

Module-level constants on a dedicated CollectorRegistry so snapshots
only contain miner metrics (not python_gc_*, process_*, etc.).
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

PREFIX = "miner_"

MINER_REGISTRY = CollectorRegistry()

# --- Training ---
TRAINING_LOSS = Gauge(
    f"{PREFIX}training_loss",
    "Latest training loss value",
    labelnames=["layer_idx"],
    registry=MINER_REGISTRY,
)

ACTIVATIONS_PROCESSED_TOTAL = Counter(
    f"{PREFIX}activations_processed_total",
    "Total activations processed",
    labelnames=["direction", "layer_idx"],
    registry=MINER_REGISTRY,
)

FORWARD_PASS_DURATION_SECONDS = Histogram(
    f"{PREFIX}forward_pass_duration_seconds",
    "Forward pass latency in seconds",
    labelnames=["layer_idx"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    registry=MINER_REGISTRY,
)

BACKWARD_PASS_DURATION_SECONDS = Histogram(
    f"{PREFIX}backward_pass_duration_seconds",
    "Backward pass latency in seconds",
    labelnames=["layer_idx"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    registry=MINER_REGISTRY,
)

# --- S3 network speed ---
S3_UPLOAD_SPEED_BYTES_PER_SEC = Gauge(
    f"{PREFIX}s3_upload_speed_bytes_per_sec",
    "Estimated S3 upload speed in bytes/sec",
    labelnames=["layer_idx"],
    registry=MINER_REGISTRY,
)

S3_DOWNLOAD_SPEED_BYTES_PER_SEC = Gauge(
    f"{PREFIX}s3_download_speed_bytes_per_sec",
    "Estimated S3 download speed in bytes/sec",
    labelnames=["layer_idx"],
    registry=MINER_REGISTRY,
)

# --- Telemetry service health ---
TELEMETRY_BUFFER_SIZE = Gauge(
    f"{PREFIX}telemetry_buffer_size",
    "Current number of messages in the telemetry buffer",
    registry=MINER_REGISTRY,
)

TELEMETRY_FLUSHES_TOTAL = Counter(
    f"{PREFIX}telemetry_flushes_total",
    "Total telemetry flushes attempted",
    labelnames=["status"],
    registry=MINER_REGISTRY,
)
