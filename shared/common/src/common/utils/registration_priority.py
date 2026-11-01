"""Hardware-aware ordering for the registration waitlist (reference for the orchestrator).

The orchestrator orders the waitlist; this module is the policy we propose it use.
Plain FIFO lets a machine that will miss upload deadlines take a seat ahead of one
that won't, so we rank by wait time scaled by a hardware score:

    priority = waited_seconds * (1 + HARDWARE_WEIGHT * hardware_score)

Because the score multiplies wait time instead of replacing it, weaker machines still
age to the front: a score-0 machine that has waited (1 + HARDWARE_WEIGHT) times longer
ties with a score-1 machine. Nobody starves.

Rejoins keep their place: a hotkey that re-registers within REJOIN_GRACE_SECONDS of
last being seen keeps its original enqueued_at.

The score uses the miner's self-reported system_data, which a miner can spoof. The
orchestrator should prefer measured throughput and on-time uploads where it has
them, and treat this score as a prior for machines it hasn't seen yet.
"""

import json
import re
from dataclasses import dataclass

HARDWARE_WEIGHT = 1.0
REJOIN_GRACE_SECONDS = 6 * 3600

# Upload is what decides whether a miner finishes its epoch in time; TAH requires
# 150 Mbps and sees little benefit past roughly 1 Gbps.
MIN_UPLOAD_MBPS = 150.0
FULL_CREDIT_UPLOAD_MBPS = 1000.0

# Relative Apple Silicon GPU throughput by tier, before the generation bonus.
_APPLE_TIER = {"": 0.25, "pro": 0.5, "max": 0.85, "ultra": 1.0}
_APPLE_GENERATION_STEP = 0.08
_APPLE_LATEST_GENERATION = 5
_APPLE_CHIP_RE = re.compile(r"apple\s+m(\d+)(?:\s+(pro|max|ultra))?", re.IGNORECASE)

# Memory past this adds nothing for the current layer sizes.
USEFUL_MEMORY_GB = 24


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def compute_score(cpu_brand: str | None) -> float:
    """Score the chip in [0, 1]. Unknown chips get a middling 0.5."""
    if not cpu_brand:
        return 0.5
    match = _APPLE_CHIP_RE.search(cpu_brand)
    if not match:
        return 0.5
    generation = int(match.group(1))
    tier = (match.group(2) or "").lower()
    generations_behind = max(0, _APPLE_LATEST_GENERATION - generation)
    return _clamp(_APPLE_TIER[tier] - _APPLE_GENERATION_STEP * generations_behind + 0.15)


def upload_score(upload_mbps: float | None) -> float:
    """0 below the TAH minimum, rising linearly to 1 at FULL_CREDIT_UPLOAD_MBPS. Unknown gets 0.5."""
    if upload_mbps is None:
        return 0.5
    if upload_mbps < MIN_UPLOAD_MBPS:
        return 0.0
    return _clamp((upload_mbps - MIN_UPLOAD_MBPS) / (FULL_CREDIT_UPLOAD_MBPS - MIN_UPLOAD_MBPS))


def hardware_score(system_data: str | dict | None) -> float:
    """Score a miner's reported hardware in [0, 1] from its MinerSystemData JSON.

    Upload bandwidth and compute weigh equally. Memory only matters as a floor:
    below USEFUL_MEMORY_GB the score is scaled down, above it there is no bonus.
    """
    if isinstance(system_data, str):
        try:
            system_data = json.loads(system_data)
        except (TypeError, ValueError):
            system_data = None
    if not isinstance(system_data, dict):
        return 0.5

    chip_info = system_data.get("chip_info") or {}
    bandwidth = system_data.get("bandwidth") or {}

    score = 0.5 * compute_score(chip_info.get("cpu")) + 0.5 * upload_score(bandwidth.get("upload_mbps"))

    memory_gb = chip_info.get("memory_gb")
    if isinstance(memory_gb, (int, float)) and 0 < memory_gb < USEFUL_MEMORY_GB:
        score *= memory_gb / USEFUL_MEMORY_GB
    return _clamp(score)


@dataclass
class QueueEntry:
    hotkey: str
    enqueued_at: float
    last_seen_at: float
    hardware_score: float = 0.5


def queue_priority(entry: QueueEntry, now: float, hardware_weight: float = HARDWARE_WEIGHT) -> float:
    waited = max(0.0, now - entry.enqueued_at)
    return waited * (1 + hardware_weight * _clamp(entry.hardware_score))


def rank_registration_queue(
    entries: list[QueueEntry], now: float, hardware_weight: float = HARDWARE_WEIGHT
) -> list[QueueEntry]:
    """Highest priority first. Ties go to whoever joined first."""
    return sorted(entries, key=lambda e: (-queue_priority(e, now, hardware_weight), e.enqueued_at))


def upsert_queue_entry(
    entries: dict[str, QueueEntry],
    hotkey: str,
    now: float,
    system_data: str | dict | None,
    rejoin_grace_seconds: float = REJOIN_GRACE_SECONDS,
) -> QueueEntry:
    """Add or refresh a hotkey's waitlist entry.

    A hotkey seen within the grace window keeps its original enqueued_at, so a
    restart or crash doesn't send it to the back. The hardware score is refreshed
    on every registration.
    """
    score = hardware_score(system_data)
    existing = entries.get(hotkey)
    if existing is not None and now - existing.last_seen_at <= rejoin_grace_seconds:
        existing.last_seen_at = now
        existing.hardware_score = score
        return existing
    entry = QueueEntry(hotkey=hotkey, enqueued_at=now, last_seen_at=now, hardware_score=score)
    entries[hotkey] = entry
    return entry
