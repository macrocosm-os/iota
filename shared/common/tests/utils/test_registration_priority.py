import json

from common.utils.registration_priority import (
    QueueEntry,
    compute_score,
    hardware_score,
    rank_registration_queue,
    upload_score,
    upsert_queue_entry,
)


def _system_data(cpu, upload_mbps, memory_gb=36):
    return json.dumps(
        {
            "gpus": [cpu],
            "chip_info": {"machine": "arm64", "cpu": cpu, "cores": 16, "memory_gb": memory_gb},
            "bandwidth": {"download_mbps": 900.0, "upload_mbps": upload_mbps},
        }
    )


def test_newer_and_bigger_chips_score_higher():
    assert compute_score("Apple M5 Max") > compute_score("Apple M5 Pro") > compute_score("Apple M1 Pro")
    assert compute_score("Apple M5 Max") > compute_score("Apple M4 Max")
    assert compute_score("Apple M1") < compute_score("Apple M1 Pro")
    assert compute_score("Some Unknown CPU") == 0.5
    assert compute_score(None) == 0.5


def test_upload_below_minimum_scores_zero():
    assert upload_score(100) == 0.0
    assert upload_score(150) == 0.0
    assert upload_score(1000) == 1.0
    assert upload_score(5000) == 1.0
    assert upload_score(None) == 0.5


def test_memory_above_useful_floor_gives_no_bonus():
    assert hardware_score(_system_data("Apple M5 Max", 500, 128)) == hardware_score(
        _system_data("Apple M5 Max", 500, 36)
    )
    assert hardware_score(_system_data("Apple M5 Max", 500, 16)) < hardware_score(_system_data("Apple M5 Max", 500, 36))


def test_hardware_score_handles_missing_or_bad_data():
    assert hardware_score(None) == 0.5
    assert hardware_score("{not json") == 0.5
    assert 0.0 <= hardware_score({"chip_info": {}}) <= 1.0


def test_better_hardware_jumps_ahead_of_equal_wait():
    now = 10_000.0
    m1 = QueueEntry(
        "m1", enqueued_at=now - 3600, last_seen_at=now, hardware_score=hardware_score(_system_data("Apple M1 Pro", 200))
    )
    m5 = QueueEntry(
        "m5", enqueued_at=now - 3600, last_seen_at=now, hardware_score=hardware_score(_system_data("Apple M5 Max", 900))
    )

    assert [e.hotkey for e in rank_registration_queue([m1, m5], now)] == ["m5", "m1"]


def test_weaker_hardware_ages_to_the_front():
    now = 100_000.0
    slow = QueueEntry("slow", enqueued_at=now - 50_000, last_seen_at=now, hardware_score=0.0)
    fast = QueueEntry("fast", enqueued_at=now - 20_000, last_seen_at=now, hardware_score=1.0)

    assert [e.hotkey for e in rank_registration_queue([fast, slow], now)] == ["slow", "fast"]


def test_ties_go_to_whoever_joined_first():
    now = 1_000.0
    a = QueueEntry("a", enqueued_at=0.0, last_seen_at=now, hardware_score=0.5)
    b = QueueEntry("b", enqueued_at=0.0, last_seen_at=now, hardware_score=0.5)

    assert [e.hotkey for e in rank_registration_queue([b, a], now)] == ["b", "a"]


def test_rejoin_within_grace_keeps_original_position():
    entries = {}
    first = upsert_queue_entry(entries, "hk", now=0.0, system_data=None)
    rejoined = upsert_queue_entry(entries, "hk", now=3600.0, system_data=_system_data("Apple M5 Max", 900))

    assert rejoined is first
    assert rejoined.enqueued_at == 0.0
    assert rejoined.hardware_score > 0.5


def test_rejoin_after_grace_starts_over():
    entries = {}
    upsert_queue_entry(entries, "hk", now=0.0, system_data=None, rejoin_grace_seconds=60)
    rejoined = upsert_queue_entry(entries, "hk", now=120.0, system_data=None, rejoin_grace_seconds=60)

    assert rejoined.enqueued_at == 120.0
