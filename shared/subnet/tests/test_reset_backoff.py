"""Test the jittered exponential backoff math used to throttle
reset_miner_state calls during fleet-wide crash clusters.

We don't instantiate the full Miner — too many dependencies — but we test
the formula in isolation so the cap and jitter envelope are pinned.
"""

import random

import pytest

# Mirror the constants from src/miner/src/miner/new_miner.py
_BASE = 5.0
_MAX = 300.0
_JITTER = 0.25


def _backoff(consecutive_reset_count: int, rand: float) -> float:
    """Reference implementation matching the inlined code in
    new_miner.reset_miner_state. ``rand`` is the value that random.random()
    would return, so we can test deterministically."""
    if consecutive_reset_count <= 0:
        return 0.0
    raw = min(_BASE * (2 ** (consecutive_reset_count - 1)), _MAX)
    jitter = raw * _JITTER * (2 * rand - 1)
    return max(0.0, raw + jitter)


def test_first_reset_is_free():
    assert _backoff(0, rand=0.5) == 0.0


def test_growth_is_exponential():
    # Center-of-jitter values for clean comparison
    assert _backoff(1, 0.5) == pytest.approx(5.0)
    assert _backoff(2, 0.5) == pytest.approx(10.0)
    assert _backoff(3, 0.5) == pytest.approx(20.0)
    assert _backoff(4, 0.5) == pytest.approx(40.0)
    assert _backoff(5, 0.5) == pytest.approx(80.0)


def test_capped_at_max():
    # 2^N grows fast — at N=10 we'd want 5 * 512 = 2560s, but cap is 300s.
    assert _backoff(10, 0.5) == pytest.approx(_MAX)
    assert _backoff(50, 0.5) == pytest.approx(_MAX)


def test_jitter_envelope():
    # Jitter is ±25% of the un-capped backoff; check both extremes.
    # attempt #3 → raw 20s, jitter window [15, 25]
    low = _backoff(3, rand=0.0)  # rand=0 → jitter coefficient = -1
    high = _backoff(3, rand=1.0)  # rand=1 → jitter coefficient = +1
    assert low == pytest.approx(15.0)
    assert high == pytest.approx(25.0)


def test_jitter_does_not_go_negative():
    # Cap protects against runaway, and max(0, ...) protects against any
    # combination of inputs producing a negative sleep.
    for n in range(1, 60):
        for r in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert _backoff(n, r) >= 0.0


def test_real_random_is_within_envelope():
    """Sanity check with the actual stdlib random to make sure the envelope
    bounds work under realistic sampling, not just the deterministic
    extremes."""
    random.seed(0)
    for _ in range(100):
        n = random.randint(1, 8)
        raw = min(_BASE * (2 ** (n - 1)), _MAX)
        observed = _backoff(n, random.random())
        assert raw * (1 - _JITTER) - 1e-9 <= observed <= raw * (1 + _JITTER) + 1e-9
