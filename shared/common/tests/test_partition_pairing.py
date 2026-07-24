"""Tests for butterfly all-reduce partition pairing (``get_pairs_for_miner``).

The pool handed to ``get_pairs_for_miner`` is the set of miners that submitted
weights this epoch (see ``fetch_partitions_for_miner`` /
``get_submitting_hotkeys_in_layer``). These tests exercise the current API,
where the function returns the list of partition indices owned by ``target_hotkey``.
"""

import pytest

from common.utils.partitions import get_pairs_for_miner


def _all_indices_by_miner(pool: list[str], n_partitions: int, seed: int) -> dict[str, set[int]]:
    """Reconstruct global ownership: hotkey -> set of partition indices it owns."""
    return {
        hk: set(get_pairs_for_miner(miner_hotkeys=pool, n_partitions=n_partitions, target_hotkey=hk, seed=seed))
        for hk in pool
    }


def test_empty_pool_returns_no_partitions() -> None:
    """No submitters this epoch -> nobody is assigned partitions (and no hang)."""
    assert get_pairs_for_miner(miner_hotkeys=[], n_partitions=8, target_hotkey="miner_0", seed=1) == []


def test_single_submitter_gets_all_partitions() -> None:
    """A lone submitter owns every partition (paired with None)."""
    n_partitions = 6
    result = get_pairs_for_miner(miner_hotkeys=["miner_0"], n_partitions=n_partitions, target_hotkey="miner_0", seed=1)
    # Single-miner branch returns a mapping partition_index -> (miner, None).
    assert len(result) == n_partitions
    assert all(pair == ("miner_0", None) for pair in result.values())


def test_non_submitter_target_gets_nothing() -> None:
    """A hotkey that is not in the submitting pool receives no partitions."""
    pool = [f"miner_{i}" for i in range(5)]
    result = get_pairs_for_miner(miner_hotkeys=pool, n_partitions=10, target_hotkey="miner_999", seed=3)
    assert result == []


@pytest.mark.parametrize("n_submitters", range(2, 9))
def test_every_partition_owned_by_exactly_two_submitters(n_submitters: int) -> None:
    """Butterfly invariant: each partition index is owned by exactly two distinct
    miners, and only miners from the submitting pool ever appear."""
    pool = [f"miner_{i}" for i in range(n_submitters)]
    n_partitions = 12
    seed = 42

    by_miner = _all_indices_by_miner(pool, n_partitions, seed)

    # No index exceeds the partition count.
    for indices in by_miner.values():
        assert all(0 <= i < n_partitions for i in indices)

    # Each partition is owned by exactly two distinct submitters.
    for p in range(n_partitions):
        owners = [hk for hk, idxs in by_miner.items() if p in idxs]
        assert len(owners) == 2, f"partition {p} owned by {owners}, expected exactly 2"


def test_pairing_is_deterministic_for_seed() -> None:
    """Same pool + seed -> identical index set for a miner (consistent across the
    two peers of every pair, and stable across orchestrator restarts)."""
    pool = [f"miner_{i}" for i in range(6)]
    a = get_pairs_for_miner(miner_hotkeys=pool, n_partitions=10, target_hotkey="miner_2", seed=7)
    b = get_pairs_for_miner(miner_hotkeys=pool, n_partitions=10, target_hotkey="miner_2", seed=7)
    assert set(a) == set(b)
