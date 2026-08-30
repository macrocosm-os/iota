"""Regression tests for capacity-weighted peer selection.

The 2026-07-22 orion stall: with every peer at exactly one free cache slot,
log(1)=0 weights summed to zero and random.choices raised
"Total of weights must be greater than zero", killing outbound sends.
"""

from common.models.compute_node import ComputeNode
from common.models.peer_status import PeerStatusBroadcast

from miner.training.peer_selection import select_by_capacity


def _node(node_id: str, cache_size: int, cache_capacity: int = 30, forward_queue_size: int = 0) -> ComputeNode:
    return ComputeNode(
        node_id=node_id,
        runtime_metrics=PeerStatusBroadcast(
            cache_size=cache_size,
            cache_capacity=cache_capacity,
            forward_queue_size=forward_queue_size,
            last_status_received=1.0,
        ),
    )


def test_all_peers_at_one_free_slot_does_not_raise():
    peers = [_node(f"n{i}", cache_size=29) for i in range(3)]
    assert select_by_capacity(peers) is not None


def test_prefers_peer_with_more_headroom():
    roomy, full = _node("roomy", cache_size=0), _node("full", cache_size=29)
    picks = {select_by_capacity([roomy, full]).node_id for _ in range(50)}
    assert "roomy" in picks


def test_all_saturated_applies_backpressure():
    saturated = [_node(f"n{i}", cache_size=30) for i in range(3)]
    # Peers report metrics but none has free cache -> don't send anywhere.
    assert select_by_capacity(saturated) is None


def test_unmeasured_peers_fall_back_to_random():
    unmeasured = [
        ComputeNode(node_id=f"n{i}", runtime_metrics=PeerStatusBroadcast(last_status_received=0.0)) for i in range(3)
    ]
    # First-contact window: no broadcasts yet -> uniform random, not deadlock.
    assert select_by_capacity(unmeasured) is not None
