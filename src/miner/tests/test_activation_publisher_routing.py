"""Tests for activation push group routing helpers on the publisher."""

from common.models.compute_node import ComputeNode
from common.models.peer_status import PeerStatusBroadcast
from miner.training.activation_publisher import _peer_matches_target_layer
from miner.training.peer_selection import select_random


def test_peer_matches_target_layer_uses_layer_group() -> None:
    node = ComputeNode(node_id="a", groups=["all", "layer-2"], p2p_node_ids=["p1"])
    assert _peer_matches_target_layer(node, 2) is True
    assert _peer_matches_target_layer(node, 1) is False


def test_peer_matches_target_layer_supports_multiple_groups() -> None:
    node = ComputeNode(
        node_id="b",
        groups=["all", "layer-3", "eval"],
        p2p_node_ids=["p2"],
    )
    assert _peer_matches_target_layer(node, 3) is True
    assert _peer_matches_target_layer(node, 2) is False


def test_select_random_excludes_initializing_registry_entry() -> None:
    """A joiner that seeded its registry entry as 'initializing' must never be picked."""
    benched = ComputeNode(
        node_id="joiner",
        groups=["all", "layer-2"],
        p2p_node_ids=["p3"],
        runtime_metrics=PeerStatusBroadcast(source_hotkey="joiner", miner_status="initializing"),
    )
    assert select_random([benched]) is None
    training = ComputeNode(node_id="worker", groups=["all", "layer-2"], p2p_node_ids=["p4"])
    assert select_random([benched, training]) is training
