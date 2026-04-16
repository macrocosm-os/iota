"""Tests for activation push layer routing helpers on the publisher."""

from miner.sync.registry import ComputeNode
from miner.training.activation_publisher import _peer_matches_target_layer


def test_peer_matches_target_layer_uses_training_layer() -> None:
    node = ComputeNode(node_id="a", training_layer=2, p2p_node_ids=["p1"])
    assert _peer_matches_target_layer(node, 2) is True
    assert _peer_matches_target_layer(node, 1) is False


def test_peer_matches_target_layer_groups_fallback() -> None:
    node = ComputeNode(
        node_id="b",
        training_layer=None,
        groups=["all", "layer-3"],
        p2p_node_ids=["p2"],
    )
    assert _peer_matches_target_layer(node, 3) is True
    assert _peer_matches_target_layer(node, 2) is False
