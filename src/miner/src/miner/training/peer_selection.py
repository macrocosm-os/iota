"""Peer selection strategies for activation routing.

Each strategy takes a list of eligible ComputeNode peers and returns the
one to send the activation to.  Swap the function passed to
``ActivationPublisher`` to change routing behaviour.
"""

from __future__ import annotations
import math

import random

from miner.sync.registry import ComputeNode


def select_random(eligible: list[ComputeNode]) -> ComputeNode:
    """Uniform random selection (current default)."""
    return random.choice(eligible)


def select_by_capacity(eligible: list[ComputeNode]) -> ComputeNode:
    """Weighted random selection favouring peers with more available capacity.

    Peers are scored by remaining cache headroom plus an inverse of forward
    queue depth.  Peers without metrics or with zero remaining capacity are
    excluded entirely.  Falls back to uniform random if no peer has capacity.
    """
    scored: list[tuple[ComputeNode, float]] = []
    for node in eligible:
        m = node.runtime_metrics
        if m.last_status_received == 0:
            # No metrics received — skip
            continue
        if m.layer_phase != "training":
            continue
        free_cache = m.cache_capacity - m.cache_size
        if free_cache <= 0:
            # No capacity — skip
            continue
        queue_score = 1.0 / (1 + m.forward_queue_size)
        scored.append((node, math.log(free_cache) + queue_score))

    if not scored:
        return random.choice(eligible)

    nodes, weights = zip(*scored)
    return random.choices(nodes, weights=weights, k=1)[0]
