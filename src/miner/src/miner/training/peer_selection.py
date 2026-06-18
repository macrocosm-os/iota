"""Peer selection strategies for activation routing.

Each strategy takes a list of eligible ComputeNode peers and returns the
one to send the activation to, or ``None`` when no peer is currently
accepting work.  Swap the function passed to ``ActivationPublisher`` to
change routing behaviour.
"""

from __future__ import annotations
import math

import random

from common.models.compute_node import ComputeNode


def _is_accepting_work(node: ComputeNode) -> bool:
    """True iff the peer's last broadcast indicates it's ready for activations.

    Peers we haven't heard from yet (default ``miner_status="training"`` via
    PeerStatusBroadcast defaults) are treated as available so first-contact
    routing isn't deadlocked. Peers that have explicitly broadcast a non-
    training status (``initializing`` / ``frozen``) are excluded.
    """
    return node.runtime_metrics.miner_status == "training"


def select_random(eligible: list[ComputeNode]) -> ComputeNode | None:
    """Uniform random selection among peers that are accepting work."""
    candidates = [n for n in eligible if _is_accepting_work(n)]
    if not candidates:
        return None
    return random.choice(candidates)


def select_by_capacity(eligible: list[ComputeNode]) -> ComputeNode | None:
    """Weighted random selection favouring peers with more available capacity.

    Peers are scored by remaining cache headroom plus an inverse of forward
    queue depth.  Peers without metrics, in a non-training layer phase, with
    a non-training miner status, or with zero remaining capacity are excluded.
    Falls back to uniform random over accepting peers if no peer has measured
    capacity, and returns ``None`` if no peer is accepting work.
    """
    scored: list[tuple[ComputeNode, float]] = []
    for node in eligible:
        m = node.runtime_metrics
        if m.last_status_received == 0:
            # No metrics received — skip
            continue
        if m.layer_phase != "training":
            continue
        if m.miner_status != "training":
            continue
        free_cache = m.cache_capacity - m.cache_size
        if free_cache <= 0:
            # No capacity — skip
            continue
        queue_score = 1.0 / (1 + m.forward_queue_size)
        scored.append((node, math.log(free_cache) + queue_score))

    if scored:
        nodes, weights = zip(*scored)
        return random.choices(nodes, weights=weights, k=1)[0]

    # Nothing measured — fall back to uniform random over peers we believe
    # are accepting work (covers the first-contact window before broadcasts arrive).
    return select_random(eligible)
