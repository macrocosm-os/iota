"""Peer selection strategies for activation routing.

Each strategy takes a list of eligible ComputeNode peers and returns the
one to send the activation to, or ``None`` when no peer is currently
accepting work.  Swap the function passed to ``ActivationPublisher`` to
change routing behaviour.
"""

from __future__ import annotations
import math

import random

from loguru import logger

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
    Returns ``None`` when peers have reported metrics but none has capacity
    (backpressure — don't send into full caches).  Falls back to uniform
    random over accepting peers only when no peer has reported metrics yet
    (first-contact window before broadcasts arrive).
    """
    scored: list[tuple[ComputeNode, float]] = []
    # Tally why peers are excluded so the routing decision's inputs are visible.
    skips = {"no_status": 0, "phase": 0, "status": 0, "no_capacity": 0}
    considered: list[str] = []  # per-peer cache/queue inputs + outcome for routing visibility
    for node in eligible:
        m = node.runtime_metrics
        tag = (
            f"{node.node_id[:8]} cache={m.cache_size}/{m.cache_capacity} "
            f"fwd_q={m.forward_queue_size} lsr={m.last_status_received} st={m.miner_status}"
        )
        if m.last_status_received == 0:
            # No metrics received — skip
            skips["no_status"] += 1
            considered.append(tag + " -> skip:no_status")
            continue
        if m.layer_phase != "training":
            skips["phase"] += 1
            considered.append(tag + f" -> skip:phase({m.layer_phase})")
            continue
        if m.miner_status != "training":
            skips["status"] += 1
            considered.append(tag + " -> skip:status")
            continue
        free_cache = m.cache_capacity - m.cache_size
        if free_cache <= 0:
            # No capacity — skip
            skips["no_capacity"] += 1
            considered.append(tag + " -> skip:no_capacity")
            continue
        queue_score = 1.0 / (1 + m.forward_queue_size)
        # log1p: plain log(1)=0 zeroes the weight at one free slot, and when
        # ALL peers sit there random.choices raises on an all-zero total.
        weight = math.log1p(free_cache) * queue_score
        scored.append((node, weight))
        considered.append(tag + f" -> scored(w={weight:.3f})")

    peers = " | ".join(considered)
    if scored:
        nodes, weights = zip(*scored)
        chosen = random.choices(nodes, weights=weights, k=1)[0]
        logger.debug(
            f"route=capacity eligible={len(eligible)} scored={len(scored)} skips={skips} "
            f"chose={chosen.node_id[:8]} | peers: {peers}"
        )
        return chosen

    if any(n.runtime_metrics.last_status_received for n in eligible):
        # Peers are reporting but none has capacity — apply backpressure.
        logger.debug(f"route=backpressure eligible={len(eligible)} skips={skips} -> None (defer) | peers: {peers}")
        return None

    # Nothing measured — fall back to uniform random over peers we believe
    # are accepting work (covers the first-contact window before broadcasts arrive).
    chosen = select_random(eligible)
    logger.debug(
        f"route=random-fallback eligible={len(eligible)} skips={skips} "
        f"chose={chosen.node_id[:8] if chosen else None} | peers: {peers}"
    )
    return chosen
