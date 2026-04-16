"""Peer runtime metrics model.

Used as both the wire format for ``/peer/status`` routed UNI broadcasts
(msgpack-serialised) and as embedded storage in :class:`ComputeNode`.
"""

from __future__ import annotations

from pydantic import BaseModel


class PeerStatusBroadcast(BaseModel):
    """Snapshot of a miner's queue and cache stats.

    Attributes:
        source_hotkey:          SS58 address of the broadcasting miner.
        forward_queue_size:     Number of forward activations currently queued.
        backward_queue_size:    Number of backward activations currently queued.
        cache_size:             Number of activations currently cached.
        cache_capacity:         Maximum activation cache capacity.
        uptime_seconds:         Seconds since miner start.
        last_status_received:   Unix timestamp set by the *recipient* on arrival
                                (zero on the wire — populated on ingest).
    """

    source_hotkey: str = ""
    forward_queue_size: int = 0
    backward_queue_size: int = 0
    cache_size: int = 0
    cache_capacity: int = 0
    uptime_seconds: float = 0.0
    layer_phase: str = "training"
    last_status_received: float = 0.0
