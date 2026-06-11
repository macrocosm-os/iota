"""ComputeNode — identity record for a single miner in the node registry."""

from __future__ import annotations

import time
from pydantic import BaseModel, Field

from common.models.peer_status import PeerStatusBroadcast


class ComputeNode(BaseModel):
    """Identity record written to the shared registry by every miner."""

    node_id: str
    p2p_node_ids: list[str] = Field(default_factory=list)
    iroh_relay_url: str | None = None
    iroh_direct_addresses: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=lambda: ["all"])
    joined_at: float = Field(default_factory=time.time)
    last_keepalive: float = Field(default_factory=time.time)
    runtime_metrics: PeerStatusBroadcast = Field(default_factory=PeerStatusBroadcast)
