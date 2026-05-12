"""Node registry utilities for SyncedNode.

ComputeNode
-----------
Pydantic model describing a single node's identity, groups, and iroh receiver
IDs.  Instances are stored inside the shared ``node_registry`` variable (a
:class:`NodeRegistry`) so every node in the same namespace can discover
each other automatically.

NodeRegistry
------------
A :class:`~miner.sync.collections.SyncedDict` subclass with utility
methods for querying the registry.  Dirty-detection and JSON Patch sync are
inherited from SyncedDict — only changed entries travel over the wire.

Wire format (flat dict)::

    {
        "node-abc": {
            "node_id":               "node-abc",
            "p2p_node_ids":          ["..."],
            "iroh_relay_url":        "https://...",
            "iroh_direct_addresses": ["192.0.2.1:54321", ...],
            "groups":                ["all", "gpu-workers"],
            "joined_at":             1234567890.123
        },
        ...
    }
"""

from __future__ import annotations

import copy
import time
from pydantic import BaseModel, Field

from miner.sync.collections import SyncedDict
from common.models.peer_status import PeerStatusBroadcast
from loguru import logger


# ── ComputeNode ───────────────────────────────────────────────────────────────


class ComputeNode(BaseModel):
    """Identity record written to the shared registry by every :class:`~miner.sync.node.SyncedNode`."""

    node_id: str
    p2p_node_ids: list[str] = Field(default_factory=list)
    #: Home relay URL of this node's receiver, used to skip n0 DNS discovery on dial.
    iroh_relay_url: str | None = None
    #: Direct sockaddr hints of this node's receiver (e.g. ["192.0.2.1:54321", ...]).
    iroh_direct_addresses: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=lambda: ["all"])
    #: Orchestrator training layer index; used for routing when ``groups`` is stale (e.g. ``["all"]`` only).
    training_layer: int | None = None
    joined_at: float = Field(default_factory=time.time)
    last_keepalive: float = Field(default_factory=time.time)
    runtime_metrics: PeerStatusBroadcast = Field(default_factory=PeerStatusBroadcast)


# ── NodeRegistry ──────────────────────────────────────────────────────────────


class NodeRegistry(SyncedDict):
    """A :class:`~miner.sync.collections.SyncedDict` subclass for the shared node registry.

    Wire format: flat dict mapping node_id → ComputeNode fields::

        {
            "node-abc": {"node_id": "node-abc", "p2p_node_ids": [...], ...},
            "node-xyz": {...},
        }
    """

    def __init__(self) -> None:
        super().__init__({})
        self._own_node: str | None = None

    # ── Write helpers ─────────────────────────────────────────────────

    def register(self, node: ComputeNode) -> None:
        """Add or update *node* in the registry (idempotent)."""
        self[node.node_id] = node.model_dump()
        self._own_node = node.node_id

    def deregister(self, node_id: str) -> None:
        """Remove *node_id* from the registry.  No-op if absent."""
        logger.warning(f"Deregistering node {node_id} from registry")
        self.pop(node_id, None)

    def update_keepalive(self) -> None:
        """Refresh the ``last_keepalive`` timestamp for the own node."""
        if self._own_node and self._own_node in self:
            self[self._own_node]["last_keepalive"] = time.time()

    # ── Read helpers ──────────────────────────────────────────────────

    @property
    def own_node(self) -> ComputeNode | None:
        """The :class:`ComputeNode` record for the current node."""
        if self._own_node is None:
            return None
        entry = self.get(self._own_node)
        return ComputeNode(**entry) if entry else None

    def all_nodes(self) -> list[ComputeNode]:
        """Return every registered node as a :class:`ComputeNode` object."""
        return [ComputeNode(**entry) for entry in self.values()]

    def get_nodes_in_group(self, group: str = "all") -> list[ComputeNode]:
        """Return all nodes whose ``groups`` list includes *group*."""
        return [n for n in self.all_nodes() if group in n.groups]

    def get_nodes_for_layer(self, layer: int) -> list[ComputeNode]:
        """Return nodes assigned to training *layer* for P2P activation routing.

        If :attr:`ComputeNode.training_layer` is set, it is authoritative (legacy
        ``groups`` may be stale). Otherwise match ``\"layer-{layer}\" in groups``.
        """
        out: list[ComputeNode] = []
        for raw in self.values():
            if not isinstance(raw, dict):
                continue
            tl = raw.get("training_layer")
            groups = raw.get("groups") or []
            if tl is not None:
                try:
                    matched = int(tl) == layer
                except (TypeError, ValueError):
                    matched = f"layer-{layer}" in groups
            else:
                matched = f"layer-{layer}" in groups
            if matched:
                out.append(ComputeNode(**raw))
        return out

    def get_lead_node(self, group: str = "all") -> ComputeNode | None:
        """Return the lead node for *group* using deterministic election.

        The lead is the node with the **earliest** ``joined_at`` timestamp.
        Ties are broken by ``node_id`` lexicographic order.
        """
        nodes = self.get_nodes_in_group(group)
        if not nodes:
            return None
        return min(nodes, key=lambda n: (n.joined_at, n.node_id))

    def is_lead(self, node_id: str, group: str = "all") -> bool:
        """Return ``True`` if *node_id* is the current lead of *group*."""
        lead = self.get_lead_node(group)
        return lead is not None and lead.node_id == node_id

    def is_alive(self, node_id: str, timeout: float = 15.0) -> bool:
        """Return ``True`` if *node_id* sent a keepalive within the last *timeout* seconds."""
        entry = self.get(node_id)
        if entry is None:
            return False
        return time.time() - entry.get("last_keepalive", 0.0) < timeout

    def apply_full_value(self, value: dict) -> None:
        """Replace the entire dict, then re-inject own node if it was removed.

        When the bridge (or another peer) evicts our entry from the
        registry, we immediately restore it locally so the next push
        re-advertises our presence. The locally-known entry (with
        ``p2p_node_ids``, ``training_layer`` and address hints) is
        preserved across the fetch so that we never push a bare
        node-id-only record back to the bridge — that would propagate
        an empty ``p2p_node_ids`` to peers and break activation routing.
        """
        local_own_entry: dict | None = None
        if self._own_node and self._own_node in self:
            existing = self[self._own_node]
            if isinstance(existing, dict):
                local_own_entry = copy.deepcopy(existing)
        super().apply_full_value(value)
        if self._own_node and self._own_node not in self:
            if local_own_entry is not None:
                self[self._own_node] = local_own_entry
            else:
                self[self._own_node] = ComputeNode(node_id=self._own_node).model_dump()
            logger.warning(f"Re-injected own node {self._own_node} after remote fetch removed it")

    def evict_stale_nodes(self, group: str, keepalive_interval: float) -> list[str]:
        """Remove nodes in *group* whose keepalive is older than 3× *keepalive_interval*.

        The own node is never evicted.  Returns the list of removed node IDs.
        """
        threshold = 3 * keepalive_interval
        now = time.time()
        to_remove = [
            n.node_id
            for n in self.get_nodes_in_group(group)
            if n.node_id != self._own_node and now - n.last_keepalive > threshold
        ]
        for node_id in to_remove:
            self.deregister(node_id)
        return to_remove
