"""SyncedNode — lifecycle wrapper that owns a ComputeNode identity.

Optionally manages keepalive and lead-election tasks when an external
:class:`~miner.sync.variable.SyncedVariable` registry is supplied via
:meth:`bind_registry`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from loguru import logger

KEEPALIVE_INTERVAL = 5.0  # seconds between keepalive updates

from miner.sync.variable import SyncedVariable
from miner.sync.registry import NodeRegistry, ComputeNode


class SyncedNode:
    """Lightweight holder for a :class:`ComputeNode` identity.

    By default no background tasks are started and no
    :class:`SyncedVariable` is created.  Call :meth:`bind_registry` to
    attach an externally-owned, run-scoped ``node_registry`` and then
    :meth:`start` to launch keepalive / lead-check loops against it.
    """

    def __init__(
        self,
        node_id: str,
        server_url: str = "http://localhost:8001",
        poll_interval: float = 2.0,
        **kwargs: Any,
    ) -> None:
        self._node_id = node_id
        self._server_url = server_url
        self._poll_interval = poll_interval
        self.compute_node = ComputeNode(node_id=node_id)
        self.peer_eviction_enabled: bool = True
        self._started = False
        self.node_registry: SyncedVariable[NodeRegistry] | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._lead_check_task: asyncio.Task | None = None
        # Optional hook supplied by subclasses/owners to attach fields the
        # SyncedNode itself doesn't know about (e.g. P2P node ids and address
        # hints) before the entry is written to the registry. Without this,
        # keepalive-driven re-registration produces an entry with empty
        # ``p2p_node_ids``, which propagates via sync and breaks activation
        # routing for peers that try to dial us.
        self._stamp_entry: Callable[[dict], dict] | None = None

    def bind_registry(self, registry: SyncedVariable[NodeRegistry]) -> None:
        """Attach an externally-owned, run-scoped node registry."""
        self.node_registry = registry

    def set_stamp_entry(self, stamp_entry: Callable[[dict], dict] | None) -> None:
        """Register a callback that attaches owner-provided fields to a registry entry."""
        self._stamp_entry = stamp_entry

    def _build_own_entry(self) -> dict:
        """Build this node's registry entry with stamp-callback fields applied.

        Starts from any existing entry in the bound registry (so timestamps
        and runtime_metrics aren't lost), falls back to ``compute_node.model_dump()``,
        then runs ``stamp_entry`` to attach owner-known fields (P2P node ids,
        address hints, training_layer).
        """
        entry: dict | None = None
        if self.node_registry is not None:
            existing = self.node_registry.value.get(self._node_id)
            if isinstance(existing, dict):
                entry = dict(existing)
        if entry is None:
            entry = self.compute_node.model_dump()
        if self._stamp_entry is not None:
            try:
                stamped = self._stamp_entry(entry)
            except Exception as exc:
                logger.warning(f"[SyncedNode] stamp_entry callback failed: {exc}")
                stamped = entry
            if isinstance(stamped, dict):
                entry = stamped
        entry["node_id"] = self._node_id
        return entry

    def _write_own_entry(self) -> None:
        """Write this node's stamped entry into the bound registry."""
        if self.node_registry is None:
            return
        registry = self.node_registry.value
        entry = self._build_own_entry()
        registry[self._node_id] = entry
        registry._own_node = self._node_id

    async def start(self) -> None:
        """Start keepalive and lead-check background tasks.

        Requires :meth:`bind_registry` to have been called first.
        Idempotent.
        """
        if self._started:
            return
        if self.node_registry is None:
            logger.warning(
                f"SyncedNode[{self._node_id}] start() called without a bound " "registry — skipping background tasks"
            )
            return
        self._started = True
        logger.debug(f"Running node {self._node_id} on server {self._server_url}")

        self._write_own_entry()

        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        self._lead_check_task = asyncio.create_task(self._lead_check_loop())
        logger.debug(f"Created keepalive and lead-check tasks for node {self._node_id}")

    async def stop(self) -> None:
        """Cancel the keepalive and lead-check tasks."""
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
        if self._lead_check_task is not None:
            self._lead_check_task.cancel()

    async def run(self) -> None:
        """Entry point for the inheritance pattern: start → run_body → stop."""
        await self.start()
        try:
            await self.run_body()
        finally:
            await self.stop()

    async def run_body(self) -> None:
        """Override in subclasses with the node's main logic."""

    async def _keepalive_loop(self) -> None:
        while True:
            try:
                if self.node_registry is not None:
                    registry = self.node_registry.value
                    if registry._own_node and registry._own_node not in registry:
                        # Bridge evicted us or local state was reset — re-insert with full metadata.
                        logger.warning(
                            f"[SyncedNode] Own node {self._node_id!r} missing from registry — re-registering"
                        )
                        self._write_own_entry()
                    else:
                        registry.update_keepalive()
            except Exception as exc:
                logger.warning(f"[SyncedNode] Keepalive loop error: {exc}")
            await asyncio.sleep(KEEPALIVE_INTERVAL)

    async def _lead_check_loop(self) -> None:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            try:
                if self.node_registry is None:
                    continue
                registry = self.node_registry.value
                all_groups: set[str] = set()
                for node in registry.all_nodes():
                    all_groups.update(node.groups)
                for group in all_groups:
                    if self.peer_eviction_enabled and registry.is_lead(self._node_id, group):
                        evicted = registry.evict_stale_nodes(group, KEEPALIVE_INTERVAL)
                        for node_id in evicted:
                            logger.info(
                                f"[lead={self._node_id}] Evicted stale node {node_id!r} "
                                f"from group {group!r} (no keepalive for >{3 * KEEPALIVE_INTERVAL:.0f}s)"
                            )
            except Exception as exc:
                logger.warning(f"[SyncedNode] Lead check loop error: {exc}")
