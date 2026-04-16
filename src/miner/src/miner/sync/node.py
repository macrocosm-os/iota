"""SyncedNode — lifecycle wrapper that owns a ComputeNode identity.

Optionally manages keepalive and lead-election tasks when an external
:class:`~miner.sync.variable.SyncedVariable` registry is supplied via
:meth:`bind_registry`.
"""

from __future__ import annotations

import asyncio
from typing import Any

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

    def bind_registry(self, registry: SyncedVariable[NodeRegistry]) -> None:
        """Attach an externally-owned, run-scoped node registry."""
        self.node_registry = registry

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

        self.node_registry.value.register(self.compute_node)

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
            if self.node_registry is not None:
                self.node_registry.value.update_keepalive()
            await asyncio.sleep(KEEPALIVE_INTERVAL)

    async def _lead_check_loop(self) -> None:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)
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
