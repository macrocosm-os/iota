"""ComputeNode — P2P-capable compute node backed by an IrohNode.

A ComputeNode represents a participant in the compute mesh.  It owns zero
or more *endpoint handlers* (registered via the ``@node.endpoint(name)``
decorator) and delegates actual P2P send/broadcast to an embedded
:class:`~common.node_mesh.iroh_node.IrohNode`.

The IrohNode is intentionally kept separate so it can be started/stopped
independently, tested in isolation, and swapped for a mock in unit tests.

Typical usage::

    from common.node_mesh.iroh_node import IrohNode

    iroh = IrohNode(pool_size=2)
    await iroh.start()

    node = ComputeNode(node_id="miner-1", iroh_receiver_ids=iroh.receiver_ids)
    node._iroh_node = iroh

    @node.endpoint("gradient")
    async def handle_gradient(msg):
        ...

    await node.send(peer_node, GradientMessage(...), endpoint="gradient")
"""

from __future__ import annotations

from typing import Any, Callable


class IrohComputeNode:
    """Compute mesh participant with P2P send/broadcast capabilities.

    Args:
        node_id:            Human-readable identifier for this node.
        iroh_receiver_ids:  List of iroh node-ID strings that remote peers
                            use to reach this node's IrohNode.
    """

    def __init__(
        self,
        node_id: str,
        iroh_receiver_ids: list[str] | None = None,
    ) -> None:
        self.node_id = node_id
        self.iroh_receiver_ids: list[str] = iroh_receiver_ids or []
        self._iroh_node: Any | None = None
        self._endpoint_handlers: dict[str, Callable] = {}

    # ── endpoint registration ──────────────────────────────────────────────────

    def endpoint(self, name: str) -> Callable:
        """Decorator that registers *fn* as the handler for endpoint *name*.

        If an IrohNode is already attached (``_iroh_node`` is set) the handler
        is also registered there immediately so incoming messages are dispatched
        correctly.
        """

        def decorator(fn: Callable) -> Callable:
            self._endpoint_handlers[name] = fn
            if self._iroh_node is not None and hasattr(self._iroh_node, "add_endpoint_handler"):
                self._iroh_node.add_endpoint_handler(name, fn)
            return fn

        return decorator

    # ── P2P operations ─────────────────────────────────────────────────────────

    async def send(self, target: "IrohComputeNode", msg: Any, endpoint: str | None = None) -> None:
        """Send *msg* to *target* via the embedded IrohNode.

        Raises:
            RuntimeError: If the IrohNode has not been set (``_iroh_node`` is None).
        """
        if self._iroh_node is None:
            raise RuntimeError(
                "P2P node not available. Call await node.start() first, " "or set node._iroh_node before sending."
            )
        await self._iroh_node.send(target, msg, endpoint)

    async def broadcast(self, targets: list["IrohComputeNode"], msg: Any, endpoint: str | None = None) -> None:
        """Send *msg* to every node in *targets* via the embedded IrohNode.

        Raises:
            RuntimeError: If the IrohNode has not been set.
        """
        if self._iroh_node is None:
            raise RuntimeError(
                "P2P node not available. Call await node.start() first, " "or set node._iroh_node before broadcasting."
            )
        await self._iroh_node.broadcast(targets, msg, endpoint)
