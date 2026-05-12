"""IrohNode — iroh-backed P2P transport for the compute mesh.

Each IrohNode spins up one or more in-memory iroh nodes (``pool_size``
controls this) and exposes a send/broadcast API that serialises arbitrary
Python objects via pickle over a custom protocol.

Wire format (uni-directional stream)::

    [2 bytes big-endian endpoint-name length]
    [endpoint-name bytes]
    [pickle.dumps(msg)]

The receiver unpickles the message and dispatches to the handler registered
for that endpoint name.

Note: this module imports ``iroh`` at the top level.  It should be imported
only inside tests that have called ``pytest.importorskip("iroh")``, or in
production code that unconditionally requires iroh.
"""

from __future__ import annotations

import asyncio
import pickle
import struct
import weakref
from typing import Any, Callable

from iroh import Iroh, NodeAddr, NodeOptions, PublicKey, iroh_ffi
from iroh import ProtocolCreator
from loguru import logger

from common.node_mesh.compute_node import IrohComputeNode

# Custom protocol ID for the compute mesh.  Must not clash with
# common.iroh.protocol.PROTOCOL_ID_UNI / PROTOCOL_ID_BI.
PROTOCOL_ID_NODEMESH = b"node-mesh/1.0"

_MAX_MSG_BYTES = 64 * 1024 * 1024  # 64 MB ceiling

# uniffi_set_event_loop must be called before *any* iroh async operation but
# calling it again mid-flight resets internal state and can race with running
# protocol handlers.  We skip redundant calls only for the *same* loop object.
# Do not key this by id(loop): after a loop is destroyed, id reuse can make a
# new asyncio.run() look like the old loop and skip uniffi_set_event_loop,
# causing "Future attached to a different loop".
_iroh_event_loop_ref: weakref.ReferenceType[asyncio.AbstractEventLoop] | None = None


# ── Protocol handler ──────────────────────────────────────────────────────────


class _NodeMeshHandler:
    """Accepts incoming iroh connections and dispatches to endpoint handlers."""

    def __init__(self, handlers: dict[str, Callable]) -> None:
        self._handlers = handlers

    async def accept(self, conn: Any) -> None:  # conn: iroh.Connection
        asyncio.get_event_loop().create_task(self._handle(conn))

    async def shutdown(self) -> None:
        pass

    async def _handle(self, conn: Any) -> None:
        while True:
            try:
                recv_stream = await conn.accept_uni()
                data: bytes = await recv_stream.read_to_end(_MAX_MSG_BYTES)

                ep_len = struct.unpack(">H", data[:2])[0]
                endpoint_name = data[2 : 2 + ep_len].decode()
                msg = pickle.loads(data[2 + ep_len :])

                handler = self._handlers.get(endpoint_name)
                if handler is not None:
                    asyncio.get_event_loop().create_task(handler(msg))
                else:
                    logger.debug(f"IrohNode: no handler for endpoint '{endpoint_name}'")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug(f"IrohNode: handler loop ended: {type(exc).__name__}: {exc}")
                break


class _NodeMeshProtocolFactory(ProtocolCreator):
    """Factory that produces a _NodeMeshHandler for every new connection."""

    def __init__(self, handlers: dict[str, Callable]) -> None:
        self._handlers = handlers

    def create(self, endpoint: Any) -> _NodeMeshHandler:
        return _NodeMeshHandler(self._handlers)


# ── IrohNode ──────────────────────────────────────────────────────────────────


class IrohNode:
    """Lightweight iroh-backed P2P transport node for the compute mesh.

    Args:
        pool_size:  Number of underlying iroh nodes to start.  Currently
                    only the first is used for receiving; all registered
                    receiver IDs are exposed via :attr:`receiver_ids`.
    """

    def __init__(self, pool_size: int = 1) -> None:
        self._pool_size = pool_size
        self._node: Iroh | None = None
        self._handlers: dict[str, Callable] = {}
        self._receiver_ids: list[str] = []

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def receiver_ids(self) -> list[str]:
        """Iroh node-ID strings that remote nodes use to reach this IrohNode."""
        return self._receiver_ids

    # ── handler registration ──────────────────────────────────────────────────

    def add_endpoint_handler(self, name: str, handler: Callable) -> None:
        """Register *handler* for endpoint *name* (called by ComputeNode.endpoint)."""
        self._handlers[name] = handler

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the underlying iroh node(s) and populate :attr:`receiver_ids`."""
        loop = asyncio.get_running_loop()
        global _iroh_event_loop_ref
        prev_loop = _iroh_event_loop_ref() if _iroh_event_loop_ref is not None else None
        if prev_loop is not loop:
            iroh_ffi.uniffi_set_event_loop(loop)
            _iroh_event_loop_ref = weakref.ref(loop)
        self._node = await Iroh.memory_with_options(
            NodeOptions(protocols={PROTOCOL_ID_NODEMESH: _NodeMeshProtocolFactory(self._handlers)})
        )
        node_id: str = await self._node.net().node_id()
        self._receiver_ids = [node_id]
        logger.debug(f"IrohNode started, node_id={node_id[:16]}...")

    async def stop(self) -> None:
        """Shut down the iroh node."""
        if self._node is not None:
            try:
                await self._node.node().shutdown()
            except Exception:
                pass
            self._node = None
        self._receiver_ids = []
        logger.debug("IrohNode stopped")

    # ── P2P operations ────────────────────────────────────────────────────────

    async def send(self, target: IrohComputeNode, msg: Any, endpoint: str | None) -> None:
        """Send *msg* to the first iroh receiver ID listed in *target*."""
        if self._node is None:
            raise RuntimeError("IrohNode is not started — call await node.start() first")
        if not target.iroh_receiver_ids:
            raise ValueError(f"Target node '{target.node_id}' has no iroh_receiver_ids")

        payload = self._encode(endpoint or "default", msg)
        ep = self._node.node().endpoint()

        receiver_id = target.iroh_receiver_ids[0]
        receiver_key = PublicKey.from_string(receiver_id)
        node_addr = NodeAddr(node_id=receiver_key, derp_url=None, addresses=[])
        conn = await ep.connect(node_addr, PROTOCOL_ID_NODEMESH)
        send_stream = await conn.open_uni()
        await send_stream.write_all(payload)
        await send_stream.finish()
        # Yield to the event loop so the receiver's accept_uni() task runs
        # before this coroutine returns and `conn` goes out of scope.
        await asyncio.sleep(0.05)

    async def broadcast(self, targets: list[IrohComputeNode], msg: Any, endpoint: str | None) -> None:
        """Send *msg* to every node in *targets* concurrently."""
        await asyncio.gather(*[self.send(t, msg, endpoint) for t in targets])

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _encode(endpoint_name: str, msg: Any) -> bytes:
        endpoint_bytes = endpoint_name.encode()
        return struct.pack(">H", len(endpoint_bytes)) + endpoint_bytes + pickle.dumps(msg)
