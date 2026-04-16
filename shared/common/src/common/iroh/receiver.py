from __future__ import annotations

import asyncio
import hashlib

from iroh import Iroh, NodeOptions, iroh_ffi
from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr

from common.iroh.cleanup import _force_free_iroh_node
from common.iroh.monitored_node import MonitoredNode, OnUnhealthy
from common.iroh.protocol import BiProtocolFactory, P2PCallback, PROTOCOL_ID_BI, PROTOCOL_ID_UNI, UniProtocolFactory
from common.iroh.router import P2PRouter
from common.iroh.serializer import Serializer
from common.iroh.settings import DEFAULT_MAX_MESSAGE_SIZE


class Receiver(BaseModel):
    max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE
    seed: str | None = None
    node_id: str | None = None
    node: Iroh | None = None
    _monitored_node: MonitoredNode | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def start(
        self,
        router: P2PRouter | None = None,
        *,
        callback_function: P2PCallback | None = None,
        serializer: Serializer | None = None,
        request_model_cls: type[BaseModel] | None = None,
        on_unhealthy: OnUnhealthy | None = None,
        health_check_interval: float = 30.0,
    ) -> Receiver:
        """Initialise the Iroh node and set ``self.node_id``.

        Pass a :class:`P2PRouter` for decorator-based dispatch, or
        ``callback_function`` for the legacy single-callback mode.

        After this returns the receiver is listening for incoming
        connections but the caller is **not** blocked.  Call
        :meth:`serve_forever` if you want to block until shutdown.
        """
        if router is None and callback_function is None:
            raise ValueError("Either router or callback_function must be provided")

        iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
        secret_key = hashlib.sha256(self.seed.encode()).digest() if self.seed else None

        if router is not None:
            protocols = {
                PROTOCOL_ID_UNI: UniProtocolFactory(
                    callback_function=router.uni_dispatch,
                    max_message_size=self.max_message_size,
                ),
                PROTOCOL_ID_BI: BiProtocolFactory(
                    callback_function=router.bi_dispatch,
                    max_message_size=self.max_message_size,
                ),
            }
        else:
            factory_kwargs = dict(
                callback_function=callback_function,
                max_message_size=self.max_message_size,
                serializer=serializer,
                request_model_cls=request_model_cls,
            )
            protocols = {
                PROTOCOL_ID_UNI: UniProtocolFactory(**factory_kwargs),
                PROTOCOL_ID_BI: BiProtocolFactory(**factory_kwargs),
            }

        # Create Iroh node with separate protocol handlers for uni and bi streams
        self.node = await Iroh.memory_with_options(
            NodeOptions(
                protocols=protocols,
                secret_key=secret_key,
            )
        )
        self.node_id = await self.node.net().node_id()

        node_label = f"receiver-{self.node_id[:16]}" if self.node_id else "receiver"
        self._monitored_node = MonitoredNode(
            node=self.node,
            on_unhealthy=on_unhealthy,
            check_interval=health_check_interval,
            label=node_label,
        )
        self._monitored_node.start_monitoring()

        logger.info(f"Receiver listening on Node ID: {self.node_id}")
        return self

    async def serve_forever(self) -> None:
        """Block until the event loop is cancelled or a keyboard interrupt.

        The Iroh node must already be started via :meth:`start`.
        """
        if self.node is None:
            raise RuntimeError("Receiver.start() must be called before serve_forever()")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("Stopping receiver...")
            try:
                await asyncio.wait_for(self.node.node().shutdown(), timeout=1.0)
            except Exception:
                logger.warning("Receiver node shutdown timed out or failed in serve_forever")

    async def shutdown(self) -> None:
        """Gracefully shut down the Iroh node."""
        if self._monitored_node is not None:
            await self._monitored_node.stop_monitoring()
        if self.node is not None:
            await self.node.node().shutdown()
            logger.success("Receiver shutdown successfully")

    async def force_destroy(self, timeout: float = 5.0) -> None:
        """Force-free the underlying Rust node synchronously via FFI.

        Use this when async ``shutdown()`` hangs or times out.  Calls Rust's
        ``Drop`` directly, bypassing the async polling machinery.
        """
        if self._monitored_node is not None:
            self._monitored_node._on_unhealthy = None
            if self._monitored_node._monitor_task and not self._monitored_node._monitor_task.done():
                self._monitored_node._monitor_task.cancel()
            self._monitored_node._node = None

        # 2. Free the Rust objects synchronously (in executor to not block event loop)
        node_obj = self.node  # Iroh instance
        self.node = None
        self.node_id = None

        if node_obj is None:
            return

        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, _force_free_iroh_node, node_obj),
            timeout=timeout,
        )
