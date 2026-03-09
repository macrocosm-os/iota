import asyncio
import inspect
import time
from typing import Any, Callable, Union

from iroh import BiStream, Connection, Endpoint, ProtocolCreator, RecvStream
from loguru import logger
from pydantic import BaseModel

from common.iroh.serializer import Serializer, unwrap_envelope, wrap_envelope

P2PCallback = Callable[[Any, str], Any]
ResponseLike = Union[bytes, bytearray, memoryview, str, BaseModel, None]

PROTOCOL_ID_UNI = b"my-simple-protocol/1.0/uni"
PROTOCOL_ID_BI = b"my-simple-protocol/1.0/bi"


class ProtocolHandler:
    def __init__(
        self,
        max_message_size: int,
        callback_function: P2PCallback,
        response_callback_function: P2PCallback | None = None,
        serializer: Serializer | None = None,
        request_model_cls: type[BaseModel] | None = None,
    ):
        self.max_message_size = max_message_size
        self.callback_function = callback_function
        self.response_callback_function: P2PCallback = response_callback_function or callback_function
        self.serializer = serializer
        self.request_model_cls = request_model_cls

    async def accept(self, conn: Connection):
        raise NotImplementedError("Use UniProtocol or BiProtocol")

    async def _handle_uni(self, conn: Connection, remote_id: str):
        while True:
            try:
                recv_stream: RecvStream = await conn.accept_uni()

                start = time.time()
                message_bytes = await recv_stream.read_to_end(self.max_message_size)
                logger.debug(
                    f"Received {len(message_bytes)} bytes in {time.time() - start:.3f}s from {remote_id[:16]}..."
                )

                if self.request_model_cls is not None:
                    message = unwrap_envelope(message_bytes, self.request_model_cls)
                    await self._maybe_await(self.callback_function(message, remote_id))
                else:
                    await self._maybe_await(self.callback_function(message_bytes, remote_id))
            except asyncio.CancelledError:
                logger.debug(f"Uni handler cancelled for {remote_id[:16]}...")
                break
            except Exception as e:
                logger.debug(f"Uni handler ended for {remote_id[:16]}...: {type(e).__name__}: {e}")
                break

    async def _handle_bi(self, conn: Connection, remote_id: str):
        while True:
            try:
                bi_stream: BiStream = await conn.accept_bi()

                request_bytes = await bi_stream.recv().read_to_end(self.max_message_size)

                if self.request_model_cls is not None:
                    request = unwrap_envelope(request_bytes, self.request_model_cls)
                    response = await self._maybe_await(self.response_callback_function(request, remote_id))
                else:
                    response = await self._maybe_await(self.response_callback_function(request_bytes, remote_id))

                response_bytes = self._coerce_response_bytes(response)

                await bi_stream.send().write_all(response_bytes)
                await bi_stream.send().finish()
                await bi_stream.send().stopped()
            except asyncio.CancelledError:
                logger.debug(f"Bi handler cancelled for {remote_id[:16]}...")
                break
            except Exception as e:
                logger.debug(f"Bi handler ended for {remote_id[:16]}...: {type(e).__name__}: {e}")
                break

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _coerce_response_bytes(self, response: ResponseLike) -> bytes:
        if isinstance(response, BaseModel) and self.serializer is not None:
            return wrap_envelope(response, self.serializer)
        if response is None:
            return b""
        if isinstance(response, bytes):
            return response
        if isinstance(response, bytearray):
            return bytes(response)
        if isinstance(response, memoryview):
            return response.tobytes()
        if isinstance(response, str):
            return response.encode()
        logger.warning("Bi response should be bytes, str, or BaseModel; got %s", type(response))
        return b""

    async def shutdown(self):
        logger.debug("Shutting down protocol")


class UniProtocol(ProtocolHandler):
    async def accept(self, conn: Connection):
        remote_id = conn.remote_node_id()
        logger.debug(f"Accepted uni connection from: {remote_id[:16]}...")
        try:
            await self._handle_uni(conn, remote_id)
        except Exception as e:
            logger.error(f"Error handling uni connection from {remote_id[:16]}...: {e}")
        finally:
            logger.debug(f"Uni connection handler finished for {remote_id[:16]}...")


class BiProtocol(ProtocolHandler):
    async def accept(self, conn: Connection):
        remote_id = conn.remote_node_id()
        logger.debug(f"Accepted bi connection from: {remote_id[:16]}...")
        try:
            await self._handle_bi(conn, remote_id)
        except Exception as e:
            logger.error(f"Error handling bi connection from {remote_id[:16]}...: {e}")
        finally:
            logger.debug(f"Bi connection handler finished for {remote_id[:16]}...")


class _BaseProtocolFactory(ProtocolCreator):
    def __init__(
        self,
        callback_function: P2PCallback,
        max_message_size: int,
        response_callback_function: P2PCallback | None = None,
        serializer: Serializer | None = None,
        request_model_cls: type[BaseModel] | None = None,
    ):
        self.callback_function = callback_function
        self.max_message_size = max_message_size
        self.response_callback_function = response_callback_function
        self.serializer = serializer
        self.request_model_cls = request_model_cls

    def _kwargs(self) -> dict:
        return dict(
            max_message_size=self.max_message_size,
            callback_function=self.callback_function,
            response_callback_function=self.response_callback_function,
            serializer=self.serializer,
            request_model_cls=self.request_model_cls,
        )


class UniProtocolFactory(_BaseProtocolFactory):
    def create(self, endpoint: Endpoint) -> UniProtocol:
        return UniProtocol(**self._kwargs())


class BiProtocolFactory(_BaseProtocolFactory):
    def create(self, endpoint: Endpoint) -> BiProtocol:
        return BiProtocol(**self._kwargs())
