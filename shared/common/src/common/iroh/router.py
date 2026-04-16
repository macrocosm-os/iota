"""P2P message router with FastAPI-like decorator registration.

Handlers are registered by route string. The request model is inferred
from the handler's first parameter annotation. UNI vs BI stream type is
inferred from the handler's return type annotation:

    @router.handler("/heartbeat")
    def on_heartbeat(msg: PeerHeartbeat, node_id: str):
        ...  # no return -> UNI

    @router.handler("/echo")
    async def on_request(req: SomeRequest, node_id: str) -> SomeResponse:
        return SomeResponse(...)  # has return -> BI

Raw-bytes handlers use ``@router.default``:

    @router.default
    def handle_activation(message: bytes, node_id: str) -> bytes:
        return response_bytes  # BI default
"""

from __future__ import annotations

import inspect
import zlib
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from loguru import logger
from pydantic import BaseModel

from common.iroh.protocol import P2PCallback
from common.iroh.serializer import (
    SERIALIZER_REGISTRY,
    MsgpackSerializer,
    Serializer,
)

# Valid serializer ID bytes for tag detection
_VALID_SERIALIZER_IDS = frozenset(SERIALIZER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Route tag helpers
# ---------------------------------------------------------------------------


def route_tag_for(route: str) -> int:
    """Deterministic 16-bit tag from a route string.

    Uses CRC32 truncated to 16 bits.  Collisions are detected at
    registration time (router startup).
    """
    return zlib.crc32(route.encode()) & 0xFFFF


def wrap_routed_envelope(route: str, model: BaseModel, serializer: Serializer | None = None) -> bytes:
    """Create a routed wire envelope: ``[2B route_tag][1B serializer_id][body]``."""
    ser = serializer or MsgpackSerializer()
    tag = route_tag_for(route)
    body = ser.serialize(model)
    return tag.to_bytes(2, "big") + ser.id + body


def unwrap_routed_envelope(data: bytes) -> tuple[int, bytes, Serializer]:
    """Parse a routed envelope -> ``(route_tag, body_bytes, serializer)``."""
    if len(data) < 3:
        raise ValueError(f"Routed envelope too short ({len(data)} bytes)")
    tag = int.from_bytes(data[:2], "big")
    ser_id = data[2:3]
    ser = SERIALIZER_REGISTRY.get(ser_id)
    if ser is None:
        raise ValueError(f"Unknown serializer ID in routed envelope: {ser_id!r}")
    return tag, data[3:], ser


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------


@dataclass
class HandlerRegistration:
    route: str
    model_cls: type[BaseModel] | None
    handler: Callable
    route_tag: int


def _is_bi(fn: Callable) -> bool:
    """Return True if *fn* has a non-None return type annotation (-> BI)."""
    try:
        hints = get_type_hints(fn)
    except Exception:
        return False
    ret = hints.get("return")
    return ret is not None and ret is not type(None)


def _infer_model_cls(fn: Callable) -> type[BaseModel] | None:
    """Infer the Pydantic model class from the first parameter's type annotation.

    Returns ``None`` if the first param is ``bytes``, unannotated, or not
    a BaseModel subclass.
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        return None
    params = list(inspect.signature(fn).parameters.keys())
    if not params:
        return None
    first_type = hints.get(params[0])
    if first_type is not None and isinstance(first_type, type) and issubclass(first_type, BaseModel):
        return first_type
    return None


# ---------------------------------------------------------------------------
# P2PRouter
# ---------------------------------------------------------------------------


class P2PRouter:
    """Decorator-based message router for P2P streams.

    Registers handlers by route string (e.g. ``"/heartbeat"``).
    The request model is inferred from the handler's first parameter
    annotation.  UNI vs BI is inferred from the return type.
    """

    def __init__(self, serializer: Serializer | None = None):
        self._serializer: Serializer = serializer or MsgpackSerializer()
        self._uni_handlers: dict[int, HandlerRegistration] = {}
        self._bi_handlers: dict[int, HandlerRegistration] = {}
        self._default_uni: Callable | None = None
        self._default_bi: Callable | None = None
        # tag -> route string, for collision detection
        self._all_tags: dict[int, str] = {}

    # ── decorator: routed handler ─────────────────────────────────

    def handler(self, route: str) -> Callable:
        """Register a handler for a route string.

        The request model is inferred from the first parameter's type
        annotation.  UNI vs BI is inferred from the return type.

        Usage::

            @router.handler("/heartbeat")
            def on_heartbeat(msg: PeerHeartbeat, node_id: str):
                ...  # no return -> UNI

            @router.handler("/echo")
            async def on_echo(req: EchoRequest, node_id: str) -> EchoResponse:
                return EchoResponse(...)  # has return -> BI
        """

        def decorator(fn: Callable) -> Callable:
            tag = route_tag_for(route)
            self._check_collision(tag, route)

            model_cls = _infer_model_cls(fn)
            reg = HandlerRegistration(route=route, model_cls=model_cls, handler=fn, route_tag=tag)
            if _is_bi(fn):
                self._bi_handlers[tag] = reg
            else:
                self._uni_handlers[tag] = reg
            return fn

        return decorator

    # ── decorator: default (raw bytes) handler ────────────────────

    def default(self, fn: Callable) -> Callable:
        """Register a default handler for untagged (raw bytes) messages.

        UNI vs BI is inferred from the return type annotation.
        """
        if _is_bi(fn):
            self._default_bi = fn
        else:
            self._default_uni = fn
        return fn

    # ── dispatch properties ───────────────────────────────────────

    @property
    def uni_dispatch(self) -> P2PCallback:
        """Callback for the UNI protocol factory."""
        return self._dispatch_uni

    @property
    def bi_dispatch(self) -> P2PCallback:
        """Callback for the BI protocol factory."""
        return self._dispatch_bi

    # ── internal dispatch ─────────────────────────────────────────

    async def _dispatch_uni(self, data: bytes, node_id: str) -> None:
        handler, message = self._resolve_uni(data)
        if handler is None:
            logger.warning(f"No UNI handler for message from {node_id[:16]}...")
            return
        result = handler(message, node_id)
        if inspect.isawaitable(result):
            await result

    async def _dispatch_bi(self, data: bytes, node_id: str) -> bytes:
        handler, message = self._resolve_bi(data)
        if handler is None:
            logger.warning(f"No BI handler for message from {node_id[:16]}...")
            return b""
        result = handler(message, node_id)
        if inspect.isawaitable(result):
            result = await result
        return self._coerce_bi_response(result)

    # ── resolution helpers ────────────────────────────────────────

    def _try_typed_lookup(self, data: bytes, handlers: dict[int, HandlerRegistration]) -> tuple[Callable, Any] | None:
        """Attempt to match the first 3 bytes as a routed envelope header.

        Returns ``(handler_fn, deserialized_model_or_body_bytes)`` on match,
        else ``None``.
        """
        if len(data) < 3:
            return None
        ser_id = data[2:3]
        if ser_id not in _VALID_SERIALIZER_IDS:
            return None
        tag = int.from_bytes(data[:2], "big")
        reg = handlers.get(tag)
        if reg is None:
            return None
        ser = SERIALIZER_REGISTRY[ser_id]
        if reg.model_cls is not None:
            message = ser.deserialize(data[3:], reg.model_cls)
        else:
            message = data[3:]
        return reg.handler, message

    def _resolve_uni(self, data: bytes) -> tuple[Callable | None, Any]:
        matched = self._try_typed_lookup(data, self._uni_handlers)
        if matched is not None:
            return matched
        if self._default_uni is not None:
            return self._default_uni, data
        return None, data

    def _resolve_bi(self, data: bytes) -> tuple[Callable | None, Any]:
        matched = self._try_typed_lookup(data, self._bi_handlers)
        if matched is not None:
            return matched
        if self._default_bi is not None:
            return self._default_bi, data
        return None, data

    def _coerce_bi_response(self, response: Any) -> bytes:
        """Convert a BI handler's return value to bytes for the wire."""
        if isinstance(response, BaseModel):
            return self._serialize_model_response(response)
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
        logger.warning("BI handler returned unexpected type %s", type(response))
        return b""

    def _serialize_model_response(self, model: BaseModel) -> bytes:
        """Serialize a BaseModel response with serializer ID prefix (no route tag)."""
        ser = self._serializer
        return ser.id + ser.serialize(model)

    # ── collision detection ───────────────────────────────────────

    def _check_collision(self, tag: int, route: str) -> None:
        existing = self._all_tags.get(tag)
        if existing is not None and existing != route:
            raise ValueError(
                f"Route tag collision: {route!r} and {existing!r} both produce "
                f"tag 0x{tag:04X}. Rename one of the routes."
            )
        self._all_tags[tag] = route
