"""Unit tests for the P2P decorator router."""

import asyncio

import pytest
from pydantic import BaseModel

from common.iroh.router import (
    P2PRouter,
    _is_bi,
    route_tag_for,
    unwrap_routed_envelope,
    wrap_routed_envelope,
)
from common.iroh.serializer import JsonSerializer, MsgpackSerializer


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class HeartbeatMsg(BaseModel):
    ts: int
    load: float


class PingRequest(BaseModel):
    seq: int


class PongResponse(BaseModel):
    seq: int
    ok: bool


class AnotherModel(BaseModel):
    value: str


# ---------------------------------------------------------------------------
# route_tag_for
# ---------------------------------------------------------------------------


class TestRouteTag:
    def test_deterministic(self):
        """Same route always produces the same tag."""
        assert route_tag_for("/heartbeat") == route_tag_for("/heartbeat")

    def test_different_routes_different_tags(self):
        """Different routes produce different tags."""
        assert route_tag_for("/heartbeat") != route_tag_for("/echo")

    def test_16_bit_range(self):
        """Tag is within 0..0xFFFF."""
        tag = route_tag_for("/heartbeat")
        assert 0 <= tag <= 0xFFFF


# ---------------------------------------------------------------------------
# wrap / unwrap routed envelope
# ---------------------------------------------------------------------------


class TestRoutedEnvelope:
    def test_roundtrip_msgpack(self):
        msg = HeartbeatMsg(ts=12345, load=0.75)
        data = wrap_routed_envelope("/heartbeat", msg)
        tag, body, ser = unwrap_routed_envelope(data)
        assert tag == route_tag_for("/heartbeat")
        assert isinstance(ser, MsgpackSerializer)
        restored = ser.deserialize(body, HeartbeatMsg)
        assert restored.ts == 12345
        assert restored.load == 0.75

    def test_roundtrip_json(self):
        msg = PingRequest(seq=42)
        json_ser = JsonSerializer()
        data = wrap_routed_envelope("/ping", msg, serializer=json_ser)
        tag, body, ser = unwrap_routed_envelope(data)
        assert tag == route_tag_for("/ping")
        assert isinstance(ser, JsonSerializer)
        restored = ser.deserialize(body, PingRequest)
        assert restored.seq == 42

    def test_envelope_structure(self):
        """First 2 bytes = tag, byte 3 = serializer ID, rest = body."""
        msg = HeartbeatMsg(ts=1, load=0.0)
        data = wrap_routed_envelope("/heartbeat", msg)
        tag_bytes = data[:2]
        ser_id = data[2:3]
        assert int.from_bytes(tag_bytes, "big") == route_tag_for("/heartbeat")
        assert ser_id == b"\x02"  # MsgpackSerializer default

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            unwrap_routed_envelope(b"\x00\x01")

    def test_unknown_serializer_raises(self):
        # Valid tag + invalid serializer byte
        data = b"\x00\x01\xFF" + b"body"
        with pytest.raises(ValueError, match="Unknown serializer"):
            unwrap_routed_envelope(data)


# ---------------------------------------------------------------------------
# _is_bi detection
# ---------------------------------------------------------------------------


class TestIsBi:
    def test_no_annotation_is_uni(self):
        def handler(msg, nid):
            pass

        assert _is_bi(handler) is False

    def test_none_return_is_uni(self):
        def handler(msg, nid) -> None:
            pass

        assert _is_bi(handler) is False

    def test_bytes_return_is_bi(self):
        def handler(msg, nid) -> bytes:
            return b""

        assert _is_bi(handler) is True

    def test_model_return_is_bi(self):
        def handler(msg, nid) -> PongResponse:
            return PongResponse(seq=0, ok=True)

        assert _is_bi(handler) is True

    def test_async_no_return_is_uni(self):
        async def handler(msg, nid):
            pass

        assert _is_bi(handler) is False

    def test_async_with_return_is_bi(self):
        async def handler(msg, nid) -> PongResponse:
            return PongResponse(seq=0, ok=True)

        assert _is_bi(handler) is True


# ---------------------------------------------------------------------------
# P2PRouter: handler registration
# ---------------------------------------------------------------------------


class TestRouterRegistration:
    def test_typed_uni_handler(self):
        router = P2PRouter()

        @router.handler("/heartbeat")
        def on_heartbeat(msg: HeartbeatMsg, node_id: str):
            pass

        tag = route_tag_for("/heartbeat")
        assert tag in router._uni_handlers
        assert tag not in router._bi_handlers
        assert router._uni_handlers[tag].model_cls is HeartbeatMsg

    def test_typed_bi_handler(self):
        router = P2PRouter()

        @router.handler("/ping")
        def on_ping(msg: PingRequest, node_id: str) -> PongResponse:
            return PongResponse(seq=msg.seq, ok=True)

        tag = route_tag_for("/ping")
        assert tag in router._bi_handlers
        assert tag not in router._uni_handlers
        assert router._bi_handlers[tag].model_cls is PingRequest

    def test_raw_bytes_handler(self):
        """Handler with bytes first param gets model_cls=None."""
        router = P2PRouter()

        @router.handler("/raw")
        def on_raw(msg: bytes, node_id: str) -> bytes:
            return b"ok"

        tag = route_tag_for("/raw")
        assert tag in router._bi_handlers
        assert router._bi_handlers[tag].model_cls is None

    def test_default_uni_handler(self):
        router = P2PRouter()

        @router.default
        def on_default(msg: bytes, node_id: str):
            pass

        assert router._default_uni is not None
        assert router._default_bi is None

    def test_default_bi_handler(self):
        router = P2PRouter()

        @router.default
        def on_default(msg: bytes, node_id: str) -> bytes:
            return b"ok"

        assert router._default_bi is not None
        assert router._default_uni is None

    def test_collision_detection(self):
        router = P2PRouter()

        @router.handler("/heartbeat")
        def h1(msg: HeartbeatMsg, node_id: str):
            pass

        # Manually inject a collision
        tag = route_tag_for("/heartbeat")
        router._all_tags[tag] = "/other-route"

        with pytest.raises(ValueError, match="collision"):

            @router.handler("/heartbeat")
            def h2(msg: HeartbeatMsg, node_id: str):
                pass

    def test_same_route_no_collision(self):
        """Registering the same route twice (overwrite) should not raise."""
        router = P2PRouter()

        @router.handler("/heartbeat")
        def h_uni(msg: HeartbeatMsg, node_id: str):
            pass

        # Re-registering same route with BI return type should not raise
        @router.handler("/heartbeat")
        def h_bi(msg: HeartbeatMsg, node_id: str) -> PongResponse:
            return PongResponse(seq=0, ok=True)

    def test_route_stored_in_registration(self):
        router = P2PRouter()

        @router.handler("/my-route")
        def h(msg: HeartbeatMsg, node_id: str):
            pass

        tag = route_tag_for("/my-route")
        assert router._uni_handlers[tag].route == "/my-route"


# ---------------------------------------------------------------------------
# P2PRouter: dispatch
# ---------------------------------------------------------------------------


class TestRouterDispatch:
    @pytest.fixture
    def router_with_handlers(self):
        router = P2PRouter()
        self.uni_calls = []
        self.bi_calls = []
        self.default_bi_calls = []

        @router.handler("/heartbeat")
        def on_heartbeat(msg: HeartbeatMsg, node_id: str):
            self.uni_calls.append((msg, node_id))

        @router.handler("/ping")
        def on_ping(msg: PingRequest, node_id: str) -> PongResponse:
            self.bi_calls.append((msg, node_id))
            return PongResponse(seq=msg.seq, ok=True)

        @router.default
        def on_default_bi(msg: bytes, node_id: str) -> bytes:
            self.default_bi_calls.append((msg, node_id))
            return b"default-response"

        return router

    def test_uni_typed_dispatch(self, router_with_handlers):
        router = router_with_handlers
        msg = HeartbeatMsg(ts=100, load=0.5)
        data = wrap_routed_envelope("/heartbeat", msg)
        asyncio.get_event_loop().run_until_complete(router._dispatch_uni(data, "node-abc"))
        assert len(self.uni_calls) == 1
        assert self.uni_calls[0][0].ts == 100
        assert self.uni_calls[0][1] == "node-abc"

    def test_bi_typed_dispatch(self, router_with_handlers):
        router = router_with_handlers
        msg = PingRequest(seq=7)
        data = wrap_routed_envelope("/ping", msg)
        result = asyncio.get_event_loop().run_until_complete(router._dispatch_bi(data, "node-xyz"))
        assert len(self.bi_calls) == 1
        assert self.bi_calls[0][0].seq == 7
        # Response should be serializer_id + body (no route tag on responses)
        assert len(result) > 0

    def test_bi_default_fallback(self, router_with_handlers):
        router = router_with_handlers
        raw = b"some-raw-activation-bytes"
        result = asyncio.get_event_loop().run_until_complete(router._dispatch_bi(raw, "node-123"))
        assert len(self.default_bi_calls) == 1
        assert self.default_bi_calls[0][0] == raw
        assert result == b"default-response"

    def test_uni_no_handler_does_not_crash(self):
        """UNI dispatch with no matching handler logs warning but doesn't raise."""
        router = P2PRouter()
        asyncio.get_event_loop().run_until_complete(router._dispatch_uni(b"unhandled", "node-000"))

    def test_bi_no_handler_returns_empty(self):
        """BI dispatch with no matching handler returns empty bytes."""
        router = P2PRouter()
        result = asyncio.get_event_loop().run_until_complete(router._dispatch_bi(b"unhandled", "node-000"))
        assert result == b""


# ---------------------------------------------------------------------------
# P2PRouter: async handler support
# ---------------------------------------------------------------------------


class TestRouterAsyncHandlers:
    def test_async_uni_handler(self):
        router = P2PRouter()
        calls = []

        @router.handler("/heartbeat")
        async def on_heartbeat(msg: HeartbeatMsg, node_id: str):
            calls.append(msg.ts)

        data = wrap_routed_envelope("/heartbeat", HeartbeatMsg(ts=999, load=0.1))
        asyncio.get_event_loop().run_until_complete(router._dispatch_uni(data, "node-async"))
        assert calls == [999]

    def test_async_bi_handler(self):
        router = P2PRouter()

        @router.handler("/ping")
        async def on_ping(msg: PingRequest, node_id: str) -> PongResponse:
            return PongResponse(seq=msg.seq, ok=True)

        data = wrap_routed_envelope("/ping", PingRequest(seq=42))
        result = asyncio.get_event_loop().run_until_complete(router._dispatch_bi(data, "node-async"))
        # Response is serializer_id + body
        assert len(result) > 1
        ser_id = result[:1]
        from common.iroh.serializer import SERIALIZER_REGISTRY

        ser = SERIALIZER_REGISTRY[ser_id]
        resp = ser.deserialize(result[1:], PongResponse)
        assert resp.seq == 42

    def test_async_default_handler(self):
        router = P2PRouter()

        @router.default
        async def on_default(msg: bytes, node_id: str) -> bytes:
            return b"async-default"

        result = asyncio.get_event_loop().run_until_complete(router._dispatch_bi(b"raw", "node-x"))
        assert result == b"async-default"


# ---------------------------------------------------------------------------
# P2PRouter: BI response coercion
# ---------------------------------------------------------------------------


class TestBiResponseCoercion:
    def test_none_returns_empty(self):
        router = P2PRouter()
        assert router._coerce_bi_response(None) == b""

    def test_bytes_passthrough(self):
        router = P2PRouter()
        assert router._coerce_bi_response(b"hello") == b"hello"

    def test_str_encoded(self):
        router = P2PRouter()
        assert router._coerce_bi_response("hello") == b"hello"

    def test_bytearray_converted(self):
        router = P2PRouter()
        assert router._coerce_bi_response(bytearray(b"hi")) == b"hi"

    def test_model_wrapped(self):
        router = P2PRouter()
        resp = PongResponse(seq=1, ok=True)
        result = router._coerce_bi_response(resp)
        # Model response: [1B serializer_id][body]
        ser_id = result[:1]
        from common.iroh.serializer import SERIALIZER_REGISTRY

        ser = SERIALIZER_REGISTRY[ser_id]
        restored = ser.deserialize(result[1:], PongResponse)
        assert restored.seq == 1
        assert restored.ok is True
