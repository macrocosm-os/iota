import pytest
from common.models.activation_push import ActivationPushMessage
from iota_sdk.p2p import P2PResponseStatus, decode_push_ack
from iota_sdk.p2p import P2PRouter, route_tag

from miner.p2p.stack import P2PStack


def test_activation_push_route_is_registered_as_bi() -> None:
    stack = P2PStack()
    router = P2PRouter()
    stack._router = router  # noqa: SLF001

    stack._wire_router_handlers()  # noqa: SLF001

    tag = route_tag("/activation/push")
    assert tag in router._bi_handlers  # noqa: SLF001
    assert tag not in router._uni_handlers  # noqa: SLF001


def _push_handler(stack: P2PStack):
    router = P2PRouter()
    stack._router = router  # noqa: SLF001
    stack._wire_router_handlers()  # noqa: SLF001
    return router._bi_handlers[route_tag("/activation/push")].handler  # noqa: SLF001


def _push_msg(direction: str) -> ActivationPushMessage:
    return ActivationPushMessage(
        activation_id="act-1",
        direction=direction,
        tensor_bytes=b"",
        run_id="run-x",
        source_hotkey="5Sender",
    )


@pytest.mark.asyncio
async def test_benched_miner_nacks_forward_pushes() -> None:
    stack = P2PStack()
    stack.miner_status_getter = lambda: "initializing"
    handler = _push_handler(stack)

    ack_bytes, after = await handler(_push_msg("forward"), "peer-1")

    assert decode_push_ack(ack_bytes) != P2PResponseStatus.SUCCESS
    await after()
    assert stack._push_queue.empty()  # noqa: SLF001


@pytest.mark.asyncio
async def test_benched_miner_still_accepts_backward_pushes() -> None:
    stack = P2PStack()
    stack.miner_status_getter = lambda: "initializing"
    handler = _push_handler(stack)

    ack_bytes, after = await handler(_push_msg("backward"), "peer-1")

    assert decode_push_ack(ack_bytes) == P2PResponseStatus.SUCCESS
    await after()
    assert stack._push_queue.qsize() == 1  # noqa: SLF001


@pytest.mark.asyncio
async def test_training_miner_accepts_forward_pushes() -> None:
    stack = P2PStack()
    stack.miner_status_getter = lambda: "training"
    handler = _push_handler(stack)

    ack_bytes, after = await handler(_push_msg("forward"), "peer-1")

    assert decode_push_ack(ack_bytes) == P2PResponseStatus.SUCCESS
    await after()
    assert stack._push_queue.qsize() == 1  # noqa: SLF001


@pytest.mark.asyncio
async def test_no_getter_disables_the_gate() -> None:
    stack = P2PStack()
    handler = _push_handler(stack)

    ack_bytes, _ = await handler(_push_msg("forward"), "peer-1")

    assert decode_push_ack(ack_bytes) == P2PResponseStatus.SUCCESS
