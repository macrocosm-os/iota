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
