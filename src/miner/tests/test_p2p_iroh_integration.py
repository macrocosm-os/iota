"""Integration tests for P2P activation transfer using iroh-cosmos."""

import asyncio
from unittest.mock import MagicMock

import pytest

from common import settings as common_settings
from miner.new_miner import Miner
from miner.sync.variable import SyncedVariable, PollingLoop
from common.iroh.p2p_protocol import P2PNotFoundError


@pytest.fixture(autouse=True)
def _stub_polling_loop(monkeypatch):
    """Replace the shared PollingLoop with a no-op mock so tests don't
    hit the bridge HTTP server (which isn't running locally)."""
    monkeypatch.setattr(SyncedVariable, "polling_loop", MagicMock(spec=PollingLoop))


@pytest.fixture(autouse=True)
def _relaxed_p2p_auth_timeout(monkeypatch):
    # Sender retries reuse the same pre-signed request bytes; under
    # heavy concurrent load on CI runners the elapsed sign→verify time
    # can exceed the 30s production default and trip UNAUTHORIZED.
    monkeypatch.setattr("common.iroh.p2p_stack.P2P_AUTH_TIMEOUT_MS", 300_000)


async def _cross_register_peer_addrs(*miners: Miner) -> None:
    """Cross-register every miner's iroh address hints with every other miner's sender.

    In production the node registry sync does this automatically; these tests
    bypass the registry and dial peers directly by p2p_node_id, so we wire it
    up by hand. Iroh's default discovery is disabled, so dials fail with
    ``PeerAddressUnknownError`` if the address book isn't populated.
    """
    for sender in miners:
        for peer in miners:
            if peer is sender:
                continue
            await sender.p2p.sender.register_peer(
                peer.p2p_node_id,
                peer.p2p.relay_url,
                peer.p2p.direct_addresses,
            )


@pytest.mark.asyncio
async def test_miner_p2p_activation_roundtrip(monkeypatch):
    """Test basic P2P activation request/response between two miners."""
    monkeypatch.setenv("BITTENSOR", "False")
    monkeypatch.setattr(common_settings, "BITTENSOR", False)

    miner_a = Miner(wallet_name="p2p_test_wallet_a", wallet_hotkey="p2p_test_hotkey_a", mock=True)
    miner_b = Miner(wallet_name="p2p_test_wallet_b", wallet_hotkey="p2p_test_hotkey_b", mock=True)

    await miner_a._start_p2p()
    await miner_b._start_p2p()
    await _cross_register_peer_addrs(miner_a, miner_b)

    try:
        activation_id = "activation-123"
        payload = b"payload-bytes"
        await miner_b.cache_activation(activation_id, payload)

        received = await miner_a.request_activation_p2p(
            activation_id=activation_id,
            source_node_id=miner_b.p2p_node_id,
        )

        assert received == payload
    finally:
        await miner_a._stop_p2p()
        await miner_b._stop_p2p()


@pytest.mark.asyncio
async def test_miner_p2p_activation_not_found(monkeypatch):
    """Test P2P request for non-existent activation returns error."""
    monkeypatch.setenv("BITTENSOR", "False")
    monkeypatch.setattr(common_settings, "BITTENSOR", False)

    miner_a = Miner(wallet_name="p2p_notfound_wallet_a", wallet_hotkey="p2p_notfound_hotkey_a", mock=True)
    miner_b = Miner(wallet_name="p2p_notfound_wallet_b", wallet_hotkey="p2p_notfound_hotkey_b", mock=True)

    await miner_a._start_p2p()
    await miner_b._start_p2p()
    await _cross_register_peer_addrs(miner_a, miner_b)

    try:
        # Request activation that doesn't exist in miner_b's cache
        with pytest.raises(P2PNotFoundError, match="not found on peer"):
            await miner_a.request_activation_p2p(
                activation_id="nonexistent-activation",
                source_node_id=miner_b.p2p_node_id,
            )
    finally:
        await miner_a._stop_p2p()
        await miner_b._stop_p2p()


@pytest.mark.asyncio
async def test_miner_p2p_large_activation(monkeypatch):
    """Test P2P transfer of larger activation data."""
    monkeypatch.setenv("BITTENSOR", "False")
    monkeypatch.setattr(common_settings, "BITTENSOR", False)

    miner_a = Miner(wallet_name="p2p_large_wallet_a", wallet_hotkey="p2p_large_hotkey_a", mock=True)
    miner_b = Miner(wallet_name="p2p_large_wallet_b", wallet_hotkey="p2p_large_hotkey_b", mock=True)

    await miner_a._start_p2p()
    await miner_b._start_p2p()
    await _cross_register_peer_addrs(miner_a, miner_b)

    try:
        activation_id = "large-activation"
        # 1MB payload
        payload = b"X" * (1024 * 1024)
        await miner_b.cache_activation(activation_id, payload)

        received = await miner_a.request_activation_p2p(
            activation_id=activation_id,
            source_node_id=miner_b.p2p_node_id,
        )

        assert received == payload
    finally:
        await miner_a._stop_p2p()
        await miner_b._stop_p2p()


@pytest.mark.asyncio
async def test_miner_p2p_multi_miner_concurrent_requests(monkeypatch):
    """Test concurrent P2P requests across multiple miners (simulates layer communication)."""
    monkeypatch.setenv("BITTENSOR", "False")
    monkeypatch.setattr(common_settings, "BITTENSOR", False)

    miner_count = 8
    layer_size = miner_count // 2
    fanout = 3
    miners = [
        Miner(wallet_name=f"p2p_multi_wallet_{i}", wallet_hotkey=f"p2p_multi_hotkey_{i}", mock=True)
        for i in range(miner_count)
    ]

    for miner in miners:
        await miner._start_p2p(timeout=30.0)
    await _cross_register_peer_addrs(*miners)

    try:
        for i, miner in enumerate(miners):
            await miner.cache_activation(f"activation-{i}", f"payload-{i}".encode())

        layer0 = miners[:layer_size]
        layer1 = miners[layer_size:]

        tasks = []
        expected = []

        for i, miner in enumerate(layer0):
            for j in range(fanout):
                target_idx = (i + j) % layer_size
                target = layer1[target_idx]
                tasks.append(
                    miner.request_activation_p2p(
                        f"activation-{layer_size + target_idx}",
                        target.p2p_node_id,
                    )
                )
                expected.append(f"payload-{layer_size + target_idx}".encode())

        for i, miner in enumerate(layer1):
            for j in range(fanout):
                target_idx = (i + j) % layer_size
                target = layer0[target_idx]
                tasks.append(
                    miner.request_activation_p2p(
                        f"activation-{target_idx}",
                        target.p2p_node_id,
                    )
                )
                expected.append(f"payload-{target_idx}".encode())

        results = await asyncio.gather(*tasks)
        assert results == expected
    finally:
        await asyncio.gather(*(miner._stop_p2p() for miner in miners))


@pytest.mark.asyncio
async def test_miner_p2p_bidirectional_same_pair(monkeypatch):
    """Test that two miners can request activations from each other."""
    monkeypatch.setenv("BITTENSOR", "False")
    monkeypatch.setattr(common_settings, "BITTENSOR", False)

    miner_a = Miner(wallet_name="p2p_bidir_wallet_a", wallet_hotkey="p2p_bidir_hotkey_a", mock=True)
    miner_b = Miner(wallet_name="p2p_bidir_wallet_b", wallet_hotkey="p2p_bidir_hotkey_b", mock=True)

    await miner_a._start_p2p()
    await miner_b._start_p2p()
    await _cross_register_peer_addrs(miner_a, miner_b)

    try:
        # Both miners cache different activations
        await miner_a.cache_activation("activation-from-a", b"data-from-a")
        await miner_b.cache_activation("activation-from-b", b"data-from-b")

        # Concurrent requests in both directions
        result_a, result_b = await asyncio.gather(
            miner_a.request_activation_p2p("activation-from-b", miner_b.p2p_node_id),
            miner_b.request_activation_p2p("activation-from-a", miner_a.p2p_node_id),
        )

        assert result_a == b"data-from-b"
        assert result_b == b"data-from-a"
    finally:
        await miner_a._stop_p2p()
        await miner_b._stop_p2p()
