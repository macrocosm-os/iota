import io

import pytest
import torch

from common.models.api_models import ActivationResponse
from common.models.run_flags import RUN_FLAGS
from common.settings import MINI_BATCH_SIZE
from common.utils.exceptions import ActivationHashMismatchError
from common.models.activation_push import ActivationPushMessage
from miner.training.activation_cache import ActivationCache
from miner.training.activation_queue import ActivationQueue
from miner.utils.activation_hash import compute_activation_hash


class _DummyHotkey:
    ss58_address = "dummy_hotkey"


class _DummyMinerAPIClient:
    def __init__(self) -> None:
        self.hotkey = _DummyHotkey()


class _DummyStateManager:
    def __init__(self, layer: int = 0) -> None:
        self.layer = layer


class _FakeMiner:
    def __init__(self, response_bytes: bytes) -> None:
        self._response_bytes = response_bytes
        self.calls: list[dict] = []

    async def request_activation_p2p(self, activation_id: str, source_node_id: str, **kwargs) -> bytes:
        self.calls.append(
            {
                "activation_id": activation_id,
                "source_node_id": source_node_id,
            }
        )
        return self._response_bytes


def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_download_activation_p2p_success():
    tensor = torch.arange(MINI_BATCH_SIZE * 100, dtype=torch.float32).reshape(MINI_BATCH_SIZE, 100)
    tensor_bytes = _serialize_tensor(tensor)
    expected_hash = compute_activation_hash(tensor_bytes)

    fake_miner = _FakeMiner(tensor_bytes)
    api_client = _DummyMinerAPIClient()
    queue = ActivationQueue(
        miner_api_client=api_client,
        state_manager=_DummyStateManager(),
        activation_cache=ActivationCache(hotkey="dummy_hotkey", cache_timeout_sec=60),
        mock=True,
        run_flags=RUN_FLAGS,
        miner=fake_miner,
    )

    response = ActivationResponse(
        activation_id="activation-123",
        source_node_id="node-abc",
        expected_input_hash=expected_hash,
    )

    downloaded_tensor, received_hash = await queue._download_activation_p2p(response)

    assert torch.equal(downloaded_tensor, tensor)
    assert received_hash == expected_hash
    assert fake_miner.calls == [{"activation_id": "activation-123", "source_node_id": "node-abc"}]


@pytest.mark.asyncio
async def test_download_activation_p2p_uses_source_activation_id():
    """When source_activation_id is provided, use it for P2P request instead of activation_id."""
    tensor = torch.arange(MINI_BATCH_SIZE * 100, dtype=torch.float32).reshape(MINI_BATCH_SIZE, 100)
    tensor_bytes = _serialize_tensor(tensor)
    expected_hash = compute_activation_hash(tensor_bytes)

    fake_miner = _FakeMiner(tensor_bytes)
    api_client = _DummyMinerAPIClient()
    queue = ActivationQueue(
        miner_api_client=api_client,
        state_manager=_DummyStateManager(),
        activation_cache=ActivationCache(hotkey="dummy_hotkey", cache_timeout_sec=60),
        mock=True,
        run_flags=RUN_FLAGS,
        miner=fake_miner,
    )

    # activation_id is the new ID for this layer, source_activation_id is what the producer cached
    response = ActivationResponse(
        activation_id="new-activation-id",
        source_activation_id="original-cached-id",
        source_node_id="node-abc",
        expected_input_hash=expected_hash,
    )

    downloaded_tensor, received_hash = await queue._download_activation_p2p(response)

    assert torch.equal(downloaded_tensor, tensor)
    # P2P request should use source_activation_id, not activation_id
    assert fake_miner.calls == [{"activation_id": "original-cached-id", "source_node_id": "node-abc"}]


@pytest.mark.asyncio
async def test_download_activation_p2p_hash_mismatch():
    tensor = torch.arange(MINI_BATCH_SIZE * 100, dtype=torch.float32).reshape(MINI_BATCH_SIZE, 100)
    tensor_bytes = _serialize_tensor(tensor)

    fake_miner = _FakeMiner(tensor_bytes)
    api_client = _DummyMinerAPIClient()
    queue = ActivationQueue(
        miner_api_client=api_client,
        state_manager=_DummyStateManager(),
        activation_cache=ActivationCache(hotkey="dummy_hotkey", cache_timeout_sec=60),
        mock=True,
        run_flags=RUN_FLAGS,
        miner=fake_miner,
    )

    response = ActivationResponse(
        activation_id="activation-456",
        source_node_id="node-def",
        expected_input_hash="deadbeef" * 8,
    )

    with pytest.raises(ActivationHashMismatchError):
        await queue._download_activation_p2p(response)


def test_validate_push_layer_accepts_matching_target() -> None:
    queue = ActivationQueue(
        miner_api_client=_DummyMinerAPIClient(),
        state_manager=_DummyStateManager(layer=2),
        activation_cache=ActivationCache(hotkey="dummy_hotkey", cache_timeout_sec=60),
        mock=True,
        run_flags=RUN_FLAGS,
        miner=None,
    )
    msg = ActivationPushMessage(
        activation_id="a1",
        direction="forward",
        source_hotkey="hk",
        tensor_bytes=b"",
        target_layer=2,
        source_layer=1,
    )
    assert queue._validate_push_layer_routing(msg) is True


def test_validate_push_layer_rejects_mismatch() -> None:
    queue = ActivationQueue(
        miner_api_client=_DummyMinerAPIClient(),
        state_manager=_DummyStateManager(layer=1),
        activation_cache=ActivationCache(hotkey="dummy_hotkey", cache_timeout_sec=60),
        mock=True,
        run_flags=RUN_FLAGS,
        miner=None,
    )
    msg = ActivationPushMessage(
        activation_id="a2",
        direction="forward",
        source_hotkey="hk",
        tensor_bytes=b"",
        target_layer=2,
        source_layer=1,
    )
    assert queue._validate_push_layer_routing(msg) is False


def test_validate_push_layer_legacy_without_target() -> None:
    queue = ActivationQueue(
        miner_api_client=_DummyMinerAPIClient(),
        state_manager=_DummyStateManager(layer=5),
        activation_cache=ActivationCache(hotkey="dummy_hotkey", cache_timeout_sec=60),
        mock=True,
        run_flags=RUN_FLAGS,
        miner=None,
    )
    msg = ActivationPushMessage(
        activation_id="a3",
        direction="forward",
        source_hotkey="hk",
        tensor_bytes=b"",
    )
    assert queue._validate_push_layer_routing(msg) is True
