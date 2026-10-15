import asyncio

import pytest

from common.models.run_flags import RUN_FLAGS
from common.utils.exceptions import LayerStateException
from common.utils.shared_states import LayerPhase
from miner.training import activation_queue as activation_queue_module
from miner.training.activation_cache import ActivationCache
from miner.training.activation_queue import ActivationQueue

pytestmark = pytest.mark.asyncio


class _DummyHotkey:
    ss58_address = "dummy_hotkey"


class _FakeMinerAPIClient:
    """Mimics MinerAPIClient: heartbeat raises once the orchestrator layer has left TRAINING."""

    def __init__(self, layer_state: LayerPhase, orchestrator_phase: LayerPhase, all_layers_training: bool) -> None:
        self.hotkey = _DummyHotkey()
        self.layer_state = layer_state
        self.orchestrator_phase = orchestrator_phase
        self.all_layers_training = all_layers_training
        self.heartbeat_calls = 0

    async def heartbeat(self, expected_phase: LayerPhase | None = None):
        self.heartbeat_calls += 1
        if expected_phase is not None and self.orchestrator_phase != expected_phase:
            self.layer_state = self.orchestrator_phase
            raise LayerStateException(f"Miner is moving state from {expected_phase} to {self.orchestrator_phase}")
        return None

    async def get_all_layers_training(self) -> bool:
        return self.all_layers_training


class _DummyStateManager:
    layer = 2


class _FakeMiner:
    def __init__(self) -> None:
        self._p2p_push_queue: asyncio.Queue = asyncio.Queue()

    async def _announce_training_status(self) -> None:
        return None


def _make_queue(api_client: _FakeMinerAPIClient) -> ActivationQueue:
    return ActivationQueue(
        miner_api_client=api_client,
        state_manager=_DummyStateManager(),
        activation_cache=ActivationCache(hotkey="dummy_hotkey", cache_timeout_sec=60),
        mock=True,
        run_flags=RUN_FLAGS,
        miner=_FakeMiner(),
    )


@pytest.fixture(autouse=True)
def _short_intervals(monkeypatch):
    monkeypatch.setattr(activation_queue_module, "PUSH_STATE_CHECK_INTERVAL_SEC", 30.0)
    monkeypatch.setattr(activation_queue_module, "PUSH_STATE_CHECK_FAST_INTERVAL_SEC", 0.2)


async def test_push_fetcher_reacts_fast_to_swallowed_layer_state_error():
    # submit_activation / report_loss already saw the LayerStateError and recorded it.
    api_client = _FakeMinerAPIClient(
        layer_state=LayerPhase.WEIGHTS_UPLOADING,
        orchestrator_phase=LayerPhase.WEIGHTS_UPLOADING,
        all_layers_training=True,
    )
    queue = _make_queue(api_client)

    with pytest.raises(LayerStateException):
        await asyncio.wait_for(queue._fetch_activations_push_based(), timeout=5)


async def test_push_fetcher_reacts_fast_when_all_layers_gate_closes():
    # No API call has failed yet, but the all-layers gate closed: the layer is moving on.
    api_client = _FakeMinerAPIClient(
        layer_state=LayerPhase.TRAINING,
        orchestrator_phase=LayerPhase.WEIGHTS_UPLOADING,
        all_layers_training=False,
    )
    queue = _make_queue(api_client)

    with pytest.raises(LayerStateException):
        await asyncio.wait_for(queue._fetch_activations_push_based(), timeout=5)


async def test_push_fetcher_keeps_slow_cadence_while_training():
    api_client = _FakeMinerAPIClient(
        layer_state=LayerPhase.TRAINING,
        orchestrator_phase=LayerPhase.TRAINING,
        all_layers_training=True,
    )
    queue = _make_queue(api_client)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue._fetch_activations_push_based(), timeout=1.5)
    assert api_client.heartbeat_calls == 0


async def test_push_fetcher_clears_stale_hint_after_successful_heartbeat():
    # A late LayerStateError left a stale hint, but the orchestrator says we are TRAINING.
    api_client = _FakeMinerAPIClient(
        layer_state=LayerPhase.WEIGHTS_UPLOADING,
        orchestrator_phase=LayerPhase.TRAINING,
        all_layers_training=True,
    )
    queue = _make_queue(api_client)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue._fetch_activations_push_based(), timeout=1.5)
    assert api_client.heartbeat_calls == 1
    assert api_client.layer_state == LayerPhase.TRAINING
