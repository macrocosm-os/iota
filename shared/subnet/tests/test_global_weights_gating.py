"""Regression tests for the gate that prevents miners from contributing with
random (un-synced) weights — see fix/restrict-miner-without-global-weights.

These tests exercise BaseNeuron.download_and_set_global_weights directly with
fakes, since the real path requires a registered miner, a configured
orchestrator client, and a fully loaded transformer split.
"""

import pytest

import subnet.base.base_neuron as base_neuron_module
from subnet.base.base_neuron import BaseNeuron
from subnet.model.model_mixin import ModelManager


class _FakeHotkey:
    def __init__(self, ss58_address: str = "5FakeHotkey00000000000000000000000000000000000000"):
        self.ss58_address = ss58_address


class _FakeWallet:
    def __init__(self):
        self.hotkey = _FakeHotkey()


class _FakeAPIClient:
    """Mirrors the surface of CommonAPIClient that download_and_set_global_weights
    needs: ``get_merged_partitions`` (the data) and ``get_run_epoch`` (the
    authoritative current epoch). The latter replaces the old local
    ``epoch_on_registration + epoch_counter`` computation."""

    def __init__(self, partitions, current_epoch: int):
        self._partitions = partitions
        self._current_epoch = current_epoch
        self.partition_calls = 0
        self.epoch_calls = 0

    async def get_merged_partitions(self, hotkey):
        self.partition_calls += 1
        return self._partitions

    async def get_run_epoch(self, run_id: str, hotkey) -> int:
        self.epoch_calls += 1
        return self._current_epoch


def _make_neuron(*, partitions, current_epoch: int) -> BaseNeuron:
    neuron = BaseNeuron()
    neuron.wallet = _FakeWallet()
    neuron.hotkey = neuron.wallet.hotkey.ss58_address
    neuron.layer = 0
    neuron.num_partitions = 4
    neuron.run_id = "test-run"
    neuron.model_manager = ModelManager()
    # global_weights_loaded defaults to False on a fresh ModelManager — exactly
    # the post-crash / post-restart state we want to test.
    neuron._api_client = _FakeAPIClient(partitions=partitions, current_epoch=current_epoch)
    return neuron


@pytest.mark.asyncio
async def test_genesis_epoch_accepts_random_init():
    # Runs start at epoch 1 (epoch 0 does not exist), and the first merge cycle
    # only completes on the transition into epoch 2 — so a miner registering on
    # a fresh run at epoch 1 legitimately finds an empty merge and must bootstrap
    # from the shared random init. This is the case the gate previously broke.
    neuron = _make_neuron(partitions=[], current_epoch=1)
    result = await neuron.download_and_set_global_weights(client=neuron._api_client, device="cpu")
    assert result is None
    assert neuron.model_manager.global_weights_loaded is True, (
        "Genesis epoch with empty merge MUST be treated as legitimately aligned "
        "so the run can bootstrap from a shared random init."
    )


@pytest.mark.asyncio
async def test_epoch_zero_defensively_accepts_random_init():
    # Defensive: epoch 0 shouldn't occur in practice, but if it ever does it is
    # unambiguously pre-merge and must be treated as genesis, not refused.
    neuron = _make_neuron(partitions=[], current_epoch=0)
    result = await neuron.download_and_set_global_weights(client=neuron._api_client, device="cpu")
    assert result is None
    assert neuron.model_manager.global_weights_loaded is True


@pytest.mark.asyncio
async def test_empty_partitions_after_restart_refuses_to_train():
    # Simulates a miner that crashed (NaN), restarted, re-registered at epoch
    # 14, and finds the orchestrator hasn't published merged partitions yet.
    neuron = _make_neuron(partitions=[], current_epoch=14)
    with pytest.raises(RuntimeError) as excinfo:
        await neuron.download_and_set_global_weights(client=neuron._api_client, device="cpu")
    assert "never loaded global weights" in str(excinfo.value)
    assert (
        neuron.model_manager.global_weights_loaded is False
    ), "Refusal must NOT silently flip the flag — that would defeat the gate on the next retry."


@pytest.mark.asyncio
async def test_empty_partitions_after_prior_sync_keeps_weights():
    # The miner already synced once this session; an empty subsequent merge is
    # tolerated (we keep the in-memory descendant of the global state).
    neuron = _make_neuron(partitions=[], current_epoch=17)
    neuron.model_manager.global_weights_loaded = True
    result = await neuron.download_and_set_global_weights(client=neuron._api_client, device="cpu")
    assert result is None
    assert neuron.model_manager.global_weights_loaded is True


@pytest.mark.asyncio
async def test_download_caches_current_epoch_on_model_manager():
    """download_and_set_global_weights should fetch the orchestrator epoch
    exactly once and stash it on model_manager.current_epoch so hot paths
    (warm-up gate, merge label) don't round-trip per call."""
    neuron = _make_neuron(partitions=[], current_epoch=14)
    neuron.model_manager.global_weights_loaded = True  # treat empty merge as benign
    assert neuron.model_manager.current_epoch is None  # cache starts empty

    await neuron.download_and_set_global_weights(client=neuron._api_client, device="cpu")

    assert neuron.model_manager.current_epoch == 14, "cache must be populated by download"
    assert neuron._api_client.epoch_calls == 1, "exactly one get_run_epoch round-trip"


@pytest.mark.asyncio
async def test_successful_download_sets_flag(monkeypatch):
    # Stub out the heavy weight-download path: pretend a download returned a
    # valid tensor, capture the set call.
    import torch

    class _StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(8))

    set_calls = []

    class _StubManager(ModelManager):
        def __init__(self):
            super().__init__()
            self.model = _StubModel()
            self.epoch_on_registration = 14

        async def set_model_weights_and_optimizer_state(self, model_weights=None, optimizer_state=None):
            set_calls.append(model_weights)

    neuron = _make_neuron(partitions=[object()], current_epoch=14)
    neuron.model_manager = _StubManager()

    async def _fake_download(**kwargs):
        return torch.ones(8)

    monkeypatch.setattr(base_neuron_module, "download_merged_partitions", _fake_download)

    await neuron.download_and_set_global_weights(client=neuron._api_client, device="cpu")
    assert len(set_calls) == 1
    assert neuron.model_manager.global_weights_loaded is True
