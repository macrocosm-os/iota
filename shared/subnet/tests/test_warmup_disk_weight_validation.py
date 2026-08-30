"""Regression tests for the warm-up disk-weight reload validation introduced
in fix/restrict-miner-without-global-weights.

The warm-up branch in ``ModelManager.local_optimization_step`` (when
``upload_optimizer_state`` is OFF and ``epoch_counter <= 2`` /
``epoch_on_registration > 1``) used to call ``load_model_weights`` and
unconditionally hand the result to ``vector_to_parameters``. That is the path
that silently reintroduced random / NaN weights *after* the
``global_weights_loaded`` gate had already flipped True, defeating the gate.

These tests verify the validation now drops invalid disk snapshots and keeps
the current in-memory parameters intact.
"""

from __future__ import annotations

import pytest
import torch

import subnet.model.model_mixin as model_mixin_module
from subnet.model.model_mixin import ModelManager


class _StubFlag:
    def __init__(self, *, on: bool):
        self._on = on

    def isOff(self) -> bool:
        return not self._on

    def isOn(self) -> bool:
        return self._on


class _StubRunFlags:
    """Match the surface that ModelManager.local_optimization_step uses."""

    def __init__(self):
        self.upload_optimizer_state = _StubFlag(on=False)  # gates the warm-up path
        self.use_AdamW = _StubFlag(on=True)
        self.clip_pseudo_gradients = _StubFlag(on=False)


def _make_manager_for_warmup(*, epoch_on_registration: int = 5) -> ModelManager:
    """Build a ModelManager whose model has a known parameter vector so we can
    assert that the warm-up reload either applied or didn't. ``current_epoch``
    is now passed as a call-time argument to ``local_optimization_step``
    (sourced from the orchestrator in production) rather than tracked on the
    manager itself."""
    mgr = ModelManager()
    mgr.model = torch.nn.Linear(4, 2, bias=False)
    # Seed with a recognizable pattern (all 7s) so we can later check whether
    # the warm-up reload overwrote it.
    with torch.no_grad():
        for p in mgr.model.parameters():
            p.fill_(7.0)
    mgr.optimizer = torch.optim.AdamW(mgr.model.parameters(), lr=1e-3)
    mgr.layer = 0
    mgr.epoch_on_registration = epoch_on_registration
    mgr.run_flags = _StubRunFlags()
    mgr.logger_attributes = {"hotkey": "5FakeHotkey00000000", "run_id": "test-run"}
    mgr.model_config = {"total_global_params": 8, "bottleneck_dim": 4, "emb_dim": 4}
    mgr.model_metadata = {"grad_clip_norm": 1.0, "n_splits": 1}
    return mgr


def _current_params(mgr: ModelManager) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(mgr.model.parameters()).detach().clone()


# Manager defaults to epoch_on_registration=5. Inside the warm-up window the
# caller passes current_epoch with epochs_since_registration <= 2; outside the
# window, anything larger. In production, ``current_epoch`` is read from
# ModelManager.current_epoch (the cache populated once per TRAINING entry by
# download_and_set_global_weights) and passed in by training.py.
_IN_WARMUP_EPOCH = 6  # 6 - 5 = 1 epoch since registration
_OUTSIDE_WARMUP_EPOCH = 15  # well past the 2-epoch warm-up window


@pytest.mark.asyncio
async def test_warmup_reload_skipped_when_disk_returns_none(monkeypatch):
    """No disk snapshot → keep current parameters, don't crash on None."""
    mgr = _make_manager_for_warmup()
    pre_warmup = _current_params(mgr)

    monkeypatch.setattr(model_mixin_module, "load_model_weights", lambda **_: None)

    await mgr.local_optimization_step(learning_rate=1e-3, current_epoch=_IN_WARMUP_EPOCH)
    after = _current_params(mgr)
    # Without a disk snapshot the warm-up branch is a no-op and parameters are
    # untouched (no None-deref crash either).
    assert torch.isfinite(after).all()
    torch.testing.assert_close(after, pre_warmup)


@pytest.mark.asyncio
async def test_warmup_reload_rejects_nan_disk_snapshot(monkeypatch):
    """A NaN-poisoned disk snapshot must NOT replace live parameters."""
    mgr = _make_manager_for_warmup()
    pre_warmup = _current_params(mgr)

    poisoned = torch.full_like(pre_warmup, float("nan"))
    monkeypatch.setattr(model_mixin_module, "load_model_weights", lambda **_: poisoned)

    await mgr.local_optimization_step(learning_rate=1e-3, current_epoch=_IN_WARMUP_EPOCH)
    after = _current_params(mgr)
    assert torch.isfinite(after).all(), (
        "Live parameters must remain finite when disk snapshot has NaN — that "
        "would silently reintroduce the failure mode global_weights_loaded gates against."
    )


@pytest.mark.asyncio
async def test_warmup_reload_rejects_inf_disk_snapshot(monkeypatch):
    """Inf is treated the same as NaN."""
    mgr = _make_manager_for_warmup()
    poisoned = torch.full_like(_current_params(mgr), float("inf"))
    monkeypatch.setattr(model_mixin_module, "load_model_weights", lambda **_: poisoned)

    await mgr.local_optimization_step(learning_rate=1e-3, current_epoch=_IN_WARMUP_EPOCH)
    after = _current_params(mgr)
    assert torch.isfinite(after).all()


@pytest.mark.asyncio
async def test_warmup_reload_rejects_wrong_shape(monkeypatch):
    """A snapshot from a different layer / cross-run file has the wrong size;
    rejecting it prevents vector_to_parameters from silently misaligning."""
    mgr = _make_manager_for_warmup()
    wrong_size = torch.zeros(_current_params(mgr).numel() + 1)
    monkeypatch.setattr(model_mixin_module, "load_model_weights", lambda **_: wrong_size)

    await mgr.local_optimization_step(learning_rate=1e-3, current_epoch=_IN_WARMUP_EPOCH)
    after = _current_params(mgr)
    assert torch.isfinite(after).all()
    # And the wrong-size tensor never got injected (no exception either).


@pytest.mark.asyncio
async def test_warmup_reload_applies_valid_snapshot(monkeypatch):
    """The happy path still works: a valid finite disk snapshot is applied."""
    mgr = _make_manager_for_warmup()
    target = torch.full_like(_current_params(mgr), -3.5)
    monkeypatch.setattr(model_mixin_module, "load_model_weights", lambda **_: target)

    await mgr.local_optimization_step(learning_rate=1e-3, current_epoch=_IN_WARMUP_EPOCH)
    after = _current_params(mgr)
    # vector_to_parameters is in-place; the AdamW step ran *before* the reload,
    # so the final params are exactly the loaded snapshot.
    torch.testing.assert_close(after, target)


@pytest.mark.asyncio
async def test_warmup_reload_preserves_param_device_and_dtype(monkeypatch):
    """load_model_weights returns a CPU tensor in whatever dtype was saved.
    vector_to_parameters re-points param.data at views of that vector, so
    applying it raw migrates the params off their device/dtype while the
    optimizer state (materialized by the step that ran just before the
    reload) stays behind — the next optimizer.step() then dies with
    "Tensors of the same index must be on the same device and the same
    dtype". Simulate with a bf16 snapshot against fp32 params (the dtype
    half of the same grouping check, testable without CUDA)."""
    mgr = _make_manager_for_warmup()
    snapshot = torch.full_like(_current_params(mgr), -3.5).to(torch.bfloat16)
    monkeypatch.setattr(model_mixin_module, "load_model_weights", lambda **_: snapshot)

    await mgr.local_optimization_step(learning_rate=1e-3, current_epoch=_IN_WARMUP_EPOCH)

    for p in mgr.model.parameters():
        assert p.dtype == torch.float32, "warm-up reload must not change parameter dtype"

    # The next optimization step must group params with the pre-existing
    # optimizer state without error.
    for p in mgr.model.parameters():
        p.grad = torch.zeros_like(p)
    mgr.optimizer.step()


@pytest.mark.asyncio
async def test_warmup_branch_skipped_outside_window(monkeypatch):
    """When epochs_since_registration > 2 (or epoch_on_registration <= 1), the
    warm-up branch must not run — so load_model_weights is never called."""
    mgr = _make_manager_for_warmup(epoch_on_registration=10)

    calls = []

    def _track(**kw):
        calls.append(kw)
        return torch.zeros_like(_current_params(mgr))

    monkeypatch.setattr(model_mixin_module, "load_model_weights", _track)

    await mgr.local_optimization_step(learning_rate=1e-3, current_epoch=_OUTSIDE_WARMUP_EPOCH)
    assert calls == [], "warm-up branch must not call load_model_weights outside its gate"


@pytest.mark.asyncio
async def test_warmup_branch_skipped_when_current_epoch_unknown(monkeypatch):
    """If the caller can't supply current_epoch (passed as None), the warm-up
    branch is skipped rather than guessing. This is the safe default."""
    mgr = _make_manager_for_warmup()

    calls = []

    def _track(**kw):
        calls.append(kw)
        return torch.zeros_like(_current_params(mgr))

    monkeypatch.setattr(model_mixin_module, "load_model_weights", _track)

    await mgr.local_optimization_step(learning_rate=1e-3, current_epoch=None)
    assert calls == [], "warm-up branch must not fire without an authoritative current_epoch"
