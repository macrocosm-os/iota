"""The periodic run-config refresh only works if updates land in-place: flag
holders keep their reference to miner.run_flags, and hyperparameter consumers
read common_settings attributes at call time."""

from types import SimpleNamespace

from common import settings as common_settings
from common.models.run_flags import RunFlags

from miner.new_miner import Miner


def _stub() -> SimpleNamespace:
    return SimpleNamespace(hotkey="hk-test-1234", run_flags=RunFlags())


def test_update_run_flags_mutates_shared_reference():
    stub = _stub()
    shared_ref = stub.run_flags  # what e.g. TrainingPhase holds

    new_flags = RunFlags()
    new_flags.sync_patches.enabled = not shared_ref.sync_patches.enabled
    Miner._update_run_flags(stub, new_flags)

    assert shared_ref.sync_patches.enabled == new_flags.sync_patches.enabled


def test_apply_run_hyperparams_updates_cache_size_live():
    stub = _stub()
    old = common_settings.MAX_ACTIVATION_CACHE_SIZE
    try:
        Miner._apply_run_hyperparams(stub, {"activation": {"max_activation_cache_size": old + 16}})
        assert common_settings.MAX_ACTIVATION_CACHE_SIZE == old + 16

        # Absent/None values must leave settings untouched
        Miner._apply_run_hyperparams(stub, {})
        assert common_settings.MAX_ACTIVATION_CACHE_SIZE == old + 16
    finally:
        common_settings.set_max_activation_cache_size(old)
