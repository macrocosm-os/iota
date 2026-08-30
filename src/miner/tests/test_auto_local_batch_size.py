"""Self-check for the OOM-driven local batch size shrink + retry logic.

Runs without a GPU: it drives the real ``TrainingPhase._reduce_local_batch_size`` and the same
retry pattern used in ``backward``/``forward``, using a fake unit of work that raises
``torch.cuda.OutOfMemoryError`` while the batch is above a pretend capacity.
"""

import types

import torch

from miner.training import training


def _make_phase(start: int):
    """A bare TrainingPhase with only the attributes the shrink logic touches."""
    phase = training.TrainingPhase.__new__(training.TrainingPhase)
    phase._local_batch_size = start
    phase._hotkey = "deadbeef"
    phase._state_manager = types.SimpleNamespace(layer=1)
    return phase


def test_shrink_converges_to_capacity():
    # _clean_gpu_memory hits CUDA; no-op it on CPU-only runners.
    training._clean_gpu_memory = lambda: None

    for capacity in (4, 2, 1):
        phase = _make_phase(4)
        attempts = 0
        while True:
            try:
                attempts += 1
                if phase._local_batch_size > capacity:
                    raise torch.cuda.OutOfMemoryError("fake OOM")
                break  # fits
            except torch.cuda.OutOfMemoryError:
                phase._reduce_local_batch_size()
        assert phase._local_batch_size == capacity, (capacity, phase._local_batch_size)
    print("shrink converges to capacity for 4/2/1 ✓")


def test_floor_reraises_when_even_one_ooms():
    training._clean_gpu_memory = lambda: None
    phase = _make_phase(1)
    raised = False
    try:
        try:
            raise torch.cuda.OutOfMemoryError("fake OOM at batch 1")
        except torch.cuda.OutOfMemoryError:
            phase._reduce_local_batch_size()  # at floor -> bare `raise` re-raises the OOM
    except torch.cuda.OutOfMemoryError:
        raised = True
    assert raised, "expected OOM to propagate when already at local_batch_size=1"
    assert phase._local_batch_size == 1
    print("floor re-raises when batch 1 still OOMs ✓")


if __name__ == "__main__":
    test_shrink_converges_to_capacity()
    test_floor_reraises_when_even_one_ooms()
    print("all ok")
