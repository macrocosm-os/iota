"""The flat chunk-slice merge in merge_partition_batch must match the old
full-model path: rebuild a model + torch.optim.SGD(momentum, nesterov) per
partition, load the momentum buffer / grads into it, step, slice.

The two paths run the same element-wise formula but through different kernels
(manual ops vs torch.optim.SGD's foreach path), and CPU architectures differ in
FMA fusion, so a handful of elements land a few bf16 ulp apart — worst where the
nesterov update nearly cancels. Bit-identity is not portable; assert instead
that almost all elements are bitwise equal and the stragglers are within
rounding distance. A real math bug (wrong lr/momentum/slice) shifts most
elements by far more than 0.05."""

import asyncio

import pytest
import torch
from common import settings as common_settings
from common.models.api_models import SubmittedWeightsAndOptimizerPresigned
from common.utils.partitions import MinerPartition, get_start_and_end_indices
from subnet.utils.partition_utils import MergingPartition

from miner.utils.partition_merging import get_outer_optimizer_warmup_momentum, merge_partition_batch

N = 1000  # deliberately not divisible by NUM_PARTITIONS
NUM_PARTITIONS = 7
EPOCH = 3


def reference_sgd_chunk(
    old_weights: torch.Tensor,
    grad_chunk: torch.Tensor,
    old_buf_chunk: torch.Tensor | None,
    start: int,
    end: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pre-refactor path: a real torch.optim.SGD over the full flat vector."""
    param = torch.nn.Parameter(old_weights.clone())
    optimizer = torch.optim.SGD(
        [param],
        lr=common_settings.NESTEROV_LEARNING_RATE,
        momentum=get_outer_optimizer_warmup_momentum(epoch),
        nesterov=True,
    )
    # create_outer_optimizer seeded the momentum buffer via a zero-grad step
    param.grad = torch.zeros_like(param)
    optimizer.step()

    if old_buf_chunk is not None:
        buf = optimizer.state[param]["momentum_buffer"]
        buf[start:end] = old_buf_chunk

    grad_full = torch.full((N,), float("inf"), dtype=torch.bfloat16)
    grad_full[start:end] = grad_chunk
    param.grad = grad_full
    optimizer.step()

    return (
        param.detach()[start:end].clone(),
        optimizer.state[param]["momentum_buffer"][start:end].clone(),
    )


@pytest.mark.parametrize("with_old_state", [True, False])
def test_flat_merge_matches_full_sgd(with_old_state: bool):
    torch.manual_seed(0)
    old_weights = torch.randn(N).to(torch.bfloat16)

    # weighted_partition_averaging is on by default — use non-uniform factors
    factors = [1, 2, 1]
    sources = [
        SubmittedWeightsAndOptimizerPresigned(layer=0, weights_path_presigned=f"s3://fake/{i}", weighting_factor=f)
        for i, f in enumerate(factors)
    ]

    partitions = []
    expected = []
    for chunk in range(NUM_PARTITIONS):
        start, end = asyncio.run(
            get_start_and_end_indices(tensor_length=N, num_sections=NUM_PARTITIONS, target_section=chunk)
        )
        pseudograds = [torch.randn(end - start).to(torch.bfloat16) for _ in sources]
        old_buf = torch.randn(end - start).to(torch.bfloat16) if with_old_state else None
        partitions.append(
            MergingPartition(
                new_partition=MinerPartition(layer=0, chunk_number=chunk, miner_hotkey="hk"),
                pseudograds=pseudograds,
                old_optimizer_state=old_buf,
            )
        )
        # Same averaging the merge does: weighted fp32 accumulate, then back to bf16
        avg = sum(g.to(torch.float32) * f for g, f in zip(pseudograds, factors)) / sum(factors)
        expected.append(reference_sgd_chunk(old_weights, avg.to(torch.bfloat16), old_buf, start, end, EPOCH))

    merged = asyncio.run(
        merge_partition_batch(
            partition_batch=partitions,
            submitted_weights_list=sources,
            old_weights=old_weights,
            num_partitions=NUM_PARTITIONS,
            epoch=EPOCH,
        )
    )

    assert len(merged) == NUM_PARTITIONS
    for partition, (expected_weights, expected_buf) in zip(merged, expected):
        assert_bf16_equivalent(partition.new_weights, expected_weights)
        assert_bf16_equivalent(partition.new_optimizer_state, expected_buf)


def assert_bf16_equivalent(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Bitwise-equal except for isolated few-ulp kernel-rounding differences."""
    assert actual.shape == expected.shape
    mismatch_fraction = (actual != expected).float().mean().item()
    max_abs_diff = (actual.float() - expected.float()).abs().max().item()
    assert mismatch_fraction < 0.05, f"{mismatch_fraction:.1%} of elements differ"
    assert max_abs_diff < 0.05, f"max abs diff {max_abs_diff}"
