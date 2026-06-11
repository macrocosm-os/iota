import asyncio
import copy
from bittensor_wallet import Keypair
from common import settings as common_settings
from common.models.api_models import SubmittedWeightsAndOptimizerPresigned
from common.utils.exceptions import WeightPartitionException
from common.utils.partitions import MinerPartition
from common.utils.s3_utils import filter_exceptions
from common.utils.partitions import get_start_and_end_indices
from common.utils.blob_format import download_blob_trailer
from subnet.utils.blob_format import download_blob_region
from loguru import logger
from miner import settings as miner_settings
from common.models.run_flags import RUN_FLAGS, RunFlags
from miner.utils.utils import upload_partition_blob

from subnet.miner_api_client import MinerAPIClient
from subnet.model.utils import log_gpu_memory_usage
from subnet.utils.partition_utils import MergingPartition, download_partition_optimizer
from subnet.utils.vector_utils import (
    add_artificial_gradients,
    extract_optimizer_state_section,
    get_optimizer_tensor_shapes,
    reconstruct_optimizer_state,
)
import torch


async def get_weight_partition_info(
    layer: int,
    miner_api_client: MinerAPIClient,
) -> tuple[list[SubmittedWeightsAndOptimizerPresigned], list[MinerPartition]]:
    """
    Get the weight partition info from the orchestrator. This calls two different API endpoints:
    - /miner/get_weight_path_per_layer (weight path for the model layer)
    - /miner/get_partition_indices_by_hotkey (partition indices for the miner)

    Returns:
        tuple[list[SubmittedWeightsPresigned], list[int]]: The weight partition info and the partition ids
    """
    weight_path_per_layer: list[
        SubmittedWeightsAndOptimizerPresigned
    ] | dict = await miner_api_client.get_weight_path_per_layer()

    if not weight_path_per_layer:
        raise WeightPartitionException("Unknown error getting weight path per layer")

    logger.debug(f"layer {layer} getting partitions")
    partitions: dict = await miner_api_client.get_partitions()

    if not partitions:
        logger.warning(f"No partitions found for layer {layer}")
        return weight_path_per_layer, []

    return weight_path_per_layer, [MinerPartition(**p) for p in partitions]


def get_total_optimizer_state_size(optimizer):
    state_dict = optimizer.state_dict()
    total_size = 0
    for group in state_dict["state"].values():
        for k, v in group.items():
            if k == "step":
                continue
            if isinstance(v, torch.Tensor):
                total_size += v.numel()
    return total_size


def get_outer_optimizer_warmup_momentum(epoch: int) -> float:
    """Compute outer optimizer momentum with linear warmup.

    Linearly interpolates from NESTEROV_MOMENTUM_WARMUP_START to NESTEROV_MOMENTUM
    over the first NESTEROV_MOMENTUM_WARMUP_EPOCHS epochs.
    """
    warmup_epochs = common_settings.NESTEROV_MOMENTUM_WARMUP_EPOCHS
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return common_settings.NESTEROV_MOMENTUM

    start = common_settings.NESTEROV_MOMENTUM_WARMUP_START
    end = common_settings.NESTEROV_MOMENTUM
    progress = epoch / warmup_epochs
    return start + (end - start) * progress


def create_outer_optimizer(model: torch.nn.Module, device: str | torch.device, epoch: int = 0):
    """Create an outer optimizer that will be used to reconstruct the optimizer state dict from a partition."""
    momentum = get_outer_optimizer_warmup_momentum(epoch)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=common_settings.NESTEROV_LEARNING_RATE,
        momentum=momentum,
        nesterov=True,
    )

    logger.debug(
        f"Creating outer optimizer with learning rate: {common_settings.NESTEROV_LEARNING_RATE} "
        f"and momentum: {momentum} (epoch {epoch}, target {common_settings.NESTEROV_MOMENTUM})"
    )
    add_artificial_gradients(model=model, device=device)
    optimizer.step()

    total_states = get_total_optimizer_state_size(optimizer)
    optimizer.zero_grad()
    return optimizer, total_states


def load_grads_from_flat_vector(model: torch.nn.Module, grad_vector: torch.Tensor):
    """
    Load a flat gradient vector into model's .grad attributes.
    grad_vector: torch.Tensor (same ordering as parameters_to_vector(model.parameters()))
    """
    # Ensure it's on the right device and dtype
    grad_vector = grad_vector.to(next(model.parameters()).device)

    # Create placeholder grads in case the model .grad is not initialized
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p, dtype=grad_vector.dtype, device=grad_vector.device)

    # Assign gradients using vector_to_parameters
    torch.nn.utils.vector_to_parameters(grad_vector, (p.grad for p in model.parameters()))


def reconstruct_outer_optimizer_from_partial_state(
    old_optimizer_state: torch.Tensor,
    start_idx: int,
    end_idx: int,
    optim_state_shapes: list[torch.Tensor],
    total_states: int,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.Optimizer:
    """Reconstruct optimizer state dict from a partition of a flattened tensor."""

    # Log this operation
    logger.debug("Reconstructing outer optimizer state from partition")

    # Construct a flat vector of infs everywhere except for the partition
    # The dtype is the one in the model config
    logger.debug(f"total_states: {total_states}, start_idx: {start_idx}, end_idx: {end_idx}")
    full_flat_vector = torch.full((total_states,), float("inf"), dtype=torch.bfloat16)
    full_flat_vector[start_idx:end_idx] = old_optimizer_state

    # Reconstruct the optimizer state dict from the flat vector
    optimizer_state_dict = reconstruct_optimizer_state(
        flat_tensor=full_flat_vector,
        tensor_shapes=optim_state_shapes,
        state_dict=optimizer.state_dict(),
    )
    optimizer.load_state_dict(optimizer_state_dict)
    return optimizer


def reconstruct_full_grads_from_partition(
    grads_partition: torch.Tensor,
    start_idx: int,
    end_idx: int,
    pseudograds_length: int,
    model: torch.nn.Module,
    device: str,
):
    """Reconstruct the full grads (.grads) of a model from a partition of grads tensor."""

    # Log this operation
    logger.debug("Reconstructing full grads from partition")

    full_flat_vector = torch.full((pseudograds_length,), float("inf"), dtype=torch.bfloat16, device=device)
    full_flat_vector[start_idx:end_idx] = grads_partition
    load_grads_from_flat_vector(model=model, grad_vector=full_flat_vector)


async def merge_partition_batch(
    partition_batch: list[MergingPartition],
    submitted_weights_list: list[SubmittedWeightsAndOptimizerPresigned],
    old_model: torch.nn.Module,
    num_partitions: int,
    weights_length: int,
    device: str,
    run_flags: RunFlags = RUN_FLAGS,
    epoch: int = 0,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, MinerPartition]]:
    optimizer_shapes = None
    valid_partitions = []

    for partition in partition_batch:
        old_model_copy = None
        weight_average = None
        outer_optimizer = None
        flat_optimizer_state = None
        try:
            logger.debug(f"merging partition {partition.new_partition.chunk_number}")

            weight_counter = 0

            for source, weights in zip(submitted_weights_list, partition.pseudograds):
                try:
                    if weights is None:
                        logger.warning(f"No weights downloaded. Partitions: {partition_batch}")
                        raise Exception(f"No weights downloaded. Partitions: {partition}")

                    weight_factor = source.weighting_factor if run_flags.weighted_partition_averaging.isOn() else 1.0
                    weighted_weights = weights.to(torch.float32) * weight_factor
                    if weight_average is None:
                        weight_average = weighted_weights
                    else:
                        weight_average += weighted_weights

                    weight_counter += weight_factor

                except Exception as e:
                    logger.exception(
                        f"Error processing chunk {partition.new_partition.chunk_number} "
                        f"from {source.weights_path_presigned}: {e}"
                    )

            if weight_average is None:
                raise Exception(f"No weights downloaded. Partitions: {partition}")

            # Average the weights
            weight_average /= weight_counter
            weight_average = weight_average.to(torch.bfloat16)
            log_gpu_memory_usage(
                note=f"after averaging partition weights on chunk {partition.new_partition.chunk_number}"
            )

            old_model_copy = copy.deepcopy(old_model).to(torch.bfloat16)
            outer_optimizer, total_states = create_outer_optimizer(model=old_model_copy, device=device, epoch=epoch)

            if optimizer_shapes is None:
                optimizer_shapes = get_optimizer_tensor_shapes(outer_optimizer)

            optimizer_start_idx, optimizer_end_idx = await get_start_and_end_indices(
                tensor_length=total_states,
                num_sections=num_partitions,
                target_section=partition.new_partition.chunk_number,
            )
            weight_start_idx, weight_end_idx = await get_start_and_end_indices(
                tensor_length=weights_length,
                num_sections=num_partitions,
                target_section=partition.new_partition.chunk_number,
            )

            logger.debug(f"weight_start_idx: {weight_start_idx}, weight_end_idx: {weight_end_idx}")
            logger.debug(f"optimizer_start_idx: {optimizer_start_idx}, optimizer_end_idx: {optimizer_end_idx}")
            log_gpu_memory_usage(
                note=f"after getting start and end indices for partition {partition.new_partition.chunk_number}"
            )

            # This should only happen after the first epoch.
            if partition.old_optimizer_state is not None:
                outer_optimizer = reconstruct_outer_optimizer_from_partial_state(
                    old_optimizer_state=partition.old_optimizer_state,
                    start_idx=optimizer_start_idx,
                    end_idx=optimizer_end_idx,
                    optim_state_shapes=optimizer_shapes,
                    total_states=total_states,
                    optimizer=outer_optimizer,
                )
                log_gpu_memory_usage(
                    note=f"after reconstructing outer optimizer state for partition {partition.new_partition.chunk_number}"
                )
            else:
                logger.warning(f"No old optimizer state found for partition {partition.new_partition.chunk_number}")

            # Reconstruct the full pseudograds vector from the partition and load it into the model
            reconstruct_full_grads_from_partition(
                grads_partition=weight_average,
                start_idx=weight_start_idx,
                end_idx=weight_end_idx,
                pseudograds_length=total_states,
                model=old_model_copy,
                device=device,
            )
            log_gpu_memory_usage(
                note=f"after reconstructing full grads for partition {partition.new_partition.chunk_number}"
            )

            # If the model has no gradients, this doesn't do anything.
            outer_optimizer.step()

            partition.new_weights = torch.nn.utils.parameters_to_vector(old_model_copy.parameters())[
                weight_start_idx:weight_end_idx
            ]
            partition.new_optimizer_state = extract_optimizer_state_section(
                outer_optimizer, optimizer_start_idx, optimizer_end_idx
            )
            log_gpu_memory_usage(
                note=f"after setting up new optimizer state for partition {partition.new_partition.chunk_number}"
            )

            # Check if the weights and optimizer state are valid and add to the list of valid partitions
            if partition.new_weights is None or partition.new_optimizer_state is None:
                logger.warning(
                    f"No weights or optimizer state found for partition {partition.new_partition.chunk_number}"
                )
                logger.warning(f"Partition: {partition}")
                continue
            valid_partitions.append(partition)
            log_gpu_memory_usage(note=f"after fininhing the merge of partition {partition.new_partition.chunk_number}")

        except Exception as e:
            logger.exception(f"Failed to get partition {partition.new_partition.chunk_number}: {e}")
        finally:
            if old_model_copy is not None:
                del old_model_copy
            if weight_average is not None:
                del weight_average
            if outer_optimizer is not None:
                del outer_optimizer
            if flat_optimizer_state is not None:
                del flat_optimizer_state

    logger.debug(f"Number of valid partitions: {len(valid_partitions)}")
    logger.debug(f"Number of invalid partitions: {len(partition_batch) - len(valid_partitions)}")
    return valid_partitions


def get_partition_batch(batch_index: int, partitions: list[MinerPartition]) -> list[MergingPartition]:
    """
    Returns a batch of partitions as a list, i.e.
    [
        MergingPartition(new_partition=MinerPartition(chunk_number=0), old_partition=None, weights=None),
        MergingPartition(new_partition=MinerPartition(chunk_number=1), old_partition=None, weights=None),
        MergingPartition(new_partition=MinerPartition(chunk_number=2), old_partition=None, weights=None),
        ...
    ]
    """
    start_index = batch_index * len(partitions) // min(miner_settings.N_PARTITION_BATCHES, len(partitions))
    end_index = (batch_index + 1) * len(partitions) // min(miner_settings.N_PARTITION_BATCHES, len(partitions))
    batch_partitions = partitions[start_index:end_index]
    batch_partitions = [
        MergingPartition(new_partition=partition, old_partition=None, pseudograds=None)
        for partition in batch_partitions
    ]
    return batch_partitions


async def download_previous_optimizer_state_for_partition_batch(
    batch_partitions: list[MergingPartition],
) -> list[MergingPartition]:
    n_with_old = sum(1 for p in batch_partitions if p.old_partition is not None)
    logger.info(
        f"download_optimizer_state: {n_with_old}/{len(batch_partitions)} partitions have a previous optimizer state to download"
    )

    async def download_previous_optimizer_state_for_partition(partition: MergingPartition) -> MergingPartition:
        if partition.old_partition is None:
            logger.warning(f"No old partition found for partition {partition.new_partition.chunk_number}")
            partition.old_partition = MinerPartition(
                chunk_number=partition.new_partition.chunk_number,
                layer=partition.new_partition.layer,
                miner_hotkey=partition.new_partition.miner_hotkey,
            )
            return partition
        partition.old_optimizer_state = await download_partition_optimizer(partition=partition.old_partition)
        return partition

    downloaded_partitions = await asyncio.gather(
        *[download_previous_optimizer_state_for_partition(partition=partition) for partition in batch_partitions]
    )
    return downloaded_partitions


async def download_pseudograds_for_partition_batch(
    batch_partitions: list[MergingPartition],
    submitted_weights_list: list[SubmittedWeightsAndOptimizerPresigned],
    run_flags: RunFlags = RUN_FLAGS,
) -> tuple[list[MergingPartition], list[SubmittedWeightsAndOptimizerPresigned]]:
    """Lazily fetch each source miner's blob trailer (cached per-URL) then range-
    download the requested chunk's weights region. Source miners whose trailer
    cannot be fetched (or whose local_optimization_steps is below threshold)
    are filtered out — the returned `submitted_weights_list` is the surviving
    subset and is positionally aligned with each `partition.pseudograds`."""

    # Step 1: fetch all trailers in parallel. async_lru on download_blob_trailer
    # dedupes repeat hits within and across batches.
    trailer_results = await asyncio.gather(
        *[download_blob_trailer(s.weights_path_presigned) for s in submitted_weights_list],
        return_exceptions=True,
    )

    valid_sources: list[SubmittedWeightsAndOptimizerPresigned] = []
    valid_trailers: list[dict] = []
    for source, trailer in zip(submitted_weights_list, trailer_results):
        if isinstance(trailer, Exception):
            logger.warning(f"Failed to fetch trailer for {source.weights_path_presigned}: {trailer}. Dropping source.")
            continue
        if run_flags.enforce_min_local_optimizer_steps.isOn():
            if trailer.get("local_optimization_steps", 0) < miner_settings.MIN_LOCAL_OPTIMIZER_STEPS:
                logger.info(
                    f"Dropping source {source.weights_path_presigned}: local_optimization_steps below threshold"
                )
                continue
        valid_sources.append(source)
        valid_trailers.append(trailer)

    if not valid_sources:
        logger.warning("No valid source miners after trailer fetch / filter")
        return [], []

    n_downloads = len(batch_partitions) * len(valid_sources)
    logger.info(
        f"download_pseudograds: {len(batch_partitions)} partitions × {len(valid_sources)} miners "
        f"= {n_downloads} S3 range downloads"
    )

    async def fetch_weights_region(
        source: SubmittedWeightsAndOptimizerPresigned,
        trailer: dict,
        chunk_number: int,
    ) -> torch.Tensor:
        section = trailer["sections"][str(chunk_number)]
        offset = section["start_byte"]
        length = section["end_byte"] - offset
        return await download_blob_region(
            source.weights_path_presigned,
            offset=offset,
            length=length,
            dtype=trailer["dtype"],
        )

    async def download_weights_for_partition(partition: MergingPartition) -> MergingPartition:
        weights: list[torch.Tensor] = await asyncio.gather(
            *[
                fetch_weights_region(s, t, partition.new_partition.chunk_number)
                for s, t in zip(valid_sources, valid_trailers)
            ]
        )
        partition.pseudograds = weights
        return partition

    downloaded_partitions: list[MergingPartition] = await asyncio.gather(
        *[download_weights_for_partition(partition=partition) for partition in batch_partitions],
        return_exceptions=True,
    )
    downloaded_partitions: list[MergingPartition] = filter_exceptions(downloaded_partitions)
    logger.info(
        f"download_pseudograds: {len(downloaded_partitions)}/{len(batch_partitions)} partitions downloaded successfully"
    )
    return downloaded_partitions, valid_sources


async def upload_partition_batch(
    miner_api_client: MinerAPIClient,
    merged_partitions: list[MergingPartition],
    hotkey: Keypair,
    run_flags: RunFlags = RUN_FLAGS,
) -> list[MinerPartition]:
    final_partitions: list[MinerPartition] = []
    try:
        upload_coros = []
        for partition in merged_partitions:
            assert partition.new_weights is not None, "New weights are None"
            assert partition.new_optimizer_state is not None, "New optimizer state is None"
            assert len(partition.new_weights) > 0, "New weights are empty"
            assert len(partition.new_optimizer_state) > 0, "New optimizer state is empty"
            upload_coros.append(
                upload_partition_blob(
                    weights=partition.new_weights.detach().cpu(),
                    optimizer_state=partition.new_optimizer_state.detach().cpu(),
                    chunk_number=partition.new_partition.chunk_number,
                    layer=partition.new_partition.layer,
                    miner_api_client=miner_api_client,
                    hotkey=hotkey,
                    run_flags=run_flags,
                )
            )

        logger.info(
            f"upload_partition_batch: uploading {len(merged_partitions)} partitions "
            f"= {len(upload_coros)} blob S3 uploads"
        )

        results = await asyncio.gather(*upload_coros, return_exceptions=True)
        for upload_result, partition in zip(results, merged_partitions):
            if isinstance(upload_result, Exception):
                logger.warning(
                    f"Failed to upload partition blob for chunk {partition.new_partition.chunk_number}: "
                    f"{upload_result}"
                )
                continue
            partition.new_partition.blob_path = upload_result.object_path
            final_partitions.append(partition.new_partition)
    except Exception as e:
        logger.exception(f"Error uploading partition batch: {e}")
        raise
    return [partition for partition in final_partitions if partition.is_valid()]
