import asyncio
import copy
from bittensor_wallet import Keypair
from common import settings as common_settings
from common.models.miner_models import ChunkMetadata
from common.models.api_models import CompleteFileUploadResponse, SubmittedWeightsAndOptimizerPresigned
from common.utils.exceptions import WeightPartitionException
from common.utils.partitions import MinerPartition
from common.utils.s3_utils import filter_exceptions
from common.utils.partitions import get_start_and_end_indices
from loguru import logger
from miner import settings as miner_settings
from common.models.run_flags import RUN_FLAGS, RunFlags
from miner.utils.utils import download_metadata, upload_tensor

from subnet.miner_api_client import MinerAPIClient
from subnet.model.utils import log_gpu_memory_usage
from subnet.utils.partition_utils import MergingPartition, download_partition_optimizer
from subnet.utils.s3_torch import download_weights_or_optimizer_state
from subnet.utils.vector_utils import (
    add_artificial_gradients,
    extract_optimizer_state_section,
    get_optimizer_tensor_shapes,
    reconstruct_optimizer_state,
)
import torch


async def get_chunk_metadata_for_all_partitions(
    submitted_weights_and_optimizer: SubmittedWeightsAndOptimizerPresigned,
    partitions: list[MinerPartition],
    run_flags: RunFlags = RUN_FLAGS,
) -> dict[int, ChunkMetadata]:
    """
    Returns a dictionary of chunk numbers to ChunkMetadata objects as follows:
    {
        0: ChunkMetadata(...),
        ...
        7: ChunkMetadata(...),
    }
    """
    weight_metadata: dict | None = None
    try:
        # Download both weight and optimizer state metadata
        weight_metadata = await download_metadata(
            metadata_path=submitted_weights_and_optimizer.weight_metadata_path_presigned
        )
        if not weight_metadata:
            raise Exception(
                f"No weight or optimizer state metadata found | {submitted_weights_and_optimizer.weights_path_presigned}"
            )

        if run_flags.enforce_min_local_optimizer_steps.isOn():
            if weight_metadata.get("local_optimization_steps", 0) < miner_settings.MIN_LOCAL_OPTIMIZER_STEPS:
                return None

        if (
            "weight" not in submitted_weights_and_optimizer.weight_metadata_path_presigned
            and "weight" not in submitted_weights_and_optimizer.weights_path_presigned
        ):
            raise Exception(
                f"Weight metadata path does not contain 'weight' | {submitted_weights_and_optimizer.weights_path_presigned}"
            )
        metadata_infos: dict[int, dict] = {}
        for partition in partitions:
            try:
                chunk_number = partition.chunk_number

                # Create separate metadata info for weights and optimizer state
                weight_metadata_info = ChunkMetadata(
                    **weight_metadata["sections"][str(chunk_number)],
                    chunk_number=chunk_number,
                    weighting_factor=submitted_weights_and_optimizer.weighting_factor,
                    tensor_path=submitted_weights_and_optimizer.weights_path_presigned,
                    metadata_path=submitted_weights_and_optimizer.weight_metadata_path_presigned,
                    chunk_dtype=weight_metadata["tensor"]["dtype"].split(".")[-1],
                    data_type="weights",
                )

                metadata_infos[chunk_number] = weight_metadata_info
            except Exception as e:
                logger.error(f"Error getting chunk metadata for partition {partition.chunk_number}: {e}")
                continue

        return metadata_infos

    except Exception as e:
        logger.exception(
            f"Error getting chunk metadata for all partitions. This is likely due to bad metadata being uploaded but is not a fatal error.: {e}"
        )
        logger.error(f"Bad metadata; WEIGHTS: {weight_metadata}")
        return None


async def download_weight_for_partition(weight_metadata: ChunkMetadata) -> torch.Tensor:
    try:
        assert weight_metadata.data_type == "weights", "Weights metadata is not of type weights"
        assert "weight" in weight_metadata.tensor_path, "Weights tensor path does not contain 'weight'"
        assert "weight" in weight_metadata.metadata_path, "Weights metadata path does not contain 'weight'"

        weights: torch.Tensor = await download_weights_or_optimizer_state(
            metadata_info=weight_metadata,
        )
    except Exception as e:
        logger.error(f"Error download partition with weight metadata: {weight_metadata}")
        logger.exception(f"Error downloading weights: {e}")
        raise

    return weights


def metadata_matches(meta1: dict[int, ChunkMetadata], meta2: dict[int, ChunkMetadata]) -> bool:
    """Check if two metadata dictionaries match."""
    # Check if the chunk numbers are the same
    if set(meta1.keys()) != set(meta2.keys()):
        return False

    # Check that all the start/end indices agree with each other
    for chunk_number in meta1.keys():
        m1, m2 = meta1[chunk_number], meta2[chunk_number]
        if not m1.compatible(m2):
            return False
    return True


async def filter_bad_metadata(
    partitions: list[MinerPartition],
    submitted_weights_and_optimizers: list[SubmittedWeightsAndOptimizerPresigned],
    run_flags: RunFlags = RUN_FLAGS,
) -> dict[str, dict[int, ChunkMetadata]]:
    """Filter out packets with bad metadata and return valid metadata info objects.

    Returns: dictionary mapping weight path to dictionary mapping chunk number to dictionary mapping data type to ChunkMetadata

    Partitions: 0,1,2,3,4,5,6,7
    Weights that were uploaded: 0,1,2

    Each weight (0,1,2) comes with its own metadata on s3 containing the start and end indices for each partition (0,1,2,3,4,5,6,7)

    All these metadata objects SHOULD agree with each other, but because miners upload them, they might not.

    This function now takes the most 'common' metadata object and filters out the rest.

    So in our case, the return would look as follows:

    {
        weight_0: {
            0: ChunkMetadata(...),
            ...
            7: ChunkMetadata(...),
        },
        ...,
        weight_2: {
            0: ChunkMetadata(...),
            ...
            7: ChunkMetadata(...),
        }
    }
    """
    # Collect valid metadata from all packets
    valid_metadata: dict[str, dict[int, ChunkMetadata]] = {}
    results = await asyncio.gather(
        *[
            get_chunk_metadata_for_all_partitions(
                submitted_weights_and_optimizer=s, partitions=partitions, run_flags=run_flags
            )
            for s in submitted_weights_and_optimizers
        ]
    )
    valid_metadata = {
        s.weights_path_presigned: r
        for s, r in zip(submitted_weights_and_optimizers, results, strict=True)
        if r is not None
    }

    if not valid_metadata:
        logger.warning("No valid metadata found")
        return {}

    # Compare all metadata objects to each other and find the most common pattern
    agreement_counts: dict[str, int] = {}
    for path1, meta1 in valid_metadata.items():
        agreement_counts[path1] = sum(
            1 for path2, meta2 in valid_metadata.items() if path1 != path2 and metadata_matches(meta1, meta2)
        )

    # Keep only metadata that matches the most common pattern
    best_path = max(agreement_counts, key=agreement_counts.get)
    best_metadata = valid_metadata[best_path]

    filtered_metadata = {path: meta for path, meta in valid_metadata.items() if metadata_matches(best_metadata, meta)}

    filtered_count = len(valid_metadata) - len(filtered_metadata)
    if filtered_count > 0:
        logger.warning(f"Filtered out {filtered_count} packets due to metadata disagreements")

    return filtered_metadata


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


def create_outer_optimizer(model: torch.nn.Module, device: str | torch.device):
    """Create an outer optimizer that will be used to reconstruct the optimizer state dict from a partition."""
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=common_settings.NESTEROV_LEARNING_RATE,
        momentum=common_settings.NESTEROV_MOMENTUM,
        nesterov=True,
    )

    logger.debug(
        f"Creating outer optimizer with learning rate: {common_settings.NESTEROV_LEARNING_RATE} and momentum: {common_settings.NESTEROV_MOMENTUM}"
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
    filtered_metadata: dict[str, dict[int, dict[str, ChunkMetadata]]],
    old_model: torch.nn.Module,
    local_optimizer_state: torch.optim.Optimizer,
    num_partitions: int,
    weights_length: int,
    device: str,
    run_flags: RunFlags = RUN_FLAGS,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, MinerPartition]]:
    # merge_results: dict[MinerPartition, tuple[torch.Tensor, torch.Tensor]] = {}

    optimizer_shapes = None
    valid_partitions = []
    local_optimizer_total_states = get_total_optimizer_state_size(local_optimizer_state)

    for partition in partition_batch:
        old_model_copy = None
        weight_average = None
        outer_optimizer = None
        flat_optimizer_state = None
        try:
            logger.debug(f"merging partition {partition.new_partition.chunk_number}")

            weight_counter = 0

            for metadata, weights in zip(filtered_metadata.values(), partition.pseudograds):
                try:
                    if weights is None:
                        logger.warning(f"No weights downloaded. Partitions: {partition_batch}")
                        raise Exception(f"No weights downloaded. Partitions: {partition}")

                    weights_metadata: ChunkMetadata = metadata[partition.new_partition.chunk_number]

                    weight_factor = (
                        weights_metadata.weighting_factor if run_flags.weighted_partition_averaging.isOn() else 1.0
                    )
                    weighted_weights = weights.to(torch.float32) * weight_factor
                    if weight_average is None:
                        weight_average = weighted_weights
                    else:
                        weight_average += weighted_weights

                    weight_counter += weight_factor

                except Exception as e:
                    logger.exception(
                        f"Error downloading chunk {partition.new_partition.chunk_number} from {metadata}: {e}"
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
            outer_optimizer, total_states = create_outer_optimizer(model=old_model_copy, device=device)

            if optimizer_shapes is None:
                optimizer_shapes = get_optimizer_tensor_shapes(outer_optimizer)
            optimizer_start_idx, optimizer_end_idx = await get_start_and_end_indices(
                tensor_length=total_states,
                num_sections=num_partitions,
                target_section=partition.new_partition.chunk_number,
            )
            local_optimizer_start_idx, local_optimizer_end_idx = await get_start_and_end_indices(
                tensor_length=local_optimizer_total_states,
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
            partition.local_optimizer_state = extract_optimizer_state_section(
                local_optimizer_state, local_optimizer_start_idx, local_optimizer_end_idx
            )
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
    batch_partitions: list[MergingPartition], filtered_metadata: dict[str, dict[int, ChunkMetadata]]
) -> list[MergingPartition]:
    downloaded_partitions = []

    async def download_weights_for_partition(partition: MergingPartition) -> MergingPartition:
        weights: list[torch.Tensor] = await asyncio.gather(
            *[
                download_weight_for_partition(
                    weight_metadata=metadata[partition.new_partition.chunk_number],
                )
                for metadata in filtered_metadata.values()
            ]
        )
        partition.pseudograds = weights
        return partition

    downloaded_partitions: list[MergingPartition] = await asyncio.gather(
        *[download_weights_for_partition(partition=partition) for partition in batch_partitions],
        return_exceptions=True,
    )
    downloaded_partitions: list[MergingPartition] = filter_exceptions(downloaded_partitions)
    return downloaded_partitions


async def upload_partition_batch(
    miner_api_client: MinerAPIClient,
    merged_partitions: list[MergingPartition],
    hotkey: Keypair,
    run_flags: RunFlags = RUN_FLAGS,
) -> list[MinerPartition]:
    weight_uploads = []
    optimizer_state_uploads = []
    final_partitions = []
    local_optimizer_state_uploads = []
    try:
        for partition in merged_partitions:
            assert partition.new_weights is not None, "New weights are None"
            assert partition.new_optimizer_state is not None, "New optimizer state is None"
            assert partition.local_optimizer_state is not None, "Local optimizer state is None"
            assert len(partition.new_weights) > 0, "New weights are empty"
            assert len(partition.new_optimizer_state) > 0, "New optimizer state is empty"
            assert len(partition.local_optimizer_state) > 0, "Local optimizer state is empty"
            weight_uploads.append(
                upload_tensor(
                    tensor=partition.new_weights.detach().cpu(),
                    miner_api_client=miner_api_client,
                    file_type="weights",
                    hotkey=hotkey,
                    run_flags=run_flags,
                )
            )
            optimizer_state_uploads.append(
                upload_tensor(
                    tensor=partition.new_optimizer_state.detach().cpu(),
                    miner_api_client=miner_api_client,
                    file_type="optimizer_state",
                    hotkey=hotkey,
                    run_flags=run_flags,
                )
            )
            logger.debug(f"Local optimizer state: {partition.local_optimizer_state.shape}")
            local_optimizer_state_uploads.append(
                upload_tensor(
                    tensor=partition.local_optimizer_state.detach().cpu(),
                    miner_api_client=miner_api_client,
                    file_type="local_optimizer_state",
                    hotkey=hotkey,
                    run_flags=run_flags,
                )
            )
        logger.debug(f"Weight uploads before upload: {len(weight_uploads)}")
        logger.debug(f"Optimizer state uploads before upload: {len(optimizer_state_uploads)}")
        logger.debug(f"Local optimizer state uploads before upload: {len(local_optimizer_state_uploads)}")

        # Upload all weights at once
        weight_uploads: list[CompleteFileUploadResponse] = filter_exceptions(
            await asyncio.gather(*weight_uploads, return_exceptions=True)
        )
        optimizer_state_uploads: list[CompleteFileUploadResponse] = filter_exceptions(
            await asyncio.gather(*optimizer_state_uploads, return_exceptions=True)
        )
        local_optimizer_state_uploads: list[CompleteFileUploadResponse] = filter_exceptions(
            await asyncio.gather(*local_optimizer_state_uploads, return_exceptions=True)
        )
        logger.debug(f"Weight uploads: {len(weight_uploads)}")
        logger.debug(f"Optimizer state uploads: {len(optimizer_state_uploads)}")
        logger.debug(f"Local optimizer state uploads: {len(local_optimizer_state_uploads)}")

        for weight_upload, optimizer_state_upload, local_optimizer_state_upload, partition in zip(
            weight_uploads, optimizer_state_uploads, local_optimizer_state_uploads, merged_partitions
        ):
            partition.new_partition.weight_path = weight_upload.object_path
            partition.new_partition.optimizer_state_path = optimizer_state_upload.object_path
            partition.new_partition.local_optimizer_state_path = local_optimizer_state_upload.object_path

            final_partitions.append(partition.new_partition)
    except Exception as e:
        logger.exception(f"Error uploading partition batch: {e}")
        raise
    return [partition for partition in final_partitions if partition.is_valid()]
