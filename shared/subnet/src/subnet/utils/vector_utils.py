from typing import Union
from common.utils.exceptions import NanInfWarning
from loguru import logger

import torch


def add_artificial_gradients(model: torch.nn.Module, device: Union[str, torch.device]):
    """Add artificial gradients to the model parameters."""
    model.to(device)
    for param in model.parameters():
        if param.requires_grad:
            # Use the same dtype as the parameter (float32 for MockModel/MPS, bfloat16 for real models)
            param.grad = torch.zeros_like(param.data).to(device)


def get_optimizer_tensor_shapes(optimizer: torch.optim.Optimizer):
    state_dict = optimizer.state_dict()
    tensor_shapes = []

    for group in state_dict["state"].values():
        for k, v in group.items():
            if k == "step":
                continue
            if isinstance(v, torch.Tensor):
                tensor_shapes.append(v.shape)
    return tensor_shapes


def extract_optimizer_state_section(optimizer: torch.optim.Optimizer, start_idx: int, end_idx: int) -> torch.Tensor:
    """Return a flattened slice of the optimizer state between start_idx (inclusive) and end_idx (exclusive)."""

    if start_idx < 0 or end_idx < start_idx:
        raise ValueError("Invalid optimizer state indices")

    state_dict = optimizer.state_dict()
    state = state_dict.get("state", {})

    total_elements = sum(
        v.numel() for group in state.values() for k, v in group.items() if k != "step" and isinstance(v, torch.Tensor)
    )

    if end_idx > total_elements:
        raise ValueError("End index exceeds optimizer state size")

    if start_idx == end_idx:
        return torch.empty((0,), dtype=torch.bfloat16, device="cpu")

    current_offset = 0
    section_tensors: list[torch.Tensor] = []

    for group in state.values():
        for key, value in group.items():
            if key == "step" or not isinstance(value, torch.Tensor):
                continue

            flattened = value.flatten()
            next_offset = current_offset + flattened.numel()

            if next_offset <= start_idx:
                current_offset = next_offset
                continue

            if current_offset >= end_idx:
                break

            slice_start = max(start_idx - current_offset, 0)
            slice_end = min(end_idx - current_offset, flattened.numel())

            if slice_start < slice_end:
                section_tensors.append(flattened[slice_start:slice_end].cpu())

            current_offset = next_offset

            if current_offset >= end_idx:
                break

        if current_offset >= end_idx:
            break

    if not section_tensors:
        return torch.empty((0,), dtype=torch.bfloat16, device="cpu")

    flat_section = torch.cat(section_tensors)
    return flat_section.to(dtype=torch.bfloat16, device="cpu")


def flatten_optimizer_state(
    optimizer: torch.optim.Optimizer, device: Union[str, torch.device], dtype=torch.bfloat16
) -> tuple[torch.Tensor, list[tuple[int, ...]], dict]:
    """Flatten all tensors in optimizer state dict into a single tensor."""
    state_dict = optimizer.state_dict()
    tensors = []
    tensor_shapes = []

    for group in state_dict["state"].values():
        for k, v in group.items():
            if k == "step":
                continue
            if isinstance(v, torch.Tensor):
                tensors.append(v.flatten().to(dtype).to(device))
                tensor_shapes.append(v.shape)

    flat_tensor = torch.cat(tensors)
    return flat_tensor, tensor_shapes, state_dict


def reconstruct_optimizer_state(
    flat_tensor: torch.Tensor, tensor_shapes: list[tuple[int, ...]], state_dict: dict
) -> dict:
    """Reconstruct optimizer state dict from flattened tensor."""

    start_idx = 0
    tensor_idx = 0
    for group in state_dict["state"].values():
        for k, v in group.items():
            if k == "step":
                continue
            if isinstance(v, torch.Tensor):
                numel = v.numel()
                tensor_data = flat_tensor[start_idx : start_idx + numel]
                group[k] = tensor_data.reshape(tensor_shapes[tensor_idx])
                start_idx += numel
                tensor_idx += 1

    return state_dict


def check_for_nans_and_infs(tensor, name: str | None = None, exception_type: type = NanInfWarning):
    # Check to see if the weights or optimizer state have any nans
    tensor = tensor.to("cpu")
    try:
        num_nans = torch.isnan(tensor).sum()
        if num_nans > 0:
            total = tensor.numel()
            percentage = (num_nans / total) * 100
            logger.error(f"❌ Miner has NaNs in {name} | {num_nans} / {total} = {percentage:.2f}%")
            raise exception_type(f"{name} has NaNs")
        num_infs = torch.isinf(tensor).sum()
        if num_infs > 0:
            total = tensor.numel()
            percentage = (num_infs / total) * 100
            logger.error(f"❌ Miner has Infs in {name} | {num_infs} / {total} = {percentage:.2f}%")
            raise exception_type(f"{name} has Infs")
    except Exception:
        raise
