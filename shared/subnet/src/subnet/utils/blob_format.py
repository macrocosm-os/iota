"""Tensor-aware helpers for the self-describing blob wire format.

The pure-Python protocol layer (header pack/parse, ChainedBuffer,
download_blob_trailer) lives in `common.utils.blob_format` so it stays
torch-free. This module adds the tensor-side composition + region-download
helpers and lives in `subnet` because `common` is forbidden from importing
torch.
"""

from __future__ import annotations

import gzip

import numpy as np
import torch

from common import settings as common_settings
from common.models.run_flags import RUN_FLAGS, RunFlags
from common.utils.blob_format import (
    ChainedBuffer,
    pack_header,
    pack_trailer,
    ranged_get,
)


def _tensor_memoryview(tensor: torch.Tensor) -> memoryview:
    """Reinterpret a contiguous tensor's storage as a uint8 memoryview (zero-copy)."""
    contig = tensor.detach().to("cpu").contiguous()
    return memoryview(contig.view(torch.uint8).numpy()).cast("B")


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Map a trailer dtype string (e.g. 'bfloat16') back to torch.dtype, or pass through."""
    if isinstance(dtype, torch.dtype):
        return dtype
    resolved = getattr(torch, dtype, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"unsupported blob dtype: {dtype!r}")
    return resolved


def build_weights_blob(
    tensor: torch.Tensor,
    num_sections: int,
    *,
    local_optimization_steps: int,
) -> tuple[ChainedBuffer, dict]:
    """Compose a weights blob: header + tensor + trailer."""
    assert tensor.dtype == torch.bfloat16, f"weights tensor must be bfloat16, got {tensor.dtype}"
    element_size = tensor.element_size()
    total_elements = tensor.numel()
    section_size = total_elements // num_sections

    sections: dict[str, dict] = {}
    for i in range(int(num_sections)):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < num_sections - 1 else total_elements
        start_byte = common_settings.BLOB_HEADER_LEN + start_idx * element_size
        end_byte = common_settings.BLOB_HEADER_LEN + end_idx * element_size
        sections[str(i)] = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_byte": start_byte,
            "end_byte": end_byte,
        }

    tensor_bytes = total_elements * element_size
    trailer = {
        "dtype": _dtype_to_str(tensor.dtype),
        "shape": list(tensor.shape),
        "sections": sections,
        "local_optimization_steps": local_optimization_steps,
    }

    trailer_offset = common_settings.BLOB_HEADER_LEN + tensor_bytes
    header_bytes = pack_header(trailer_offset=trailer_offset)
    trailer_bytes = pack_trailer(trailer)

    tensor_mv = _tensor_memoryview(tensor)
    buf = ChainedBuffer(header_bytes, tensor_mv, trailer_bytes)
    return buf, trailer


def build_partition_blob(
    weights: torch.Tensor,
    optimizer_state: torch.Tensor,
    *,
    chunk_number: int,
    layer: int,
) -> tuple[ChainedBuffer, dict]:
    """Compose a partition blob: header + weights bytes + optimizer-state bytes + trailer."""
    assert weights.dtype == torch.bfloat16, f"weights must be bfloat16, got {weights.dtype}"
    assert optimizer_state.dtype == torch.bfloat16, f"optimizer_state must be bfloat16, got {optimizer_state.dtype}"
    weights_bytes = weights.numel() * weights.element_size()
    opt_bytes = optimizer_state.numel() * optimizer_state.element_size()

    weights_offset = common_settings.BLOB_HEADER_LEN
    opt_offset = weights_offset + weights_bytes

    trailer = {
        "chunk_number": chunk_number,
        "layer": layer,
        "regions": {
            "weights": {
                "offset": weights_offset,
                "length": weights_bytes,
                "dtype": _dtype_to_str(weights.dtype),
                "shape": list(weights.shape),
            },
            "optimizer_state": {
                "offset": opt_offset,
                "length": opt_bytes,
                "dtype": _dtype_to_str(optimizer_state.dtype),
                "shape": list(optimizer_state.shape),
            },
        },
    }

    trailer_offset = opt_offset + opt_bytes
    header_bytes = pack_header(trailer_offset=trailer_offset)
    trailer_bytes = pack_trailer(trailer)

    weights_mv = _tensor_memoryview(weights)
    opt_mv = _tensor_memoryview(optimizer_state)
    buf = ChainedBuffer(header_bytes, weights_mv, opt_mv, trailer_bytes)
    return buf, trailer


async def download_blob_region(
    presigned_url: str,
    *,
    offset: int,
    length: int,
    dtype: torch.dtype | str,
    run_flags: RunFlags = RUN_FLAGS,
) -> torch.Tensor:
    """Range-fetch a region of a blob and reinterpret as a tensor of `dtype`.

    `dtype` may be a `torch.dtype` or the trailer's string name (e.g. 'bfloat16');
    callers should pass it through from the blob trailer so the wire format is
    self-describing in practice, not just on paper.
    """
    resolved_dtype = _resolve_dtype(dtype)
    byte_range = f"bytes={offset}-{offset + length - 1}"
    data = await ranged_get(presigned_url, byte_range)
    if run_flags.compress_s3_files.isOn():
        data = gzip.decompress(data)
    arr = np.frombuffer(data, dtype=np.uint8)
    tensor = torch.from_numpy(arr.copy()).view(resolved_dtype)
    return tensor
