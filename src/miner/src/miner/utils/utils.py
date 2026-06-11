import asyncio
import json
import platform
import psutil
import subprocess
import sys

from typing import Literal

from common.utils.exceptions import LayerStateException, NanInfException
from common.utils.formulas import compute_multipart_layout
from common.utils.shared_states import LayerPhase
from common.utils.blob_format import ChainedBuffer
from subnet.utils.blob_format import build_partition_blob, build_weights_blob
from miner.utils.timer_logger import TimerLoggerMiner
import torch
from bittensor_wallet import Keypair
from common.models.api_models import (
    CompleteFileUploadResponse,
    FileUploadCompletionRequest,
    FileUploadRequest,
    FileUploadResponse,
)
from loguru import logger
from subnet.utils.vector_utils import check_for_nans_and_infs
from subnet.miner_api_client import MinerAPIClient
from common.models.run_flags import RUN_FLAGS, RunFlags


def _sysctl(key: str) -> str | None:
    """Read a single sysctl value, returning None on failure."""
    try:
        return subprocess.check_output(["sysctl", "-n", key], timeout=5).decode("utf-8").strip()
    except Exception:
        return None


def _log_upload_ram(tag: str, file_type: str, payload_bytes: int | None = None) -> None:
    """Log CPU RAM at instrumented points in the blob upload path.

    The wire-format path goes tensor -> memoryview -> ChainedBuffer ->
    per-part slice, so RAM between `entry` and `after_s3_upload` should stay
    within a few MB of the baseline. A jump of ~tensor-size at any tag points
    at an accidental materialization (e.g. memoryview falling back to bytes,
    or a slice crossing every segment boundary).
    """
    vm = psutil.virtual_memory()
    ram_used_gb = vm.used / 1024**3
    ram_total_gb = vm.total / 1024**3
    payload_msg = f" | payload={payload_bytes / 1024**3:.2f}GB" if payload_bytes is not None else ""
    logger.info(f"[_log_upload_ram:{tag}:{file_type}] RAM {ram_used_gb:.2f}/{ram_total_gb:.2f}GB{payload_msg}")


def run_speedtest() -> dict[str, float] | None:
    """Run an upload/download speed test.

    Returns ``{"download_mbps": ..., "upload_mbps": ...}`` or None on failure.
    """
    try:
        import speedtest

        logger.info("Running speedtest...")

        st = speedtest.Speedtest(secure=True)
        st.get_best_server()
        download_bps = st.download()
        upload_bps = st.upload()
        bandwidth = {
            "download_mbps": round(download_bps / 1_000_000, 2),
            "upload_mbps": round(upload_bps / 1_000_000, 2),
        }

        logger.info(f"Speedtest completed with results: {bandwidth}")

        return bandwidth
    except Exception as e:
        logger.debug(f"Failed to run speedtest: {e}")
        return None


def collect_hardware_info() -> dict:
    """Collect static hardware info. No speedtest; never raises.

    Returns dict with keys: machine, cpu_brand, cpu_cores, memory_bytes,
    gpu_count, gpu_name. Missing fields are None/0. Safe to call from
    telemetry hot paths.
    """
    info: dict = {
        "machine": platform.machine(),
        "cpu_brand": None,
        "cpu_cores": None,
        "memory_bytes": None,
        "gpu_count": 0,
        "gpu_name": None,
    }
    try:
        if sys.platform == "darwin":
            info["cpu_brand"] = _sysctl("machdep.cpu.brand_string") or platform.processor() or None
            ncpu = _sysctl("hw.ncpu")
            if ncpu:
                info["cpu_cores"] = int(ncpu)
            memsize = _sysctl("hw.memsize")
            if memsize:
                info["memory_bytes"] = int(memsize)
        else:
            info["cpu_brand"] = platform.processor() or None
            try:
                import os as _os

                info["cpu_cores"] = _os.cpu_count()
            except Exception:
                pass
            try:
                import psutil as _psutil

                info["memory_bytes"] = int(_psutil.virtual_memory().total)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"collect_hardware_info: host probe failed: {e}")

    try:
        if torch.cuda.is_available():
            info["gpu_count"] = int(torch.cuda.device_count())
            if info["gpu_count"] > 0:
                info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        logger.debug(f"collect_hardware_info: cuda probe failed: {e}")

    return info


def collect_system_data() -> str | None:
    """Collect device info and return as MinerSystemData-compatible JSON.

    Thin wrapper around collect_hardware_info() that preserves the legacy
    {gpus, chip_info, bandwidth?} shape used by older callers.
    """
    try:
        hw = collect_hardware_info()
        gpus: list[str] = []
        if sys.platform == "darwin" and hw["cpu_brand"]:
            gpus.append(hw["cpu_brand"])
        if hw["gpu_name"]:
            gpus.append(hw["gpu_name"])
        memory_gb = hw["memory_bytes"] // (1024**3) if hw["memory_bytes"] else None
        data: dict = {
            "gpus": gpus,
            "chip_info": {
                "machine": hw["machine"],
                "cpu": hw["cpu_brand"],
                "cores": hw["cpu_cores"],
                "memory_gb": memory_gb,
            },
        }
        bandwidth = run_speedtest()
        if bandwidth:
            data["bandwidth"] = bandwidth
        return json.dumps(data)
    except Exception as e:
        logger.debug(f"Failed to collect system data: {e}")
        return None


async def _initiate_and_upload_blob(
    *,
    miner_api_client: MinerAPIClient,
    hotkey: Keypair,
    buf: ChainedBuffer,
    file_type: Literal["weights", "optimizer_state", "partition"],
) -> CompleteFileUploadResponse:
    """Shared multipart-upload plumbing for blob payloads. The ChainedBuffer is
    fed straight into the multipart slicer — only the segment-boundary slices
    materialize a copy."""
    total_bytes = len(buf)
    _log_upload_ram("blob_upload_entry", file_type, payload_bytes=total_bytes)
    num_parts, multipart = compute_multipart_layout(total_bytes)

    async with TimerLoggerMiner(
        name="initiate_blob_upload", metadata={"file_type": file_type}, hotkey=hotkey.ss58_address[:8]
    ):
        initiate_response: FileUploadResponse | dict = await miner_api_client.initiate_file_upload_request(
            hotkey=hotkey,
            file_upload_request=FileUploadRequest(
                file_type=file_type,
                num_parts=num_parts,
                multipart=multipart,
            ),
        )
        if not isinstance(initiate_response, FileUploadResponse):
            raise Exception(f"Error initiating file upload for blob: {initiate_response}")
        logger.info(
            f"Initiated upload | file_type={file_type} object_name={initiate_response.object_name} "
            f"upload_id={initiate_response.upload_id} urls={len(initiate_response.urls)} "
            f"num_parts={num_parts} total_bytes={total_bytes}"
        )

    # `upload_parts` slices its `data` arg per multipart chunk; ChainedBuffer's
    # slice protocol returns a memoryview when a slice stays within one segment
    # (header / tensor / trailer), so the tensor body is zero-copy. For
    # single-part uploads aiohttp needs a bytes-like payload directly, so we
    # materialize once at that boundary.
    if multipart:
        payload = buf
    else:
        payload = bytes(buf[0 : len(buf)])

    _log_upload_ram("before_s3_upload", file_type, payload_bytes=total_bytes)
    async with TimerLoggerMiner(
        name="upload_multipart_blob_to_s3", metadata={"file_type": file_type}, hotkey=hotkey.ss58_address[:8]
    ):
        parts: list[dict] | None = await MinerAPIClient.upload_to_s3(
            urls=initiate_response.urls,
            data=payload,
            upload_id=initiate_response.upload_id,
        )
    _log_upload_ram("after_s3_upload", file_type, payload_bytes=total_bytes)

    if multipart:
        async with TimerLoggerMiner(
            name="complete_file_upload_request",
            metadata={"file_type": file_type},
            hotkey=hotkey.ss58_address[:8],
        ):
            await miner_api_client.complete_file_upload_request(
                hotkey=hotkey,
                file_upload_completion_request=FileUploadCompletionRequest(
                    object_name=initiate_response.object_name,
                    upload_id=initiate_response.upload_id,
                    parts=parts,
                ),
            )
    return CompleteFileUploadResponse(object_path=initiate_response.object_name)


async def upload_weights_blob(
    *,
    tensor: torch.Tensor,
    num_sections: int,
    file_type: Literal["weights", "optimizer_state"],
    miner_api_client: MinerAPIClient,
    hotkey: Keypair,
    local_optimization_steps: int,
    run_flags: RunFlags = RUN_FLAGS,
) -> CompleteFileUploadResponse:
    """Build the self-describing weights blob and upload it."""
    _log_upload_ram("upload_weights_blob_entry", file_type)
    check_for_nans_and_infs(
        tensor=tensor,
        name=f"Uploading {file_type} blob",
        exception_type=NanInfException,
    )
    _log_upload_ram("after_nan_inf_check", file_type)
    buf, _trailer = build_weights_blob(
        tensor,
        num_sections=num_sections,
        local_optimization_steps=local_optimization_steps,
    )
    _log_upload_ram("after_build_weights_blob", file_type, payload_bytes=len(buf))
    return await _initiate_and_upload_blob(
        miner_api_client=miner_api_client,
        hotkey=hotkey,
        buf=buf,
        file_type=file_type,
    )


async def upload_partition_blob(
    *,
    weights: torch.Tensor,
    optimizer_state: torch.Tensor,
    chunk_number: int,
    layer: int,
    miner_api_client: MinerAPIClient,
    hotkey: Keypair,
    run_flags: RunFlags = RUN_FLAGS,
) -> CompleteFileUploadResponse:
    """Build the self-describing partition blob (weights + opt_state) and upload it."""
    _log_upload_ram("upload_partition_blob_entry", "partition")
    check_for_nans_and_infs(tensor=weights, name="partition weights", exception_type=NanInfException)
    check_for_nans_and_infs(tensor=optimizer_state, name="partition optimizer_state", exception_type=NanInfException)
    _log_upload_ram("after_nan_inf_check", "partition")
    buf, _trailer = build_partition_blob(
        weights=weights,
        optimizer_state=optimizer_state,
        chunk_number=chunk_number,
        layer=layer,
    )
    _log_upload_ram("after_build_partition_blob", "partition", payload_bytes=len(buf))
    return await _initiate_and_upload_blob(
        miner_api_client=miner_api_client,
        hotkey=hotkey,
        buf=buf,
        file_type="partition",
    )


async def wait_for_state(state: LayerPhase, miner_api_client: MinerAPIClient, raise_bad_sync: bool = True):
    while True:
        await asyncio.sleep(1)
        logger.info(f"Waiting for state {state}")
        response = await miner_api_client.heartbeat()
        if response.phase == state.value:
            logger.info(f"Orchestrator is finally in state {state}")
            miner_api_client.layer_state = LayerPhase(response.phase)
            break
        elif LayerPhase(response.phase).next() == state:
            continue
        else:
            miner_api_client.layer_state = LayerPhase.TRAINING
            if raise_bad_sync:
                raise LayerStateException(
                    f"Miner is out of sync with the orchestrator. Miner is waiting for orchestrator to be in state {state}, but orchestrator is in state {response.phase}, setting state to training"
                )
