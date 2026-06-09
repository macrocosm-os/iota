import asyncio
import json
import gzip
import platform
import psutil
import subprocess
import sys
from urllib.parse import urlparse

from typing import Literal, Optional

from common.utils.cache import async_lru
from common.utils.exceptions import LayerStateException, NanInfException
from common.utils.formulas import calculate_num_parts
from common.utils.shared_states import LayerPhase
from miner.utils.timer_logger import TimerLoggerMiner
import torch
from bittensor_wallet import Keypair
from common import settings as common_settings
from common.models.api_models import (
    CompleteFileUploadResponse,
    FileUploadCompletionRequest,
    FileUploadRequest,
    FileUploadResponse,
)
from common.utils.s3_utils import download_file
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
    """Log CPU RAM at instrumented points in upload_tensor / upload_file.

    Used to validate that the memoryview refactor actually reduces peak memory
    relative to the old .tobytes() path. With the old path, RAM at
    `after_memoryview` would jump by ~tensor-size relative to `before_memoryview`.
    With the new memoryview path, the two should be ~identical.
    """
    vm = psutil.virtual_memory()
    ram_used_gb = vm.used / 1024**3
    ram_total_gb = vm.total / 1024**3
    payload_msg = f" | payload={payload_bytes / 1024**3:.2f}GB" if payload_bytes is not None else ""
    logger.info(
        f"[_log_upload_ram:upload_tensor:{tag}:{file_type}] RAM {ram_used_gb:.2f}/{ram_total_gb:.2f}GB{payload_msg}"
    )


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


# OBSOLETE
async def get_start_and_end_indices(tensor_length: int, num_sections: int, target_section: int) -> tuple[int, int]:
    """Get the start and end indices for a tensor.

    Args:
        tensor_length (int): The length of the tensor to get the start and end indices for.
        num_sections (int): The number of sections to split the tensor into.
        target_section (int): The target section to get the start and end indices for.

    Returns:
        tuple[int, int]: The start and end indices for the target section.
    """
    assert target_section < num_sections, "Target section is greater than the number of sections"
    section_size = tensor_length // num_sections
    for i in range(int(min(target_section + 1, num_sections))):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < num_sections - 1 else tensor_length
        assert start_idx is not None and end_idx is not None, "Start idx and end idx are missing"
    return start_idx, end_idx


def create_metadata(tensor: torch.Tensor, num_sections: int) -> dict:
    """Create metadata for a tensor.

    Args:
        weights_tensor (torch.Tensor): The tensor to create metadata for.
        num_sections (int): The number of sections to split the tensor into.

    Returns:
        dict: The metadata for the tensor.
    """
    num_elements = tensor.numel()
    element_size = tensor.itemsize

    tensor_metadata = {
        "dtype": str(tensor.dtype),
        "size": list(tensor.shape),  # plain list, no torch.Size reference
        "num_elements": num_elements,
        "element_size": element_size,
        "total_bytes": num_elements * element_size,
    }

    section_size = num_elements // num_sections
    sections_metadata = {}

    for i in range(int(num_sections)):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < num_sections - 1 else num_elements
        sections_metadata[i] = {
            "start_byte": start_idx * element_size,
            "end_byte": end_idx * element_size,
            "start_idx": start_idx,
            "end_idx": end_idx,
        }

    return {"tensor": tensor_metadata, "sections": sections_metadata}


@async_lru(maxsize=5000)
async def download_metadata(metadata_path: str) -> dict:
    """Download metadata from a presigned url.

    Args:
        metadata_path (str): The path to the metadata.

    Returns:
        dict: The metadata.
    """
    metadata_bytes: bytes = await download_file(presigned_url=metadata_path)
    if len(metadata_bytes) > 1_000_000:
        logger.warning(f"Metadata is too large: {len(metadata_bytes)} bytes")
        raise ValueError(f"Metadata is too large: {len(metadata_bytes)} bytes")

    metadata: dict = json.loads(metadata_bytes)
    return metadata


async def upload_file(
    miner_api_client: MinerAPIClient,
    hotkey: Keypair,
    data: bytes,
    file_type: Literal["weights", "optimizer_state", "weights_metadata", "optimizer_state_metadata"],
    file_upload_response: Optional[FileUploadResponse | FileUploadResponse] = None,
    run_flags: RunFlags = RUN_FLAGS,
) -> str | dict:
    """
    Uploads a file to the orchestrator. To upload, we need to:
    1. Initiate a file upload by getting a FileUploadResponse from the orchestrator
    2. Upload the data using the presigned urls
    3. Complete the file upload

    Args:
        miner_api_client (MinerAPIClient): The miner API client.
        hotkey (Keypair): The hotkey of the miner.
        data (bytes): The data to upload
        file_type (Literal["weights", "optimizer_state"]): The type of file to upload
        file_upload_response (Optional[FileUploadResponse], optional): The response from the orchestrator. Defaults to None.

    Raises:
        ValueError: If the number of parts is greater than the maximum number of parts
        e: If there is an error uploading the file

    Returns:
        str: The path to the uploaded file
    """
    # TODO: We may want to set this to a more optimal value, for now we just make each part 10MB
    try:
        num_parts = calculate_num_parts(data=data)
        if num_parts > common_settings.MAX_NUM_PARTS:
            raise ValueError(
                f"Number of parts must be less than {common_settings.MAX_NUM_PARTS}. Your file with {len(data)} bytes doesn't fit within {common_settings.MAX_NUM_PARTS} part of 10MB each"
            )

        if file_upload_response is None:
            # Get presigned urls from orchestrator
            file_upload_response: FileUploadResponse | dict = await miner_api_client.initiate_file_upload_request(
                hotkey=hotkey,
                file_upload_request=FileUploadRequest(
                    file_type=file_type, num_parts=num_parts, multipart=num_parts > 1
                ),
            )

            if isinstance(file_upload_response, FileUploadResponse):
                logger.info(
                    f"Initiated multipart upload | file_type={file_type} "
                    f"object_name={file_upload_response.object_name} "
                    f"upload_id={file_upload_response.upload_id} "
                    f"urls={len(file_upload_response.urls)} num_parts={num_parts}"
                )

            # Need to return to check the parsing of the response
            if isinstance(file_upload_response, dict):
                return file_upload_response

        if run_flags.compress_s3_files.isOn():
            data = gzip.compress(data)

        # Upload data to presigned urls
        logger.debug(f"Uploading file {file_type} to presigned urls: {file_upload_response.urls}")
        parts: list[dict] = await MinerAPIClient.upload_to_s3(
            urls=file_upload_response.urls, data=data, upload_id=file_upload_response.upload_id
        )

        # Complete file upload. Necessary to notify orchestrator that all parts have been uploaded.
        if isinstance(file_upload_response, FileUploadResponse) and file_upload_response.upload_id is not None:
            logger.info(
                f"Completing multipart upload | file_type={file_type} "
                f"object_name={file_upload_response.object_name} "
                f"upload_id={file_upload_response.upload_id} parts_count={len(parts)} "
                f"part_numbers={[p.get('PartNumber') for p in parts][:5]}"
            )
            complete_file_upload_response: CompleteFileUploadResponse | dict = (
                await miner_api_client.complete_file_upload_request(
                    hotkey=hotkey,
                    file_upload_completion_request=FileUploadCompletionRequest(
                        object_name=file_upload_response.object_name,
                        upload_id=file_upload_response.upload_id,
                        parts=parts,
                    ),
                )
            )

            if isinstance(complete_file_upload_response, dict):
                return complete_file_upload_response

        return file_upload_response.object_name

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise


async def upload_tensor(
    miner_api_client: MinerAPIClient,
    tensor: torch.Tensor,
    hotkey: Keypair,
    file_type: Literal["activation", "weights", "optimizer_state"] = "activation",
    upload_urls: list[str] | None = None,
    object_name: str | None = None,
    run_flags: RunFlags = RUN_FLAGS,
) -> CompleteFileUploadResponse:
    """
    Upload a tensor to the orchestrator.
    TODO: Make this function properly handle single and multipart uploads

    Args:
        miner_api_client (MinerAPIClient): The miner API client.
        tensor (torch.Tensor): The tensor to upload.
        hotkey (Keypair): The hotkey of the miner.
        file_type (Literal["activation", "weights", "optimizer_state"]): The type of file to upload.
        upload_urls (list[str] | None): The upload urls to use for the upload.
        object_name (str | None): The object name to use for the upload.

    Returns:
        CompleteFileUploadResponse: The response from the orchestrator.
    """

    assert tensor.dtype == torch.bfloat16, f"Tensor must be bfloat16, got {tensor.dtype}"

    assert (object_name is None and upload_urls is None) or (
        object_name is not None and upload_urls is not None
    ), "Object name and upload urls have to be provided together if provided at all"

    upload_id = None
    initiate_response = None
    existing_upload_urls = upload_urls is not None

    _log_upload_ram("entry", file_type)

    # Reinterpret tensor memory as bytes in a consistent format (bfloat16 → uint8 bytes)
    # Always upload as bfloat16-backed bytes to match the downloader's default expectation.
    check_for_nans_and_infs(
        tensor=tensor,
        name=f"Uploading tensor of file type {file_type}",
        exception_type=NanInfException,
    )
    _log_upload_ram("after_nan_inf_check", file_type)

    tensor = tensor.detach().to("cpu").to(torch.bfloat16).contiguous()
    _log_upload_ram("after_contiguous", file_type)

    # Zero-copy buffer-protocol view of the tensor's raw bytes — avoids the
    # ~11 GB Python `bytes` copy that .tobytes() would create here. The
    # memoryview keeps the underlying tensor alive via its reference chain,
    # so the buffer stays valid for the entire upload. gzip.compress, len(),
    # slicing, and aiohttp.session.put(data=...) all accept memoryview.
    #
    # Validation signal: with .tobytes() the next log line would jump by
    # ~len(tensor) GB vs. after_contiguous. With memoryview it should be flat.
    _log_upload_ram("before_memoryview", file_type)
    tensor = memoryview(tensor.view(torch.uint8).numpy())
    _log_upload_ram("after_memoryview", file_type, payload_bytes=len(tensor))

    num_parts = calculate_num_parts(data=tensor)
    logger.info(f"Uploading {file_type} tensor with {num_parts} parts")
    multipart = num_parts > 1

    # Sanity checks (should not be triggered)
    if multipart:
        assert upload_urls is None, "Passing upload_urls which are only valid for single part uploads"

    if run_flags.compress_s3_files.isOn():
        payload = gzip.compress(tensor)
        del tensor
        _log_upload_ram("after_gzip_compress", file_type, payload_bytes=len(payload))
    else:
        payload = tensor

    try:
        # If we don't already have an upload url, we need to initiate a file upload request
        if existing_upload_urls:
            logger.debug(f"Using already provided upload URL for {file_type} tensor")
        else:
            async with TimerLoggerMiner(
                name="initiate_activation_upload", metadata={"file_type": file_type}, hotkey=hotkey.ss58_address[:8]
            ):
                initiate_response: FileUploadResponse | dict = await miner_api_client.initiate_file_upload_request(
                    hotkey=hotkey,
                    file_upload_request=FileUploadRequest(
                        file_type=file_type,
                        num_parts=num_parts,
                    ),
                )
                assert len(payload) > 0, "Tensor is empty"
                upload_urls = initiate_response.urls
                upload_id = initiate_response.upload_id
                if isinstance(initiate_response, FileUploadResponse):
                    logger.info(
                        f"Initiated upload | file_type={file_type} object_name={initiate_response.object_name} "
                        f"upload_id={initiate_response.upload_id} urls={len(initiate_response.urls)} "
                        f"num_parts={num_parts}"
                    )
            if not initiate_response:
                raise Exception("Error initiating file upload")

        # Upload data to presigned urls
        _log_upload_ram("before_s3_upload", file_type, payload_bytes=len(payload))
        async with TimerLoggerMiner(
            name="upload_multipart_to_s3", metadata={"file_type": file_type}, hotkey=hotkey.ss58_address[:8]
        ):
            logger.debug(f"Uploading tensor {file_type} to presigned urls: {upload_urls}")
            parts: list[dict] | None = await MinerAPIClient.upload_to_s3(
                urls=upload_urls, data=payload, upload_id=upload_id
            )
        _log_upload_ram("after_s3_upload", file_type)

        # for multipart uploads, we need to manually complete the upload request
        if multipart:
            async with TimerLoggerMiner(
                name="complete_file_upload_request", metadata={"file_type": file_type}, hotkey=hotkey.ss58_address[:8]
            ):
                logger.info(
                    f"Completing multipart upload | file_type={file_type} "
                    f"object_name={initiate_response.object_name} upload_id={initiate_response.upload_id} "
                    f"parts_count={len(parts) if parts else 0} part_numbers={[p.get('PartNumber') for p in parts][:5]}"
                )
                await miner_api_client.complete_file_upload_request(
                    hotkey=hotkey,
                    file_upload_completion_request=FileUploadCompletionRequest(
                        object_name=initiate_response.object_name,
                        upload_id=initiate_response.upload_id,
                        parts=parts,
                    ),
                )
        else:
            logger.debug(f"Skipped completing file upload request for {file_type} tensor as it is a single part upload")
        return CompleteFileUploadResponse(
            object_path=initiate_response.object_name if initiate_response else object_name
        )
    except Exception as e:
        logger.exception(f"Error uploading multipart to S3: {e}")
        raise


# OBSOLETE
def extract_filename_from_url(url):
    """
    Extract the filename from a URL, handling both regular paths and query parameters.

    Args:
        url: The URL to extract filename from


    Returns:
        str: The extracted filename
    """
    # Parse the URL
    parsed_url = urlparse(url)

    # Get the path component
    path = parsed_url.path

    # Extract filename from path
    filename = path.split("/")[-1]

    return filename


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
