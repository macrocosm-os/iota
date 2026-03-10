import asyncio
import json
import gzip
import platform
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


def collect_system_data() -> str | None:
    """Collect device info and return as MinerSystemData-compatible JSON."""
    try:
        machine = platform.machine()
        gpus: list[str] = []
        cpu_brand: str | None = None
        cores: int | None = None
        memory_gb: int | None = None

        if sys.platform == "darwin":
            cpu_brand = _sysctl("machdep.cpu.brand_string")
            if not cpu_brand:
                cpu_brand = platform.processor() or None
            if cpu_brand:
                gpus.append(cpu_brand)

            ncpu = _sysctl("hw.ncpu")
            if ncpu:
                cores = int(ncpu)

            memsize = _sysctl("hw.memsize")
            if memsize:
                memory_gb = int(memsize) // (1024**3)
        else:
            cpu_brand = platform.processor() or None

        # CUDA: get GPU name
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                if gpu_name:
                    gpus.append(gpu_name)
        except Exception:
            pass

        data = {
            "gpus": gpus,
            "chip_info": {
                "machine": machine,
                "cpu": cpu_brand,
                "cores": cores,
                "memory_gb": memory_gb,
            },
        }
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
    # Create metadata about the tensor
    tensor_metadata = {
        "dtype": str(tensor.dtype),
        "size": tensor.size(),
        "num_elements": tensor.numel(),
        "element_size": tensor.itemsize,
        "total_bytes": tensor.nbytes,  # this is just weights_tensor.numel() * weights_tensor.itemsize
    }

    # Number of sections to split into (in elements, or indices)
    section_size = tensor.numel() // num_sections
    # Create section metadata
    sections_metadata = {}

    for i in range(int(num_sections)):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < num_sections - 1 else tensor.numel()

        # Calculate corresponding tensor indices
        start_byte = start_idx * tensor_metadata["element_size"]
        end_byte = end_idx * tensor_metadata["element_size"]

        assert start_byte is not None and end_byte is not None, "Start byte and end byte are missing"
        assert start_idx is not None and end_idx is not None, "Start idx and end idx are missing"

        sections_metadata[i] = {
            "start_byte": start_byte,
            "end_byte": end_byte,
            "start_idx": start_idx,  # e.g for a (100,100) matrix divided into 10 sections, the indices are: 0-999. 1000 - 1999. 2000 - 2999. 3000 - 3999. 4000 - 4999. 5000 - 5999. 6000 - 6999. 7000 - 7999. 8000 - 8999. 9000 - 9999.
            "end_idx": end_idx,
        }

    # Save full tensor metadata
    full_metadata = {"tensor": tensor_metadata, "sections": sections_metadata}

    return full_metadata


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
    file_type: Literal["activation", "weights", "optimizer_state", "local_optimizer_state"] = "activation",
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
        file_type (Literal["activation", "weights", "optimizer_state", "local_optimizer_state"]): The type of file to upload.
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

    # Reinterpret tensor memory as bytes in a consistent format (bfloat16 → uint8 bytes)
    # Always upload as bfloat16-backed bytes to match the downloader's default expectation.
    check_for_nans_and_infs(
        tensor=tensor,
        name=f"Uploading tensor of file type {file_type}",
        exception_type=NanInfException,
    )

    tensor = tensor.detach().to("cpu").to(torch.bfloat16).contiguous()
    tensor = tensor.view(torch.uint8).numpy().tobytes()
    num_parts = calculate_num_parts(data=tensor)
    logger.info(f"Uploading {file_type} tensor with {num_parts} parts")
    multipart = num_parts > 1

    # Sanity checks (should not be triggered)
    if multipart:
        assert upload_urls is None, "Passing upload_urls which are only valid for single part uploads"

    if run_flags.compress_s3_files.isOn():
        payload = gzip.compress(tensor)
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
        async with TimerLoggerMiner(
            name="upload_multipart_to_s3", metadata={"file_type": file_type}, hotkey=hotkey.ss58_address[:8]
        ):
            logger.debug(f"Uploading tensor {file_type} to presigned urls: {upload_urls}")
            parts: list[dict] | None = await MinerAPIClient.upload_to_s3(
                urls=upload_urls, data=payload, upload_id=upload_id
            )

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
        response = await miner_api_client.get_layer_state_request()
        if response == state.value:
            logger.info(f"Orchestrator is finally in state {state}")
            miner_api_client.layer_state = LayerPhase.from_str(response)
            break
        elif LayerPhase.from_str(response).next() == state:
            continue
        else:
            miner_api_client.layer_state = LayerPhase.TRAINING
            if raise_bad_sync:
                raise LayerStateException(
                    f"Miner is out of sync with the orchestrator. Miner is waiting for orchestrator to be in state {state}, but orchestrator is in state {response}, setting state to training"
                )
