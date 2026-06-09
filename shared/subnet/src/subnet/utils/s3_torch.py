import gzip
from time import time
from loguru import logger

import asyncio
from asyncio.exceptions import TimeoutError

import aiohttp
from common.utils.exceptions import NanInfWarning
from common.models.miner_models import ChunkMetadata
from common.models.run_flags import RUN_FLAGS, RunFlags
from subnet.model.utils import log_gpu_memory_usage
from subnet.utils.vector_utils import check_for_nans_and_infs

import torch
import numpy as np


def _should_skip_ssl(url: str) -> bool:
    """Return True if SSL verification should be skipped for localhost/minio URLs."""
    return "localhost" in url or "minio" in url or "127.0.0.1" in url


class AioHttpClientWithOpenSession:
    def __init__(self):
        self.SESSION_REFRESH_TIME = 300
        self.session = None
        self.session_created_at = time()
        self._current_ssl_skip = None  # Track if current session has SSL skip enabled

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
            self._current_ssl_skip = None

    async def _get_or_create_session(self, skip_ssl: bool):
        """Get existing session or create new one with appropriate SSL settings."""
        needs_refresh = (
            not self.session
            or time() - self.session_created_at > self.SESSION_REFRESH_TIME
            or self._current_ssl_skip != skip_ssl
        )
        if needs_refresh:
            await self.close()
            connector = aiohttp.TCPConnector(ssl=False) if skip_ssl else None
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300), connector=connector)
            self.session_created_at = time()
            self._current_ssl_skip = skip_ssl
        return self.session

    async def get(self, url: str):
        skip_ssl = _should_skip_ssl(url)
        session = await self._get_or_create_session(skip_ssl)
        response = await session.get(url)
        return response


async def process_response(
    response: aiohttp.ClientResponse, dtype: torch.dtype, device: str = "cuda", run_flags: RunFlags = RUN_FLAGS
) -> torch.Tensor:
    """Process the response from aiohttp and return a tensor.

    Memory note: the previous implementation used ``torch.tensor(np.frombuffer(content, ...))``,
    which empirically allocates ~1.5× the source size (PyTorch makes intermediate copies
    when constructing a tensor from a numpy array). With 32 partitions downloaded
    concurrently for first/last-layer miners (which have vocab-size weights — large
    ``tok_emb`` / ``out_head``), this caused ~20 GB of transient host RAM during the
    download phase and contributed to OOM kills on the most heap-pressured miners.

    ``torch.from_numpy`` is zero-copy: it constructs a tensor that shares storage with
    the numpy array, which itself shares storage with ``content`` (the bytes object).
    The whole chain stays alive via PyTorch's storage refcount until the returned
    tensor is dropped. The resulting tensor is read-only (because ``bytes`` is
    immutable), but we only ever read from it downstream (nan/inf check and an
    ``index_copy`` into ``target_tensor``), so that is safe.
    """
    content = await response.read()
    if run_flags.compress_s3_files.isOn():
        content = gzip.decompress(content)
    np_view = np.frombuffer(content, dtype=np.uint8)
    # Zero-copy: torch.from_numpy shares storage with np_view, which shares
    # storage with `content`. ``.view(dtype)`` reinterprets bytes without copying.
    # ``.to(device)`` is a no-op when device is already CPU (the common case here).
    loaded_tensor = torch.from_numpy(np_view).view(dtype).to(device)
    return loaded_tensor


async def download_tensor(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    max_retries: int = 3,
    run_flags: RunFlags = RUN_FLAGS,
) -> torch.Tensor:
    """Download bytes and cast into a tensor from S3 storage with retry logic.

    Args:
        path: URL path to download tensor from
        dtype: PyTorch data type for the tensor
        device: Device to load tensor to ('cuda', 'cpu', etc.)
        max_retries: Maximum number of retry attempts
        session: Optional aiohttp session to reuse. If None, creates a new session.

    Returns:
        Downloaded tensor
    """
    log_gpu_memory_usage(note="before downloading tensor")

    # Determine if we need to manage the session ourselves
    for attempt in range(max_retries + 1):
        try:
            # Create new session for single download
            response = await s3_client.get(path)
            response.raise_for_status()
            loaded_tensor = await process_response(response=response, dtype=dtype, device=device, run_flags=run_flags)

            assert isinstance(
                loaded_tensor, torch.Tensor
            ), f"Downloaded tensor is not a torch.Tensor: {type(loaded_tensor)}, path: {path}"

            check_for_nans_and_infs(loaded_tensor, f"tensor downloaded from {path}", exception_type=NanInfWarning)

            log_gpu_memory_usage(note="after downloading tensor")

            return loaded_tensor

        except (aiohttp.ClientResponseError, aiohttp.ClientConnectorDNSError, aiohttp.ClientConnectorError) as e:
            # Determine if error is retryable
            is_retryable = False
            error_type = "Unknown error"

            if isinstance(e, aiohttp.ClientResponseError):
                if e.status >= 500 or e.status == 429:
                    is_retryable = True
                    error_type = f"HTTP {e.status}"
                else:
                    logger.error(f"HTTP error downloading tensor: {e}")
                    raise
            else:  # ClientConnectorDNSError or ClientConnectorError
                is_retryable = True
                error_type = "DNS/network"

            if is_retryable:
                if attempt < max_retries:
                    delay = 2**attempt
                    logger.warning(
                        f"Retryable error ({error_type}) downloading tensor from R2: {e}. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.warning(
                        f"Retryable error ({error_type}) downloading tensor from R2: {e}. Failed after {max_retries + 1} attempts. This is likely a temporary issue."
                    )
                    raise

        except Exception as e:
            logger.error(f"Error downloading tensor: {e}")
            raise


async def download_weights_or_optimizer_state(
    metadata_info: ChunkMetadata,
    max_retries: int = 3,
) -> torch.Tensor:
    """Download weights from S3 storage with retry logic."""
    start_time = time()
    timeout = aiohttp.ClientTimeout(total=3000)  # 5 minute timeout
    # Skip SSL verification for localhost/minio (self-signed certs in local dev)
    connector = aiohttp.TCPConnector(ssl=False) if _should_skip_ssl(metadata_info.tensor_path) else None

    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                # if partition is not specified, download the full tensor
                byte_range = f"bytes={metadata_info.start_byte}-{metadata_info.end_byte - 1}"
                async with session.get(metadata_info.tensor_path, headers={"Range": byte_range}) as response:
                    if response.status > 299:
                        response.raise_for_status()
                    binary_data = await response.read()

            if RUN_FLAGS.compress_s3_files.isOn():
                binary_data = gzip.decompress(binary_data)

            section_numpy = np.frombuffer(binary_data, dtype=np.uint8)
            section_torch = torch.from_numpy(section_numpy.copy())
            # assumes default dtype if not specified
            section_torch = section_torch.view(getattr(torch, metadata_info.chunk_dtype))
            return section_torch

        except TimeoutError as e:
            logger.error(
                f"Timeout error downloading weights or optimizer state: {e}. Time taken: {time() - start_time} seconds. Started download at {start_time}."
            )
            raise
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectorDNSError, aiohttp.ClientConnectorError) as e:
            # Determine if error is retryable
            is_retryable = False
            error_type = "Unknown error"

            if isinstance(e, aiohttp.ClientResponseError):
                if e.status >= 500 or e.status == 429:
                    is_retryable = True
                    error_type = f"HTTP {e.status}"
                else:
                    logger.error(f"HTTP error downloading weights or optimizer state: {e}")
                    raise
            else:  # ClientConnectorDNSError or ClientConnectorError
                is_retryable = True
                error_type = "DNS/network"

            if is_retryable:
                if attempt < max_retries:
                    delay = 2**attempt
                    logger.warning(
                        f"Retryable error ({error_type}) downloading weights or optimizer state from R2: {e}. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.warning(
                        f"Retryable error ({error_type}) downloading weights or optimizer state from R2: {e}. Failed after {max_retries + 1} attempts. This is likely a temporary issue."
                    )
                    raise
        except Exception as e:
            logger.error(f"Error downloading weights or optimizer state: {e}")
            raise


s3_client = AioHttpClientWithOpenSession()
