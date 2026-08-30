import asyncio
import gzip
import math
from typing import Any

import aiohttp
from loguru import logger
from common.models.run_flags import RUN_FLAGS, RunFlags
from common import settings as common_settings


def should_skip_ssl(url: str) -> bool:
    """Return True if SSL verification should be skipped for localhost/minio URLs."""
    return "localhost" in url or "minio" in url or "127.0.0.1" in url


async def upload_parts(urls: list[str], data: Any, upload_id: str | None, max_retries: int = 3) -> list[dict]:
    """Upload parts to S3 storage with retry logic.

    Args:
        urls (list[str]): The URLs to upload the parts to.
        data: The data to upload. bytes/bytearray/memoryview, or a ChainedBuffer
            (zero-copy sliceable wrapper around header + tensor + trailer segments).
        upload_id (str): The upload ID.
        max_retries (int): Maximum number of retry attempts per part (default: 3).

    Returns:
        list[dict]: The parts that were uploaded.
    """
    # Wrap raw bytes/bytearray in a memoryview so per-part slicing `data[i:j]`
    # is zero-copy. memoryview and ChainedBuffer (the wire-format blob wrapper)
    # already return memoryview-backed slices and are passed through as-is —
    # ChainedBuffer in particular does not implement the C buffer protocol, so
    # an unconditional memoryview(data) would TypeError.
    if isinstance(data, (bytes, bytearray)):
        data = memoryview(data)

    if len(urls) > 1 and upload_id is None:
        logger.exception("Upload ID is required for multipart uploads")
    if len(urls) == 1 and upload_id is None:
        return await upload_part(urls=urls, data=data, upload_id=upload_id, max_retries=max_retries)
    else:
        try:
            assert upload_id is not None, "Upload ID is required for multipart uploads"
        except Exception as e:
            logger.exception(f"Error uploading parts: {e}")
            raise

    # Configure timeout for S3 uploads - allow for larger files with reasonable timeout
    timeout = aiohttp.ClientTimeout(total=common_settings.S3_UPLOAD_TIMEOUT, connect=30)
    # Skip SSL verification for localhost/minio (self-signed certs in local dev)
    connector = aiohttp.TCPConnector(ssl=False) if urls and should_skip_ssl(urls[0]) else None
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        part_size = int(math.ceil(len(data) / len(urls)))
        assert part_size > 0, "Part size is 0"

        chunk_indices = range(0, len(data), part_size)

        logger.info(
            f"uploading {len(chunk_indices)} chunks with part size {part_size} "
            f"(concurrency {common_settings.S3_UPLOAD_MAX_CONCURRENCY})"
        )
        semaphore = asyncio.Semaphore(common_settings.S3_UPLOAD_MAX_CONCURRENCY)

        async def upload_one(part_number: int, url: str, chunk_index: int) -> dict:
            # Retry logic for each part
            async with semaphore:
                for attempt in range(max_retries + 1):  # +1 to include initial attempt
                    try:
                        start_time = asyncio.get_event_loop().time()
                        async with session.put(url, data=data[chunk_index : chunk_index + part_size]) as response:
                            upload_time = asyncio.get_event_loop().time() - start_time

                            if not response.ok:
                                # Get detailed error information from S3
                                error_body = await response.text()
                                error_headers = dict(response.headers)

                                logger.error(f"HTTP Status: {response.status} {response.reason}")
                                logger.error(f"Response Headers: {error_headers}")
                                logger.error(f"Response Body: {error_body}")
                                logger.error(f"Request URL: {url}")
                                logger.error(f"Upload ID: {upload_id}")

                            response.raise_for_status()

                            # Extract ETag from response headers (remove quotes if present)
                            etag = response.headers.get("ETag", "").strip('"')

                            # Log upload performance
                            upload_speed_mbps = (part_size / (1024 * 1024)) / max(upload_time, 0.001)
                            logger.debug(
                                f"🏎️ Part {part_number} upload completed in {upload_time:.2f}s ({upload_speed_mbps:.2f} MB/s) 🏎️"
                            )

                            return {
                                "PartNumber": part_number,
                                "ETag": etag,
                            }

                    except (
                        aiohttp.ClientError,
                        aiohttp.ServerTimeoutError,
                        aiohttp.ClientResponseError,
                        asyncio.TimeoutError,
                        TimeoutError,  # Python built-in TimeoutError
                        ConnectionError,
                        Exception,  # Catch RequestTimeout and other S3-specific errors
                    ) as e:
                        if attempt < max_retries:
                            # Calculate exponential backoff delay (1s, 2s, 4s, ...)
                            delay = 2**attempt
                            logger.warning(
                                f"Upload failed for part {part_number} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                                f"Retrying in {delay}s..."
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"Upload failed for part {part_number} after {max_retries + 1} attempts: {e}")
                            raise

        # Parts upload concurrently (bounded by the semaphore); gather preserves
        # input order so `parts` stays sorted by PartNumber for the S3 complete call.
        parts = list(
            await asyncio.gather(
                *[upload_one(i + 1, url, chunk_index) for i, (url, chunk_index) in enumerate(zip(urls, chunk_indices))]
            )
        )
    return parts


async def upload_part(urls: list[str], data: bytes | memoryview, upload_id: str, max_retries: int = 3) -> list[dict]:
    """Upload a single file to S3 storage with retry logic (non-multipart upload).

    Args:
        urls (list[str]): The URL to upload to (should contain a single URL).
        data (bytes | memoryview): The data to upload.
        upload_id (str): The upload ID.
        max_retries (int): Maximum number of retry attempts (default: 3).

    Returns:
        list[dict]: A list containing a single part info dict with PartNumber and ETag.
    """
    assert len(urls) == 1, "Single part upload should only have one URL"
    url = urls[0]

    # Configure timeout for S3 uploads
    timeout = aiohttp.ClientTimeout(total=common_settings.S3_UPLOAD_TIMEOUT, connect=30)
    # Skip SSL verification for localhost/minio (self-signed certs in local dev)
    connector = aiohttp.TCPConnector(ssl=False) if should_skip_ssl(url) else None
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Retry logic for the upload
        for attempt in range(max_retries + 1):  # +1 to include initial attempt
            try:
                start_time = asyncio.get_event_loop().time()
                async with session.put(url, data=data) as response:
                    upload_time = asyncio.get_event_loop().time() - start_time

                    if not response.ok:
                        # Get detailed error information from S3
                        error_body = await response.text()
                        error_headers = dict(response.headers)

                        logger.error(f"HTTP Status: {response.status} {response.reason}")
                        logger.error(f"Response Headers: {error_headers}")
                        logger.error(f"Response Body: {error_body}")
                        logger.error(f"Request URL: {url}")
                        logger.error(f"Upload ID: {upload_id}")

                    response.raise_for_status()

                    # Log upload performance
                    upload_speed_mbps = (len(data) / (1024 * 1024)) / max(upload_time, 0.001)
                    logger.debug(
                        f"🏎️ Single part upload completed in {upload_time:.2f}s ({upload_speed_mbps:.2f} MB/s) 🏎️"
                    )
                break

            except (
                aiohttp.ClientError,
                aiohttp.ServerTimeoutError,
                aiohttp.ClientResponseError,
                asyncio.TimeoutError,
                TimeoutError,  # Python built-in TimeoutError
                ConnectionError,
                Exception,  # Catch RequestTimeout and other S3-specific errors
            ) as e:
                if attempt < max_retries:
                    # Calculate exponential backoff delay (1s, 2s, 4s, ...)
                    delay = 2**attempt
                    logger.warning(
                        f"Upload failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Upload failed after {max_retries + 1} attempts: {e}")
                    raise


# Per-event-loop download session cache: a fresh ClientSession per call means a new
# TCP+TLS handshake to R2 for every sample download (~100-300ms of a ~700ms download).
# Keyed by (loop, skip_ssl) so keep-alive connections are reused within a process.
_download_sessions: dict = {}


def _get_download_session(skip_ssl: bool) -> "aiohttp.ClientSession":
    loop = asyncio.get_running_loop()
    key = (id(loop), skip_ssl)
    session = _download_sessions.get(key)
    if session is None or session.closed:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=common_settings.S3_DOWNLOAD_TIMEOUT),
            connector=aiohttp.TCPConnector(ssl=False) if skip_ssl else None,
        )
        _download_sessions[key] = session
    return session


async def download_file(presigned_url: str, max_retries: int = 3, run_flags: RunFlags = RUN_FLAGS):
    """Download a file from S3 storage with retry logic."""
    # Skip SSL verification for localhost/minio (self-signed certs in local dev)
    session = _get_download_session(should_skip_ssl(presigned_url))

    for attempt in range(max_retries + 1):
        try:
            async with session.get(presigned_url) as response:
                response.raise_for_status()
                if run_flags.compress_s3_files.isOn():
                    return gzip.decompress(await response.read())
                else:
                    return await response.read()
        except aiohttp.ClientResponseError as e:
            if e.status >= 500 or e.status == 429:
                if attempt < max_retries:
                    delay = 2**attempt
                    logger.warning(
                        f"Retryable error (HTTP {e.status}), retrying in {delay}s... (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.warning(
                        f"Server error (HTTP {e.status}) downloading file from R2: {e}. Failed after {max_retries + 1} attempts. This is likely a temporary R2 issue."
                    )
                    raise
            else:
                logger.error(f"HTTP error downloading file: {e}")
                raise
        except (
            aiohttp.ClientPayloadError,  # truncated body (ContentLengthError) — frequent on slow R2 links
            aiohttp.ClientConnectionError,
            aiohttp.ServerTimeoutError,
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
        ) as e:
            if attempt < max_retries:
                delay = 2**attempt
                logger.warning(
                    f"Transient download error ({type(e).__name__}: {e}), retrying in {delay}s... "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )
                await asyncio.sleep(delay)
                continue
            logger.error(f"Download failed after {max_retries + 1} attempts: {type(e).__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error downloading file from presigned URL: {e}")
            raise


def filter_exceptions(*args) -> list[Any]:
    bad_indices = set()
    # Track actual exceptions for logging without relying on indices across tuples
    collected_exceptions: list[Exception] = []
    for arg in args:
        for i, element in enumerate(arg):
            if isinstance(element, Exception):
                bad_indices.add(i)
                collected_exceptions.append(element)

    # Filter each iterable by bad indices
    result = tuple([[e for i, e in enumerate(arg) if i not in bad_indices] for arg in args])

    # Log the collected exceptions safely
    if collected_exceptions:
        logger.error(collected_exceptions)

    if len(result) == 1:
        return result[0]
    return result
