"""Self-describing binary blob wire format — protocol layer (no torch).

Layout:
    [ header (12B) ][ payload bytes ][ trailer JSON (UTF-8) ]
    header:
      bytes 0..1   : version u16 LE
      bytes 2..3   : reserved u16  (zero)
      bytes 4..11  : trailer_offset u64 LE
    trailer_len is recovered as object_size - trailer_offset.

This module is the pure-Python protocol layer: header pack/parse, trailer
encoding, the sliceable ChainedBuffer producers compose, and the cached
trailer-fetch downloader. The tensor-aware helpers (build_*_blob,
download_blob_region) live in `subnet/utils/blob_format` because the
`common` package is forbidden from importing torch.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Union

import aiohttp
from loguru import logger

from common import settings as common_settings
from common.utils.cache import async_lru
from common.utils.s3_utils import should_skip_ssl


_HEADER_STRUCT = struct.Struct("<HHQ")
assert _HEADER_STRUCT.size == common_settings.BLOB_HEADER_LEN


def pack_header(*, trailer_offset: int) -> bytes:
    return _HEADER_STRUCT.pack(common_settings.BLOB_VERSION, 0, trailer_offset)


def parse_header(header: bytes) -> int:
    if len(header) < common_settings.BLOB_HEADER_LEN:
        raise ValueError(f"blob header too short: got {len(header)} bytes, need {common_settings.BLOB_HEADER_LEN}")
    version, _reserved, trailer_offset = _HEADER_STRUCT.unpack(header[: common_settings.BLOB_HEADER_LEN])
    if version != common_settings.BLOB_VERSION:
        raise ValueError(f"unsupported blob version {version}; expected {common_settings.BLOB_VERSION}")
    return trailer_offset


def pack_trailer(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


class ChainedBuffer:
    """Sliceable, len-able concatenation of (memoryview | bytes | bytearray) segments.

    Slices entirely inside one segment return a memoryview (zero-copy). Slices that
    cross a boundary materialize a small bytes object — for our 12-byte header +
    tensor + trailer composition, at most two multipart slices per upload ever hit
    that path.
    """

    def __init__(self, *segments: Union[bytes, bytearray, memoryview]):
        self._segments: list[memoryview] = []
        self._offsets: list[int] = [0]
        total = 0
        for s in segments:
            if isinstance(s, memoryview):
                mv = s.cast("B") if s.format != "B" or s.itemsize != 1 else s
            else:
                mv = memoryview(s).cast("B")
            self._segments.append(mv)
            total += len(mv)
            self._offsets.append(total)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def _locate(self, pos: int) -> int:
        """Index of the segment containing absolute byte `pos`."""
        lo, hi = 0, len(self._segments) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def __getitem__(self, item: slice) -> Union[memoryview, bytes]:
        if not isinstance(item, slice):
            raise TypeError("ChainedBuffer only supports slice indexing")
        start, stop, step = item.indices(self._length)
        if step != 1:
            raise ValueError("ChainedBuffer does not support stepped slices")
        if start >= stop:
            return b""

        first = self._locate(start)
        # Find the last segment that contains a byte < stop.
        last = self._locate(stop - 1)

        seg_start_offset = self._offsets[first]
        local_start = start - seg_start_offset

        if first == last:
            local_stop = stop - seg_start_offset
            return self._segments[first][local_start:local_stop]

        out = bytearray()
        out += self._segments[first][local_start:]
        for i in range(first + 1, last):
            out += self._segments[i]
        last_offset = self._offsets[last]
        out += self._segments[last][: stop - last_offset]
        return bytes(out)


async def ranged_get(presigned_url: str, byte_range: str, max_retries: int = 3) -> bytes:
    """Single ranged GET against a presigned URL with retries. Public helper —
    the tensor-aware downloader in subnet/utils/blob_format wraps this."""
    timeout = aiohttp.ClientTimeout(total=common_settings.S3_DOWNLOAD_TIMEOUT)
    connector = aiohttp.TCPConnector(ssl=False) if should_skip_ssl(presigned_url) else None
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(presigned_url, headers={"Range": byte_range}) as response:
                    if response.status >= 400:
                        response.raise_for_status()
                    return await response.read()
        except aiohttp.ClientResponseError as e:
            last_exc = e
            if (e.status >= 500 or e.status == 429) and attempt < max_retries:
                await asyncio.sleep(2**attempt)
                continue
            raise
        except (aiohttp.ClientConnectorError, aiohttp.ClientConnectorDNSError) as e:
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(2**attempt)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


@async_lru(maxsize=5000)
async def download_blob_trailer(presigned_url: str) -> dict:
    """Two ranged GETs: header (12B) then trailer JSON. Cached by URL."""
    header_range = f"bytes=0-{common_settings.BLOB_HEADER_LEN - 1}"
    header = await ranged_get(presigned_url, header_range)
    trailer_offset = parse_header(header)

    trailer_range = f"bytes={trailer_offset}-"
    trailer_bytes = await ranged_get(presigned_url, trailer_range)
    try:
        return json.loads(trailer_bytes)
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse blob trailer at offset {trailer_offset}: {e}. "
            f"First 200 bytes: {trailer_bytes[:200]!r}"
        )
        raise


def buffer_to_bytes(buf: ChainedBuffer) -> bytes:
    """Materialize a ChainedBuffer to a single bytes object. Used only when a caller
    must hand off a contiguous buffer (e.g. compression). Multipart upload paths
    should slice the ChainedBuffer directly to preserve zero-copy."""
    return bytes(buf[0 : len(buf)])
