import asyncio
import gzip
import torch
from loguru import logger

from common import settings as common_settings
from common.models.run_flags import RUN_FLAGS, RunFlags
from common.utils.s3_utils import download_file
from subnet.model.utils import log_gpu_memory_usage


def _decode_and_tokenize(data: bytes, tokenizer, device: str) -> torch.Tensor:
    # Some objects may be gzip-compressed without content-encoding header in R2; try transparently
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = gzip.decompress(data).decode("utf-8")
        except Exception as e:
            raise Exception(f"Failed to decode sample as utf-8 (and gzip fallback failed): {e}")
    return torch.tensor(tokenizer.encode(text)).to(device)


async def download_sample(
    download_url: str,
    tokenizer,
    device: str = "cpu",
    mock: bool = False,
    run_flags: RunFlags = RUN_FLAGS,
) -> torch.Tensor:
    """
    Downloads the sample from the given URL and returns it as a tensor.

    Args:
        download_url: The URL of the sample to download.
        tokenizer: The tokenizer to use to decode the sample.
    """
    log_gpu_memory_usage(note="before downloading sample")
    data = await download_file(presigned_url=download_url, run_flags=run_flags)

    if mock:
        return torch.randn(
            size=(common_settings.MINI_BATCH_SIZE, 100),
            dtype=torch.bfloat16,
        ).to("cpu")

    # Decode + tokenize off the event loop: encoding ~16K+ tokens takes 50-300ms of
    # pure CPU, and this coroutine shares the loop with the P2P server — blocking it
    # here delays activation serving to downstream peers (their download time).
    sample = await asyncio.to_thread(_decode_and_tokenize, data, tokenizer, device)
    if len(sample) < common_settings.SEQUENCE_LENGTH * common_settings.MINI_BATCH_SIZE:
        raise Exception(
            f"Sample is too short: {len(sample)} < {common_settings.SEQUENCE_LENGTH * common_settings.MINI_BATCH_SIZE}"
        )

    sample = sample[: common_settings.SEQUENCE_LENGTH * common_settings.MINI_BATCH_SIZE]
    sample = sample.reshape(common_settings.MINI_BATCH_SIZE, common_settings.SEQUENCE_LENGTH)

    log_gpu_memory_usage(note="after downloading sample")

    logger.info(f"Sample shape: {sample.shape}")
    return sample
