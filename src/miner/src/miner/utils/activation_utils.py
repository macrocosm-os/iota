import gzip
import torch
from loguru import logger

from common import settings as common_settings
from common.models.run_flags import RUN_FLAGS, RunFlags
from common.utils.s3_utils import download_file
from subnet.model.utils import log_gpu_memory_usage


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

    # Some objects may be gzip-compressed without content-encoding header in R2; try transparently
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = gzip.decompress(data).decode("utf-8")
        except Exception as e:
            raise Exception(f"Failed to decode sample as utf-8 (and gzip fallback failed): {e}")

    if mock:
        return torch.randn(
            size=(common_settings.MINI_BATCH_SIZE, 100),
            dtype=torch.bfloat16,
        ).to("cpu")

    sample = torch.tensor(tokenizer.encode(text)).to(device)
    if len(sample) < common_settings.SEQUENCE_LENGTH * common_settings.MINI_BATCH_SIZE:
        raise Exception(
            f"Sample is too short: {len(sample)} < {common_settings.SEQUENCE_LENGTH * common_settings.MINI_BATCH_SIZE}"
        )

    sample = sample[: common_settings.SEQUENCE_LENGTH * common_settings.MINI_BATCH_SIZE]
    sample = sample.reshape(common_settings.MINI_BATCH_SIZE, common_settings.SEQUENCE_LENGTH)

    log_gpu_memory_usage(note="after downloading sample")

    logger.info(f"Sample shape: {sample.shape}")
    return sample
