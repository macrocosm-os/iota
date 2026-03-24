import sys
import os
import asyncio
from loguru import logger

from miner.new_miner import Miner
from miner import settings as miner_settings
from common import settings as common_settings
from common.utils.gpu_process_utils import kill_stale_gpu_processes, cleanup_stale_shared_memory

# Setup logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <yellow><b>[MINER]</b></yellow> | <level>{level: <8}</level> | <cyan>{name}:{line}</cyan> | <level>{message}</level> | <magenta>{extra}</magenta>",
    level=common_settings.LOG_LEVEL,
    colorize=True,
)
if common_settings.LOG_FILE_ENABLED:
    logger.add(
        f"../../logs/{miner_settings.WALLET_NAME}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | [MINER] | {level: <8} | {name}:{line} | {message} | {extra}",
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        colorize=False,
    )


def main():
    """Main entry point for the miner."""
    kill_stale_gpu_processes()
    cleanup_stale_shared_memory()
    logger.info("Starting miner")
    logger.info(f"Wallet: {miner_settings.WALLET_NAME}")
    logger.info(f"Hotkey: {miner_settings.WALLET_HOTKEY}")
    resolved_device = os.getenv("DEVICE") or miner_settings.detect_device()
    logger.info(f"Device: {resolved_device}")
    logger.info(f"Timeout: {miner_settings.TIMEOUT}s")

    try:
        # Create miner instance
        miner = Miner(
            wallet_name=miner_settings.WALLET_NAME,
            wallet_hotkey=miner_settings.WALLET_HOTKEY,
            device=resolved_device,
        )

        # Run the miner
        asyncio.run(miner.run_miner())

    except KeyboardInterrupt:
        logger.info("Miner stopped by user")
    except Exception as e:
        logger.error(f"Error running miner: {e}")
        raise


if __name__ == "__main__":
    main()
