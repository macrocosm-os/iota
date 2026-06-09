import argparse
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iota miner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("MINER_DRY_RUN", "").lower() in ("1", "true", "yes"),
        help="Import everything and exit 0 without starting the miner. "
        "Also enabled via MINER_DRY_RUN=1. Used by liquid-compute to "
        "smoke-test packaged tarballs.",
    )
    return parser.parse_args(argv)


def main():
    """Main entry point for the miner."""
    args = _parse_args()
    if args.dry_run:
        logger.info(
            "Dry-run: imports OK (run_id={}, wallet={}). Exiting.",
            os.environ.get("RUN_ID") or os.environ.get("LC_RUN_ID", ""),
            miner_settings.WALLET_NAME,
        )
        return

    kill_stale_gpu_processes()
    cleanup_stale_shared_memory()
    logger.info("Starting miner")
    logger.info(f"Wallet: {miner_settings.WALLET_NAME}")
    logger.info(f"Hotkey: {miner_settings.WALLET_HOTKEY}")
    resolved_device = os.getenv("DEVICE") or miner_settings.detect_device()
    logger.info(f"Device: {resolved_device}")
    logger.info(f"Timeout: {miner_settings.TIMEOUT}s")

    async def run():
        miner = Miner(
            wallet_name=miner_settings.WALLET_NAME,
            wallet_hotkey=miner_settings.WALLET_HOTKEY,
            device=resolved_device,
        )
        await miner.run_miner()

    try:
        asyncio.run(run())

    except KeyboardInterrupt:
        logger.info("Miner stopped by user")
    except Exception as e:
        logger.error(f"Error running miner: {e}")
        raise


if __name__ == "__main__":
    main()
