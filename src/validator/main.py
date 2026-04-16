import sys
import asyncio
from loguru import logger

from validator.validator import Validator
from validator import settings as validator_settings
from common import settings as common_settings

# Setup logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <cyan><b>[VALID]</b></cyan> | <level>{level: <8}</level> | <cyan>{name}:{line}</cyan> | <level>{message}</level> | <magenta>{extra}</magenta>",
    level=common_settings.LOG_LEVEL,
    colorize=True,
)
if common_settings.LOG_FILE_ENABLED:
    logger.add(
        f"../../logs/{validator_settings.WALLET_NAME}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | [VALID] | {level: <8} | {name}:{line} | {message} | {extra}",
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        colorize=False,
    )


def main():
    """Main entry point for the validator."""
    logger.info("Starting validator")
    logger.info(f"Wallet: {validator_settings.WALLET_NAME}")
    logger.info(f"Hotkey: {validator_settings.WALLET_HOTKEY}")
    logger.info(f"Device: {validator_settings.DEVICE}")

    try:
        # Create validator instance
        validator = Validator(
            wallet_name=validator_settings.WALLET_NAME, wallet_hotkey=validator_settings.WALLET_HOTKEY
        )

        # Run the validator
        asyncio.run(validator.run_validator())

    except KeyboardInterrupt:
        logger.info("Validator stopped by user")
    except Exception as e:
        logger.error(f"Error running validator: {e}")
        raise


if __name__ == "__main__":
    main()
