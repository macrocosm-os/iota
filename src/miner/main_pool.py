import multiprocessing

# this is neccessary to avoid accidental multiprocessing fork-bomb when compiled
if __name__ == "__main__":
    multiprocessing.freeze_support()
    APP_NAME = "iota-miner"

import argparse
import asyncio
import logging
import os
import warnings
from loguru import logger

from common.utils.gpu_process_utils import kill_stale_gpu_processes, cleanup_stale_shared_memory

_ORIGINAL_LOGURU_ADD = logger.add
_ORIGINAL_LOGURU_REMOVE = logger.remove


def _install_loguru_null_sink() -> None:
    """Permanently silence loguru output for the pool miner process."""
    _ORIGINAL_LOGURU_REMOVE()
    _ORIGINAL_LOGURU_ADD(lambda message: None)

    def _noop_add(*_args, **_kwargs):
        return 0

    def _noop_remove(*_args, **_kwargs):
        return None

    logger.add = _noop_add  # type: ignore[assignment]
    logger.remove = _noop_remove  # type: ignore[assignment]


# _install_loguru_null_sink()

from miner.pool.miner import Miner


def main():
    """Main entry point for the miner pool miner."""
    parser = argparse.ArgumentParser(description="Run the miner pool miner.")
    parser.add_argument(
        "--auto-start",
        dest="auto_start",
        action="store_true",
        help="Start mining immediately using saved configuration.",
    )
    parser.add_argument(
        "--no-auto-start",
        dest="auto_start",
        action="store_false",
        help="Require interactive confirmation before starting even if --auto-start was provided earlier.",
    )
    parser.add_argument(
        "--wallet",
        dest="wallet",
        help="Coldkey name to use. If omitted, defaults to the built-in pool miner wallet.",
    )
    parser.add_argument(
        "--hotkey",
        dest="hotkey",
        help="Hotkey name to use. If omitted, defaults to the built-in pool miner hotkey.",
    )
    parser.add_argument(
        "--payout-coldkey",
        dest="payout_coldkey",
        help="Payout coldkey ss58 address. Use 'creators' to keep default routing.",
    )
    parser.add_argument(
        "--dashboard",
        dest="show_dashboard",
        action="store_true",
        help="Enable the live terminal dashboard (default).",
    )
    parser.add_argument(
        "--no-dashboard",
        dest="show_dashboard",
        action="store_false",
        help="Disable the live terminal dashboard and re-enable log output.",
    )
    parser.add_argument(
        "--no-btcli",
        dest="use_btcli",
        action="store_false",
        help="Skip all btcli integrations and load the local IOTA wallet naively.",
    )
    parser.add_argument(
        "--use-btcli",
        dest="use_btcli",
        action="store_true",
        help="Force-enable btcli integrations if previously disabled.",
    )
    parser.set_defaults(auto_start=False, show_dashboard=True, use_btcli=True)
    args = parser.parse_args()

    # Fall back to MINER_WALLET / MINER_HOTKEY env vars when the CLI flags
    # aren't provided. Lets pm2 (or any orchestrator) pin a unique wallet/hotkey
    # per instance without re-templating the launch script — required for running
    # multiple pool miners on the same host with distinct hotkeys.
    if not args.wallet:
        args.wallet = os.environ.get("MINER_WALLET") or None
    if not args.hotkey:
        args.hotkey = os.environ.get("MINER_HOTKEY") or None

    kill_stale_gpu_processes()
    cleanup_stale_shared_memory()

    payout_override: str | None = None
    if args.payout_coldkey is not None:
        value = args.payout_coldkey.strip()
        if value and value.lower() not in {"creators", "default"}:
            payout_override = value
        else:
            payout_override = None

    btcli_disabled = not args.use_btcli

    try:
        if args.show_dashboard:
            _install_loguru_null_sink()
            # Disable all standard library logging (including uvicorn/FastAPI)
            logging.disable(logging.CRITICAL)
            # Explicitly suppress uvicorn and FastAPI loggers (they may set up handlers before disable)
            uvicorn_logger = logging.getLogger("uvicorn")
            uvicorn_access_logger = logging.getLogger("uvicorn.access")
            fastapi_logger = logging.getLogger("fastapi")
            uvicorn_logger.setLevel(logging.CRITICAL)
            uvicorn_access_logger.setLevel(logging.CRITICAL)
            fastapi_logger.setLevel(logging.CRITICAL)
            # Remove any existing handlers from these loggers
            uvicorn_logger.handlers.clear()
            uvicorn_access_logger.handlers.clear()
            fastapi_logger.handlers.clear()
            logging.captureWarnings(False)
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module=r"huggingface_hub\.file_download",
            )

        async def _run():
            miner = Miner(
                wallet_name=args.wallet,
                wallet_hotkey=args.hotkey,
                auto_start=args.auto_start,
                payout_coldkey=payout_override,
                show_dashboard=args.show_dashboard,
                btcli_disabled=btcli_disabled,
            )
            await miner.run_miner()

        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    except Exception:
        # Re-raise with suppressed logging so upstream tooling can handle the failure.
        raise


if __name__ == "__main__":
    main()
