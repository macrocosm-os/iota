#!/usr/bin/env python3
"""
Simple script to launch multiple miners concurrently.
"""

import argparse
import asyncio
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import signal
from datetime import datetime
from multiprocessing import Process
import time
from urllib.parse import urlparse
import io
import contextlib

from loguru import logger
import bittensor as bt


# The multiprocessing logging QueueListener crashes with EOFError when the queue
# disappears during interpreter shutdown. That shows up as a noisy traceback in
# start-miners-uv. Patch dequeue to treat EOF as a sentinel so the monitor
# thread exits quietly.
def _patch_queue_listener_eof_handling() -> None:
    listener_cls = logging.handlers.QueueListener
    if getattr(listener_cls, "_iota_eof_patched", False):
        return

    original_dequeue = listener_cls.dequeue

    def _safe_dequeue(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return original_dequeue(self, *args, **kwargs)
        except EOFError:
            return self._sentinel

    listener_cls.dequeue = _safe_dequeue  # type: ignore[assignment]
    listener_cls._iota_eof_patched = True  # type: ignore[attr-defined]


_patch_queue_listener_eof_handling()

# Add the miner package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
# Add the shared common package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "common", "src"))

from common import settings as common_settings
from common.settings import BITTENSOR

from miner.new_miner import Miner


def get_available_hotkeys(wallet_name: str = "swarm-test") -> list[str]:
    """
    Get available hotkeys for a given wallet from btcli w list command.

    Args:
        wallet_name: The wallet name to search for hotkeys

    Returns:
        List of hotkey names
    """
    try:
        logger.info(f"Getting hotkeys for wallet '{wallet_name}' from btcli...")
        result = subprocess.run(["btcli", "w", "list"], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            logger.error(f"btcli command failed: {result.stderr}")
            return []

        output = result.stdout
        hotkeys = []

        # Look for the wallet section and extract hotkeys
        lines = output.split("\n")
        in_target_wallet = False

        for line in lines:
            # Check if we're entering the target wallet section
            if f"Coldkey {wallet_name}" in line:
                in_target_wallet = True
                continue

            # Check if we're entering a different wallet section
            if in_target_wallet and "Coldkey " in line and wallet_name not in line:
                break

            # Extract hotkey names when in the target wallet section
            if in_target_wallet and "Hotkey " in line:
                # Pattern to match: "│   ├── Hotkey miner-XX"
                match = re.search(r"Hotkey\s+([^\s]+)", line)
                if match:
                    hotkey_name = match.group(1)
                    hotkeys.append(hotkey_name)
                    logger.debug(f"Found hotkey: {hotkey_name}")

        logger.info(f"Found {len(hotkeys)} hotkeys for wallet '{wallet_name}': {hotkeys}")
        return hotkeys

    except subprocess.TimeoutExpired:
        logger.error("btcli command timed out")
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"btcli command error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error getting hotkeys: {e}")
        return []


def run_single_miner_process(wallet_name: str, wallet_hotkey: str, miner_id: int):
    """Run a single miner instance in a separate process."""
    # Put this child (and anything it spawns — torch DataLoader workers,
    # bittensor subprocesses, etc.) in its own process group so the parent's
    # signal handler can `killpg` the full tree atomically. Without this, a
    # hard kill of the parent (e.g. go-task's post-grace-period SIGKILL on
    # Ctrl+C) leaves the child and any of *its* children orphaned to init.
    try:
        os.setpgrp()
    except OSError:
        pass

    # Belt-and-suspenders: if the parent launcher is killed so abruptly that
    # it doesn't get to signal us (go-task SIGKILL racing against our
    # signal_handler), we'd be left running forever under init (PPID=1).
    # macOS has no PR_SET_PDEATHSIG, so we poll getppid() from a daemon
    # thread and self-terminate when it flips to 1.
    import threading

    _original_ppid = os.getppid()

    def _parent_death_watchdog() -> None:
        while True:
            time.sleep(2.0)
            try:
                if os.getppid() != _original_ppid or os.getppid() == 1:
                    # Parent went away — kill our whole process group (this
                    # child + any torch/bt subprocesses we spawned) and exit.
                    try:
                        os.killpg(os.getpgrp(), signal.SIGKILL)
                    except Exception:  # noqa: BLE001
                        pass
                    os._exit(0)
            except Exception:  # noqa: BLE001
                return

    _wd = threading.Thread(target=_parent_death_watchdog, daemon=True, name=f"miner_{miner_id}_pdwatchdog")
    _wd.start()

    logger.remove()
    logger.add(
        sys.stderr,
        format=f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | <level>{{level: <8}}</level> | <cyan>MINER-{miner_id}</cyan> | <level>{{message}}</level>",
        level=common_settings.LOG_LEVEL,
        colorize=True,
    )
    if common_settings.LOG_FILE_ENABLED:
        log_file = f"logs/miner_{wallet_hotkey}.log"
        if os.path.exists(log_file):
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_name = f"logs/miner_{wallet_hotkey}_archived_at_{current_time}.log"
            os.rename(log_file, archived_name)

        logger.add(
            log_file,
            format=f"{{time:YYYY-MM-DD HH:mm:ss.SSS}} | {{level: <8}} | MINER-{miner_id} | {{message}}",
            level="DEBUG",  # Log file always captures DEBUG for post-hoc analysis
            rotation="10 MB",
            retention="10 days",
            colorize=False,
        )

    def _is_local_subtensor() -> bool:
        endpoint = common_settings.SUBTENSOR_ENDPOINT or ""
        parsed = urlparse(endpoint)
        host = parsed.hostname or ""
        return common_settings.NETWORK == "local" or host in {"127.0.0.1", "localhost"}

    # Provision local wallets/hotkeys if missing when pointing at a local subtensor (helps local dev).
    wallet: bt.wallet | None = None
    if common_settings.BITTENSOR and _is_local_subtensor():
        wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        if not wallet.coldkey_file.exists_on_device():
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                wallet.create_new_coldkey(use_password=False, overwrite=True)
            logger.info(f"Created coldkey for wallet {wallet_name}")
        if not wallet.hotkey_file.exists_on_device():
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                wallet.create_new_hotkey(use_password=False, overwrite=True)
            logger.info(f"Created hotkey {wallet_hotkey} for wallet {wallet_name}")

    async def run_miner():
        try:
            logger.info(f"Starting miner {miner_id} with wallet_name={wallet_name}, wallet_hotkey={wallet_hotkey}")
            miner = Miner(wallet_name=wallet_name, wallet_hotkey=wallet_hotkey, wallet=wallet)
            await miner.run_miner()
        except Exception as e:
            logger.exception(f"Error in miner {miner_id}: {e}")
            raise

    try:
        asyncio.run(run_miner())
    except KeyboardInterrupt:
        logger.info(f"Miner {miner_id} received shutdown signal")
    except Exception as e:
        logger.error(f"Miner {miner_id} failed: {e}")
        sys.exit(1)


def launch_multiple_miners(num_miners: int, wallet_name_prefix: str = "miner", wallet_hotkey_prefix: str = "hotkey"):
    """Launch multiple miners in separate processes."""
    processes = []

    # Get available hotkeys if BITTENSOR mode is enabled
    available_hotkeys = []
    if BITTENSOR:
        available_hotkeys = get_available_hotkeys("swarm-test")
        if not available_hotkeys:
            logger.error("No hotkeys found for wallet 'swarm-test'. Falling back to prefix-based naming.")

    for i in range(num_miners):
        if BITTENSOR and available_hotkeys:
            # Use predefined wallet and hotkeys when BITTENSOR is enabled
            wallet_name = "swarm-test"
            if i < len(available_hotkeys):
                wallet_hotkey = available_hotkeys[i]
            else:
                logger.warning(f"Not enough available hotkeys for miner {i}. Using fallback naming.")
                wallet_hotkey = f"{wallet_hotkey_prefix}_{i}"
        else:
            # Use the original naming convention
            wallet_name = f"{wallet_name_prefix}_{i}"
            wallet_hotkey = f"{wallet_hotkey_prefix}_{i}"

        # Create a process for each miner
        process = Process(target=run_single_miner_process, args=(wallet_name, wallet_hotkey, i), name=f"miner_{i}")
        processes.append(process)
        process.start()
        logger.info(f"Started miner {i} in process {process.pid}")

        # Small delay between launches to prevent overwhelming the system
        time.sleep(0.1)

    logger.info(f"Launched {num_miners} miners in separate processes. Waiting for completion...")

    # Guard against re-entry: if a second Ctrl+C / SIGTERM arrives while the
    # handler is still running we want to `os._exit` immediately rather than
    # start the grace period over.
    _shutting_down = {"flag": False}

    # Launcher-level parent-death watchdog. Workers already have their own
    # (run_single_miner_process), but the LAUNCHER itself can be orphaned in
    # the e2e flow: the e2e driver Popens us with start_new_session=True, so
    # if the driver is SIGKILL'd (e.g. go-task post-grace), our ppid flips
    # to 1 (launchd) but no signal is delivered. Without this watchdog the
    # launcher keeps `process.join()`-ing forever and workers are orphaned.
    # macOS has no PR_SET_PDEATHSIG, so we poll. Manual `task start-miners-uv`
    # is unaffected: ppid stays stable until the user Ctrl+Cs, at which point
    # the regular SIGINT path tears things down before the watchdog ticks.
    import threading as _threading

    _launcher_initial_ppid = os.getppid()

    def _launcher_parent_watchdog() -> None:
        while True:
            time.sleep(2.0)
            try:
                ppid = os.getppid()
            except Exception:  # noqa: BLE001
                return
            if ppid == _launcher_initial_ppid and ppid != 1:
                continue
            # Parent went away. Best-effort kill of every worker pgrp, then
            # exit. We deliberately do NOT call signal_handler() here — it
            # logs and runs a 3s grace period, which is wasted work when
            # the driver is already gone.
            for p in processes:
                if p.pid is None:
                    continue
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            os._exit(0)

    _threading.Thread(target=_launcher_parent_watchdog, daemon=True, name="launcher_pdwatchdog").start()

    def _killpg(pid: int, sig: int) -> None:
        """Best-effort killpg of the pgrp owned by `pid`.

        Children call `os.setpgrp()` in `run_single_miner_process` so each
        has its own group; signalling that group tears down torch DataLoader
        workers / bittensor subprocesses / anything else the miner spawned.
        """
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return
        # Refuse to signal our own group — would nuke the launcher itself.
        if pgid == os.getpgrp():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            return
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def signal_handler(signum, frame):
        if _shutting_down["flag"]:
            logger.warning("Second shutdown signal received — exiting immediately")
            os._exit(1)
        _shutting_down["flag"] = True
        logger.info(f"Received shutdown signal ({signum}). Stopping all miners...")

        for process in processes:
            if process.pid is None:
                continue
            logger.info(f"SIGTERM miner pgid of pid {process.pid}")
            _killpg(process.pid, signal.SIGTERM)

        # Short grace period — anything clean exits here. We intentionally do
        # NOT consult process.is_alive() before the SIGKILL sweep; it has
        # been observed to return False while the OS process is still
        # running, causing the kill to be skipped and the child to survive.
        time.sleep(3)

        for process in processes:
            if process.pid is None:
                continue
            logger.warning(f"SIGKILL miner pgid of pid {process.pid}")
            _killpg(process.pid, signal.SIGKILL)

        # Reap whatever exited so the process table stays tidy.
        for process in processes:
            try:
                process.join(timeout=1.0)
            except Exception:  # noqa: BLE001
                pass

        logger.info("All miners stopped")
        # os._exit bypasses atexit handlers, threading teardown, and any
        # blocking destructors in torch / bittensor that could hang us long
        # enough for go-task to SIGKILL us with children still running.
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # SIGHUP fires when the controlling terminal closes (e.g. user closes
    # the tab with miners still running) — same cleanup path.
    signal.signal(signal.SIGHUP, signal_handler)

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch multiple miners concurrently")
    parser.add_argument("--num-miners", type=int, default=3, help="Number of miners to launch (default: 3)")
    parser.add_argument(
        "--wallet-name-prefix", type=str, default="miner", help="Prefix for wallet names (default: 'miner')"
    )
    parser.add_argument(
        "--wallet-hotkey-prefix", type=str, default="hotkey", help="Prefix for wallet hotkeys (default: 'hotkey')"
    )
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log level (default: DEBUG)")

    args = parser.parse_args()

    # Configure logging for the main process
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>MAIN</cyan> | <level>{message}</level>",
        level=common_settings.LOG_LEVEL,
        colorize=True,
    )
    if common_settings.LOG_FILE_ENABLED:
        log_file = "logs/multiple_miners_main.log"
        if os.path.exists(log_file):
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_name = f"logs/multiple_miners_main_archived_at_{current_time}.log"
            os.rename(log_file, archived_name)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | MAIN | {message}",
            level="DEBUG",  # Log file always captures DEBUG for post-hoc analysis
            rotation="10 MB",
            retention="10 days",
            colorize=False,
        )

    if BITTENSOR:
        logger.info("BITTENSOR mode enabled - using wallet 'swarm-test' with dynamically retrieved hotkeys")
        # Get hotkeys early to show count in logs
        hotkeys = get_available_hotkeys("swarm-test")
        if hotkeys:
            logger.info(f"Available hotkeys: {len(hotkeys)}")
            if args.num_miners > len(hotkeys):
                logger.warning(f"Requested {args.num_miners} miners but only {len(hotkeys)} hotkeys available")
        else:
            logger.warning("No hotkeys found - will fall back to prefix-based naming")
    else:
        logger.info("BITTENSOR mode disabled - using prefix-based naming")

    logger.info(f"Launching {args.num_miners} miners in separate processes...")
    logger.info(f"Wallet name prefix: {args.wallet_name_prefix}")
    logger.info(f"Wallet hotkey prefix: {args.wallet_hotkey_prefix}")

    try:
        launch_multiple_miners(
            num_miners=args.num_miners,
            wallet_name_prefix=args.wallet_name_prefix,
            wallet_hotkey_prefix=args.wallet_hotkey_prefix,
        )
    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
