"""Remember this miner's registration queue entry across restarts.

Every POST /miner/register creates a fresh entry at the back of the orchestrator's
waitlist, so a miner that restarts (crash, update, reboot) used to lose its place.
We persist the queue_id per hotkey and, on the next registration, poll that entry
first. Only when the orchestrator reports it failed, expired or unknown do we join
the queue again.

Everything here is best effort: a missing or corrupt file just means a normal
registration.
"""

import json
import os
import time
from pathlib import Path

from loguru import logger
from platformdirs import user_data_dir

APP_NAME = "IOTA Train at Home"

# Entries older than this are assumed to have been dropped server-side.
QUEUE_ENTRY_MAX_AGE_SECONDS = float(os.getenv("REGISTRATION_QUEUE_ENTRY_MAX_AGE_SECONDS", str(7 * 24 * 3600)))


def store_path() -> Path:
    override = os.getenv("REGISTRATION_QUEUE_STORE_PATH")
    if override:
        return Path(override)
    return Path(user_data_dir(APP_NAME, "Macrocosmos")) / "registration_queue.json"


def _read_all() -> dict:
    try:
        data = json.loads(store_path().read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.debug(f"Ignoring unreadable registration queue store: {e}")
        return {}


def _write_all(data: dict) -> None:
    path = store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
    except Exception as e:
        logger.debug(f"Failed to write registration queue store: {e}")


def load_queue_id(hotkey: str | None, p2p_node_id: str | None = None) -> str | None:
    """Return the saved queue_id for this hotkey, or None if absent, stale or for another node."""
    if not hotkey:
        return None
    entry = _read_all().get(hotkey)
    if not isinstance(entry, dict) or not entry.get("queue_id"):
        return None
    if time.time() - float(entry.get("saved_at", 0)) > QUEUE_ENTRY_MAX_AGE_SECONDS:
        clear_queue_id(hotkey)
        return None
    # The queue entry carries the P2P node id it was created with; resuming it from a
    # different node would register an address peers can't reach.
    if p2p_node_id and entry.get("p2p_node_id") and entry["p2p_node_id"] != p2p_node_id:
        clear_queue_id(hotkey)
        return None
    return entry["queue_id"]


def save_queue_id(hotkey: str | None, queue_id: str, p2p_node_id: str | None = None) -> None:
    if not hotkey or not queue_id:
        return
    data = _read_all()
    existing = data.get(hotkey)
    if isinstance(existing, dict) and existing.get("queue_id") == queue_id:
        return
    data[hotkey] = {"queue_id": queue_id, "p2p_node_id": p2p_node_id, "saved_at": time.time()}
    _write_all(data)


def clear_queue_id(hotkey: str | None) -> None:
    if not hotkey:
        return
    data = _read_all()
    if data.pop(hotkey, None) is not None:
        _write_all(data)
