import os
from dotenv import load_dotenv
from loguru import logger

from common import settings as common_settings


DOTENV_PATH = os.getenv("DOTENV_PATH", ".env")
load_dotenv(dotenv_path=DOTENV_PATH)


def detect_device() -> str:
    """Detect the most capable torch device available on the host."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch import failure on non-runtime environments
        logger.debug(f"Unable to import torch for device detection: {exc}")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch, "backends", None)
    if mps_backend is not None:
        mps = getattr(mps_backend, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"

    mps_module = getattr(torch, "mps", None)
    if mps_module is not None:
        is_available = getattr(mps_module, "is_available", None)
        if callable(is_available) and is_available():
            return "mps"

    return "cpu"


# Wallet
WALLET_NAME = os.getenv("MINER_WALLET", "test")
WALLET_HOTKEY = os.getenv("MINER_HOTKEY", "m1")

MINER_HEALTH_HOST = os.getenv("MINER_HEALTH_HOST", "0.0.0.0")
MINER_HEALTH_PORT = int(os.getenv("MINER_HEALTH_PORT", 9000))
MINER_HEALTH_ENDPOINT = os.getenv("MINER_HEALTH_ENDPOINT", "/health")

LAUNCH_HEALTH = os.getenv("LAUNCH_HEALTH") == "True"

DEFAULT_DEVICE = os.getenv("DEVICE") or detect_device()
os.environ.setdefault("DEVICE", DEFAULT_DEVICE)

# Training settings
TIMEOUT = int(os.getenv("MINER_TIMEOUT", "300"))  # 5 minutes default
PACK_SAMPLES = os.getenv("PACK_SAMPLES", "True") == "True"  # not for miner's to change
N_PARTITION_BATCHES = int(os.getenv("N_PARTITION_BATCHES", "20"))  # not for miner's to change
PREVIOUS_WEIGHTS = os.getenv("MODEL_DIR", "./weights")

# Activation settings
P2P_ACTIVATION_CACHE_TTL = int(os.getenv("P2P_ACTIVATION_CACHE_TTL", 3000))  # seconds
MAX_FORWARD_ACTIVATIONS_IN_QUEUE = int(
    os.getenv("MAX_FORWARD_ACTIVATIONS_IN_QUEUE", common_settings.MAX_FORWARD_ACTIVATIONS_IN_QUEUE)
)
MIN_FORWARD_ACTIVATIONS_IN_QUEUE = int(
    os.getenv("MIN_FORWARD_ACTIVATIONS_IN_QUEUE", common_settings.MIN_FORWARD_ACTIVATIONS_IN_QUEUE)
)


ACTIVATION_SEND_MAX_TRIES = int(os.getenv("ACTIVATION_SEND_MAX_TRIES", "5"))
ACTIVATION_SEND_DEADLINE_SECONDS = float(os.getenv("ACTIVATION_SEND_DEADLINE_SECONDS", str(30 * 60)))
ACTIVATION_PUSH_TIMEOUT_SECONDS = float(os.getenv("ACTIVATION_PUSH_TIMEOUT_SECONDS", "60"))
ACTIVATION_SEND_CONCURRENCY = int(os.getenv("ACTIVATION_SEND_CONCURRENCY", "8"))

PEER_STATUS_BROADCAST_INTERVAL_SECONDS = float(os.getenv("PEER_STATUS_BROADCAST_INTERVAL_SECONDS", "5.0"))

# How often the miner re-pulls the run's flags + hyperparameters from the
# orchestrator so operator changes (e.g. cache size) apply without a restart.
RUN_CONFIG_REFRESH_INTERVAL_SECONDS = float(os.getenv("RUN_CONFIG_REFRESH_INTERVAL_SECONDS", "300"))

VISUALIZATION_API_URL = os.getenv("VISUALIZATION_API_URL", "http://localhost:8009")
VISUALIZATION_AUTO_OPEN = os.getenv("VISUALIZATION_AUTO_OPEN", "true").lower() in ("1", "true", "yes", "on")

# Miner contribution filtering - minimum local optimizer steps required for contribution to be included
MIN_LOCAL_OPTIMIZER_STEPS = int(os.getenv("MIN_LOCAL_OPTIMIZER_STEPS", "5"))

# Training settings
LOCAL_BATCH_SIZE = int(
    os.getenv("LOCAL_BATCH_SIZE", "8")
)  # Splits the minibatch further into even smaller local batches to avoid running out of memory
# Probing the largest local batch that fits (and shrinking on OOM) is gated by the
# `auto_local_batch_size` run flag; see TrainingPhase.calibrate_local_batch_size.
PSEUDO_GRADIENTS_BATCH_SIZE = int(os.getenv("PSEUDO_GRADIENTS_BATCH_SIZE", "100"))

# Determines whether the miner is mounted within a host electron app
_electron_host_pid_raw = os.getenv("ELECTRON_HOST_PID")
try:
    ELECTRON_HOST_PID = int(_electron_host_pid_raw) if _electron_host_pid_raw else None
except ValueError:
    ELECTRON_HOST_PID = None
IS_MOUNTED = ELECTRON_HOST_PID is not None or os.getenv("IS_MOUNTED") == "true"
ELECTRON_VERSION = os.getenv("ELECTRON_VERSION")
NODE_CONTROL_TOKEN = os.getenv("NODE_CONTROL_TOKEN")
REGISTRATION_SPEED_TEST_CACHE_TTL_SEC = float(os.getenv("REGISTRATION_SPEED_TEST_CACHE_TTL_SEC", "3600"))

# Telemetry settings
TELEMETRY_ENABLED = os.getenv("TELEMETRY_ENABLED", "true").lower() in ("1", "true", "yes", "on")
SYNC_POLL_TICK = float(os.getenv("SYNC_POLL_TICK", "2.0"))

TELEMETRY_FLUSH_INTERVAL_SEC = float(os.getenv("TELEMETRY_FLUSH_INTERVAL_SEC", "15"))
TELEMETRY_MAX_BUFFER_SIZE = int(os.getenv("TELEMETRY_MAX_BUFFER_SIZE", "1000"))
