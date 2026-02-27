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

# Activation settings - miners can reduce if they are OOM'ing but can't surpass common settings
MAX_ACTIVATION_CACHE_SIZE = int(os.getenv("MAX_ACTIVATION_CACHE_SIZE", common_settings.MAX_ACTIVATION_CACHE_SIZE))
MAX_FORWARD_ACTIVATIONS_IN_QUEUE = int(
    os.getenv("MAX_FORWARD_ACTIVATIONS_IN_QUEUE", common_settings.MAX_FORWARD_ACTIVATIONS_IN_QUEUE)
)
MIN_FORWARD_ACTIVATIONS_IN_QUEUE = int(
    os.getenv("MIN_FORWARD_ACTIVATIONS_IN_QUEUE", common_settings.MIN_FORWARD_ACTIVATIONS_IN_QUEUE)
)


VISUALIZATION_API_URL = os.getenv("VISUALIZATION_API_URL", "http://localhost:8009")
VISUALIZATION_AUTO_OPEN = os.getenv("VISUALIZATION_AUTO_OPEN", "true").lower() in ("1", "true", "yes", "on")

# Miner contribution filtering - minimum local optimizer steps required for contribution to be included
MIN_LOCAL_OPTIMIZER_STEPS = int(os.getenv("MIN_LOCAL_OPTIMIZER_STEPS", "5"))

# Training settings
LOCAL_BATCH_SIZE = int(
    os.getenv("LOCAL_BATCH_SIZE", "8")
)  # Splits the minibatch further into even smaller local batches to avoid running out of memory
PSEUDO_GRADIENTS_BATCH_SIZE = int(os.getenv("PSEUDO_GRADIENTS_BATCH_SIZE", "100"))

# Determines whether the miner is mounted within a host electron app
IS_MOUNTED = os.getenv("IS_MOUNTED") == "true"
ELECTRON_VERSION = os.getenv("ELECTRON_VERSION")

# Telemetry settings
TELEMETRY_ENABLED = os.getenv("TELEMETRY_ENABLED", "true").lower() in ("1", "true", "yes", "on")
TELEMETRY_FLUSH_INTERVAL_SEC = float(os.getenv("TELEMETRY_FLUSH_INTERVAL_SEC", "15"))
TELEMETRY_MAX_BUFFER_SIZE = int(os.getenv("TELEMETRY_MAX_BUFFER_SIZE", "1000"))
