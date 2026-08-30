import os
from pathlib import Path

from dotenv import load_dotenv

COMMON_DOTENV_PATH = os.getenv("COMMON_DOTENV_PATH", ".env")
_dotenv_path = Path(COMMON_DOTENV_PATH)
if _dotenv_path.exists():
    load_dotenv(dotenv_path=_dotenv_path)

# Generic settings
MOCK = os.getenv("MOCK") == "True"
LOG_FILE_ENABLED = os.getenv("LOG_FILE_ENABLED") == "True"
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
RUNTIME_IMAGE_VERSION = os.getenv("RUNTIME_IMAGE_VERSION")

# Bittensor settings
__SPEC_VERSION__ = 30017
__VALIDATOR_SPEC_VERSION__ = 5065
BITTENSOR = os.getenv("BITTENSOR", "True") == "True"
SUBTENSOR_ENDPOINT = os.getenv("SUBTENSOR_ENDPOINT")
MAX_NUM_PARTS = int(os.getenv("MAX_NUM_PARTS", 10000))
NETUID = int(os.getenv("NETUID", "9"))
NETWORK = os.getenv("NETWORK", "finney")
OWNER_UID = 209
FALLBACK_BURN_FACTOR = 0.8

# Orchestrator settings (common)
if NETWORK == "local":
    # Local development
    ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 8000))
    ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "localhost")
    ORCHESTRATOR_SCHEMA = os.getenv("ORCHESTRATOR_SCHEME", "http")

    BRIDGE_URL = os.getenv("BRIDGE_URL", "http://localhost:8001/sync")
elif NETWORK == "test":
    # Testnet
    ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 443))
    ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "iota-branch-main.api.macrocosmos.ai")
    ORCHESTRATOR_SCHEMA = os.getenv("ORCHESTRATOR_SCHEME", "https")

    BRIDGE_URL = os.getenv("BRIDGE_URL", "https://iota-branch-main.api.macrocosmos.ai/sync")
else:
    # Mainnet
    ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 443))
    ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "iota.api.macrocosmos.ai")
    ORCHESTRATOR_SCHEMA = os.getenv("ORCHESTRATOR_SCHEME", "https")

    BRIDGE_URL = os.getenv("BRIDGE_URL", "https://iota.api.macrocosmos.ai/sync")

BRIDGE_V2_URL = BRIDGE_URL.replace("/sync", "/bridge", 1).rstrip("/")

ORCHESTRATOR_URL = f"{ORCHESTRATOR_SCHEMA}://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}"
REQUEST_RETRY_COUNT = int(os.getenv("REQUEST_RETRY_COUNT", "3"))

# Bridge settings
CLIENT_REQUEST_TIMEOUT = int(os.getenv("CLIENT_REQUEST_TIMEOUT", "40"))  # TODO: Make this 20s

MIN_PART_SIZE = 10 * 1024 * 1024  # TODO: Change this to 16 -> 64 for Orion runs
MAX_PART_SIZE = 100 * 1024 * 1024  # 100MB
S3_UPLOAD_MAX_CONCURRENCY = int(os.getenv("S3_UPLOAD_MAX_CONCURRENCY", "16"))  # concurrent part PUTs per blob

# Self-describing blob wire format (see shared/common/src/common/utils/blob_format.py)
BLOB_VERSION = 1
BLOB_HEADER_LEN = 12  # version u16 + reserved u16 + trailer_offset u64

# System Settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
LRU_CACHE_TIMEOUT = 20  # seconds
ALL_LAYERS_TRAINING_CACHE_TTL_SEC = float(os.getenv("ALL_LAYERS_TRAINING_CACHE_TTL_SEC", "2"))

# Model Training Settings - not for miner's to change
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_NAME = "HuggingFaceFW/fineweb-edu-2"
SHUFFLE_DATASET = True
WEIGHT_DECAY = 1e-1
GRAD_CLIP_NORM = 1.0
LEARNING_RATE = 2 * 1e-4
MOMENTUM_DECAY = 1.0
BETAS = (0.9, 0.95)  # IMPORTANT: Beta1 is stage-dependent for NAdam, it will not be read from the settings.
EPS = 1e-8
TOTAL_TRAIN_STEPS = 100_000_000
LR_WARMUP_START_FACTOR = 1  # 5e-3
LR_WARMUP_STEPS = 1
LR_CONST_STEPS = 90_999_999
LR_TAIL_STEPS_FRAC = 0.02
LR_FINAL_FACTOR = 0.10
LR_SAW_CYCLE_LENGTH = 1000
NESTEROV_LEARNING_RATE = 0.7
NESTEROV_MOMENTUM = 0.9
NESTEROV_MOMENTUM_WARMUP_START = 0.5
NESTEROV_MOMENTUM_WARMUP_EPOCHS = 20

# Optimizer state sharing: how many completed epochs to search backwards for the
# most recently uploaded optimizer state. Floored at the upload interval so the
# window always covers at least one scheduled optimizer upload.
OPTIMIZER_UPLOAD_EPOCH_INTERVAL = int(os.getenv("OPTIMIZER_UPLOAD_EPOCH_INTERVAL", "1"))
OPTIMIZER_LOOKBACK_EPOCHS = max(
    int(os.getenv("OPTIMIZER_LOOKBACK_EPOCHS", "4")),
    OPTIMIZER_UPLOAD_EPOCH_INTERVAL,
)

# Activation settings - not for miner's to change
MAX_ACTIVATION_CACHE_SIZE = 16
N_BACKWARDS_FOR_CACHE_INCREASE_STOP = int(os.getenv("N_BACKWARDS_FOR_CACHE_INCREASE_STOP", 3))
MAX_FORWARD_ACTIVATIONS_IN_QUEUE = 2
MIN_FORWARD_ACTIVATIONS_IN_QUEUE = 1
MINI_BATCH_SIZE = 4  # default; per-run value is applied via set_mini_batch_size() from run metadata
MINI_BATCH_ACCUMULATION_COUNT = 4
SEQUENCE_LENGTH = 800


def set_mini_batch_size(value: int | None) -> None:
    """Apply a run's mini batch size process-wide (mirrors the RUN_FLAGS per-run pattern).

    Consumers read ``common_settings.MINI_BATCH_SIZE`` by attribute, so mutating it here makes
    them all see the per-run value. Callers must invoke this once per active run, at the point
    they learn the run's metadata (miner registration, validator task execution). No-op if
    ``value`` is falsy or unchanged.
    """
    global MINI_BATCH_SIZE
    if value and value != MINI_BATCH_SIZE:
        MINI_BATCH_SIZE = value


def set_mini_batch_accumulation_count(value: int | None) -> None:
    """Apply a run's mini batch accumulation count process-wide. No-op if value is falsy or unchanged."""
    global MINI_BATCH_ACCUMULATION_COUNT
    if value and value != MINI_BATCH_ACCUMULATION_COUNT:
        MINI_BATCH_ACCUMULATION_COUNT = value


def set_max_activation_cache_size(value: int | None) -> None:
    """Apply a run's max activation cache size process-wide. No-op if value is falsy or unchanged."""
    global MAX_ACTIVATION_CACHE_SIZE
    if value and value != MAX_ACTIVATION_CACHE_SIZE:
        MAX_ACTIVATION_CACHE_SIZE = value


def set_sequence_length(value: int | None) -> None:
    """Apply a run's sequence length process-wide. No-op if value is falsy or unchanged."""
    global SEQUENCE_LENGTH
    if value and value != SEQUENCE_LENGTH:
        SEQUENCE_LENGTH = value


ACTIVATION_TIMEOUT_SEC = int(os.getenv("ACTIVATION_TIMEOUT", "100") if not MOCK else 600)
ACTIVATION_CACHE_TIMEOUT_SEC = int(os.getenv("ACTIVATION_CACHE_TIMEOUT", "300"))

# P2P connection pool
P2P_MAX_SENDER_CONNECTIONS = int(os.getenv("P2P_MAX_SENDER_CONNECTIONS", "128"))
MAX_ACTIVATION_PROCESS_COUNT = 3

# Local mock model settings
MOCK_MODEL_INPUT_DIM = int(os.getenv("MOCK_MODEL_INPUT_DIM", "100"))
MOCK_MODEL_HIDDEN_DIM = int(os.getenv("MOCK_MODEL_HIDDEN_DIM", "32"))
MOCK_MODEL_BOTTLENECK_DIM = int(os.getenv("MOCK_MODEL_BOTTLENECK_DIM", "16"))

# Epoch level sync settings
DOWNLOAD_BATCH_SIZE = 50

S3_DOWNLOAD_TIMEOUT = 300
S3_UPLOAD_TIMEOUT = 300
CREATORS_PAYOUT_COLDKEY_PLACEHOLDER = (
    "5F4w5QQzf1aqiRjkzbZDTV9Ams9YdrBguPdLcui5FnMSVBjx"
    if NETWORK == "test"
    else "5DUS9gRskdxSLES9y4pF5iay2f8DWSFHe1ArT8CySGs36NMH"
)

MIN_PARTITION_DOWNLOAD_SUCCESS_PCT = 90  # TODO: Make this dynamic based on the model config?

# Merge quality validation thresholds
MIN_COSINE_SIMILARITY_DOWNLOAD = float(os.getenv("MIN_COSINE_SIMILARITY_DOWNLOAD", "0.9"))


def activation_cache_timeout_sec(
    num_layers: int, layer_idx: int, interval_sec: int = ACTIVATION_CACHE_TIMEOUT_SEC
) -> int:
    """
    Seconds to retain forward activations in the miner cache; longer for earlier layers.
    layer_idx is 0-indexed.
    """
    return interval_sec * (num_layers - layer_idx)


# Buffer window settings
BUFFER_WINDOW_BATCH_FRACTION = 0.05
