import os

from common import configs as model_configs
from dotenv import load_dotenv
from loguru import logger

COMMON_DOTENV_PATH = os.getenv("COMMON_DOTENV_PATH", ".env")
if not load_dotenv(dotenv_path=COMMON_DOTENV_PATH):
    logger.warning("No .env file found for common settings")

# Generic settings
MOCK = os.getenv("MOCK") == "True"
if MOCK:
    logger.warning("RUNNING IN MOCK MODE!")

LOG_FILE_ENABLED = os.getenv("LOG_FILE_ENABLED") == "True"
TEST_MODE = os.getenv("TEST_MODE") == "True"

# Bittensor settings
__SPEC_VERSION__ = 10_006
__VALIDATOR_SPEC_VERSION__ = 4065
BITTENSOR = os.getenv("BITTENSOR") == "True"
MAX_NUM_PARTS = int(os.getenv("MAX_NUM_PARTS", 10000))
NETUID = int(os.getenv("NETUID", "9"))
NETWORK = os.getenv("NETWORK", "finney")
OWNER_UID = 209

# Orchestrator settings (common)
if TEST_MODE:
    # Local testing
    ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 8000))
    ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "localhost")
    ORCHESTRATOR_SCHEMA = os.getenv("ORCHESTRATOR_SCHEME", "http")
elif NETWORK == "test":
    # Testnet
    ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 443))
    ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "iota-branch-main.api.macrocosmos.ai")
    ORCHESTRATOR_SCHEMA = os.getenv("ORCHESTRATOR_SCHEME", "https")
else:
    # Mainnet
    ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 443))
    ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "iota.api.macrocosmos.ai")
    ORCHESTRATOR_SCHEMA = os.getenv("ORCHESTRATOR_SCHEME", "https")

ORCHESTRATOR_URL = f"{ORCHESTRATOR_SCHEMA}://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}"
REQUEST_RETRY_COUNT = int(os.getenv("REQUEST_RETRY_COUNT", "3"))
CLIENT_REQUEST_TIMEOUT = int(os.getenv("CLIENT_REQUEST_TIMEOUT", "40"))  # TODO: Make this 20s

MIN_PART_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PART_SIZE = 100 * 1024 * 1024  # 100MB

# System Settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
ACTIVATION_CACHE_TIMEOUT = 60 * 20
MAX_ACTIVATION_CACHE_SIZE = 2
LRU_CACHE_TIMEOUT = 20  # seconds
MINER_SCORES_TIME_WINDOW = 2  # hours

# LLM Model Settings
MODEL_NAME = os.getenv("MODEL_NAME", "1b")
match MODEL_NAME.lower():
    case "100m":
        MODEL_CFG = model_configs.LLAMA32_CONFIG_100M
    case "1b":
        MODEL_CFG = model_configs.LLAMA32_CONFIG_1B
    case "3b":
        MODEL_CFG = model_configs.LLAMA32_CONFIG_3B
    case "5b":
        MODEL_CFG = model_configs.LLAMA32_CONFIG_5B
    case "7b":
        MODEL_CFG = model_configs.LLAMA32_CONFIG_7B
    case "8b":
        MODEL_CFG = model_configs.LLAMA32_CONFIG_8B
    case "12b":
        MODEL_CFG = model_configs.LLAMA32_CONFIG_12B
    case "15b":
        MODEL_CFG = model_configs.LLAMA32_CONFIG_15B
    case "21b":
        MODEL_CFG = model_configs.LLAMA32_CONFIG_21B
    case _:
        raise ValueError(f"Unknown model size: {MODEL_NAME}")

MODEL_SPLITS = MODEL_CFG.pop("model_splits")
N_LAYERS = len(MODEL_SPLITS)
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"

# Model Training Settings
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_NAME = "HuggingFaceFW/fineweb"
SHUFFLE_DATASET = True
EFFECTIVE_BATCH_SIZE = 2 if MOCK else 500
SEQUENCE_LENGTH = 512
WEIGHT_DECAY = 1e-1
GRAD_CLIP_NORM = 1.0
LEARNING_RATE = 2 * 1e-4
BETAS = (0.9, 0.95)
EPS = 1e-8
TOTAL_TRAIN_STEPS = 100_000_000
LR_WARMUP_START_FACTOR = 1  # 5e-3
LR_WARMUP_STEPS = 1
LR_CONST_STEPS = 90_999_999
LR_TAIL_STEPS_FRAC = 0.02
LR_FINAL_FACTOR = 0.10
LR_SAW_CYCLE_LENGTH = 1000
