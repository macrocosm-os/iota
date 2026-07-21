from hashlib import sha256
import bittensor as bt
from bittensor.core.subtensor import Subtensor
from bittensor_wallet import Keypair
from bittensor_wallet.mock import get_mock_wallet
from loguru import logger
import tenacity

from common import settings as common_settings


def _log_retry_attempt(retry_state):
    """Log when a retry attempt is made."""
    attempt_number = retry_state.attempt_number
    logger.warning(f"🔄 Retry attempt {attempt_number} for getting subtensor on network {common_settings.NETWORK}")


def create_subtensor_client() -> bt.Subtensor:
    """Build a subtensor client honoring custom endpoints if provided."""
    # When SUBTENSOR_ENDPOINT is set, pass it directly as the network parameter
    # This is required because bt.Subtensor(network="local") hardcodes 127.0.0.1:9944
    # and ignores the config's chain_endpoint
    if common_settings.SUBTENSOR_ENDPOINT:
        logger.info(f"🔄 Using custom subtensor endpoint: {common_settings.SUBTENSOR_ENDPOINT}")
        return Subtensor(network=common_settings.SUBTENSOR_ENDPOINT)

    return bt.Subtensor(network=common_settings.NETWORK)


# retry but if it fails, it will raise an error
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=_log_retry_attempt,
)
def get_subtensor() -> bt.Subtensor:
    logger.info(f"🔄 Getting subtensor for network: {common_settings.NETWORK}")
    if common_settings.BITTENSOR:
        logger.info("🔄 Using subtensor")
        return create_subtensor_client()
    else:
        raise Exception("No subtensor found")


def get_wallet(wallet_name: str, wallet_hotkey: str) -> bt.Wallet:
    """Get a Bittensor wallet.

    Args:
        wallet_name: The name of the wallet
        wallet_hotkey: The hotkey of the wallet
    """
    logger.info(
        f"Initializing Bittensor wallet: {wallet_name} and hotkey: {wallet_hotkey}. Bittensor is set to {common_settings.BITTENSOR}"
    )
    if common_settings.BITTENSOR:
        wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey)
        return wallet
    else:
        return get_mock_wallet(
            hotkey=Keypair.create_from_seed(seed=sha256(wallet_name.encode()).hexdigest()),
            coldkey=Keypair.create_from_seed(seed=sha256(wallet_hotkey.encode()).hexdigest()),
        )
