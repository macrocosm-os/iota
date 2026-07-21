from common import settings as common_settings
from common.models.api_models import SubnetScores
from subnet.validator_api_client import ValidatorAPIClient
import torch
import numpy as np
import bittensor as bt
from loguru import logger
from bittensor import Wallet, Subtensor


async def weight_setting_step(wallet: Wallet, subtensor: Subtensor, metagraph: bt.Metagraph):
    global_weights = await ValidatorAPIClient.get_global_miner_scores(hotkey=wallet.hotkey)

    if not global_weights or "error_name" in global_weights:
        if global_weights and "error_name" in global_weights:
            logger.error(f"Error getting global weights: {global_weights['error_name']}")
        logger.warning("Falling back to copying weights from the chain")
        weights = copy_weights_from_chain(metagraph=metagraph)
        await set_weights(wallet=wallet, subtensor=subtensor, weights=weights, metagraph=metagraph)
        return

    global_weights = SubnetScores.model_validate(global_weights)
    logger.debug(f"GRADIENT VALIDATOR: GLOBAL MINER SCORES: {global_weights}")

    # Safer type conversion
    # TODO for Nick: Brian would like this simplified to what it was before (as a dictionary comprehension)
    # the reason being, that the logic should be on orchestartor side - not validator.  However,
    # doing this, would require a refactor the MinerScore data type which expects a single run_id
    # for each miner score record.
    global_weights_dict: dict[int, float] = {}
    try:
        for m in global_weights.miner_scores:
            k = int(m.uid)
            global_weights_dict[k] = global_weights_dict.get(k, 0.0) + (m.weight or 0.0)
    except Exception:
        logger.error("Error converting global weights to dictionary")
        raise

    await set_weights(wallet=wallet, subtensor=subtensor, weights=global_weights_dict, metagraph=metagraph)


async def set_weights(wallet: Wallet, subtensor: Subtensor, weights: dict[int, float], metagraph: bt.Metagraph):
    """
    Sets the validator weights to the metagraph hotkeys based on the global weights.
    """

    logger.info("Attempting to set weights to Bittensor.")
    if not common_settings.BITTENSOR:
        logger.warning("Bittensor is not enabled, skipping weight submission")
        return

    if not wallet:
        logger.warning("Wallet not initialized, skipping weight submission")
        return

    if not subtensor:
        logger.warning("Subtensor not initialized, skipping weight submission")
        return

    if not metagraph:
        logger.warning("Metagraph not initialized, skipping weight submission")
        return

    try:
        # Convert global weights to tensor, Global state of scores is on the orchestrator
        scores = torch.zeros(len(metagraph.uids), dtype=torch.float32)
        for uid, weight in weights.items():
            scores[uid] = weight

        # Check if scores contains any NaN values
        if torch.isnan(scores).any():
            logger.warning("Scores contain NaN values. Replacing with 0.")
            scores = torch.nan_to_num(scores, 0)

        # Check if we have any non-zero scores
        if torch.sum(scores) == 0:
            logger.warning("All scores are zero, skipping weight submission")
            return

        # Normalize weights
        raw_weights = torch.nn.functional.normalize(scores, p=1, dim=0)

        # Fetch the burn factor from the weights
        try:
            burn_factor = next((weight for uid, weight in weights.items() if uid == common_settings.OWNER_UID), None)
        except Exception as e:
            logger.warning(f"Error fetching burn factor: {e}")
            burn_factor = None
        if burn_factor is None:
            burn_factor = 0

        # Process the raw weights to final_weights via subtensor limitations
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=metagraph.uids,
            weights=raw_weights.detach().cpu().float().numpy(force=True).astype(np.float32),
            netuid=int(common_settings.NETUID),
            subtensor=subtensor,
            metagraph=metagraph,
        )

        # Log the weights being set
        weight_dict = dict(zip(processed_weight_uids.tolist(), processed_weights.tolist()))
        logger.info(f"Setting weights for {len(weight_dict)} miners")
        logger.debug(f"Weight details: {weight_dict}")

        # Submit weights to Bittensor chain
        success, response = subtensor.set_weights(
            wallet=wallet,
            netuid=int(common_settings.NETUID),
            uids=processed_weight_uids,
            weights=processed_weights,
            wait_for_finalization=False,
            version_key=common_settings.__VALIDATOR_SPEC_VERSION__,
        )

        if success:
            logger.success("Successfully submitted weights to Bittensor.")
            logger.debug(f"Response: {response}")
        else:
            logger.error("Failed to submit weights to Bittensor")
            logger.error(f"Response: {response}")

    except Exception as e:
        logger.exception(f"Error submitting weights to Bittensor: {e}")


def copy_weights_from_chain(metagraph: bt.Metagraph) -> dict[int, float]:
    """Copy weights from the chain to the validator.

    Returns:
        dict[int, float]: A dictionary of weights for each miner.
    """

    valid_indices = np.where(metagraph.validator_permit)[0]
    valid_weights = metagraph.weights[valid_indices]
    valid_stakes = metagraph.stake[valid_indices]
    normalized_stakes = valid_stakes / np.sum(valid_stakes)
    stake_weighted_average = np.dot(normalized_stakes, valid_weights).astype(float).tolist()

    # This is for the special case of testnet.
    if len(metagraph.uids) == 0:
        logger.warning("No valid indices found in metagraph, returning empty weights")
        return {}

    return dict(zip(metagraph.uids, list(stake_weighted_average)))
