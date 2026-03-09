from datetime import datetime
from typing import Optional
import bittensor as bt
import torch
from loguru import logger

from common.models.error_models import BaseErrorModel
from common.models.miner_models import MinerStatus
from common.utils.partitions import MinerPartition
from common.utils.shared_states import LayerPhase
from subnet.model.model_mixin import ModelManager
from subnet.utils.bt_utils import get_wallet
from subnet.common_api_client import CommonAPIClient
from subnet.utils.partition_utils import download_merged_partitions


class BaseNeuron:
    def __init__(self):
        super().__init__()
        self.wallet: bt.wallet | None = None
        self.layer: Optional[int] = None
        self.status: str = MinerStatus.IDLE.value
        self.registration_time: str = datetime.now().isoformat()
        self.merge_status: str = LayerPhase.TRAINING.value
        self.epoch_on_registration: int | None = None
        self.run_id: str | None = None

        self.model_manager = ModelManager()
        self.num_partitions: int | None = None

    def init_neuron(self, wallet_name: str = None, wallet_hotkey: str = None, wallet: bt.wallet | None = None):
        self.wallet = wallet or get_wallet(
            wallet_name=wallet_name,
            wallet_hotkey=wallet_hotkey,
        )
        self.hotkey = self.wallet.hotkey.ss58_address
        return self

    async def _setup_local_model(
        self,
        model_config: dict,
        model_metadata: dict,
        model_weights: torch.Tensor | None,
        optimizer_state: dict | None,
        layer: int,
        device: str,
    ) -> bool:
        try:
            logger.info(f"📥 Attempting to load model for layer {layer} for {self.hotkey[:8]}")
            await self.model_manager.initialize_model_manager(
                model_config=model_config,
                model_metadata=model_metadata,
                model_weights=model_weights,
                optimizer_state=optimizer_state,
                layer=layer,
                device=device,
                logger_attributes={
                    "hotkey": self.hotkey,
                    "run_id": self.run_id,
                },
            )

            # this is a deterministic number based on the config.
            logger.info(f"Number of partitions used for butterfly-reduce: {self.num_partitions}")

        except Exception as e:
            logger.error(f"❌ Error loading model for miner {self.hotkey[:8]}: {e}")
            return False
        return True

    async def download_and_set_global_weights(self, client: CommonAPIClient, device: str) -> torch.Tensor | None:
        """
        Downloads the weights and optimizer state from the db and sets them to the model

        Args:
            client: The client to use to get the merged partitions
            device: The device to use to download the weights and optimizer state
        """
        try:
            logger.info(f"Downloading and setting weights for miner {self.hotkey[:8]}")
            old_weights = torch.nn.utils.parameters_to_vector(self.model_manager.model.parameters()).clone()

            merged_partitions: list[MinerPartition] | BaseErrorModel = await client.get_merged_partitions(
                hotkey=self.wallet.hotkey
            )
            logger.debug(f"Merged partitions: {len(merged_partitions) if merged_partitions else 'None'}")
            if isinstance(merged_partitions, BaseErrorModel):
                logger.error(
                    f"Error getting merged partitions {merged_partitions.error_name}: {merged_partitions.error_dict}"
                )
                raise RuntimeError(f"Failed to get merged partitions: {merged_partitions.error_name}") from None

            # Download new weights
            new_weights = await download_merged_partitions(
                merged_partitions=merged_partitions,
                target_tensor=old_weights,
                device=device,
                layer=self.layer,
                num_partitions=self.num_partitions,
                download_type="weights",
            )

            if new_weights is None:
                logger.warning("No new weights or optimizer state downloaded")
                raise RuntimeError("Failed to download global weights: no weights returned")

            # Set new weights and optimizer state to model
            await self.model_manager.set_model_weights_and_optimizer_state(
                model_weights=new_weights, optimizer_state=None
            )

        except Exception as e:
            logger.exception(f"Error downloading and setting weights and optimizer state: {e}")
            raise
