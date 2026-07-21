import os
import time
import asyncio
from typing import Optional
from aiohttp import web
from bittensor_wallet import Wallet
from loguru import logger
import bittensor as bt

from common import settings as common_settings
from common.models.api_models import ValidatorTask
from subnet.base.base_neuron import BaseNeuron
from subnet.utils.bt_utils import get_subtensor
from subnet.validator_api_client import ValidatorAPIClient
from validator import settings as validator_settings
from validator.utils.task_execution import execute_task
from validator.utils.weight_setting import set_weights, copy_weights_from_chain, weight_setting_step


class HealthServerMixin:
    health_app_runner: Optional[web.AppRunner] = None
    health_site: Optional[web.TCPSite] = None

    async def _start_health_server(self):
        """Starts the aiohttp web server for healthchecks."""
        app = web.Application()

        async def health_handler(request):
            return web.json_response(
                {
                    "status": "healthy",
                    "hotkey": getattr(self, "hotkey", "N/A"),
                    "layer": getattr(self, "layer", "N/A"),
                    "uid": getattr(self, "uid", "N/A"),
                    "registered": getattr(self, "reregister_needed", True) is False,
                    "timestamp": time.time(),
                }
            )

        app.router.add_get(validator_settings.VALIDATOR_HEALTH_ENDPOINT, health_handler)

        self.health_app_runner = web.AppRunner(app)
        await self.health_app_runner.setup()

        self.health_site = web.TCPSite(
            self.health_app_runner, validator_settings.VALIDATOR_HEALTH_HOST, validator_settings.VALIDATOR_HEALTH_PORT
        )
        if validator_settings.LAUNCH_HEALTH:
            await self.health_site.start()
            logger.info(
                f"Miner {getattr(self, 'hotkey', 'N/A')} healthcheck API started on "
                f"http://{validator_settings.VALIDATOR_HEALTH_HOST}:{validator_settings.VALIDATOR_HEALTH_PORT}{validator_settings.VALIDATOR_HEALTH_ENDPOINT}"
            )


class Validator(BaseNeuron, HealthServerMixin):
    def __init__(self, wallet_name: str | None = None, wallet_hotkey: str | None = None, wallet: Wallet | None = None):
        super().__init__()
        self.init_neuron(wallet_name=wallet_name, wallet_hotkey=wallet_hotkey, wallet=wallet)

        if common_settings.BITTENSOR:
            self.subtensor = get_subtensor()
            self.metagraph = bt.Metagraph(
                netuid=int(common_settings.NETUID), lite=False, network=common_settings.NETWORK
            )

    async def weight_loop(self):
        """
        Enhanced weight loop with better error handling and logging.
        """
        loop_count = 0
        logger.info(f"🔄 Starting weight loop for validator {self.hotkey[:8]}")

        while True:
            try:
                loop_count += 1
                self.metagraph.sync(subtensor=self.subtensor, lite=False)

                try:
                    logger.debug(f"Weight loop iteration {loop_count} starting")
                    await weight_setting_step(wallet=self.wallet, subtensor=self.subtensor, metagraph=self.metagraph)

                except TimeoutError as e:
                    logger.error(f"TimeoutError in weight loop iteration {loop_count}: {e}")

                except Exception as e:
                    logger.exception(f"Error in weight loop iteration {loop_count}: {e}")
                    weights = copy_weights_from_chain(metagraph=self.metagraph)
                    await set_weights(
                        wallet=self.wallet, subtensor=self.subtensor, weights=weights, metagraph=self.metagraph
                    )

                logger.info(
                    f"💤 Weight submission loop sleeping for {validator_settings.WEIGHT_SUBMIT_INTERVAL} seconds 💤"
                )
                await asyncio.sleep(validator_settings.WEIGHT_SUBMIT_INTERVAL)

            except Exception as e:
                logger.exception(f"Error in weight loop: {e}")

    async def task_loop(self):
        """
        Task loop for the validator.
        """
        logger.info(f"Getting validator code for validator {self.hotkey[:8]}")
        validator_code = await ValidatorAPIClient.get_validator_code(hotkey=self.wallet.hotkey)
        logger.info(f"Validator code: {validator_code}")

        # Unzip the validator code into a temporary directory
        import tempfile
        import zipfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the zip file to a temporary location
            temp_zip_path = os.path.join(temp_dir, "validator_code.zip")
            with open(temp_zip_path, "wb") as f:
                f.write(validator_code)

            # Extract the zip file
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            logger.info(f"Extracted validator code to {temp_dir}")

            # The extracted files are now in temp_dir
            # They will be automatically deleted when exiting this context
            while True:
                try:
                    task: ValidatorTask | None = await ValidatorAPIClient.fetch_task(hotkey=self.wallet.hotkey)
                    if task is not None:
                        result = await execute_task(task, validator_code_dir=temp_dir)
                        await ValidatorAPIClient.submit_task_result(hotkey=self.wallet.hotkey, task_result=result)
                    else:
                        logger.warning(f"No task found for validator {self.hotkey[:8]}")
                        await asyncio.sleep(validator_settings.FETCH_TASKS_INTERVAL)
                except Exception as e:
                    logger.exception(f"Error in task loop: {e}")

    async def run_validator(self):
        """
        Run the validator with robust task management. Responsible for:
        - Starting the healthcheck server
        - Managing both weight_loop and validator_loop tasks
        - Monitoring tasks for failures and restarting them
        - Proper error logging and recovery
        """

        logger.info("🚀 Starting validator with robust task management")

        # Start the healthcheck server
        if validator_settings.LAUNCH_HEALTH:
            await self._start_health_server()
            logger.info(f"🏥 Health server started for validator {self.hotkey[:8]}")
        else:
            logger.warning(
                "⚠️ Validator healthcheck API not configured in settings (VALIDATOR_HEALTH_PORT missing). Skipping."
            )

        # Main task monitoring loop
        if common_settings.BITTENSOR:
            asyncio.create_task(self.weight_loop())

        asyncio.create_task(self.task_loop())
        while True:
            await asyncio.sleep(1)
