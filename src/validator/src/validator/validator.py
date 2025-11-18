import asyncio
import random
import time
from typing import Literal, Optional
import bittensor as bt
import numpy as np
import torch
from aiohttp import web
from bittensor_wallet import Wallet
from loguru import logger

from common import settings as common_settings
from common.models.api_models import ValidationTaskResponse, ValidatorRegistrationResponse, ValidatorTask, SubnetScores
from common.models.run_flags import RunFlags
from common.validator.base_validator import BaseValidator
from common.models.ml_models import ModelConfig, ModelMetadata
from subnet.base.base_neuron import BaseNeuron
from subnet.test_client import TestAPIClient
from subnet.utils.bt_utils import get_subtensor
from subnet.utils.s3_torch import download_tensor
from subnet.validator_api_client import ValidatorAPIClient
from validator import settings as validator_settings
from validator.utils.utils import compute_cosine_similarity, compute_magnitude_ratio

PENALTY_RATE = 3


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


class Validator(BaseNeuron, HealthServerMixin, BaseValidator):
    def __init__(self, wallet_name: str | None = None, wallet_hotkey: str | None = None, wallet: Wallet | None = None):
        super().__init__()
        self.init_neuron(wallet_name=wallet_name, wallet_hotkey=wallet_hotkey, wallet=wallet)

        self.available: bool = True
        self.tracked_miner_hotkey: str | None = None  # hotkey
        self.model_cfg: ModelConfig | None = None
        self.model_metadata: ModelMetadata | None = None
        self.weight_version: str | None = None
        self.external_ip: str | None = None
        self.run_flags: RunFlags | None = None

        # Circuit breaker state
        self._orchestrator_failure_count: int = 0
        self._last_orchestrator_failure_time: float = 0

        # Metrics
        self._tasks_processed: int = 0
        self._tasks_failed: int = 0
        self._last_heartbeat: float = time.time()

        self.subtensor = get_subtensor()
        self.metagraph = bt.metagraph(netuid=common_settings.NETUID, lite=False, network=common_settings.NETWORK)

        self.burn_factor: float = common_settings.FALLBACK_BURN_FACTOR

        if common_settings.BITTENSOR:
            try:
                uid = (
                    self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address) if not common_settings.MOCK else None
                )
            except ValueError:
                logger.warning(
                    f"Hotkey {self.wallet.hotkey.ss58_address} not registered on Subnet {common_settings.NETUID} // network: {common_settings.NETWORK} // mock: {common_settings.MOCK}"
                )
        else:
            logger.info(f"Validator {self.hotkey[:8]} registered on metagraph")

    async def test_task(self, reason: str) -> ValidationTaskResponse:
        # randoms score between 0 and 1
        score: float = random.random()
        return ValidationTaskResponse(success=True, score=score, reason=reason, task_type="test_task")

    async def validate_activations(
        self,
        validator_activation_path: str,
        miner_activation_path: str,
        direction: Literal["forward", "backward"],
    ) -> ValidationTaskResponse:
        """
        Validate the activations of the miner against the validator's activations.
        First checks magnitude ratio as a gatekeeper, then cosine similarity if magnitude check passes.
        """

        validator_activations: torch.Tensor = await download_tensor(
            path=validator_activation_path, device=validator_settings.DEVICE
        )
        miner_activations: torch.Tensor = await download_tensor(
            path=miner_activation_path, device=validator_settings.DEVICE
        )

        # Flatten tensors for validation
        validator_flat = validator_activations.flatten().to(validator_settings.DEVICE)
        miner_flat = miner_activations.flatten().to(validator_settings.DEVICE)

        # Calculate norms for logging
        validator_norm = torch.norm(validator_flat).item()
        miner_norm = torch.norm(miner_flat).item()

        if common_settings.MOCK:
            # In mock mode, use simple cosine similarity
            cosine_similarity = compute_cosine_similarity(validator_flat, miner_flat)
            passed = (cosine_similarity > validator_settings.COSINE_SIMILARITY_THRESHOLD).item()
            score = 1 if passed else -1
            return ValidationTaskResponse(success=passed, score=score, task_type="validate_activations")

        # Step 1: Magnitude ratio check (gatekeeper) to minimize computation
        magnitude_ratio_value = compute_magnitude_ratio(validator_flat=validator_flat, miner_flat=miner_flat, eps=1e-8)

        if magnitude_ratio_value < validator_settings.ACTIVATION_MAGNITUDE_THRESHOLD:
            logger.warning(
                f"GRADIENT VALIDATOR [MINER {self.tracked_miner_hotkey}]: MAGNITUDE CHECK FAILED - "
                f"ratio: {magnitude_ratio_value:.4f}, validator_norm: {validator_norm:.4f}, "
                f"miner_norm: {miner_norm:.4f}, direction: {direction}"
            )

            # self.miner_scores[self.tracked_miner] -= PENALTY_RATE #TODO: Add penalty rate for miners that don't do well...
            return ValidationTaskResponse(
                success=False,
                score=-PENALTY_RATE,
                task_type="validate_activations",
                reason=f"Magnitude ratio: {magnitude_ratio_value:.4f}",
            )

        # Step 2: Cosine similarity check (only if magnitude check passed)
        cosine_similarity = compute_cosine_similarity(validator_flat=validator_flat, miner_flat=miner_flat)
        passed = (cosine_similarity > validator_settings.COSINE_SIMILARITY_THRESHOLD).item()
        score = 1 if passed else -PENALTY_RATE
        return ValidationTaskResponse(
            success=passed,
            score=score,
            task_type="validate_activations",
            reason=f"Cosine similarity: {cosine_similarity:.4f}",
        )

    async def validate_optimizer_state(
        self, validator_optimizer_state: torch.Tensor, miner_optimizer_state: torch.Tensor
    ) -> bool:
        similarity: float = compute_cosine_similarity(
            validator_flat=validator_optimizer_state.flatten().to(validator_settings.DEVICE),
            miner_flat=miner_optimizer_state.flatten().to(validator_settings.DEVICE),
        )
        passed = similarity > validator_settings.OPTIMIZER_SIMILARITY_THRESHOLD

        logger.debug(
            f"Validator optimizer state validation for {self.tracked_miner_hotkey[:8]}: {passed}, similarity: {similarity}"
        )
        return passed

    async def _validator_loop(self):
        """
        Main validator loop that handles registration, health checks, and task processing.
        """
        logger.info(f"🔄 Starting validator loop for {self.hotkey[:8]}")

        while True:
            try:
                # Check if we need to register or re-register (also triggered when the orchestrator tells us to reset)
                if not self._is_properly_registered():
                    await self._handle_registration()
                    continue

                # Check orchestrator health before proceeding
                logger.debug("Checking orchestrator health")
                if not await self._check_orchestrator_health():
                    logger.warning(
                        f"⏳ Orchestrator health check failed for validator {self.hotkey[:8]}, sleeping for {validator_settings.ORCHESTRATOR_HEALTH_CHECK_INTERVAL} seconds"
                    )
                    await asyncio.sleep(validator_settings.ORCHESTRATOR_HEALTH_CHECK_INTERVAL)
                    continue

                # Fetch and execute tasks
                await self._process_tasks()

            except Exception as e:
                logger.exception(f"Error in validator main loop: {e}")

            finally:
                logger.info(f"🔄 Validator loop sleeping for {validator_settings.FETCH_TASKS_INTERVAL} seconds")
                await asyncio.sleep(validator_settings.FETCH_TASKS_INTERVAL)

    async def weight_loop(self):
        """
        Enhanced weight loop with better error handling and logging.
        """
        loop_count = 0
        logger.info(f"🔄 Starting weight loop for validator {self.hotkey[:8]}")

        while True:
            loop_count += 1
            try:
                logger.debug(f"Weight loop iteration {loop_count} starting")

                # Reload the metagraph to get the latest weights, must use lite=False to get the latest weights
                self.metagraph = bt.metagraph(
                    netuid=int(common_settings.NETUID), lite=False, network=common_settings.NETWORK
                )

                logger.debug(f"GRADIENT VALIDATOR [MINER {self.tracked_miner_hotkey}]: WEIGHT LOOP RUNNING")
                if await ValidatorAPIClient.check_orchestrator_health(hotkey=self.wallet.hotkey):
                    logger.debug(f"GRADIENT VALIDATOR [MINER {self.tracked_miner_hotkey}]: GETTING GLOBAL MINER SCORES")
                    global_weights: dict | None = await ValidatorAPIClient.get_global_miner_scores(
                        hotkey=self.wallet.hotkey
                    )

                    if not global_weights or not isinstance(global_weights, dict):
                        raise Exception("No global weights received from orchestrator")

                    if "error_name" in global_weights:
                        logger.error(f"Error getting global weights: {global_weights['error_name']}")
                        global_weights = {}
                    else:
                        global_weights = SubnetScores.model_validate(global_weights)

                        logger.debug(
                            f"GRADIENT VALIDATOR [MINER {self.tracked_miner_hotkey}]: GLOBAL MINER SCORES: {global_weights}"
                        )

                        # Safer type conversion
                        try:
                            global_weights = {int(m.uid): m.weight for m in global_weights.miner_scores}
                        except (ValueError, TypeError) as e:
                            logger.error(f"Invalid UID in global_weights: {e}")
                            global_weights = {}

                else:
                    logger.warning("Orchestrator is not healthy, skipping weight submission")
                    global_weights = {}

                # Submit global weights to Bittensor
                if len(global_weights) > 0:
                    logger.debug(f"Received global weights: {global_weights}")
                    await self.set_weights(weights=global_weights)
                else:
                    logger.warning("No global weights received, temporarily copying weights from the chain")
                    await self.set_weights(weights=self.copy_weights_from_chain())

                logger.debug(f"Weight loop iteration {loop_count} completed successfully")

            except TimeoutError as e:
                logger.error(f"TimeoutError in weight loop iteration {loop_count}: {e}")
                await self.set_weights(weights=self.copy_weights_from_chain())

            except Exception as e:
                logger.exception(f"Error in weight loop iteration {loop_count}: {e}")
                await self.set_weights(weights=self.copy_weights_from_chain())

            finally:
                logger.info(
                    f"💤 Weight submission loop sleeping for {validator_settings.WEIGHT_SUBMIT_INTERVAL} seconds 💤"
                )
                await asyncio.sleep(validator_settings.WEIGHT_SUBMIT_INTERVAL)

    async def run_validator(self):
        """
        Run the validator with robust task management. Responsible for:
        - Starting the healthcheck server
        - Managing both weight_loop and validator_loop tasks
        - Monitoring tasks for failures and restarting them
        - Proper error logging and recovery
        """

        logger.info("🚀 Starting validator with robust task management")

        # Initial setup - this only happens once
        if not common_settings.BITTENSOR:
            logger.info(f"🔄 Registering validator {self.hotkey[:8]} to metagraph")
            await TestAPIClient.register_to_metagraph(hotkey=self.wallet.hotkey, role="validator")

        # Start the healthcheck server
        if validator_settings.LAUNCH_HEALTH:
            await self._start_health_server()
            logger.info(f"🏥 Health server started for validator {self.hotkey[:8]}")
        else:
            logger.warning(
                "⚠️ Validator healthcheck API not configured in settings (VALIDATOR_HEALTH_PORT missing). Skipping."
            )

        # Task management state
        self._weight_task = None
        self._validator_task = None
        task_restart_count = {"weight_loop": 0, "validator_loop": 0}
        max_restarts = 10  # Prevent infinite restart loops
        restart_delay = 5  # Seconds to wait before restarting a failed task
        status_log_interval = 300  # Log status every 5 minutes
        last_status_log = 0

        # Main task monitoring loop
        while True:
            try:
                current_time = time.time()

                # Log task status periodically
                if current_time - last_status_log > status_log_interval:
                    self._log_task_status(
                        weight_task=self._weight_task,
                        validator_task=self._validator_task,
                        task_restart_count=task_restart_count,
                    )
                    last_status_log = current_time

                # Create tasks if they don't exist or have completed/failed
                if self._weight_task is None or self._weight_task.done():
                    if self._weight_task is not None and self._weight_task.done():
                        try:
                            # Check if the task completed with an exception
                            self._weight_task.result()
                            logger.info("Weight loop task completed normally")
                        except Exception as e:
                            logger.exception(f"❌ Weight loop task failed: {e}")
                            task_restart_count["weight_loop"] += 1

                            if task_restart_count["weight_loop"] >= max_restarts:
                                logger.critical(f"Weight loop has failed {max_restarts} times, giving up")
                                raise Exception(f"Weight loop exceeded maximum restart attempts ({max_restarts})")

                    logger.info(
                        f"🔄 Starting/restarting weight loop task (attempt {task_restart_count['weight_loop'] + 1})"
                    )
                    self._weight_task = asyncio.create_task(self.weight_loop())

                if self._validator_task is None or self._validator_task.done():
                    if self._validator_task is not None and self._validator_task.done():
                        try:
                            # Check if the task completed with an exception
                            self._validator_task.result()
                            logger.info("Validator loop task completed normally")
                        except Exception as e:
                            logger.exception(f"❌ Validator loop task failed: {e}")
                            task_restart_count["validator_loop"] += 1

                            if task_restart_count["validator_loop"] >= max_restarts:
                                logger.critical(f"Validator loop has failed {max_restarts} times, giving up")
                                raise Exception(f"Validator loop exceeded maximum restart attempts ({max_restarts})")

                    logger.info(
                        f"🔄 Starting/restarting validator loop task (attempt {task_restart_count['validator_loop'] + 1})"
                    )
                    self._validator_task = asyncio.create_task(self._validator_loop())

                # Wait for either task to complete (indicating failure since they run forever)
                logger.debug("🔍 Monitoring tasks for failures...")
                done, pending = await asyncio.wait(
                    [self._weight_task, self._validator_task], return_when=asyncio.FIRST_COMPLETED
                )

                # Log which task(s) completed
                for task in done:
                    if task == self._weight_task:
                        logger.warning("⚠️ Weight loop task completed unexpectedly")
                    elif task == self._validator_task:
                        logger.warning("⚠️ Validator loop task completed unexpectedly")

                # Wait a bit before restarting to prevent rapid restart loops
                if restart_delay > 0:
                    logger.info(f"⏳ Waiting {restart_delay} seconds before restarting failed tasks...")
                    await asyncio.sleep(restart_delay)

            except Exception as e:
                logger.exception(f"Critical error in validator task manager: {e}")

                # Cancel any running tasks before retrying
                if self._weight_task and not self._weight_task.done():
                    self._weight_task.cancel()
                    try:
                        await self._weight_task
                    except asyncio.CancelledError:
                        pass

                if self._validator_task and not self._validator_task.done():
                    self._validator_task.cancel()
                    try:
                        await self._validator_task
                    except asyncio.CancelledError:
                        pass

                # Reset tasks to None so they get recreated
                self._weight_task = None
                self._validator_task = None

                # Wait before retrying
                await asyncio.sleep(10)

    async def set_weights(self, weights: dict[int, float]):
        """
        Sets the validator weights to the metagraph hotkeys based on the global weights.
        """
        logger.info("Attempting to set weights to Bittensor.")
        if not common_settings.BITTENSOR:
            logger.warning("Bittensor is not enabled, skipping weight submission")
            return

        if not hasattr(self, "wallet") or not self.wallet:
            logger.warning("Wallet not initialized, skipping weight submission")
            return

        if not hasattr(self, "subtensor") or not self.subtensor:
            logger.warning("Subtensor not initialized, skipping weight submission")
            return

        if not hasattr(self, "metagraph") or not self.metagraph:
            logger.warning("Metagraph not initialized, skipping weight submission")
            return

        try:
            # Convert global weights to tensor, Global state of scores is on the orchestrator
            scores = torch.zeros(len(self.metagraph.uids), dtype=torch.float32)
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
                burn_factor = next(
                    (weight for uid, weight in weights.items() if uid == common_settings.OWNER_UID), None
                )
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
                uids=self.metagraph.uids,
                weights=raw_weights.detach().cpu().float().numpy(force=True).astype(np.float32),
                netuid=int(common_settings.NETUID),
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )

            # Log the weights being set
            weight_dict = dict(zip(processed_weight_uids.tolist(), processed_weights.tolist()))
            logger.info(f"Setting weights for {len(weight_dict)} miners")
            logger.debug(f"Weight details: {weight_dict}")

            # Submit weights to Bittensor chain
            success, response = self.subtensor.set_weights(
                wallet=self.wallet,
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

    def copy_weights_from_chain(self) -> dict[int, float]:
        """Copy weights from the chain to the validator.

        Returns:
            dict[int, float]: A dictionary of weights for each miner.
        """
        meta: bt.metagraph = bt.metagraph(
            netuid=int(common_settings.NETUID), lite=False, network=common_settings.NETWORK
        )
        valid_indices = np.where(meta.validator_permit)[0]
        valid_weights = meta.weights[valid_indices]
        valid_stakes = meta.stake[valid_indices]
        normalized_stakes = valid_stakes / np.sum(valid_stakes)
        stake_weighted_average = np.dot(normalized_stakes, valid_weights).astype(float).tolist()

        # This is for the special case of testnet.
        if len(meta.uids) == 0:
            logger.warning("No valid indices found in metagraph, returning empty weights")
            return {}

        return dict(zip(meta.uids, list(stake_weighted_average)))

    async def register_with_orchestrator(self) -> None:
        try:
            response: ValidatorRegistrationResponse = await ValidatorAPIClient.register_validator_request(
                hotkey=self.wallet.hotkey
            )

            self.layer = int(response.layer)
            self.tracked_miner_hotkey = response.miner_hotkey_to_track
            self.available = False
            self.model_cfg = response.model_cfg
            self.model_metadata = response.model_metadata
            self.run_id = response.run_id
            self.run_flags = response.run_flags

        except Exception as e:
            raise e

    def _is_properly_registered(self) -> bool:
        """
        Check if the validator is properly registered and configured.
        """
        return (
            hasattr(self, "layer")
            and self.layer is not None
            and hasattr(self, "tracked_miner_hotkey")
            and self.tracked_miner_hotkey is not None
            # and hasattr(self, "available")
            # and self.available is not True
        )

    async def _handle_registration(self) -> None:
        """
        Handle initial registration or re-registration with the orchestrator.
        """
        logger.info(f"🔄 Attempting to register validator {self.hotkey[:8]} with orchestrator...")

        try:
            await self.register_with_orchestrator()
            logger.success(f"✅ Validator {self.hotkey[:8]} registered successfully in layer {self.layer}")

            # Setup local model
            if not await self._setup_local_model(
                layer=self.layer,
                device=validator_settings.DEVICE,
                model_weights=None,
                model_config=self.model_cfg.model_dump(),
                model_metadata=self.model_metadata.model_dump(),
                optimizer_state=None,
            ):
                raise Exception("Error setting up local model")

            logger.success(f"🖥️  Validator {self.hotkey[:8]} model setup completed for layer {self.layer}")
            return

        except Exception as e:
            logger.exception(f"Error during registration: {e}")
            await asyncio.sleep(validator_settings.FETCH_TASKS_INTERVAL)

    async def _check_orchestrator_health(self) -> bool:
        """
        Check if the orchestrator is healthy.
        """
        logger.info(f"🔄 Checking orchestrator health for validator {self.hotkey[:8]}")
        current_time = time.time()

        try:
            is_healthy = await ValidatorAPIClient.check_orchestrator_health(hotkey=self.wallet.hotkey)

            if is_healthy:
                logger.success(f"✅ Orchestrator health check passed for validator {self.hotkey[:8]}")
                return True
            else:
                self._orchestrator_failure_count += 1
                self._last_orchestrator_failure_time = current_time
                return False

        except Exception as e:
            logger.warning(f"Orchestrator health check failed: {e}")
            self._orchestrator_failure_count += 1
            self._last_orchestrator_failure_time = current_time
            return False

    async def _process_tasks(self):
        """
        Fetch and process tasks from the orchestrator.
        """
        try:
            logger.info(f"🔄 Fetching tasks for validator {self.hotkey[:8]}")
            tasks: list[dict] = await self.fetch_tasks()

            logger.debug(f"tasks: {tasks}, length: {len(tasks)}")

            if tasks is None or len(tasks) == 0:
                logger.debug(
                    f"⏳ No tasks found for validator {self.hotkey[:8]}, sleeping for {validator_settings.FETCH_TASKS_INTERVAL} seconds"
                )
                return

            logger.info(f"📋 Processing {len(tasks)} tasks for validator {self.hotkey[:8]}")

            # Process tasks sequentially to avoid conflicts
            for task in tasks:
                try:
                    logger.debug(f"Executing task: {task['function_name']} with args: {task['inputs']}")
                    result: ValidationTaskResponse = await self._execute_task(task=ValidatorTask(**task))
                    await ValidatorAPIClient.submit_task_result(hotkey=self.wallet.hotkey, task_result=result)
                except Exception as e:
                    logger.exception(f"Error executing task {task['function_name']}: {e}")
                    # Continue with next task instead of failing completely

        except Exception as e:
            logger.exception(f"Error fetching tasks: {e}")

    async def _execute_task(self, task: ValidatorTask) -> ValidationTaskResponse:
        """
        Execute a single task with proper validation.
        """
        task_name = task.function_name

        # Validate task before execution
        if not self._is_properly_registered():
            logger.warning(f"Cannot execute task {task_name}: validator not properly registered")
            self._tasks_failed += 1
            return ValidationTaskResponse(
                success=False, score=0, reason="Validator not properly registered", task_type=task_name
            )

        # Check the method defined in the task_name
        if hasattr(self, task_name):
            task_func = getattr(self, task_name)
            try:
                result: ValidationTaskResponse = await task_func(**task.inputs)

                if not result.success:
                    logger.warning(f"Task {task_name} failed: {result.reason}")
                    self._tasks_failed += 1
                    return result

                logger.debug(f"Task {task_name} completed successfully")
                self._tasks_processed += 1
                self._last_heartbeat = time.time()
                return result
            except Exception as e:
                logger.exception(f"Task {task_name} failed: {e}")
                self._tasks_failed += 1
                raise
        else:
            logger.warning(f"Task function {task_name} not found on validator")
            self._tasks_failed += 1
            return ValidationTaskResponse(
                success=False, score=0, reason=f"Task function {task_name} not found on validator", task_type=task_name
            )

    async def fetch_tasks(self):
        tasks: list[dict] = await ValidatorAPIClient.fetch_tasks(hotkey=self.wallet.hotkey)
        return tasks

    async def reset_validator(self) -> ValidationTaskResponse:
        """
        reset_validator is a generic function that can be called to clear the validator state, but
        is typically called when the validator needs to start tracking a new miner, change layer, etc..

        Upon reset, the validator will submit its current miner scores to the orchestrator, and then clear its state.
        """
        logger.info(f"🧽 Orchestrator requested validator reset on {self.hotkey[:8]} 🧽")

        try:
            self.model_manager.reset()

            self.layer = None
            self.available = True
            self.tracked_miner_hotkey = None

            return ValidationTaskResponse(
                success=True, reason="Validator reset successfully", task_type="reset_validator", score=0
            )

        except Exception as e:
            logger.exception(f"GRADIENT VALIDATOR [MINER {self.tracked_miner_hotkey}]: Error resetting validator: {e}")
            return ValidationTaskResponse(
                success=False,
                reason=f"Error resetting validator {self.hotkey[:8]} with error: {e}",
                task_type="reset_validator",
                score=0,
            )

    async def get_validator_status(self) -> dict:
        """
        Get current validator status for monitoring, including task states.
        """
        status = {
            "hotkey": self.hotkey[:8] if hasattr(self, "hotkey") else "N/A",
            "layer": getattr(self, "layer", None),
            "tracked_miner_hotkey": getattr(self, "tracked_miner_hotkey", None),
            "available": getattr(self, "available", True),
            "registered": self._is_properly_registered(),
            "orchestrator_failure_count": self._orchestrator_failure_count,
            "tasks_processed": self._tasks_processed,
            "tasks_failed": self._tasks_failed,
            "last_heartbeat": self._last_heartbeat,
            "uptime": time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 0,
        }

        # Add task status if available
        if hasattr(self, "_weight_task") and self._weight_task:
            status["weight_task_running"] = not self._weight_task.done()
            status["weight_task_cancelled"] = self._weight_task.cancelled()
        else:
            status["weight_task_running"] = False

        if hasattr(self, "_validator_task") and self._validator_task:
            status["validator_task_running"] = not self._validator_task.done()
            status["validator_task_cancelled"] = self._validator_task.cancelled()
        else:
            status["validator_task_running"] = False

        return status

    def _log_task_status(self, weight_task: asyncio.Task, validator_task: asyncio.Task, task_restart_count: dict):
        """
        Log the current status of both tasks for debugging.
        """
        weight_status = "None"
        if weight_task:
            if weight_task.done():
                weight_status = "Done/Failed"
            elif weight_task.cancelled():
                weight_status = "Cancelled"
            else:
                weight_status = "Running"

        validator_status = "None"
        if validator_task:
            if validator_task.done():
                validator_status = "Done/Failed"
            elif validator_task.cancelled():
                validator_status = "Cancelled"
            else:
                validator_status = "Running"

        logger.info(
            f"📊 Task Status - Weight: {weight_status} (restarts: {task_restart_count['weight_loop']}), "
            f"Validator: {validator_status} (restarts: {task_restart_count['validator_loop']})"
        )


if __name__ == "__main__":
    gradient_validator = Validator(
        wallet_name=validator_settings.WALLET_NAME, wallet_hotkey=validator_settings.WALLET_HOTKEY
    )
    asyncio.run(gradient_validator.run_validator())
