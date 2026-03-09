import asyncio
import copy
import json
import multiprocessing
import os
import sys
import threading
import time
import webbrowser
from common.utils.verify_enclave_signature import payload_base64_from_obj
from loguru import logger
from miner.utils.node_control_mixin import NodeControlMixin
from miner.utils.miner_dashboard_api import start_visualization_server
from miner.utils.partition_merging import download_previous_optimizer_state_for_partition_batch, merge_partition_batch
from miner.utils.partition_merging import get_partition_batch
from miner.utils.partition_merging import download_pseudograds_for_partition_batch
from miner.utils.partition_merging import upload_partition_batch
from subnet.utils.partition_utils import save_model_weights_and_optimizer_state
from subnet.utils.vector_utils import reconstruct_optimizer_state, get_optimizer_tensor_shapes
from miner.utils.timer_logger import TimerLoggerMiner
from miner.telemetry import TelemetryBufferService
from miner.utils.stats import StatsTracker
import torch
import aiohttp
from bittensor import Wallet
from subnet.common_api_client import CommonAPIClient
from miner.health_server import HealthServerMixin
from miner.utils.partition_merging import (
    filter_bad_metadata,
    get_weight_partition_info,
)
from miner import settings as miner_settings
from miner.state_manager import StateManager
from miner.utils.utils import (
    create_metadata,
    upload_file,
    upload_tensor,
    wait_for_state,
)
from miner.utils.run_utils import identify_best_run
from miner.utils.attestation_utils import collect_attestation_payload, AttestationUnavailableError
from common.iroh.p2p_protocol import (
    P2PExpiredError,
    P2PNotFoundError,
    P2PRequestError,
    P2PResponseStatus,
    P2PUnauthorizedError,
    encode_activation_request,
    decode_activation_response,
)
from common.iroh import (
    DEFAULT_MAX_MESSAGE_SIZE,
    P2POperationTimings,
    P2PStack,
)
from common.models.api_models import (
    AttestationChallengeResponse,
    EnclaveSignResponse,
    MinerRegistrationResponse,
    MinerAttestationPayload,
    RegisterMinerRequest,
    SubmittedWeightsAndOptimizerPresigned,
    WeightSubmitResponse,
    WeightUpdate,
)
from subnet.utils.vector_utils import flatten_optimizer_state
from common.models.miner_models import ChunkMetadata
from common.utils.exceptions import (
    APIException,
    LayerStateException,
    RateLimitException,
    MinerNotRegisteredException,
    RunFullException,
    NanInfException,
    NanInfWarning,
    SpecVersionException,
    SubmittedWeightsError,
    WeightPartitionException,
    MinerBlockedException,
    MinerFrozenException,
    MinerInitializingException,
)
from common.utils.partitions import MinerPartition
from common.utils.shared_states import LayerPhase
from common.models.run_flags import RUN_FLAGS, RunFlags
from common import settings as common_settings
from subnet.base.base_neuron import BaseNeuron
from subnet.miner_api_client import MinerAPIClient
from subnet.model.utils import _clean_gpu_memory, log_gpu_memory_usage
from subnet.utils.partition_utils import (
    MergingPartition,
    load_model_weights,
    load_model_weights_and_optimizer_state,
)
from subnet.utils.vector_utils import check_for_nans_and_infs
from subnet.model import gpu_device
from subnet.utils.s3_torch import download_tensor

from miner.training import TrainingPhase


class Miner(BaseNeuron, HealthServerMixin, NodeControlMixin):
    def __init__(
        self,
        wallet_name: str | None = None,
        wallet_hotkey: str | None = None,
        wallet: Wallet | None = None,
        device: str | None = None,
        run_flags: RunFlags | None = None,
        mock: bool | None = None,
        health_host: str | None = None,
        health_port: int | None = None,
        health_endpoint: str | None = None,
        launch_health: bool | None = None,
        visualization_port: int | None = None,
        visualization_auto_open: bool | None = None,
        node_control_port: int | None = None,
    ):
        super().__init__()
        self.device = device or os.getenv("DEVICE") or miner_settings.detect_device()
        self.run_flags: RunFlags = run_flags.model_copy(deep=True) if run_flags else RUN_FLAGS.model_copy(deep=True)
        self.model_manager.run_flags = self.run_flags
        self.mock = mock if mock is not None else common_settings.MOCK
        self.health_host = health_host or miner_settings.MINER_HEALTH_HOST
        self.health_port = health_port or miner_settings.MINER_HEALTH_PORT
        self.health_endpoint = health_endpoint or miner_settings.MINER_HEALTH_ENDPOINT
        self.launch_health = miner_settings.LAUNCH_HEALTH if launch_health is None else launch_health
        self.visualization_port = visualization_port or 8009
        self.visualization_auto_open = (
            miner_settings.VISUALIZATION_AUTO_OPEN if visualization_auto_open is None else visualization_auto_open
        )
        self.init_neuron(wallet_name=wallet_name, wallet_hotkey=wallet_hotkey, wallet=wallet)
        self.state_manager: StateManager = StateManager(wallet=self.wallet)
        self.weights_submitted: bool = False
        self.partitions_submitted: bool = False
        self.miner_api_client: MinerAPIClient = MinerAPIClient(
            hotkey=self.wallet.hotkey,
            is_mounted=miner_settings.IS_MOUNTED,
            electron_version=miner_settings.ELECTRON_VERSION,
        )
        self.need_to_pull_weights = True
        self._needs_local_optimizer_state_download: bool = False
        self.training_phase: TrainingPhase = TrainingPhase(
            miner_api_client=self.miner_api_client,
            state_manager=self.state_manager,
            model_manager=self.model_manager,
            device=self.device,
            run_flags=self.run_flags,
            mock=self.mock,
            is_mounted=miner_settings.IS_MOUNTED,
            miner=self,
        )
        self.stats_tracker = StatsTracker()
        self.training_phase.attach_stats_tracker(self.stats_tracker)
        self._latest_attestation_payloads: dict[str, MinerAttestationPayload | EnclaveSignResponse] = {}
        self.visualization_process: multiprocessing.Process | None = None

        # Telemetry
        self.telemetry_service: TelemetryBufferService | None = None
        if miner_settings.TELEMETRY_ENABLED:
            self.telemetry_service = TelemetryBufferService(
                hotkey=self.wallet.hotkey,
                max_buffer_size=miner_settings.TELEMETRY_MAX_BUFFER_SIZE,
                flush_interval_sec=miner_settings.TELEMETRY_FLUSH_INTERVAL_SEC,
                is_mounted=miner_settings.IS_MOUNTED,
                electron_version=miner_settings.ELECTRON_VERSION,
            )

        self.node_control_port = node_control_port or 8010
        self.node_control_process: multiprocessing.Process | None = None
        self.is_mounted = miner_settings.IS_MOUNTED

        # P2P lifecycle manager (receiver subprocess + sender)
        self.p2p: P2PStack | None = None

        # Input hash tracking: activation_id -> hash of input we received
        self.input_activation_hashes: dict[str, str] = {}
        self.input_hash_lock = asyncio.Lock()

        # Per-peer semaphores to limit concurrent P2P requests to any single peer
        self._peer_semaphores: dict[str, asyncio.Semaphore] = {}
        self._peer_semaphores_lock = asyncio.Lock()
        self._max_concurrent_per_peer = 2  # Based on benchmark: BI degrades at concurrency >= 5

    async def _collect_attestation_payload(self, action: str) -> MinerAttestationPayload | EnclaveSignResponse | None:
        if self.run_flags.attest.isOff():
            return None

        challenge_response = await self.miner_api_client.request_attestation_challenge(action=action)
        if challenge_response is None:
            logger.debug(f"No attestation challenge issued for action {action}")
            return None

        try:
            challenge = AttestationChallengeResponse(
                challenge_blob=challenge_response.attestation_challenge_blob,
                self_checks=challenge_response.self_checks,
                crypto=challenge_response.crypto,
            )

            if self.is_mounted:
                challenge_id = json.loads(challenge_response.attestation_challenge_blob)["challenge_id"]
                payload = await self.enclave_sign_with_purpose(
                    purpose="attestation",
                    payload=payload_base64_from_obj(challenge),
                    challenge_id=challenge_id,
                )
                logger.debug(f"Signing attestation challenge {challenge_id} for action {action}")
            else:
                payload = await asyncio.to_thread(collect_attestation_payload, challenge)

            self._latest_attestation_payloads[action] = payload
            logger.info(f"Collected attestation payload for action {action}")
            return payload
        except AttestationUnavailableError as exc:
            error_code = getattr(exc, "error_code", None)
            suffix = f" (error_code={error_code})" if error_code is not None else ""
            logger.error(f"Attestation unavailable for action {action}{suffix}: {exc}")
        except Exception as exc:
            logger.exception(f"Error collecting attestation for action {action}: {exc}")
        return None

    def _start_visualization_server_process(self, port: int | None = None):
        """Start the visualization server in a separate process."""
        try:
            target_port = port or self.visualization_port
            self.visualization_process = multiprocessing.Process(
                target=start_visualization_server, args=(target_port,), daemon=True, name="VisualizationServer"
            )
            self.visualization_process.start()
            logger.info(f"✅ Visualization server started in separate process (PID: {self.visualization_process.pid})")
        except Exception as e:
            logger.exception(f"Error starting visualization server process: {e}")

    def _stop_visualization_server_process(self):
        """Stop the visualization server process."""
        if self.visualization_process and self.visualization_process.is_alive():
            logger.info("Stopping visualization server process...")
            self.visualization_process.terminate()
            self.visualization_process.join(timeout=5)
            if self.visualization_process.is_alive():
                logger.warning("Visualization server did not terminate gracefully, forcing kill...")
                self.visualization_process.kill()
                self.visualization_process.join()
            logger.info("✅ Visualization server stopped")

    def _open_visualization_tab(self, url: str, delay: float = 2.0) -> None:
        """Open the visualization UI in the user's default browser after a short delay."""

        def _open() -> None:
            time.sleep(delay)
            try:
                webbrowser.open(url, new=2)
            except Exception as exc:  # pragma: no cover - depends on host browser support
                logger.warning(f"Could not auto-open visualization tab: {exc}")

        threading.Thread(target=_open, name="VisualizationTabOpener", daemon=True).start()

    def _update_run_flags(self, new_flags: RunFlags) -> None:
        """Update this miner's run flags in-place."""
        for field_name in new_flags.model_fields:
            new_flag = getattr(new_flags, field_name)
            current_flag = getattr(self.run_flags, field_name, None)
            if current_flag is not None:
                current_flag.enabled = new_flag.enabled
                current_flag.version = new_flag.version
            else:
                setattr(self.run_flags, field_name, new_flag)

    async def _start_p2p(self) -> None:
        """Initialize and start the P2P stack (receiver subprocess + sender)."""
        self.p2p = P2PStack(
            cache_ttl=float(miner_settings.P2P_ACTIVATION_CACHE_TTL),
            max_cache_size=miner_settings.MAX_ACTIVATION_CACHE_SIZE,
        )
        seed = f"iota-miner-{self.wallet.hotkey.ss58_address}"
        await self.p2p.start(seed=seed)

    async def _stop_p2p(self) -> None:
        """Shutdown the P2P stack."""
        if self.p2p is not None:
            await self.p2p.stop()
            self.p2p = None

    @property
    def p2p_node_id(self) -> str | None:
        """Get this miner's P2P node ID."""
        if self.p2p is not None:
            return self.p2p.node_id
        return None

    async def cache_activation(self, activation_id: str, tensor_bytes: bytes) -> None:
        """Cache activation for P2P retrieval by other miners.

        Delegates to the P2PStack which writes to SharedMemory for the
        receiver subprocess to read.
        """
        if self.p2p is not None:
            self.p2p.cache_activation(activation_id, tensor_bytes)

    async def _get_peer_semaphore(self, node_id: str) -> asyncio.Semaphore:
        """Get or create a semaphore for limiting concurrent requests to a peer."""
        async with self._peer_semaphores_lock:
            if node_id not in self._peer_semaphores:
                self._peer_semaphores[node_id] = asyncio.Semaphore(self._max_concurrent_per_peer)
            return self._peer_semaphores[node_id]

    async def request_activation_p2p(
        self,
        activation_id: str,
        source_node_id: str,
        timings: P2POperationTimings | None = None,
    ) -> bytes:
        """
        Request activation from another miner via P2P using bidirectional messaging.

        Sends request and receives response on the same connection.
        Uses per-peer semaphore to limit concurrent requests (benchmark showed
        bidirectional communication degrades at concurrency >= 5).

        Retry and per-phase timeout logic is handled by the Sender's
        built-in ``P2PRetry`` executor — no manual retry loop needed here.

        Args:
            activation_id: The activation to request from the remote peer.
            source_node_id: Iroh node ID of the peer that has the activation.
            timings: Optional mutable timing record.  When provided,
                per-phase durations (connection, send, receive) and retry
                metadata are recorded by the Sender for later stats
                consumption.
        """
        if self.p2p is None:
            raise P2PRequestError("P2P stack not available (shutting down?)", P2PResponseStatus.ERROR)

        # Acquire per-peer semaphore to limit concurrent requests
        semaphore = await self._get_peer_semaphore(source_node_id)

        async with semaphore:
            epistula_start = time.time()
            request = encode_activation_request(activation_id, hotkey=self.wallet.hotkey)
            epistula_end = time.time()

            stats = self.stats_tracker.ensure_activation_stats(activation_id)
            stats.timing.epistula.start = epistula_start
            stats.timing.epistula.end = epistula_end
            stats.timing.epistula.duration = epistula_end - epistula_start

            # All retry + timeout logic lives inside Sender.send_message_bi()
            response = await self.p2p.sender.send_message_bi(
                source_node_id,
                request,
                DEFAULT_MAX_MESSAGE_SIZE,
                timings=timings,
            )

            tensor_bytes, status = decode_activation_response(response)

            if status == P2PResponseStatus.SUCCESS:
                logger.debug(f"Received activation {activation_id} from {source_node_id[:16]}...")
                return tensor_bytes

            # Definitive failures — no retry at this layer
            if status == P2PResponseStatus.NOT_FOUND:
                raise P2PNotFoundError(f"Activation {activation_id} not found on peer {source_node_id[:16]}...")
            elif status == P2PResponseStatus.EXPIRED:
                raise P2PExpiredError(f"Activation {activation_id} expired on peer {source_node_id[:16]}...")
            elif status == P2PResponseStatus.UNAUTHORIZED:
                raise P2PUnauthorizedError(f"Activation {activation_id} auth failed on peer {source_node_id[:16]}...")
            else:
                raise P2PRequestError(
                    f"Activation {activation_id} peer error ({status.name}) on {source_node_id[:16]}...",
                    status,
                )

    async def store_input_hash(self, activation_id: str, input_hash: str) -> None:
        """Store hash of received input for later submission."""
        async with self.input_hash_lock:
            self.input_activation_hashes[activation_id] = input_hash

    def get_input_hash(self, activation_id: str) -> str | None:
        """Get stored input hash for an activation."""
        return self.input_activation_hashes.get(activation_id)

    async def clear_input_hash(self, activation_id: str) -> None:
        """Clean up after activation is fully processed."""
        async with self.input_hash_lock:
            self.input_activation_hashes.pop(activation_id, None)

    async def training_loop_tick(self):
        """Single iteration of the training loop, handling state-specific work."""
        with logger.contextualize(
            hotkey=self.hotkey[:8],
            run_id=self.state_manager.run_id,
            layer=self.state_manager.layer,
        ):
            if not await CommonAPIClient.check_orchestrator_health(hotkey=self.wallet.hotkey):
                logger.info(f"🔄 Orchestrator health check failed for miner {self.wallet.hotkey.ss58_address[:8]}")
                await asyncio.sleep(5)
                return

            allocated_memory = gpu_device.allocated_memory() / 1024**3  # GB
            logger.debug(f"💾 GPU memory: {allocated_memory:.2f}GB")

            logger.info(
                f"🔄 Miner {self.hotkey[:8]} in Layer {self.state_manager.layer} is in state: {self.miner_api_client.layer_state}"
            )

            if self.miner_api_client.layer_state == LayerPhase.TRAINING:
                if self.need_to_pull_weights:
                    if self.model_manager.epoch_on_registration == 1:
                        logger.info(
                            f"Miner {self.hotkey[:8]} registered on epoch 1 - no merged weights to download, proceeding with current model weights"
                        )
                    else:
                        try:
                            async with TimerLoggerMiner(
                                name="download_and_set_global_weights",
                                metadata={"hotkey": self.hotkey[:8], "layer": self.state_manager.layer},
                                hotkey=self.hotkey[:8],
                            ):
                                await self.download_and_set_global_weights(
                                    device=self.device,
                                    client=self.miner_api_client,
                                )
                        except MinerNotRegisteredException:
                            # Re-raise so training_loop can reset and re-register; do not retry download
                            raise
                        except Exception as e:
                            logger.exception(f"Error downloading and setting weights: {e}")
                            logger.warning(
                                f"Miner {self.hotkey[:8]} will NOT train until global weights are downloaded successfully... Retrying"
                            )
                            await asyncio.sleep(1)
                            return

                        # If miner is new to this layer, download global optimizer state (if feature enabled)
                        if self._needs_local_optimizer_state_download and self.run_flags.upload_optimizer_state.isOn():
                            try:
                                await self._download_and_apply_local_optimizer_state()
                            except Exception as e:
                                logger.warning(f"Failed to download global optimizer state (non-fatal): {e}")
                            finally:
                                self._needs_local_optimizer_state_download = False
                        elif self._needs_local_optimizer_state_download:
                            # Feature disabled, skip download
                            self._needs_local_optimizer_state_download = False

                    # Always persist a snapshot at epoch start so submit_weights has previous weights
                    save_model_weights_and_optimizer_state(
                        model_weights=torch.nn.utils.parameters_to_vector(self.model_manager.model.parameters()),
                        optimizer_state_dict=self.model_manager.optimizer.state_dict(),
                        hotkey=self.hotkey,
                        run_id=self.state_manager.run_id,
                        layer_idx=self.state_manager.layer,
                    )
                    logger.info(f"Saved current model weights and optimizer state for miner {self.hotkey[:8]}")

                self.need_to_pull_weights = False
                self.weights_submitted = False
                self.partitions_submitted = False
                await self.training_phase.run()
                await asyncio.sleep(1.1)
                return

            if self.miner_api_client.layer_state == LayerPhase.WEIGHTS_UPLOADING:
                self.need_to_pull_weights = True
                logger.info(
                    f"\n\n\n\n\n\n\n\n 🔄 Miner in layer {self.state_manager.layer} submitting weights state!\n\n\n\n\n\n\n\n"
                )
                if self.weights_submitted:
                    logger.debug(f"Weights already submitted for miner {self.hotkey[:8]}, skipping")
                else:
                    await self.submit_weights()
                    self.weights_submitted = True
                logger.info("🔄 Miner submitted weights, switching to merging partitions")
                await wait_for_state(state=LayerPhase.MERGING_PARTITIONS, miner_api_client=self.miner_api_client)
                return

            if self.miner_api_client.layer_state == LayerPhase.MERGING_PARTITIONS:
                self.need_to_pull_weights = True
                logger.info(
                    f"\n\n\n\n\n\n\n\n 🔄 Miner in layer {self.state_manager.layer} merging partitions state!\n\n\n\n\n\n\n\n"
                )
                if not self.partitions_submitted:
                    logger.info("🔄 Miner getting weight partition info")
                    weight_path_per_layer, partitions = await get_weight_partition_info(
                        layer=self.state_manager.layer, miner_api_client=self.miner_api_client
                    )

                    if not partitions:
                        logger.info("🔄 Miner has no partitions to merge")
                        await asyncio.sleep(1.1)
                        return

                    logger.info(f"🔄 Miner starting merging partitions: {[p.chunk_number for p in partitions]}")
                    await self.merge_partitions(
                        weight_path_per_layer=weight_path_per_layer,
                        partitions=partitions,
                    )
                    logger.info("🔄 Miner finished merged partitions")

                    self.partitions_submitted = True
                    await wait_for_state(state=LayerPhase.TRAINING, miner_api_client=self.miner_api_client)

                else:
                    logger.info(f"🔄 Miner {self.hotkey[:8]} already submitted partitions, skipping...")
                    await wait_for_state(state=LayerPhase.TRAINING, miner_api_client=self.miner_api_client)

                self.model_manager.epoch_counter += 1
                self.training_phase.local_optimization_steps = 0
                return

            await asyncio.sleep(1.1)

    async def training_loop(self):
        """Main training loop delegating to tick with existing error handling."""
        while True:
            try:
                await self.report_training_state(
                    state="training_tick",
                    run_id=self.state_manager.run_id,
                    layer=self.state_manager.layer,
                )
                await self.training_loop_tick()
            except RunFullException as e:
                logger.warning(
                    f"🔄 Miner {self.hotkey[:8]} cannot join run because it is full. Retrying in 60 seconds: {e}"
                )
                await asyncio.sleep(60)
                await self.reset_miner_state()
                continue
            except LayerStateException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} layer state change...: {e}")
                await self.report_training_state(
                    state="layer_state_change",
                    detail=str(e),
                    run_id=self.state_manager.run_id,
                    layer=self.state_manager.layer,
                )
                continue
            except MinerNotRegisteredException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} miner not registered error: {e}")
                await self.reset_miner_state()
                continue
            except APIException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} API exception: {e}")
                continue
            except RateLimitException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} Rate limit exception: {e}")
                continue
            except aiohttp.ClientResponseError as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} Client response error: {e}")
                continue
            except (aiohttp.ClientConnectorDNSError, aiohttp.ClientConnectorError) as e:
                logger.warning(f"🔄 Miner {self.hotkey[:8]} Connection error (DNS/network): {e}. Retrying...")
                await asyncio.sleep(5)
                continue
            except (asyncio.TimeoutError, TimeoutError) as e:
                logger.warning(f"🔄 Miner {self.hotkey[:8]} Timeout error: {e}")
                continue
            except SubmittedWeightsError as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} Submitted weights error: {e}")
                continue

            except MinerInitializingException as e:
                logger.warning(
                    f"🔄Miner {self.hotkey[:8]} has been temporarily blocked (initializing) and cannot perform work: {e}"
                )
                await self.register_set_status(status="initializing")
                await asyncio.sleep(60)
                continue

            except MinerFrozenException as e:
                logger.warning(
                    f"🔄Miner {self.hotkey[:8]} has been temporarily blocked (initializing) and cannot perform work: {e}"
                )
                await self.register_set_status(status="frozen")
                await asyncio.sleep(60)
                continue

            except MinerBlockedException as e:
                logger.warning(f"🔄 Miner {self.hotkey[:8]} has been temporarily blocked and cannot perform work: {e}")
                await asyncio.sleep(60)
                continue
            except WeightPartitionException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} Partition exception: {e}")
                continue
            except NanInfWarning as e:
                logger.info(f"⚠️ Miner {self.hotkey[:8]} NaN/Inf warning: {e}")
                continue
            except NanInfException as e:
                logger.error(f"❌ Miner {self.hotkey[:8]} NaN/Inf exception: {e}")
                raise
            except Exception:
                raise

    async def run(self):
        self._start_visualization_server_process(port=self.visualization_port)
        if self.visualization_auto_open:
            self._open_visualization_tab(f"http://localhost:{self.visualization_port}/vis.html")

        if self.telemetry_service:
            try:
                await self.telemetry_service.start()
            except Exception as e:
                logger.error(f"Failed to start telemetry service, continuing without telemetry: {e}")
                self.telemetry_service = None

        await self.report_training_state(state="resetting")
        await self.reset_miner_state()
        logger.info(f"🚀 Starting miner {self.hotkey[:8]} on layer {self.layer} | Timeout: {miner_settings.TIMEOUT}s")

        await self.report_training_state(
            state="waiting_training", run_id=self.state_manager.run_id, layer=self.state_manager.layer
        )
        await wait_for_state(state=LayerPhase.TRAINING, miner_api_client=self.miner_api_client, raise_bad_sync=False)
        await self.report_training_state(
            state="training", run_id=self.state_manager.run_id, layer=self.state_manager.layer
        )
        await self.training_loop()

    async def register(self) -> tuple[dict, dict]:
        """Single registration attempt. Raises on failure for caller to retry."""
        logger.info(f"🔄 Attempting to fetch run info for miner {self.hotkey[:8]}...")
        run_info_list = await self.miner_api_client.fetch_run_info_request()
        if not run_info_list:
            raise Exception("Fatal Error: Could not fetch run info")

        best_run = identify_best_run(run_info_list=run_info_list)
        logger.info(f"✅ Best run for miner {self.hotkey[:8]} is {best_run.run_id}")

        logger.info(f"🔄 Attempting to register miner {self.hotkey[:8]} on run {best_run.run_id} with orchestrator...")

        # P2P node ID is required for registration
        if not self.p2p_node_id:
            raise RuntimeError(
                f"P2P node ID not available for miner {self.hotkey[:8]}. "
                "P2P must be initialized before registration."
            )

        register_request = RegisterMinerRequest(
            run_id=best_run.run_id,
            register_as_metagraph_miner=True,
            p2p_node_id=self.p2p_node_id,
        )
        response: MinerRegistrationResponse = await self.miner_api_client.register_miner_request(
            register_miner_request=register_request
        )
        logger.info(f"Registered with P2P node ID: {self.p2p_node_id[:16]}...")

        assigned_layer = int(response.layer)
        current_epoch = int(response.current_epoch)

        logger.debug(f"Number of partitions for miner {self.hotkey[:8]}: {response.num_partitions}")

        self.model_manager.num_partitions = int(response.num_partitions)
        self.num_partitions = int(response.num_partitions)

        if response.layer is None:
            raise Exception(f"Miner {self.hotkey[:8]} registered with no layer assigned, this should not happen")

        # TODO: clean these up
        self.layer = assigned_layer
        self.state_manager.layer = assigned_layer
        self.state_manager.training_epoch_when_registered = current_epoch
        self.state_manager.run_id = response.run_id
        self.run_id = response.run_id
        self.model_manager.epoch_on_registration = current_epoch

        self._update_run_flags(response.run_flags)
        await self.register_set_status(status="registered")

        self.stats_tracker.reset()
        self.stats_tracker.set_layer(self.state_manager.layer)
        self.stats_tracker.set_remote_epoch(current_epoch)
        self.stats_tracker.set_run_id(response.run_id)

        logger.success(
            f"✅ Miner {self.hotkey[:8]} registered successfully in layer {self.state_manager.layer} on training epoch {current_epoch}"
        )
        logger.debug(f"Run flags for miner {self.hotkey[:8]}: {self.run_flags}")
        return response.model_cfg.model_dump(), response.model_metadata.model_dump()

    async def register_loop(self) -> tuple[dict, dict]:
        """
        Register the miner with the orchestrator, acquiring a layer during the process.
        If the miner is not registered, it will try to register every 60 seconds
        """
        while True:
            try:
                return await self.register()
            except RunFullException as e:
                logger.warning(f"Run is full for miner {self.hotkey[:8]}: {e}")
                await asyncio.sleep(60)
                continue
            except SpecVersionException as e:
                logger.error(f"Spec version mismatch: {e}")
                raise

            except Exception as e:
                logger.exception(f"Error registering miner: {e}")
                await asyncio.sleep(10)

    async def _download_and_apply_local_optimizer_state(self) -> None:
        """
        Download the stage's local optimizer state for a miner new to this layer.

        This is called when a miner first joins a layer (brand new registration or layer change).
        The local optimizer state is uploaded by the top K productive miners.
        """
        logger.info(f"🔄 Miner {self.hotkey[:8]} downloading layer {self.state_manager.layer} optimizer state")

        # Get the presigned URL for the global optimizer state
        response = await self.miner_api_client.get_layer_optimizer_state()

        if not response.available:
            logger.info(
                f"No layer {self.state_manager.layer} optimizer state available for layer {self.state_manager.layer} yet - skipping"
            )
            return

        optimizer_state_tensor = await download_tensor(
            path=response.optimizer_state_url,
            dtype=torch.bfloat16,
            device="cpu",
            run_flags=self.run_flags,
        )

        if optimizer_state_tensor is None:
            logger.warning(f"Failed to download layer {self.state_manager.layer} optimizer state tensor")
            return

        tensor_shapes = get_optimizer_tensor_shapes(self.model_manager.optimizer)
        optimizer_state_dict = reconstruct_optimizer_state(
            flat_tensor=optimizer_state_tensor,
            tensor_shapes=tensor_shapes,
            state_dict=self.model_manager.optimizer.state_dict(),
        )

        self.model_manager.optimizer.load_state_dict(optimizer_state_dict)

        logger.success(
            f"✅ Miner {self.hotkey[:8]} successfully downloaded and applied local optimizer state from {response.optimizer_state_url} for layer {self.state_manager.layer}"
        )

    async def submit_weights(self):
        """
        Uploads the weights to the orchestrator and submits them to the database

        Raises:
            SubmittedWeightsError: If the weights are not submitted successfully
            e: If there is an error submitting the weights
        """
        async with TimerLoggerMiner(
            name="submit_weights",
            metadata={"hotkey": self.hotkey[:8], "layer": self.state_manager.layer},
            hotkey=self.hotkey[:8],
        ):
            if self.training_phase.backwards_since_reset == 0:
                logger.warning(f"Backwards since reset for miner {self.hotkey[:8]} is 0, skipping")
                return

            current_weights = (
                torch.nn.utils.parameters_to_vector(parameters=self.model_manager.model.parameters()).detach().to("cpu")
            )
            previous_weights = load_model_weights(
                hotkey=self.hotkey, run_id=self.state_manager.run_id, layer_idx=self.state_manager.layer
            )

            # For diloco we want to upload the pseudo gradients to the orchestrator
            if previous_weights is None:
                raise Exception(f"Previous weights are None for miner {self.hotkey[:8]}")

            # creating changes
            pseudo_gradients = torch.zeros_like(previous_weights).to(torch.bfloat16)

            # iterate over pseudo gradients in batches and fill them - this avoids using unnecessary memory usage
            for i in range(miner_settings.PSEUDO_GRADIENTS_BATCH_SIZE):
                logger.debug(f"Getting pseudo gradients for batch {i}")
                previous_weights_batch = previous_weights[i :: miner_settings.PSEUDO_GRADIENTS_BATCH_SIZE]
                current_weights_batch = current_weights[i :: miner_settings.PSEUDO_GRADIENTS_BATCH_SIZE]
                pseudo_gradients_batch = previous_weights_batch.to(torch.float32) - current_weights_batch.to(
                    torch.float32
                )
                pseudo_gradients[i :: miner_settings.PSEUDO_GRADIENTS_BATCH_SIZE] = pseudo_gradients_batch.to(
                    torch.bfloat16
                )

            if self.run_flags.clip_pseudo_gradients.isOn():
                pseudo_gradients = await self.model_manager.clip_pseudo_gradients(pseudo_gradients)

            # Log some stats about the pseudo gradients
            logger.info(
                f"Pseudo gradients for miner {self.hotkey[:8]} have mean {pseudo_gradients.mean():.6f} and std {pseudo_gradients.std():.6f}"
            )
            logger.info(
                f"Previous weights for miner {self.hotkey[:8]} have mean {previous_weights.mean():.6f} and std {previous_weights.std():.6f}"
            )
            logger.info(
                f"New weights for miner {self.hotkey[:8]} have mean {current_weights.mean():.6f} and std {current_weights.std():.6f}"
            )
            logger.info(f"Pseudo gradients shape: {pseudo_gradients.shape}")

            try:
                self.model_manager.optimizer.zero_grad()
                await self.training_phase.optimization_reset()

                try:
                    await self.miner_api_client.notify_orchestrator_of_state_call()
                except Exception as e:
                    logger.warning(f"Error notifying orchestrator of state call: {e}")

                attestation_payload: MinerAttestationPayload | None = await self._collect_attestation_payload(
                    action="weights"
                )

                check_for_nans_and_infs(
                    tensor=pseudo_gradients,
                    name=f"pseudo gradients for miner {self.hotkey[:8]}",
                    exception_type=NanInfException,
                )

                metadata: dict = create_metadata(tensor=pseudo_gradients, num_sections=self.num_partitions)
                metadata["local_optimization_steps"] = self.training_phase.local_optimization_steps

                # Convert tensor to bytes, handling bfloat16 compatibility
                path = await upload_tensor(
                    tensor=pseudo_gradients,
                    file_type="weights",
                    hotkey=self.wallet.hotkey,
                    miner_api_client=self.miner_api_client,
                    run_flags=self.run_flags,
                )

                # Upload metadata as activation type since orchestrator doesn't have a metadata type
                metadata_path = await upload_file(
                    miner_api_client=self.miner_api_client,
                    data=json.dumps(metadata).encode(),
                    file_type="weights_metadata",
                    hotkey=self.wallet.hotkey,
                    run_flags=self.run_flags,
                )

                response: WeightSubmitResponse = await self.miner_api_client.submit_weights(
                    weight_update=WeightUpdate(
                        weights_path=path.object_path,
                        weights_metadata_path=metadata_path,
                        attestation=attestation_payload,
                    ),
                )

                if not response:
                    raise SubmittedWeightsError("Error submitting weights")

                if response.should_upload_optimizer_state:
                    logger.info(f"Miner {self.hotkey[:8]} selected to upload optimizer state")
                    try:
                        # Flatten optimizer state
                        flat_optimizer_state, _, _ = flatten_optimizer_state(
                            optimizer=self.model_manager.optimizer,
                            device="cpu",
                            dtype=torch.bfloat16,
                        )

                        # Upload optimizer state tensor
                        optimizer_state_upload = await upload_tensor(
                            tensor=flat_optimizer_state,
                            file_type="optimizer_state",
                            hotkey=self.wallet.hotkey,
                            miner_api_client=self.miner_api_client,
                            run_flags=self.run_flags,
                        )

                        # Notify orchestrator of the optimizer state path
                        await self.miner_api_client.submit_optimizer_state(
                            optimizer_state_path=optimizer_state_upload.object_path,
                        )
                        logger.info(f"Miner {self.hotkey[:8]} successfully uploaded optimizer state")
                    except Exception as e:
                        logger.warning(f"Failed to upload optimizer state (non-fatal): {e}")

            except LayerStateException as e:
                logger.debug(f"Layer state exception submitting weights: {e}")
                raise

            except Exception as e:
                logger.error(f"Generic error submitting weights: {e}")
                raise

    async def run_miner(self):
        """
        Run the miner. Responsible for:
        - Starting the healthcheck server
        - Registering the miner
        - Setting up the local model
        - Running the miner loop

        The method runs in a loop and retries on failures with a fixed delay.
        """

        logger.info("🚀 Starting miner 🚀")
        try:
            # Start P2P before registration so we have a node_id to register with
            await self._start_p2p()
            logger.info("P2P communication started")

            # Start the healthcheck server
            if self.launch_health:
                try:
                    killed_process = self._kill_process_on_port(self.health_port)
                    if killed_process:
                        logger.warning(f"Terminated existing process using healthcheck port {self.health_port}")
                except Exception as e:
                    logger.error(f"Failed to clear healthcheck port {self.health_port}: {e}")
                await self._start_health_server()
                logger.info("🏥 Health server started")
            else:
                logger.warning(
                    "⚠️ Miner healthcheck API not configured in settings (MINER_HEALTH_PORT missing). Skipping."
                )

                # Reset the entire miner state, which also downloads the weights and optimizer state.
            await self.run()

        except KeyboardInterrupt:
            logger.info("Gracefully shutting down miner")

        except SpecVersionException:
            logger.error("Spec version mismatch. Please pull the latest code and restart the miner")
            raise

        except LayerStateException as e:
            logger.warning(f"Layer state exception: {e}")

        except Exception as e:
            logger.exception(f"❌ Critical error in run_miner: {e}")
            await asyncio.sleep(5)

        finally:
            logger.info("Cleaning up miner on shutdown...")
            try:
                _clean_gpu_memory()

                try:
                    await self._stop_health_server()
                    logger.info("🏥 Health server stopped")
                except Exception as e:
                    logger.error(f"Failed to stop health server: {e}")

                try:
                    self._stop_visualization_server_process()
                except Exception as e:
                    logger.error(f"Failed to stop visualization server: {e}")

                # Stop P2P communication
                try:
                    await self._stop_p2p()
                    logger.info("P2P stopped")
                except Exception as e:
                    logger.error(f"Failed to stop P2P: {e}")
                try:
                    if self.telemetry_service:
                        await self.telemetry_service.stop()
                        logger.info("Telemetry service stopped")
                except Exception as e:
                    logger.error(f"Failed to stop telemetry service: {e}")

            except Exception as e:
                logger.error(f"Failed to shutdown miner: {e}")

        # Final cleanup when exiting the loop (only reached on KeyboardInterrupt)
        logger.info("🛑 Miner shutdown complete")

        # Miners can sometimes not clean themselves up properly. Therefore, lets force kill the process.
        sys.exit(0)

    async def reset_miner_state(self):
        """
        Reset the entire miner state, including the API client, health server, and all other state.
        """
        logger.info("🔄 Resetting miner entire state!")
        self.need_to_pull_weights = True

        old_run_id = self.state_manager.run_id
        old_layer = self.state_manager.layer

        await self.training_phase.reset()

        # We provide the model config and metadata so that all miners are aligned.
        model_config, model_metadata = await self.register_loop()

        # Determine if miner is new to this layer (needs to download layer-specific local optimizer state)
        # This is true if:
        # - Brand new registration (old_run_id is None)
        # - Different run (old_run_id != new run_id)
        # - Layer change (old_layer != new layer)
        is_same_layer = old_run_id == self.state_manager.run_id and old_layer == self.state_manager.layer
        self._needs_local_optimizer_state_download = not is_same_layer

        if self._needs_local_optimizer_state_download:
            logger.info(
                f"🆕 Miner {self.hotkey[:8]} is new to layer {self.state_manager.layer} "
                f"(old: run={old_run_id}, layer={old_layer}) - will download layer-specifc local optimizer state"
            )

        # if we continue on the same run and layer, save off what we've done so far and load weights
        current_model_weights: torch.Tensor = None
        current_model_optimizer_state: dict = None

        if is_same_layer:
            if self.model_manager.model is not None and self.model_manager.optimizer is not None:
                current_model_weights = torch.nn.utils.parameters_to_vector(self.model_manager.model.parameters())
                current_model_optimizer_state = self.model_manager.optimizer.state_dict()

            else:
                current_model_weights, current_model_optimizer_state = load_model_weights_and_optimizer_state(
                    hotkey=self.hotkey,
                    run_id=self.state_manager.run_id,
                    layer_idx=self.state_manager.layer,
                )

        self.model_manager.reset()

        if not await self._setup_local_model(
            model_config=model_config,
            model_metadata=model_metadata,
            model_weights=current_model_weights,
            optimizer_state=current_model_optimizer_state,
            layer=self.state_manager.layer,
            device=self.device,
        ):
            raise Exception("Error setting up local model")

        logger.success("✅ Successfully setup local model")

    async def get_old_partition_for_partition_batch(
        self, batch_partitions: list[MergingPartition]
    ) -> list[MergingPartition]:
        previous_partitions = await self.miner_api_client.get_previous_partitions(
            partition_indices=[partition.new_partition.chunk_number for partition in batch_partitions]
        )
        for partition in batch_partitions:
            previous_partition = [
                p for p in previous_partitions if p.chunk_number == partition.new_partition.chunk_number
            ]
            if not previous_partition:
                logger.warning(f"No previous partition found for partition {partition.new_partition.chunk_number}")
                partition.old_partition = None
            else:
                partition.old_partition = previous_partition[0]
        logger.debug(f"{len(batch_partitions)} batch partitions got old partition")
        return batch_partitions

    async def merge_partitions(
        self, weight_path_per_layer: list[SubmittedWeightsAndOptimizerPresigned], partitions: list[MinerPartition]
    ) -> list[MinerPartition]:
        """Merge the models from the other miners.

        Args:
            weight_path_per_layer (list[SubmittedWeightsPresigned]): The paths to the other miners' partitions
            partition_ids (list[int]): The partition indices to merge

        Returns:
            list[Partition]: The merged partitions
        """
        async with TimerLoggerMiner(
            name="merge_partitions",
            metadata={"hotkey": self.hotkey[:8], "layer": self.state_manager.layer},
            hotkey=self.hotkey[:8],
        ):
            filtered_metadata: dict[str, dict[int, dict[str, ChunkMetadata]]] = await filter_bad_metadata(
                partitions=partitions,
                submitted_weights_and_optimizers=weight_path_per_layer,
                run_flags=self.run_flags,
            )
            # Grab a batch of partitions to download the weights for
            for batch in range(min(miner_settings.N_PARTITION_BATCHES, len(partitions))):
                logger.debug(f"Merging batch {batch} of {min(miner_settings.N_PARTITION_BATCHES, len(partitions))}")

                # Grab a batch of partitions to merge (no downloading yet)
                batch_partitions: list[MergingPartition] = get_partition_batch(batch_index=batch, partitions=partitions)
                logger.debug(f"{len(batch_partitions)} batch partitions grabbed")

                # Download the weights for the batch (fills partitions.weights with a list of all pseudograds from all the other miners)
                merging_partitions: list[MergingPartition] = await download_pseudograds_for_partition_batch(
                    batch_partitions=batch_partitions, filtered_metadata=filtered_metadata
                )
                logger.debug(f"{len(merging_partitions)} batch partitions downloaded successfully")

                # Gets the old partition for the batch (which point us to the previous optimizer state)
                merging_partitions = await self.get_old_partition_for_partition_batch(merging_partitions)
                logger.debug(f"{len(merging_partitions)} batch partitions got old partition")

                # Download the previous optimizer state for the batch (fills partitions.old_optimizer_state with the previous optimizer state)
                merging_partitions = await download_previous_optimizer_state_for_partition_batch(merging_partitions)
                logger.debug(f"{len(merging_partitions)} batch partitions downloaded previous optimizer state")

                # Determine if we have enough memory in the GPU to merge the partitions on GPU or CPU
                device = self.device
                if device != "cpu":
                    gpu_device.synchronize()
                    gpu_device.empty_cache()
                    avail_memory = gpu_device.available_memory()
                    # TODO: @cassova: correct this calculation - 100x is just to push it to cpu for now
                    need_to_merge_on_gpu = (
                        100
                        * torch.nn.utils.parameters_to_vector(self.model_manager.model.parameters()).numel()
                        * len(merging_partitions)
                    )
                    if need_to_merge_on_gpu > avail_memory:
                        logger.warning(
                            "Not enough memory available to merge partitions on GPU"
                            f" - needed {need_to_merge_on_gpu / 1024**3:.2f}GB, available {avail_memory / 1024**3:.2f}GB"
                        )
                        device = "cpu"

                # Load old weights into model
                if device == "cpu":
                    old_model = copy.deepcopy(self.model_manager.model).cpu()
                else:
                    old_model = copy.deepcopy(self.model_manager.model)
                    log_gpu_memory_usage(note="after copying old model")
                torch.nn.utils.vector_to_parameters(
                    load_model_weights(
                        hotkey=self.hotkey, run_id=self.state_manager.run_id, layer_idx=self.state_manager.layer
                    ),
                    old_model.parameters(),
                )

                # Do the actual merging (apply the optimizer state to the weights)
                weights_length = sum([p.numel() for p in old_model.parameters()])
                merged_partitions = await merge_partition_batch(
                    partition_batch=merging_partitions,
                    filtered_metadata=filtered_metadata,
                    old_model=old_model,
                    local_optimizer_state=self.model_manager.optimizer,
                    weights_length=weights_length,
                    num_partitions=self.num_partitions,
                    device=device,
                    run_flags=self.run_flags,
                )
                logger.debug(f"{len(merged_partitions)} batch partitions merged")
                log_gpu_memory_usage(note=f"after merging partitions on {device}")

                # Upload the merged partitions to the database and return list of MinerPartition
                final_partitions = await upload_partition_batch(
                    merged_partitions=merged_partitions,
                    hotkey=self.wallet.hotkey,
                    miner_api_client=self.miner_api_client,
                    run_flags=self.run_flags,
                )
                logger.debug(f"{len(final_partitions)} batch partitions uploaded")

                # Submit the merged partitions to the database
                attestation_payload = await self._collect_attestation_payload(action="merged_partitions")
                await self.miner_api_client.submit_merged_partitions(
                    merged_partitions=final_partitions,
                    attestation=attestation_payload,
                )
                logger.debug(f"{len(final_partitions)} batch partitions submitted")

                self.model_manager.model = self.model_manager.model.to(self.device)

                del old_model
                del merged_partitions  # TODO: @cassova: do a better job of cleaning this up
                del final_partitions  # TODO: @cassova: do a better job of cleaning this up
                log_gpu_memory_usage(note="after merging partitions")
