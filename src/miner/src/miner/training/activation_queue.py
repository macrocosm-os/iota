from __future__ import annotations

import io
import asyncio
from collections import deque
from typing import Optional
from loguru import logger
from miner.utils.timer_logger import TimerLoggerMiner
import torch
import time
from pydantic import BaseModel

from common.models.api_models import ActivationResponse, GetActivationRequest
from common import settings as common_settings
from common.utils.exceptions import ActivationHashMismatchError, RateLimitException
from subnet.miner_api_client import MinerAPIClient
from miner.training.activation_cache import ActivationCache, ActivationData
from miner.state_manager import StateManager
from miner.utils.activation_utils import download_sample
from miner.utils.activation_hash import compute_activation_hash, verify_activation_hash
from subnet.model.model_mixin import ModelManager
from common.utils.exceptions import LayerStateException, MinerNotRegisteredException
from common.utils.shared_states import LayerPhase
from miner import settings as miner_settings

from common.iroh.activation_push import ActivationPushMessage
from common.iroh.timings import P2POperationTimings
from miner.utils.stats import StatsTracker, tensor_num_bytes
from common.iroh.p2p_protocol import P2PRequestError
from miner.telemetry.metric_registry import S3_DOWNLOAD_SPEED_BYTES_PER_SEC
from common.models.run_flags import RunFlags
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miner.new_miner import Miner


class DownloadedData(BaseModel):
    activation_response: ActivationResponse
    input_activations: torch.Tensor
    sample_activations: torch.Tensor | None = None

    class Config:
        arbitrary_types_allowed = True


class ActivationQueue:
    """
    The ActivationQueue is responsible for fetching and storing activations before they are ready
    to be processed by the training phase. We specify the number of forward activations but not that
    backward activations since those are tied to the miner who processed the forward activation.
    """

    def __init__(
        self,
        miner_api_client: MinerAPIClient,
        state_manager: StateManager,
        activation_cache: ActivationCache,
        mock: bool,
        run_flags: RunFlags,
        miner: "Miner | None" = None,
    ):
        self._miner_api_client: MinerAPIClient = miner_api_client
        self._state_manager: StateManager = state_manager
        self._cache: ActivationCache = activation_cache
        self._stats_tracker: StatsTracker | None = None
        self._mock = mock
        self._run_flags = run_flags
        self._miner = miner  # Reference to miner for P2P operations

        self._queue_lock: asyncio.Lock = asyncio.Lock()
        self._forward_queue: deque[ActivationData] = deque()
        self._backward_queue: deque[ActivationData] = deque()
        self._activation_fetcher_task: asyncio.Task | None = None
        self._model_manager: ModelManager | None = None
        self._max_forwards_in_queue: int = miner_settings.MAX_FORWARD_ACTIVATIONS_IN_QUEUE
        self._min_forwards_in_queue: int = miner_settings.MIN_FORWARD_ACTIVATIONS_IN_QUEUE

        self._all_layers_training_cached: bool = False
        self._all_layers_training_cached_at: float = 0.0
        self._all_layers_training_ttl_sec: float = common_settings.ALL_LAYERS_TRAINING_CACHE_TTL_SEC

    def attach_stats_tracker(self, tracker: StatsTracker | None) -> None:
        """Attach a stats tracker for dashboard metrics."""
        self._stats_tracker = tracker

    def _reshape_mock_activations(self, input_activations: torch.Tensor) -> torch.Tensor:
        """Normalize downloaded activations to the configured mock input width."""
        batch_size = common_settings.MINI_BATCH_SIZE
        target_dim = common_settings.MOCK_MODEL_INPUT_DIM

        if input_activations.numel() % batch_size != 0:
            raise ValueError(
                "Mock activation size is not divisible by MINI_BATCH_SIZE. "
                f"numel={input_activations.numel()}, mini_batch_size={batch_size}"
            )

        input_activations = input_activations.reshape(batch_size, -1)
        current_dim = input_activations.shape[-1]

        if current_dim == target_dim:
            return input_activations

        if current_dim > target_dim:
            logger.warning(f"Mock activation width mismatch; truncating features from {current_dim} to {target_dim}")
            return input_activations[:, :target_dim].contiguous()

        logger.warning(f"Mock activation width mismatch; right-padding features from {current_dim} to {target_dim}")
        padding = torch.zeros(
            (batch_size, target_dim - current_dim),
            dtype=input_activations.dtype,
            device=input_activations.device,
        )
        return torch.cat([input_activations, padding], dim=1).contiguous()

    def __len__(self) -> int:
        """Get the number of activations in the queue."""
        return len(self._backward_queue) + len(self._forward_queue)

    def next_activation_is_forward(self) -> bool:
        """Peek at the next activation in the queue without removing it."""
        if len(self._backward_queue) == 0 and len(self._forward_queue) > 0:
            return True
        return False

    async def check_if_training_is_complete(self) -> bool:
        """Check if training is complete by checking if the activation fetcher task has completed."""
        # If the activation fetcher is done, stop it.
        # this will raise any errors that were raised by the activation fetcher
        # (i.e. layer state change errors)
        if self.activation_fetcher_is_done():
            logger.debug("Activation fetcher is done")
            await self.stop_activation_fetcher()  # This should raise LayerStateException
            raise Exception("Unexpected error: Activation fetcher is done, it should have raised LayerStateException")

        # logger.debug("Activation fetcher is not done, training will continue")  # produces too many logs
        return False

    async def get_activation(self, timeout=-1) -> ActivationData:
        """Get an activation from the queue. If the queue is empty, wait for a new activation to be added."""
        start_time = time.time()
        try:
            # Check if the activation fetcher task has completed with an exception
            await self.check_if_training_is_complete()  # This will raise any exception from the background task

            async with self._queue_lock:
                if len(self._backward_queue) + len(self._forward_queue) > 0:
                    logger.debug(
                        f"Activation queue length: {len(self._backward_queue) + len(self._forward_queue)}: "
                        f"backward: {[a.activation_id for a in self._backward_queue]} "
                        f"forward: {[a.activation_id for a in self._forward_queue]}"
                    )
                    logger.debug(f"Cache status: {len(self._cache)}")
                    if len(self._backward_queue) > 0:
                        logger.debug(f"Took {time.time() - start_time} seconds to get backward activation")
                        return self._backward_queue.popleft()
                    if len(self._forward_queue) > 0 and not self._cache.is_full():
                        logger.debug(f"Took {time.time() - start_time} seconds to get forward activation")
                        return self._forward_queue.popleft()

                # Wait for more activations
                if timeout > 0 and time.time() - start_time > timeout:
                    raise Exception("Timeout getting activation")
                await asyncio.sleep(0.1)  # prevent CPU from being blocked
        except Exception as e:
            logger.error(f"Error getting activation from queue: {e}")
            raise

    def activation_fetcher_is_done(self) -> bool:
        """Check if the activation fetcher task has completed."""
        if not self._activation_fetcher_task:
            return True
        return self._activation_fetcher_task.done()

    async def start_activation_fetcher(self, model_manager: ModelManager):
        """Start the activation fetcher task if it's not already running."""
        self._model_manager = model_manager
        if self._activation_fetcher_task and not self._activation_fetcher_task.done():
            logger.warning("Activation fetcher task already running")
            return
        self._backward_queue.clear()  # Clear the backward queue
        self._forward_queue.clear()  # Clear the forward queue to avoid processing expired activations from previous epoch
        self._activation_fetcher_task = asyncio.create_task(self._fetch_activations())
        logger.debug("Activation fetcher task started")

    async def stop_activation_fetcher(self):
        """Stop the activation fetcher task if it's running and await it."""
        if self._activation_fetcher_task:
            try:
                logger.debug("Awaiting activation fetcher task")
                await self._activation_fetcher_task
                logger.error(
                    "Activation fetcher task completed - this message should never be logged bcs we expect a LayerStateException"
                )
            except Exception as e:
                # Handle the error from the task
                logger.warning(f"Activation fetcher task returned an exception: {e}")
                raise

    async def _fetch_activations(self):
        """Route to the appropriate fetcher based on layer."""
        if self._state_manager.layer == 0:
            await self._fetch_activations_layer0()
        else:
            await self._fetch_activations_push_based()

    async def _all_layers_training(self) -> bool:
        """Return True iff all layers in the run are currently in TRAINING."""
        now = time.time()
        if now - self._all_layers_training_cached_at < self._all_layers_training_ttl_sec:
            return self._all_layers_training_cached
        try:
            result = await self._miner_api_client.get_all_layers_training()
        except (LayerStateException, MinerNotRegisteredException):
            raise
        except RateLimitException:
            return self._all_layers_training_cached
        except Exception as exc:
            logger.debug(f"all_layers_training check failed; treating as False: {exc}")
            result = False
        self._all_layers_training_cached = result
        self._all_layers_training_cached_at = now
        return result

    async def _fetch_activations_layer0(self):
        """Layer-0 fetcher: forward samples from orchestrator, backward from push queue."""
        last_state_check = time.time()
        while True:
            # This loop will only break if an exception is raised (i.e. LayerStateException)
            await asyncio.sleep(0.51)  # comply with the orchestrator rate limit

            # ── Periodic heartbeat for state-change detection ────────────────
            if time.time() - last_state_check >= 30.0:
                try:
                    await self._miner_api_client.heartbeat(expected_phase=LayerPhase.TRAINING)
                except RateLimitException:
                    pass
                except (LayerStateException, MinerNotRegisteredException):
                    raise
                except Exception as exc:
                    logger.debug(f"Heartbeat check failed (non-fatal): {exc}")
                last_state_check = time.time()

            # Gate: do not request activations until every layer is in TRAINING.
            # Keeps layer-0 from producing forwards that peers on later layers
            # would reject because they are still uploading weights or merging.
            if not await self._all_layers_training():
                logger.debug("Not all layers are in TRAINING yet; skipping activation fetch tick")
                continue

            # Drain backward activations that arrived via push from layer 1
            await self._drain_layer0_push_queue()

            # Keep cache clean
            self._cache.cleanup()

            # Log cache and queue status
            effective_max_cache = self._cache.effective_max_for_queue
            cache_vacancy = effective_max_cache - len(self._cache)
            logger.debug(f"Cache size: {len(self._cache)}/{effective_max_cache} (vacancy: {cache_vacancy})")
            logger.debug(
                f"Backward activations in queue: {len(self._backward_queue)}"
                f" - Forward activations in queue: {len(self._forward_queue)}"
                f" (min/max fwds: {self._min_forwards_in_queue}/{self._max_forwards_in_queue})"
            )
            queue_status = f"backward: {[a.activation_id for a in self._backward_queue]} forward: {[a.activation_id for a in self._forward_queue]}"
            logger.debug(f"Queue status: {queue_status}")

            max_allowed_total = effective_max_cache + miner_settings.MIN_FORWARD_ACTIVATIONS_IN_QUEUE
            used = len(self._cache) + len(self._forward_queue)
            n_fwd_activations = min(max_allowed_total - used, miner_settings.MAX_FORWARD_ACTIVATIONS_IN_QUEUE)
            missing_backwards = len(self._cache) - len(self._backward_queue)

            if n_fwd_activations < 0:
                n_fwd_activations = 0

            logger.debug(
                f"Max allowed forwards: {max_allowed_total}"
                f" -- Used: {used}"
                f" -- Forward activation reqs: {n_fwd_activations}"
                f" -- Missing backwards: {missing_backwards}"
            )

            if n_fwd_activations == 0:
                # This can happen if our queue is full and we haven't yet processed anything into the cache
                logger.debug("No forward activations needed and no backwards activations needed")
                continue

            try:
                response: list[ActivationResponse] = await self._miner_api_client.get_activations(
                    get_activation_request=GetActivationRequest(n_fwd_activations=n_fwd_activations)
                )
                logger.debug(f"Received activations: {len(response)}")
            except RateLimitException:
                logger.warning("Rate limit exceeded")
                await asyncio.sleep(1)
                continue
            except LayerStateException as e:
                logger.warning(f"Layer state changing while getting activations: {e}")
                raise
            except MinerNotRegisteredException as e:
                logger.warning(f"Miner no longer registered while getting activations: {e}")
                raise
            except Exception as e:
                logger.exception(f"Error getting activations from orchestrator: {e}")
                raise

            if len(response) == 0:
                logger.debug("No activations received from orchestrator")
                continue

            logger.debug(f"Response contains: {[(a.activation_id, a.direction) for a in response]}")

            # Filter the response
            response = await self._filter_duplicates(response=response)  # do this before we split
            backward_response, forward_response = await self._split_responses(response=response)
            if self._stats_tracker is not None:
                for forward in forward_response:
                    self._stats_tracker.ensure_activation_stats(
                        forward.activation_id,
                        direction="forward",
                        time_received=time.time(),
                    )
                for backward in backward_response:
                    self._stats_tracker.ensure_activation_stats(
                        backward.activation_id,
                        direction="backward",
                        time_received=time.time(),
                    )
            logger.debug(
                f"Forward response prior to excess filtering: {[(a.activation_id, a.direction) for a in forward_response]}"
            )
            forward_response = await self._filter_excess_forwards(forward_response=forward_response)

            if len(backward_response) == 0 and len(forward_response) == 0:
                logger.debug("No activations to download after filtering")
                continue

            logger.debug(
                f"After filtering, downloading activations {len(backward_response)} backward: {[(a.activation_id, a.direction) for a in backward_response]}"
            )
            logger.debug(
                f"After filtering, downloading activations: {len(forward_response)} forward: {[(a.activation_id, a.direction) for a in forward_response]}"
            )

            # Download the activations
            download_tasks = [
                asyncio.create_task(self._download_activations(activation_response=r)) for r in backward_response
            ]
            download_tasks.extend(
                [asyncio.create_task(self._download_activations(activation_response=r)) for r in forward_response]
            )
            logger.debug(f"Downloading {len(download_tasks)} activations")

            completed_tasks = set()
            for task in asyncio.as_completed(download_tasks):
                try:
                    downloaded_data: DownloadedData = await task
                    completed_tasks.add(task)
                except asyncio.TimeoutError:
                    logger.warning("Timeout downloading activation -- skipping")
                    continue
                except asyncio.CancelledError:
                    logger.warning("Download task cancelled -- propagating for shutdown")
                    raise
                except (LayerStateException, MinerNotRegisteredException) as e:
                    logger.warning(f"Anticipated exception has occurred while downloading activations: {e}")
                    self._cancel_tasks(tasks=download_tasks, completed_tasks=completed_tasks)
                    raise
                except P2PRequestError as e:
                    logger.warning(f"P2P activation unavailable -- skipping: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error downloading activation -- skipping: {e}")
                    continue
                activation_response = downloaded_data.activation_response
                entry = ActivationData(
                    activation_id=activation_response.activation_id,
                    direction=activation_response.direction,
                    input_activations=downloaded_data.input_activations,
                    sample_activations=downloaded_data.sample_activations,
                    output_activations=None,
                    state=None,
                    upload_time=time.time(),
                    upload_url=activation_response.presigned_upload_url,
                    activation_upload_path=activation_response.activation_upload_path,
                    # For layer-0 forward activations, presigned_download_url IS the sample URL.
                    # Carry it through so the last-layer miner can download target labels via push.
                    target_download_url=activation_response.presigned_download_url,
                )
                logger.debug(
                    f"Downloaded activation {activation_response.activation_id} going {activation_response.direction}"
                )
                async with self._queue_lock:
                    if self._stats_tracker is not None:
                        stats = self._stats_tracker.ensure_activation_stats(
                            activation_response.activation_id,
                            direction=activation_response.direction,
                        )
                        stats.timing.queue.start = time.time()
                    if activation_response.direction == "backward":
                        self._backward_queue.append(entry)
                    else:
                        self._forward_queue.append(entry)

    async def _download_activations(self, activation_response: ActivationResponse) -> DownloadedData:
        """Download an activation from the API (via P2P or S3) and return it."""
        with logger.contextualize(activation_id=activation_response.activation_id):
            async with TimerLoggerMiner(
                name="download_activations",
                metadata={"activation_id": activation_response.activation_id},
                hotkey=self._miner_api_client.hotkey.ss58_address[:8],
            ):
                try:
                    start_time = time.time()
                    input_hash = None

                    # Download the input activations
                    if activation_response.direction == "forward" and self._state_manager.layer == 0:
                        # Layer 0 always downloads samples from S3 (no P2P for initial data)
                        input_activations = await asyncio.wait_for(
                            download_sample(
                                download_url=activation_response.presigned_download_url,
                                tokenizer=self._model_manager.tokenizer,
                                device="cpu",
                                mock=self._mock,
                                run_flags=self._run_flags,
                            ),
                            timeout=common_settings.S3_DOWNLOAD_TIMEOUT,
                        )
                    else:
                        # P2P download for all inter-miner activations
                        if not activation_response.source_node_id:
                            raise RuntimeError(
                                f"No source_node_id for activation {activation_response.activation_id} - "
                                f"P2P routing required but orchestrator did not provide producer node ID"
                            )
                        if not self._miner:
                            raise RuntimeError(
                                f"P2P not initialized for activation {activation_response.activation_id} - "
                                f"miner reference not set in activation queue"
                            )
                        input_activations, input_hash = await self._download_activation_p2p(activation_response)

                    # Store input hash for later submission (if we got one from P2P)
                    if input_hash and self._miner:
                        await self._miner.store_input_hash(activation_response.activation_id, input_hash)

                    # Download the sample for last layer miners as well
                    sample_activations = None
                    sample_download_start = None
                    sample_download_end = None
                    if (
                        activation_response.direction == "forward"
                        and self._state_manager.layer == self._model_manager.model_metadata["n_splits"] - 1
                    ):
                        logger.debug("Last layer miner, downloading sample activations")
                        sample_download_start = time.time()
                        sample_activations = await asyncio.wait_for(
                            download_sample(
                                download_url=activation_response.target_download_url,
                                tokenizer=self._model_manager.tokenizer,
                                device="cpu",
                                mock=self._mock,
                                run_flags=self._run_flags,
                            ),
                            timeout=common_settings.S3_DOWNLOAD_TIMEOUT,
                        )
                        sample_download_end = time.time()
                    total_bytes = tensor_num_bytes(input_activations) + tensor_num_bytes(sample_activations)
                    if self._stats_tracker is not None:
                        self._stats_tracker.record_download(total_bytes)
                    end_time = time.time()
                    if self._stats_tracker is not None:
                        stats = self._stats_tracker.ensure_activation_stats(
                            activation_response.activation_id,
                            direction=activation_response.direction,
                        )
                        stats.timing.download.start = start_time
                        stats.timing.download.end = end_time
                        stats.timing.download.duration = end_time - start_time
                        if sample_download_start is not None:
                            stats.timing.sample_download.start = sample_download_start
                            stats.timing.sample_download.end = sample_download_end
                            stats.timing.sample_download.duration = sample_download_end - sample_download_start
                    download_duration = end_time - start_time
                    if download_duration > 0 and total_bytes > 0:
                        S3_DOWNLOAD_SPEED_BYTES_PER_SEC.labels(layer_idx=str(self._state_manager.layer)).set(
                            total_bytes / download_duration
                        )
                    return DownloadedData(
                        activation_response=activation_response,
                        input_activations=input_activations,
                        sample_activations=sample_activations,
                    )
                except (
                    asyncio.TimeoutError,
                    asyncio.CancelledError,
                    LayerStateException,
                    MinerNotRegisteredException,
                    ActivationHashMismatchError,
                    P2PRequestError,
                ):
                    # Just raise these expected errors to be caught by the caller
                    raise
                except Exception as e:
                    # For these unexpected errors, we want the stack trace
                    logger.exception(f"Failed downloading activation {activation_response.activation_id}: {e}")
                    raise

    async def _download_activation_p2p(self, activation_response: ActivationResponse) -> tuple[torch.Tensor, str]:
        """Download activation via P2P and verify hash."""
        activation_id = activation_response.activation_id
        # Use source_activation_id for P2P request - this is the ID the producer cached
        # activation_id is the new ID assigned by orchestrator for this layer
        source_activation_id = activation_response.source_activation_id or activation_id
        source_node_id = activation_response.source_node_id
        expected_hash = activation_response.expected_input_hash

        logger.debug(
            f"Downloading activation {activation_id} via P2P from {source_node_id[:16]}... "
            f"(requesting as {source_activation_id})"
        )

        # Retry logic and per-phase timeouts live inside
        # Sender.send_message_bi() — we just pass a timings record so
        # the Sender can populate it for us.
        p2p_timings = P2POperationTimings()
        tensor_bytes = await self._miner.request_activation_p2p(
            activation_id=source_activation_id,  # Use the ID the producer knows
            source_node_id=source_node_id,
            timings=p2p_timings,
        )

        # Feed the per-phase breakdown into the stats tracker
        if self._stats_tracker is not None:
            self._stats_tracker.record_p2p_operation(
                activation_id,
                p2p_timings,
                direction=activation_response.direction,
            )

        # Compute hash of received bytes
        received_hash = compute_activation_hash(tensor_bytes)

        # Verify against expected hash if provided
        if expected_hash:
            if not verify_activation_hash(tensor_bytes, expected_hash):
                logger.error(
                    f"HASH MISMATCH for activation {activation_id}: "
                    f"expected={expected_hash[:16]}... received={received_hash[:16]}..."
                )
                raise ActivationHashMismatchError(
                    activation_id=activation_id,
                    expected_hash=expected_hash,
                    received_hash=received_hash,
                )
            logger.debug(f"Hash verified for activation {activation_id}")

        buffer = io.BytesIO(tensor_bytes)
        input_activations = torch.load(buffer, map_location="cpu", weights_only=True)

        if not self._mock:
            input_activations = input_activations.reshape(
                common_settings.MINI_BATCH_SIZE,
                common_settings.SEQUENCE_LENGTH,
                self._model_manager.model_config.get("bottleneck_dim") or self._model_manager.model_config["emb_dim"],
            )
        else:
            input_activations = input_activations.reshape(
                common_settings.MINI_BATCH_SIZE,
                100,
            )

        return input_activations, received_hash

    async def _split_responses(
        self, response: list[ActivationResponse]
    ) -> tuple[list[ActivationResponse], list[ActivationResponse]]:
        """Split the response into backward and forward activations."""
        backward_response = [resp for resp in response if resp.direction == "backward"] if response else []
        forward_response = [resp for resp in response if resp.direction == "forward"] if response else []
        return backward_response, forward_response

    async def _filter_duplicates(self, response: list[ActivationResponse]) -> list[ActivationResponse]:
        """Filter the response to remove any activations that we already have in the cache or queue."""
        # Remove any forward activations that we already have in the cache
        async with self._cache._lock:
            response = [
                resp for resp in response if resp.direction == "backward" or resp.activation_id not in self._cache
            ]
        logger.debug(
            f"After filtering with cache, response contains: {[(a.activation_id, a.direction) for a in response]}"
        )

        # Remove any activations that we already have in the queue, with special handling for backward activations
        # ex1. we already have the forward in the queue so we can remove it from the response
        # ex2. we already have the forward in the queue but we received its backward for it so we should remove the forward from the queue and add the backward
        async with self._queue_lock:
            # Build a map for items in queue
            forward_queue_activation_map = {a.activation_id: i for i, a in enumerate(self._forward_queue)}
            backward_queue_activation_map = {a.activation_id: i for i, a in enumerate(self._backward_queue)}

            filtered_response = []
            indices_to_remove = []

            for resp in response:
                activation_id = resp.activation_id
                response_direction = resp.direction

                if (
                    activation_id not in forward_queue_activation_map
                    and activation_id not in backward_queue_activation_map
                ):
                    filtered_response.append(resp)
                else:
                    forward_queue_index = forward_queue_activation_map.get(activation_id, None)
                    backward_queue_index = backward_queue_activation_map.get(activation_id, None)

                    if response_direction == "forward" and forward_queue_index is not None:
                        # Same direction, skip the response activation (keep the one already in queue)
                        continue
                    elif response_direction == "backward" and backward_queue_index is not None:
                        # Same direction, skip the response activation (keep the one already in queue)
                        continue
                    elif response_direction == "backward" and forward_queue_index is not None:
                        # Backward activation takes priority over forward activation - delete the forward and store the backward
                        # Mark the forward activation for removal from queue and keep the backward response
                        indices_to_remove.append(forward_queue_index)
                        filtered_response.append(resp)
                    else:
                        # Forward response with backward in queue, skip the response (keep backward in queue)
                        continue

            # Remove marked activations from forward queue (in reverse order to maintain indices)
            for index in sorted(indices_to_remove, reverse=True):
                removed_activation = self._forward_queue[index]
                del self._forward_queue[index]
                logger.debug(
                    f"Removed forward activation {removed_activation.activation_id} from queue to make way for backward activation"
                )

            response = filtered_response

        return response

    async def _filter_excess_forwards(self, forward_response: list[ActivationResponse]) -> list[ActivationResponse]:
        """Remove excess forward activations to make sure we leave room for backwards activations in the queue."""
        if len(forward_response) == 0:
            return forward_response

        async with self._queue_lock:
            forwards_in_queue = len(self._forward_queue)
            received_forwards = len(forward_response)

            # If we have too many forwards in process, discard all forward responses
            if forwards_in_queue >= self._max_forwards_in_queue:
                logger.debug("Removing all forward activations from response to make way for backward activations")
                return []

            # If the response contains too many forwards, crop it down to the max
            # e.g. received_forwards = 8 - max_forwards_in_queue = 6 - forwards_in_queue = 1 = 3 (remove 3)
            # e.g. received_forwards = 3 - max_forwards_in_queue = 6 - forwards_in_queue = 1 = -4 (remove 0)
            removal_amount = received_forwards - self._max_forwards_in_queue - forwards_in_queue
            if removal_amount > 0:
                forward_response = forward_response[:-removal_amount]
                logger.debug(
                    f"Removed {removal_amount} forward activations from response to make way for backward activations"
                )
            return forward_response

    async def _drain_layer0_push_queue(self) -> None:
        """Pull all currently available messages from the P2P push queue into the activation queues."""
        if not self._miner:
            return
        while True:
            try:
                msg = self._miner._p2p_push_queue.get_nowait()
                logger.info(
                    f"Activation push RECV | layer0 drain ingress | id={msg.activation_id} "
                    f"dir={msg.direction} src_layer={msg.source_layer}"
                )
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error receiving push message: {e}")
                break

            if msg.direction != "backward":
                # Wrong direction -- discard the message
                logger.warning(
                    f"Discarding {msg.direction} push {msg.activation_id} because we are layer 0 and only expect to receive backwards activations"
                )
                continue

            entry = await self._push_message_to_activation_data(msg)
            if entry is None:
                continue

            async with self._queue_lock:
                if self._stats_tracker is not None:
                    self._stats_tracker.ensure_activation_stats(
                        msg.activation_id,
                        direction=msg.direction,
                        time_received=time.time(),
                    )
                    stats = self._stats_tracker.ensure_activation_stats(msg.activation_id, direction=msg.direction)
                    stats.timing.queue.start = time.time()
                if msg.direction == "backward":
                    self._backward_queue.append(entry)
                else:
                    self._forward_queue.append(entry)
            logger.info(
                f"Activation push RECV | in-process queue | id={msg.activation_id} dir={msg.direction} "
                f"(layer0 backward slot)"
            )

    def _validate_push_layer_routing(self, msg: ActivationPushMessage) -> bool:
        """Reject pushes whose target_layer does not match this miner's training layer."""
        if msg.target_layer is None:
            if msg.source_layer is not None:
                logger.warning(
                    f"Push {msg.activation_id} has source_layer={msg.source_layer} but no target_layer "
                    "(legacy sender); accepting without target check"
                )
            return True
        my_layer = self._state_manager.layer
        if msg.target_layer != my_layer:
            logger.error(
                f"Rejecting {msg.direction} push {msg.activation_id}: target_layer={msg.target_layer} "
                f"but this miner is layer {my_layer}"
            )
            return False
        return True

    async def _push_message_to_activation_data(self, msg: ActivationPushMessage) -> "ActivationData | None":
        """Decode an :class:`ActivationPushMessage` into an :class:`ActivationData`."""
        if not self._validate_push_layer_routing(msg):
            return None
        try:
            import io as _io

            logger.info(
                f"Activation push RECV | materialize start | id={msg.activation_id} dir={msg.direction} "
                f"my_layer={self._state_manager.layer} tensor_bytes={len(msg.tensor_bytes)}"
            )

            buffer = _io.BytesIO(msg.tensor_bytes)
            input_activations = torch.load(buffer, map_location="cpu", weights_only=True)

            if not self._mock and self._model_manager is not None:
                if msg.direction == "forward":
                    input_activations = input_activations.reshape(
                        common_settings.MINI_BATCH_SIZE,
                        common_settings.SEQUENCE_LENGTH,
                        self._model_manager.model_config.get("bottleneck_dim")
                        or self._model_manager.model_config["emb_dim"],
                    )
                else:  # backward gradient has the same shape as the forward input
                    input_activations = input_activations.reshape(
                        common_settings.MINI_BATCH_SIZE,
                        common_settings.SEQUENCE_LENGTH,
                        self._model_manager.model_config.get("bottleneck_dim")
                        or self._model_manager.model_config["emb_dim"],
                    )
            elif self._mock:
                input_activations = self._reshape_mock_activations(input_activations)

            logger.info(
                f"Activation push RECV | tensor decoded | id={msg.activation_id} "
                f"shape={tuple(input_activations.shape)}"
            )

            if self._miner:
                input_hash = compute_activation_hash(msg.tensor_bytes)
                await self._miner.store_input_hash(msg.activation_id, input_hash)

            # Download sample activations for last-layer forward passes
            sample_activations = None
            if (
                msg.direction == "forward"
                and msg.sample_path
                and self._model_manager is not None
                and self._state_manager.layer == self._model_manager.model_metadata["n_splits"] - 1
            ):
                logger.info(
                    f"Activation push RECV | last-layer sample download start | id={msg.activation_id} "
                    f"(presigned path present)"
                )
                sample_activations = await asyncio.wait_for(
                    download_sample(
                        download_url=msg.sample_path,
                        tokenizer=self._model_manager.tokenizer,
                        device="cpu",
                        mock=self._mock,
                        run_flags=self._run_flags,
                    ),
                    timeout=common_settings.S3_DOWNLOAD_TIMEOUT,
                )
                logger.info(f"Activation push RECV | last-layer sample download done | id={msg.activation_id}")

            logger.info(
                f"Activation push RECV | materialize done | id={msg.activation_id} dir={msg.direction} "
                f"sample={'yes' if sample_activations is not None else 'no'}"
            )

            return ActivationData(
                activation_id=msg.activation_id,
                direction=msg.direction,
                input_activations=input_activations,
                sample_activations=sample_activations,
                output_activations=None,
                state=None,
                upload_time=time.time(),
                source_hotkey=msg.source_hotkey,
                source_p2p_node_ids=msg.source_p2p_node_ids,
                target_download_url=msg.sample_path,
            )
        except Exception as exc:
            aid = getattr(msg, "activation_id", "<unknown>")
            logger.error(f"Activation push RECV | materialize failed | id={aid}: {exc}")
            return None

    async def _fetch_activations_push_based(self) -> None:
        """For layer > 0: receive all activations via P2P push instead of polling the orchestrator.

        A lightweight orchestrator heartbeat runs every 30 s to detect layer-state
        changes (which raises :class:`LayerStateException` and resets training).
        """
        last_state_check = time.time()
        while True:
            # ── Periodic layer-state check ────────────────────────────────────
            if time.time() - last_state_check >= 30.0:
                try:
                    await self._miner_api_client.heartbeat(expected_phase=LayerPhase.TRAINING)
                except RateLimitException:
                    pass
                except (LayerStateException, MinerNotRegisteredException):
                    raise
                except Exception as exc:
                    logger.debug(f"Heartbeat check failed (non-fatal): {exc}")
                last_state_check = time.time()

            # Gate: don't pull pushed activations off the queue until every
            # layer is in TRAINING. Pushes queued before the gate opens remain
            # in self._miner._p2p_push_queue and are picked up on a later tick.
            if not await self._all_layers_training():
                logger.debug("Not all layers are in TRAINING yet; deferring push consumption")
                await asyncio.sleep(1.0)
                continue

            # ── Wait for the next push (1 s timeout so the state check fires) ─
            try:
                msg = await asyncio.wait_for(
                    self._miner._p2p_push_queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise

            logger.info(
                f"Activation push RECV | pulled ingress async queue | id={msg.activation_id} "
                f"dir={msg.direction} src_layer={msg.source_layer} tgt_layer={msg.target_layer}"
            )

            entry = await self._push_message_to_activation_data(msg)
            if entry is None:
                continue

            async with self._queue_lock:
                if self._stats_tracker is not None:
                    self._stats_tracker.ensure_activation_stats(
                        msg.activation_id,
                        direction=msg.direction,
                        time_received=time.time(),
                    )
                    stats = self._stats_tracker.ensure_activation_stats(msg.activation_id, direction=msg.direction)
                    stats.timing.queue.start = time.time()
                if msg.direction == "backward":
                    self._backward_queue.append(entry)
                else:
                    self._forward_queue.append(entry)
            logger.info(
                f"Activation push RECV | in-process queue | id={msg.activation_id} dir={msg.direction} "
                f"(layer>0 training queue)"
            )

    def _cancel_tasks(self, tasks: list[asyncio.Task], completed_tasks: Optional[set[asyncio.Task]] = None):
        if completed_tasks is None:
            completed_tasks = set()

        # Cancel all tasks
        for t in tasks:
            if t not in completed_tasks and not t.done():
                t.cancel()
