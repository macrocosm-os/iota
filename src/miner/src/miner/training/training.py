from __future__ import annotations

import torch
from loguru import logger
import asyncio
import time

from common.models.run_flags import RunFlags
from common import settings as common_settings
from common.utils.exceptions import NanInfException
from common.utils.lr_scheduler import make_lr_scheduler
from subnet.utils.vector_utils import check_for_nans_and_infs
from subnet.miner_api_client import MinerAPIClient
from subnet.model.model_mixin import ModelManager
from subnet.model.utils import compute_loss, log_gpu_memory_usage
from subnet.model import gpu_device


from miner.utils.stats import StatsTracker
from miner.utils.timer_logger import TimerLoggerMiner
from miner import settings as miner_settings
from miner.state_manager import StateManager
from miner.training.activation_cache import ActivationData, ActivationCache
from miner.training.activation_queue import ActivationQueue
from miner.training.activation_publisher import ActivationPublisher
from miner.training.peer_selection import select_by_capacity
from miner.sync import DistributedCounter, NodeRegistry, SyncedVariable
from miner.sync.variable import sync_run_sync_prefix
from miner.telemetry.metric_registry import (
    ACTIVATIONS_PROCESSED_TOTAL,
    BACKWARD_PASS_DURATION_SECONDS,
    FORWARD_PASS_DURATION_SECONDS,
    TRAINING_LOSS,
)

if False:  # for typing purposes only
    from miner.new_miner import Miner


class TrainingPhase:
    def __init__(
        self,
        miner_api_client: MinerAPIClient,
        state_manager: StateManager,
        model_manager: ModelManager,
        device: str,
        run_flags: RunFlags,
        mock: bool,
        node_registry: SyncedVariable[NodeRegistry],
        miner: Miner,  # ty: ignore[invalid-type-form]
        is_mounted: bool = False,
    ):
        self._miner_api_client = miner_api_client
        self._state_manager = state_manager
        self._model_manager = model_manager
        self._hotkey = miner_api_client.hotkey.ss58_address
        self._device = device
        self._run_flags = run_flags
        self._mock = mock
        self._cache: ActivationCache = ActivationCache(
            hotkey=self._hotkey,
            cache_timeout_sec=common_settings.activation_cache_timeout_sec(
                num_layers=self._model_manager.model_metadata["n_splits"],
                layer_idx=self._state_manager.layer,
            ),
        )
        self._queue: ActivationQueue = ActivationQueue(
            miner_api_client=self._miner_api_client,
            state_manager=self._state_manager,
            activation_cache=self._cache,
            mock=self._mock,
            run_flags=self._run_flags,
            miner=miner,
        )

        self._publisher = ActivationPublisher(
            miner_api_client=self._miner_api_client,
            run_flags=self._run_flags,
            miner=miner,
            node_registry=node_registry,
            peer_selector=select_by_capacity,
        )
        self._stats_tracker: StatsTracker | None = None
        self.backwards_since_reset = 0
        self.backwards_since_last_optim = 0
        self.local_optimization_steps = 0
        self.is_mounted = is_mounted
        self._auto_max_cache_frozen: bool = False
        if self._run_flags.auto_max_cache.isOn():
            self._cache._warmup_active = True
        self.miner = miner
        self._loss_on_cpu: bool = False
        self.node_registry = node_registry
        # Counter is created lazily after registration so Redis keys include ``run_id``.
        self._backward_counter: DistributedCounter | None = None
        self._counter_started: bool = False
        self._lr_scheduler = make_lr_scheduler()

    async def rebind_backward_counter_for_run(self) -> None:
        """Drop the distributed counter so the next opt step binds a run-scoped Redis key."""
        if self._backward_counter is not None:
            await self._backward_counter.stop()
            self._backward_counter = None
        self._counter_started = False

    async def _ensure_backward_counter(self) -> DistributedCounter:
        """Build or replace the backward counter for the current run and layer."""
        ns = sync_run_sync_prefix(self._state_manager.run_id)
        name = f"global_backward_count_{self._state_manager.layer}"
        if (
            self._backward_counter is not None
            and getattr(self._backward_counter, "_namespace", None) == ns
            and getattr(self._backward_counter, "_name", None) == name
        ):
            return self._backward_counter
        if self._backward_counter is not None:
            await self._backward_counter.stop()
        self._backward_counter = DistributedCounter(
            name=name,
            server_url=common_settings.BRIDGE_URL.rstrip("/"),
            namespace=ns,
            poll_interval=5,
        )
        return self._backward_counter

    def _record_forward_timing(self, activation_data: ActivationData, start_time: float, end_time: float) -> None:
        if self._stats_tracker is None:
            return
        stats = self._stats_tracker.ensure_activation_stats(
            activation_data.activation_id,
            direction=activation_data.direction,
        )
        stats.timing.forward.start = start_time
        stats.timing.forward.end = end_time
        stats.timing.forward.duration = end_time - start_time
        stats.timing.forward.cache_len = len(self._cache)
        stats.timing.forward.forward_queue_len = len(self._queue._forward_queue)
        stats.timing.forward.backward_queue_len = len(self._queue._backward_queue)

    def attach_stats_tracker(self, tracker: StatsTracker | None) -> None:
        """Attach a stats tracker and propagate to child components."""
        self._stats_tracker = tracker
        self._queue.attach_stats_tracker(tracker)
        self._publisher.attach_stats_tracker(tracker)

    async def run(self):
        try:
            await self._queue.start_activation_fetcher(model_manager=self._model_manager)

            last_activation_time = time.time()
            while True:
                await asyncio.sleep(0.01)  # yield control back to the event loop
                if self._stats_tracker is not None:
                    self._stats_tracker.set_phase(self._miner_api_client.layer_state)
                    self._stats_tracker.set_layer(self._state_manager.layer)
                    self._stats_tracker.set_local_epoch(getattr(self._model_manager, "epoch_counter", None))
                self._publisher.layer_idx = str(self._state_manager.layer)

                # Check if training phase is complete
                await self._queue.check_if_training_is_complete()

                if self._cache.is_full() and self._queue.next_activation_is_forward():
                    logger.debug(f"Activation cache is full ({len(self._cache)}), waiting for backward activations")
                    await asyncio.sleep(1)
                    continue

                # Get next activation to process
                activation = await self._queue.get_activation()
                if activation is None:
                    continue
                if self._stats_tracker is not None:
                    stats = self._stats_tracker.ensure_activation_stats(
                        activation.activation_id,
                        direction=activation.direction,
                    )
                    stats.timing.queue.end = time.time()
                    if stats.timing.queue.start is not None:
                        stats.timing.queue.duration = stats.timing.queue.end - stats.timing.queue.start

                with logger.contextualize(
                    activation_id=activation.activation_id,
                    time_since_last_activation=time.time() - last_activation_time,
                ):
                    last_activation_time = time.time()
                    # initialisation completed
                    if self.miner and self.is_mounted:
                        await self.miner.register_set_status(status="initialized")

                    if activation.direction == "forward":
                        await self.forward(activation)
                    elif activation.direction == "backward":
                        await self.backward(activation)

                # Loop until LayerStateException is raised by `get_activation`
                logger.debug(
                    f"Node registry contains nodes: {[node.node_id for node in self.node_registry.value.all_nodes()]}"
                )
        except Exception:
            logger.info("Finishing training phase")
            raise
        finally:
            # TODO: @cassova: determine if we want to add an optimization step here too
            # considering the last activation submission may have failed.
            await self.optimization_reset()
            log_gpu_memory_usage(note="after training phase cleanup")

    async def forward(self, activation_data: ActivationData):
        """
        Performs the forward pass.

        If the layer is 0, it will load the data and upload the initial activation to the API.
        If the layer is not 0, it will download a random forward activation from the API and perform the forward pass.

        The forward pass contains:
        - Downloading the forward activation from the API
        - Performing the forward pass
        - Reporting the loss to the API
        - Performing the backward pass
        """
        logger.debug(
            f"Forward pass for activation {activation_data.activation_id} on layer {self._state_manager.layer}"
        )
        with logger.contextualize(cache_size=len(self._cache)):
            async with TimerLoggerMiner(
                name="forward",
                metadata={
                    "hotkey": self._hotkey[:8],
                    "activation_id": activation_data.activation_id,
                    "layer": self._state_manager.layer,
                },
                hotkey=self._hotkey[:8],
            ):
                if self._stats_tracker is not None:
                    self._stats_tracker.record_forward()
                start_time = time.time()
                logger.debug(
                    f"Starting FORWARD pass | layer={self._state_manager.layer} activation={activation_data.activation_id} hotkey={self._hotkey[:8]}"
                )
                log_gpu_memory_usage(note="starting training forward pass")
                if self._state_manager.layer == 0:
                    logger.debug(f"Got sample shape: {activation_data.input_activations.shape}")
                else:
                    logger.debug(f"Got activation shape: {activation_data.input_activations.shape}")

                # Move to GPU
                self._model_manager.model = self._model_manager.model.to(self._device)
                input_activations_gpu = activation_data.input_activations.to(self._device)

                # populate cache
                # activation_data.state = state
                activation_data.input_activations = input_activations_gpu  # keep it on the gpu while in cache
                activation_data.output_activations = None
                activation_data.upload_time = time.time()
                self._cache[activation_data.activation_id] = activation_data

                if self._state_manager.layer == self._model_manager.model_metadata["n_splits"] - 1:
                    logger.debug(
                        f"Last layer miner, performing backward pass for activation {activation_data.activation_id}"
                    )
                    self._record_forward_timing(activation_data, start_time, end_time=time.time())
                    return await self.backward(activation_data=activation_data)

                logger.debug(f"Forwarding activation of size {activation_data.input_activations.shape}")
                output_activations_gpu, _ = await self._model_manager._forward_no_intermittent_activations(
                    input_activations=input_activations_gpu, processing_batch_size=miner_settings.LOCAL_BATCH_SIZE
                )

                # Cleanup GPU memory
                output_activations_cpu = output_activations_gpu.detach().cpu()
                del output_activations_gpu
                logger.debug(f"Activation shape after forward: {output_activations_cpu.shape}")
                log_gpu_memory_usage(note="after training forward pass")

                end_time = time.time()
                self._record_forward_timing(activation_data, start_time, end_time)

                layer_idx = str(self._state_manager.layer)
                FORWARD_PASS_DURATION_SECONDS.labels(layer_idx=layer_idx).observe(end_time - start_time)
                ACTIVATIONS_PROCESSED_TOTAL.labels(direction="forward", layer_idx=layer_idx).inc()

                self._publisher.publish_activation(
                    tensor=output_activations_cpu,
                    activation_id=activation_data.activation_id,
                    direction="forward",
                    upload_url=activation_data.upload_url,
                    activation_path=activation_data.activation_upload_path,
                    source_p2p_node_ids=None,  # forward: publisher picks next-layer peer from registry
                    sample_path=activation_data.target_download_url,
                )

                log_gpu_memory_usage(note="after training forward pass cleaning on non-last layer miner")
                logger.success(
                    f"✅ FORWARD complete | layer={self._state_manager.layer} activation={activation_data.activation_id} hotkey={self._hotkey[:8]}"
                )
                logger.debug(
                    f"Node registry in forward contains: {[m.node_id for m in self.node_registry.value.all_nodes()]}"
                )

    async def backward(self, activation_data: ActivationData):
        """
        Performs the backward pass.
        """
        with logger.contextualize(cache_size=len(self._cache)):
            async with TimerLoggerMiner(
                name="backward",
                metadata={"hotkey": self._hotkey[:8], "activation_id": activation_data.activation_id},
                hotkey=self._hotkey[:8],
            ):
                if self._stats_tracker is not None:
                    self._stats_tracker.record_backward()
                start_time = time.time()
                last_layer = self._state_manager.layer == self._model_manager.model_metadata["n_splits"] - 1
                all_input_activations_grads = []
                losses = []

                logger.info(
                    f"🔄 BACKWARD pass | layer={self._state_manager.layer} activation={activation_data.activation_id} hotkey={self._hotkey[:8]}"
                )

                # Sub-phase timing accumulators (accumulated across local batch iterations)
                gpu_setup_total = 0.0
                bwd_fwd_total = 0.0
                bwd_loss_total = 0.0
                bwd_pass_total = 0.0
                grad_extract_total = 0.0

                for i in range(0, len(activation_data.input_activations), miner_settings.LOCAL_BATCH_SIZE):
                    log_gpu_memory_usage(
                        note=f"after training forward pass cleaning on last layer miner with cache size of {len(self._cache)}"
                    )
                    gpu_setup_start = time.time()
                    async with TimerLoggerMiner(name="moving to gpu", hotkey=self._hotkey[:8]):
                        log_gpu_memory_usage(note="starting training backward pass")

                        # Check if activation is in cache
                        if activation_data.activation_id not in self._cache:
                            logger.warning(
                                f"⚠️ Activation {activation_data.activation_id} not found in cache, skipping backward pass"
                            )
                            return

                        # Move to GPU and enable gradients only for floating point tensors
                        self._model_manager.model = self._model_manager.model.to(self._device)
                        cached_input_activation = self._cache[activation_data.activation_id].input_activations.to(
                            self._device
                        )
                        backwards_grads_from_previous_miner = activation_data.input_activations.to(self._device)

                        # Take slices
                        sliced_cached_input_activation = (
                            cached_input_activation[i : i + miner_settings.LOCAL_BATCH_SIZE].clone().contiguous()
                        )

                        if last_layer:
                            sliced_backwards_grads_from_previous_miner = None
                            sliced_targets = activation_data.sample_activations[
                                i : i + miner_settings.LOCAL_BATCH_SIZE
                            ].contiguous()
                        else:
                            sliced_backwards_grads_from_previous_miner = (
                                backwards_grads_from_previous_miner[i : i + miner_settings.LOCAL_BATCH_SIZE]
                                .clone()
                                .contiguous()
                            )

                        if self._state_manager.layer > 0:
                            sliced_cached_input_activation.requires_grad_(True)
                    gpu_setup_total += time.time() - gpu_setup_start

                    bwd_fwd_start = time.time()
                    output_activations_gpu, state = await self._model_manager._forward(
                        layer=self._state_manager.layer,
                        input_activations=sliced_cached_input_activation,
                    )
                    bwd_fwd_total += time.time() - bwd_fwd_start

                    log_gpu_memory_usage(note="after preparing activations on training backward pass")

                    # Compute loss; if targets download or loss computation fails, skip backward gracefully
                    if last_layer:
                        try:
                            bwd_loss_start = time.time()
                            logger.debug(
                                f"Computing loss for last layer miner with shape {output_activations_gpu.shape} and targets shape {sliced_targets.shape} on local batch {i} of {len(activation_data.input_activations)/miner_settings.LOCAL_BATCH_SIZE}"
                            )
                            # logger.debug(f"Targets (shape: {sliced_targets.shape}): {sliced_targets}")
                            loss = await self.compute_last_layer_loss(
                                activation_data=activation_data, logits=output_activations_gpu, targets=sliced_targets
                            )
                            # Ex: batch = 8, local_batch = 2, so we divide by 4
                            output_activations_gpu = loss / (
                                len(activation_data.input_activations) / miner_settings.LOCAL_BATCH_SIZE
                            )
                            # output_activations_gpu = loss
                            losses.append(loss.item())
                            bwd_loss_total += time.time() - bwd_loss_start
                        except Exception as e:
                            logger.exception(
                                f"Skipping backward for activation {activation_data.activation_id} due to loss/target fetch error: {e}"
                            )
                            return

                    bwd_pass_start = time.time()
                    async with TimerLoggerMiner(name="backward pass", hotkey=self._hotkey[:8]):
                        await self._model_manager._backward(
                            layer=self._state_manager.layer,
                            output_activations=output_activations_gpu,
                            activation_grads=sliced_backwards_grads_from_previous_miner,
                            state=state,
                        )
                    bwd_pass_total += time.time() - bwd_pass_start
                    log_gpu_memory_usage(note="after backward pass")

                    # Handle different cases for input activation gradients
                    grad_extract_start = time.time()
                    if self._mock:
                        input_activation_grads = sliced_cached_input_activation.detach().to(torch.bfloat16).cpu()

                    elif self._state_manager.layer == 0:
                        # Get the embedding layer weight grads instead of the input activations grads
                        # This is because input activation grads of the first layer do not exist.
                        emb_weight = self._model_manager.model.tok_emb.weight
                        embedding_dim = (
                            self._model_manager.model_config["bottleneck_dim"]
                            or self._model_manager.model_config["emb_dim"]
                        )
                        n_grad_elems = common_settings.SEQUENCE_LENGTH * embedding_dim * common_settings.MINI_BATCH_SIZE
                        # Same values as flatten().cpu()[:n] before, but slice + bf16 on GPU then D2H only for the prefix.
                        input_activation_grads = (
                            emb_weight.grad.detach().reshape(-1)[:n_grad_elems].to(torch.bfloat16).cpu()
                        )
                    else:
                        input_activation_grads = sliced_cached_input_activation.grad.detach().cpu()
                    grad_extract_total += time.time() - grad_extract_start

                    log_gpu_memory_usage(note="after moving input activation grads to GPU")
                    all_input_activations_grads.append(input_activation_grads)
                end_time = time.time()
                if self._stats_tracker is not None:
                    stats = self._stats_tracker.ensure_activation_stats(
                        activation_data.activation_id,
                        direction=activation_data.direction,
                    )
                    stats.timing.backward.start = start_time
                    stats.timing.backward.end = end_time
                    stats.timing.backward.duration = end_time - start_time
                    stats.timing.backward.cache_len = len(self._cache)
                    stats.timing.backward.forward_queue_len = len(self._queue._forward_queue)
                    stats.timing.backward.backward_queue_len = len(self._queue._backward_queue)

                    # Sub-phase durations (accumulated across local batch iterations)
                    stats.timing.backward_gpu_setup.duration = gpu_setup_total
                    stats.timing.backward_forward.duration = bwd_fwd_total
                    stats.timing.backward_loss.duration = bwd_loss_total
                    stats.timing.backward_pass.duration = bwd_pass_total
                    stats.timing.backward_grad_extract.duration = grad_extract_total
                layer_idx = str(self._state_manager.layer)
                BACKWARD_PASS_DURATION_SECONDS.labels(layer_idx=layer_idx).observe(end_time - start_time)
                ACTIVATIONS_PROCESSED_TOTAL.labels(direction="backward", layer_idx=layer_idx).inc()

                async with TimerLoggerMiner(name="publishing_backwards", hotkey=self._hotkey[:8]):
                    logger.debug(f"Backwards since reset for miner {self._hotkey[:8]}: {self.backwards_since_reset}")

                    logger.debug(f"All input activations grads shape: {len(all_input_activations_grads)}")
                    self.backwards_since_reset += 1
                    mean_loss: float | None = None
                    if losses:
                        mean_loss = sum(losses) / len(losses)
                        TRAINING_LOSS.labels(layer_idx=layer_idx).set(mean_loss)
                        self._publisher.publish_loss(
                            loss=mean_loss,
                            activation_id=activation_data.activation_id,
                            layer_idx=self._state_manager.layer,
                        )
                        if self._stats_tracker is not None:
                            self._stats_tracker.record_loss(mean_loss)
                    self._publisher.publish_activation(
                        tensor=torch.cat(all_input_activations_grads, dim=0),
                        activation_id=activation_data.activation_id,
                        direction="backward",
                        upload_url=activation_data.upload_url,
                        activation_path=activation_data.activation_upload_path,
                        source_p2p_node_ids=self._cache[activation_data.activation_id].source_p2p_node_ids or None,
                    )

                async with TimerLoggerMiner(name="cleaning up cache", hotkey=self._hotkey[:8]):
                    # Cleanup cache
                    del self._cache[activation_data.activation_id]

                    # End of activation lifecycle: drop its stats entry so the
                    # StatsTracker dict doesn't grow unboundedly with microbatches.
                    if self._stats_tracker is not None:
                        self._stats_tracker.discard_activation_stats(activation_data.activation_id)

                    # auto_max_cache: freeze max cache size after N backward activations
                    if self._run_flags.auto_max_cache.isOn() and not self._auto_max_cache_frozen:
                        if self.backwards_since_reset >= common_settings.N_BACKWARDS_FOR_CACHE_INCREASE_STOP:
                            frozen_size = len(self._cache)
                            logger.info(
                                f"🔒 auto_max_cache: freezing max cache size at {frozen_size} "
                                f"after {self.backwards_since_reset} backward activations | hotkey={self._hotkey[:8]}"
                            )
                            self._cache._warmup_active = False
                            self._cache._frozen_max_size = frozen_size
                            self._auto_max_cache_frozen = True

                    # Cleanup GPU memory
                    del output_activations_gpu, cached_input_activation, backwards_grads_from_previous_miner

                    with logger.contextualize(cache_size=len(self._cache)):
                        log_gpu_memory_usage(note="after training backward pass cleaning")

                        # Check if we need to perform a local optimization step
                        self.backwards_since_last_optim += 1
                        mini_batch_accumulation_count = (
                            self._model_manager.model_metadata.get("mini_batch_accumulation_count")
                            or common_settings.MINI_BATCH_ACCUMULATION_COUNT
                        )
                        if self.backwards_since_last_optim >= mini_batch_accumulation_count:
                            logger.info(
                                f"🔄 Local optimization step after {mini_batch_accumulation_count} backward passes | hotkey={self._hotkey[:8]}"
                            )
                            if not self._counter_started:
                                counter = await self._ensure_backward_counter()
                                await counter.start()
                                self._counter_started = True
                            global_step = await self._backward_counter.increment()
                            logger.debug(f"Global step for miner {self._hotkey[:8]}: {global_step}")
                            learning_rate = self._lr_scheduler.step(global_step)
                            logger.debug(f"LR at global step {global_step}: {learning_rate}")
                            await self._model_manager.local_optimization_step(learning_rate=learning_rate)
                            await self.optimization_reset()

                            log_gpu_memory_usage(note="after local optimization step")

                            self.local_optimization_steps += 1
                            logger.success(
                                f"✅ Optimization step #{self.local_optimization_steps} | hotkey={self._hotkey[:8]}"
                            )

                        logger.success(
                            f"✅ BACKWARD complete | layer={self._state_manager.layer} activation={activation_data.activation_id} hotkey={self._hotkey[:8]}"
                        )

                if self._stats_tracker is not None:
                    self._stats_tracker.set_local_epoch(getattr(self._model_manager, "epoch_counter", None))

    async def compute_last_layer_loss(
        self, activation_data: ActivationData, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the backward pass for the last layer.
        """
        async with TimerLoggerMiner(
            name="compute_last_layer_loss",
            metadata={
                "hotkey": self._hotkey[:8],
                "activation_id": activation_data.activation_id,
                "layer": self._state_manager.layer,
            },
            hotkey=self._hotkey[:8],
        ):
            # NOTE: targets are on the CPU at this point
            # the problem is that loss calculation is very heavy on the GPU memory
            # on A4000 a 1B, when on GPU, it took 0.03s to compute the loss - on the CPU performance it took 0.5s
            if self._loss_on_cpu:
                device = "cpu"  # If I failed on the gpu once, just calculate on the cpu from now on
            else:
                device = self._device
            try:
                loss: torch.Tensor = self._compute_loss_on_device(logits, targets, device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.exception(f"Out of memory error while computing loss on GPU: {e}")
                    gpu_device.empty_cache()
                    self._loss_on_cpu = True
                    loss = self._compute_loss_on_device(logits, targets, "cpu")
                else:
                    raise e

            check_for_nans_and_infs(
                tensor=loss, name=f"Loss for miner {self._hotkey[:8]}", exception_type=NanInfException
            )

            # Update cache with loss before attempting to report it to handle API errors gracefully
            activation_data.upload_time = time.time()
            self._cache[activation_data.activation_id] = activation_data

            return loss

    def _compute_loss_on_device(self, logits: torch.Tensor, targets: torch.Tensor, device: str) -> torch.Tensor:
        """Compute loss on the CPU regardless of current device setting."""
        loss: torch.Tensor = compute_loss(
            mock=self._mock,
            logits=logits,
            targets=targets,
            vocab_size=self._model_manager.vocab_size,
            pad_token_id=self._model_manager.eos_token_id,
            pack=miner_settings.PACK_SAMPLES,
            device=device,
        )
        return loss

    async def shutdown(self) -> None:
        """Tear down background tasks and release resources held by this instance.

        Must be called before discarding a ``TrainingPhase`` (e.g. on re-registration)
        so the old publisher send loop, activation fetcher task, and distributed
        backward counter don't linger alongside a freshly built replacement.
        """
        logger.debug("🧹 Shutting down TrainingPhase")

        # Stop the publisher's outbound send loop and cancel any in-flight publish tasks.
        try:
            self._publisher.stop_send_loop()
        except Exception as e:
            logger.error(f"Failed to stop publisher send loop: {e}")

        pending_publishes = [t for t in self._publisher._publishing_tasks if not t.done()]
        for task in pending_publishes:
            task.cancel()
        if pending_publishes:
            await asyncio.gather(*pending_publishes, return_exceptions=True)
        self._publisher._publishing_tasks.clear()

        # Cancel the activation fetcher if it's somehow still alive. By normal flow
        # it has already exited via LayerStateException, but make the teardown
        # idempotent so we never leave a dangling task.
        fetcher = self._queue._activation_fetcher_task
        if fetcher is not None and not fetcher.done():
            fetcher.cancel()
            try:
                await fetcher
            except (asyncio.CancelledError, Exception):
                pass

        # Stop the distributed counter so its background poller/Redis connection exits.
        if self._backward_counter is not None:
            try:
                await self._backward_counter.stop()
            except Exception as e:
                logger.error(f"Failed to stop backward counter: {e}")
            self._backward_counter = None
        self._counter_started = False

        # Drop cached activations to release GPU memory before the new instance loads weights.
        try:
            await self._cache.reset()
        except Exception as e:
            logger.error(f"Failed to reset activation cache on shutdown: {e}")

    async def epoch_reset(self):
        """Reset cache and queues between epochs.

        Model weights change entirely after a merge, so all cached activations
        and queued work from the previous epoch are stale and must be discarded.
        """
        logger.debug("🗑️ Resetting for new epoch")
        logger.info(
            f"epoch_reset clearing: publishing_tasks={len(self._publisher._publishing_tasks)} "
            f"outbound_qsize={self._publisher._outbound.qsize()} "
            f"removal_tasks={len(self._cache._removal_tasks)} "
            f"forward_queue={len(self._queue._forward_queue)} "
            f"backward_queue={len(self._queue._backward_queue)}"
        )
        self.local_optimization_steps = 0
        await self._cache.reset()
        self._queue._forward_queue.clear()
        self._queue._backward_queue.clear()

        # Publisher state (outbound queue, _publishing_tasks, peer-lookup dict)
        # otherwise only clears on re-registration, so it accumulates host RAM
        # across the whole run. publisher.reset() cancels pending publishes,
        # drains the outbound queue, clears the task list, and restarts the
        # send loop so new activations can be drained after epoch reset.
        await self._publisher.reset()

        log_gpu_memory_usage(note="after epoch reset")

    async def optimization_reset(self):
        """Reset the cache and backward pass counter after performing optimization step."""
        logger.debug("🗑️ Resetting after optimization step")
        self.backwards_since_last_optim = 0

        # we can't process backwards activations on forwards processed before the optimization step
        if self._run_flags.keep_cache_on_local_step.isOff():
            await self._cache.reset()
        log_gpu_memory_usage(note="after cache reset")
