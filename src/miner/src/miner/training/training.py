from __future__ import annotations

from common.models.run_flags import RunFlags
from typing import Any
from miner.utils.timer_logger import TimerLoggerMiner
import torch
from loguru import logger
import asyncio
import time

from common import settings as common_settings
from common.utils.exceptions import NanInfException
from subnet.utils.vector_utils import check_for_nans_and_infs
from miner import settings as miner_settings
from miner.state_manager import StateManager
from miner.training.activation_cache import ActivationData, ActivationCache
from miner.training.activation_queue import ActivationQueue
from miner.training.activation_publisher import ActivationPublisher
from subnet.miner_api_client import MinerAPIClient
from subnet.model.model_mixin import ModelManager
from subnet.model.utils import compute_loss, log_gpu_memory_usage
from subnet.model import gpu_device

from miner.utils.stats import StatsTracker
from miner.telemetry.metric_registry import (
    ACTIVATIONS_PROCESSED_TOTAL,
    BACKWARD_PASS_DURATION_SECONDS,
    FORWARD_PASS_DURATION_SECONDS,
    TRAINING_LOSS,
)


class TrainingPhase:
    def __init__(
        self,
        miner_api_client: MinerAPIClient,
        state_manager: StateManager,
        model_manager: ModelManager,
        device: str,
        run_flags: RunFlags,
        mock: bool,
        is_mounted: bool = False,
        miner: Any | None = None,
    ):
        self._miner_api_client = miner_api_client
        self._state_manager = state_manager
        self._model_manager = model_manager
        self._hotkey = miner_api_client.hotkey.ss58_address
        self._device = device
        self._run_flags = run_flags
        self._mock = mock
        self._cache: ActivationCache = ActivationCache(miner_api_client=self._miner_api_client)
        self._queue: ActivationQueue = ActivationQueue(
            miner_api_client=self._miner_api_client,
            state_manager=self._state_manager,
            activation_cache=self._cache,
            mock=self._mock,
            run_flags=self._run_flags,
        )
        self._publisher = ActivationPublisher(
            miner_api_client=self._miner_api_client, run_flags=self._run_flags, miner=miner
        )
        self._stats_tracker: StatsTracker | None = None
        self.backwards_since_reset = 0
        self.backwards_since_last_optim = 0
        self.local_optimization_steps = 0
        self.is_mounted = is_mounted
        self.miner = miner
        self._loss_on_cpu: bool = False

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
                    logger.info(f"Activation cache is full. Waiting for backwards activations: {len(self._cache)}")
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
                logger.info(
                    f"🚀 Starting FORWARD pass for layer {self._state_manager.layer} | Processing activation {activation_data.activation_id} | Miner: {self._hotkey[:8]}"
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

                # If we are not on the last layer, we just need to upload the activations
                logger.info(
                    f"output activations before upload with shape {output_activations_cpu.shape} for {self._hotkey[:8]} on layer {self._state_manager.layer}"
                )
                end_time = time.time()
                if self._stats_tracker is not None:
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

                layer_idx = str(self._state_manager.layer)
                FORWARD_PASS_DURATION_SECONDS.labels(layer_idx=layer_idx).observe(end_time - start_time)
                ACTIVATIONS_PROCESSED_TOTAL.labels(direction="forward", layer_idx=layer_idx).inc()

                self._publisher.publish_activation(
                    tensor=output_activations_cpu,
                    activation_id=activation_data.activation_id,
                    direction="forward",
                    attestation_challenge_blob=activation_data.attestation_challenge_blob,
                    attestation_self_checks=activation_data.attestation_self_checks,
                    attestation_crypto=activation_data.attestation_crypto,
                    upload_url=activation_data.upload_url,
                    activation_path=activation_data.activation_upload_path,
                )

                log_gpu_memory_usage(note="after training forward pass cleaning on non-last layer miner")
                logger.success(
                    f"✅ Successfully completed FORWARD pass for activation {activation_data.activation_id} on layer {self._state_manager.layer} | Miner: {self._hotkey[:8]}"
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
                for i in range(0, len(activation_data.input_activations), miner_settings.LOCAL_BATCH_SIZE):
                    log_gpu_memory_usage(
                        note=f"after training forward pass cleaning on last layer miner with cach size of {len(self._cache)}"
                    )
                    logger.info(
                        f"🔄 Starting BACKWARD pass for activation {activation_data.activation_id} | Layer: {self._state_manager.layer} | Miner: {self._hotkey[:8]}"
                    )
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

                        output_activations_gpu, state = await self._model_manager._forward(
                            layer=self._state_manager.layer,
                            input_activations=sliced_cached_input_activation,
                        )

                        log_gpu_memory_usage(note="after preparing activations on training backward pass")

                        logger.info(
                            f"output activations before backward with shape {output_activations_gpu.shape} for {self._hotkey[:8]} on layer {self._state_manager.layer}"
                        )

                    # Compute loss; if targets download or loss computation fails, skip backward gracefully
                    if last_layer:
                        try:
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
                        except Exception as e:
                            logger.exception(
                                f"Skipping backward for activation {activation_data.activation_id} due to loss/target fetch error: {e}"
                            )
                            return

                    logger.debug(f"State: {state}")
                    async with TimerLoggerMiner(name="backward pass", hotkey=self._hotkey[:8]):
                        await self._model_manager._backward(
                            layer=self._state_manager.layer,
                            output_activations=output_activations_gpu,
                            activation_grads=sliced_backwards_grads_from_previous_miner,
                            state=state,
                        )
                    log_gpu_memory_usage(note="after backward pass")

                    # Handle different cases for input activation gradients
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
                        # Detach and convert to bfloat16 to ensure we only save the values
                        # NOTE: if we cast to bfloat16 after moving to CPU, there will be extra load if the grad is huge
                        # but if we do it before, we'll be taking up additional GPU memory
                        grad_flattened = emb_weight.grad.detach().cpu().to(torch.bfloat16).flatten()
                        input_activation_grads = grad_flattened[
                            : common_settings.SEQUENCE_LENGTH * embedding_dim * common_settings.MINI_BATCH_SIZE
                        ]
                    else:
                        input_activation_grads = sliced_cached_input_activation.grad.detach().cpu()

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
                        self._publisher.publish_loss(loss=mean_loss, activation_id=activation_data.activation_id)
                        if self._stats_tracker is not None:
                            self._stats_tracker.record_loss(mean_loss)
                    self._publisher.publish_activation(
                        tensor=torch.cat(all_input_activations_grads, dim=0),
                        activation_id=activation_data.activation_id,
                        direction="backward",
                        attestation_challenge_blob=activation_data.attestation_challenge_blob,
                        attestation_self_checks=activation_data.attestation_self_checks,
                        attestation_crypto=activation_data.attestation_crypto,
                        upload_url=activation_data.upload_url,
                        activation_path=activation_data.activation_upload_path,
                    )

                async with TimerLoggerMiner(name="cleaning up cache", hotkey=self._hotkey[:8]):
                    # Cleanup cache
                    del self._cache[activation_data.activation_id]

                    # Cleanup GPU memory
                    del output_activations_gpu, cached_input_activation, backwards_grads_from_previous_miner

                    with logger.contextualize(cache_size=len(self._cache)):
                        log_gpu_memory_usage(note="after training backward pass cleaning")

                        # Check if we need to perform a local optimization step
                        self.backwards_since_last_optim += 1
                        if self.backwards_since_last_optim >= common_settings.MINI_BATCH_ACCUMULATION_COUNT:
                            logger.info(
                                f"🔄 Miner {self._hotkey[:8]} performing local optimization step after {common_settings.MINI_BATCH_ACCUMULATION_COUNT} backward passes"
                            )
                            learning_rate = await self._miner_api_client.get_learning_rate()
                            await self._model_manager.local_optimization_step(learning_rate=learning_rate)
                            await self.optimization_reset()

                            log_gpu_memory_usage(note="after local optimization step")

                            self.local_optimization_steps += 1
                            logger.success(
                                f"✅ Miner {self._hotkey[:8]} completed local optimization step #{self.local_optimization_steps}"
                            )

                        logger.success(
                            f"✅ Successfully completed BACKWARD pass for activation {activation_data.activation_id} | Layer: {self._state_manager.layer} | Miner: {self._hotkey[:8]}"
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

    async def reset(self):
        """Reset the training phase."""
        logger.debug("🗑️ Performing full reset of training")
        self.backwards_since_reset = 0
        self.local_optimization_steps = 0
        self._loss_on_cpu = False
        await self.optimization_reset()
        await self._publisher.reset()

    async def optimization_reset(self):
        """Reset the cache and backward pass counter after performing optimization step."""
        logger.debug("🗑️ Resetting after optimization step")
        self.backwards_since_last_optim = 0

        # we can't process backwards activations on forwards processed before the optimization step
        if self._run_flags.keep_cache_on_local_step.isOff():
            await self._cache.reset()
        log_gpu_memory_usage(note="after cache reset")
