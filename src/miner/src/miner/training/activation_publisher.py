from __future__ import annotations

import asyncio
import io
import json
from typing import Any
from common.utils.verify_enclave_signature import payload_base64_from_obj
from loguru import logger
from miner.utils.timer_logger import TimerLoggerMiner
import torch
import time
from common.models.api_models import (
    AttestationChallengeResponse,
    LossReportRequest,
    MinerAttestationPayload,
    SubmitActivationRequest,
)
from common.models.run_flags import RunFlags, RUN_FLAGS
from common.utils.exceptions import LayerStateException, MinerNotRegisteredException
from miner.utils.attestation_utils import AttestationUnavailableError, collect_attestation_payload
from miner.utils.activation_hash import compute_activation_hash
from subnet.miner_api_client import MinerAPIClient

from miner.utils.stats import StatsTracker


class ActivationPublisher:
    def __init__(self, miner_api_client: MinerAPIClient, run_flags: RunFlags | None = None, miner: Any | None = None):
        self._miner_api_client = miner_api_client
        self._publishing_tasks: list[asyncio.Task] = []
        self._stats_tracker: StatsTracker | None = None
        self._run_flags: RunFlags = run_flags or RUN_FLAGS
        self.miner = miner
        self.layer_idx: str | None = None

    def attach_stats_tracker(self, tracker: StatsTracker | None) -> None:
        """Attach a stats tracker for dashboard metrics."""
        self._stats_tracker = tracker

    def publish_activation(
        self,
        tensor: torch.Tensor,
        activation_id: str,
        direction: str,
        attestation_challenge_blob: str | None,
        attestation_self_checks: list[str] | None,
        attestation_crypto: str | None,
        upload_url: list[str] | None,
        activation_path: str | None,
    ):
        """Publish an activation to the orchestrator."""
        task = asyncio.create_task(
            self._publish_activation(
                tensor=tensor,
                activation_id=activation_id,
                direction=direction,
                attestation_challenge_blob=attestation_challenge_blob,
                attestation_self_checks=attestation_self_checks,
                attestation_crypto=attestation_crypto,
                upload_url=upload_url,
                activation_path=activation_path,
            )
        )
        self._publishing_tasks.append(task)

    def publish_loss(self, loss: float, activation_id: str):
        """Publish a loss to the orchestrator."""
        task = asyncio.create_task(self._publish_loss(loss=loss, activation_id=activation_id))
        self._publishing_tasks.append(task)

    async def _publish_activation(
        self,
        tensor: torch.Tensor,
        activation_id: str,
        direction: str,
        attestation_challenge_blob: str | None,
        attestation_self_checks: list[str] | None,
        attestation_crypto: str | None,
        upload_url: list[str] | None,
        activation_path: str | None,
    ):
        """Upload an activation to the orchestrator and cache for P2P retrieval."""
        try:
            publish_start = time.time()

            # Serialize tensor to bytes for hashing and P2P caching
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            tensor_bytes = buffer.getvalue()

            # Compute hash of output activation
            output_hash = compute_activation_hash(tensor_bytes)

            # Cache locally for P2P requests from other miners
            if self.miner:
                await self.miner.cache_activation(activation_id, tensor_bytes)
                logger.debug(f"Cached activation {activation_id} for P2P retrieval, hash={output_hash[:16]}...")

            # Get input hash from when we received this activation's input
            input_hash = None
            if self.miner:
                input_hash = self.miner.get_input_hash(activation_id)

            # Spot-check: upload activation to S3 if the orchestrator selected it
            if upload_url is not None:
                logger.info(f"Spot-check: uploading activation {activation_id} to S3 ({len(tensor_bytes)} bytes)")
                try:
                    await MinerAPIClient.upload_to_s3(urls=upload_url, data=tensor_bytes, upload_id=None)
                    logger.success(f"Spot-check upload complete for {activation_id}")
                except Exception as e:
                    logger.error(f"Spot-check upload failed for {activation_id}: {e}")
                    # Don't raise -- spot-check failure must not block normal activation flow

            logger.debug(f"Activation {activation_id} ready for P2P serving (tensor shape: {tensor.shape})")

            async with TimerLoggerMiner(
                name="upload_activation",  # Name kept for metrics compatibility (no actual S3 upload)
                metadata={
                    "activation_id": activation_id,
                    "direction": direction,
                },
                hotkey=self._miner_api_client.hotkey.ss58_address[:8],
            ):
                attestation_payload: MinerAttestationPayload | None = None
                if self._run_flags.attest.isOn():
                    try:
                        challenge = AttestationChallengeResponse(
                            challenge_blob=attestation_challenge_blob,
                            self_checks=attestation_self_checks,
                            crypto=attestation_crypto,
                        )
                        challenge_id = json.loads(attestation_challenge_blob)["challenge_id"]

                        if self.miner and self.miner.is_mounted:
                            attestation_payload = await self.miner.enclave_sign_with_purpose(
                                purpose="attestation",
                                payload=payload_base64_from_obj(challenge),
                                challenge_id=challenge_id,
                            )
                        else:
                            attestation_payload = await asyncio.to_thread(
                                collect_attestation_payload,
                                challenge,
                            )
                            logger.info(
                                f"Collected attestation payload for activation {activation_id}",
                            )
                    except AttestationUnavailableError as exc:
                        error_code = getattr(exc, "error_code", None)
                        code_suffix = f" (error_code={error_code})" if error_code is not None else ""
                        logger.error(
                            f"Attestation unavailable while submitting activation {activation_id} {code_suffix}: {exc}"
                        )
                    except Exception as exc:
                        logger.exception(
                            f"Error collecting attestation for activation {activation_id}: {exc}",
                        )

            # Record publish timing BEFORE fetching the stats payload so that
            # publish timing is included in the data sent to the orchestrator.
            publish_end = time.time()
            if self._stats_tracker is not None:
                stats = self._stats_tracker.ensure_activation_stats(activation_id, direction=direction)
                stats.timing.publish.start = publish_start
                stats.timing.publish.end = publish_end
                stats.timing.publish.duration = publish_end - publish_start

            async with TimerLoggerMiner(
                name="submit_activation",
                metadata={"activation_id": activation_id, "direction": direction},
                hotkey=self._miner_api_client.hotkey.ss58_address[:8],
            ):
                activation_stats = None
                if self._stats_tracker is not None:
                    activation_stats = self._stats_tracker.get_activation_stats_payload(activation_id)
                await self._miner_api_client.submit_activation_request(
                    submit_activation_request=SubmitActivationRequest(
                        activation_id=activation_id,
                        activation_path=activation_path,  # Non-None when spot-check upload occurred
                        direction=direction,
                        activation_stats=activation_stats,
                        attestation=attestation_payload,
                        input_activation_hash=input_hash,
                        output_activation_hash=output_hash,
                    ),
                )
                logger.success(f"✅ Successfully published activation {activation_id} direction {direction}")

            # Clean up input hash after submission
            if self.miner:
                await self.miner.clear_input_hash(activation_id)

        except (LayerStateException, MinerNotRegisteredException) as e:
            # Swallow expected exceptions
            logger.warning(f"Anticipated exception has occurred while publishing activations (swallowed): {e}")
            pass
        except Exception as e:
            logger.exception(f"Failed to publish activation to orchestrator: {e}")
            raise

    async def _publish_loss(self, loss: float, activation_id: str):
        """Report a loss to the orchestrator."""
        try:
            async with TimerLoggerMiner(
                name="publish_loss",
                metadata={"activation_id": activation_id},
                hotkey=self._miner_api_client.hotkey.ss58_address[:8],
            ):
                await self._miner_api_client.report_loss(
                    loss_report=LossReportRequest(activation_id=activation_id, loss=loss),
                )
                logger.success(f"✅ Successfully published loss for activation {activation_id}")

        except (LayerStateException, MinerNotRegisteredException) as e:
            # Swallow expected exceptions
            logger.warning(f"Anticipated exception has occurred while publishing loss (swallowed): {e}")
            pass
        except Exception as e:
            logger.error(f"Failed to publish loss to orchestrator: {e}")
            raise

    async def reset(self):
        """Cancel any in-progress publishing tasks."""
        if len(self._publishing_tasks) > 0:
            for task in self._publishing_tasks:
                if not task.done():
                    task.cancel()

            results = await asyncio.gather(*self._publishing_tasks, return_exceptions=True)
            for result in results:
                try:
                    if isinstance(result, Exception):
                        raise result
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Failed to publish message to orchestrator: {e}")
                    pass
            self._publishing_tasks.clear()
