from __future__ import annotations

import asyncio
import json
from typing import Any
from common.utils.verify_enclave_signature import payload_base64_from_obj
from loguru import logger
from miner.utils.timer_logger import TimerLoggerMiner
import torch
import time
from common.models.api_models import (
    AttestationChallengeResponse,
    CompleteFileUploadResponse,
    LossReportRequest,
    MinerAttestationPayload,
    SubmitActivationRequest,
)
from common.models.run_flags import RunFlags, RUN_FLAGS
from common.utils.exceptions import LayerStateException, MinerNotRegisteredException
from miner.utils.attestation_utils import AttestationUnavailableError, collect_attestation_payload
from miner.utils.utils import upload_tensor
from subnet.miner_api_client import MinerAPIClient

from miner.utils.stats import StatsTracker, tensor_num_bytes
from miner.telemetry.metric_registry import S3_UPLOAD_SPEED_BYTES_PER_SEC


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
        """Upload an activation to the orchestrator."""
        try:
            async with TimerLoggerMiner(
                name="upload_activation",
                metadata={
                    "activation_id": activation_id,
                    "direction": direction,
                },
                hotkey=self._miner_api_client.hotkey.ss58_address[:8],
            ):
                start_time = time.time()
                upload_response: CompleteFileUploadResponse = await upload_tensor(
                    miner_api_client=self._miner_api_client,
                    tensor=tensor,
                    hotkey=self._miner_api_client.hotkey,
                    upload_urls=upload_url,
                    object_name=activation_path,
                    run_flags=self._run_flags,
                )
                logger.debug(f"tensor shape before upload:{tensor.shape}")
                if self._stats_tracker is not None:
                    bytes_uploaded = tensor_num_bytes(tensor)
                    self._stats_tracker.record_upload(bytes_uploaded)

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
                end_time = time.time()
                if self._stats_tracker is not None:
                    stats = self._stats_tracker.ensure_activation_stats(activation_id, direction=direction)
                    stats.timing.upload.start = start_time
                    stats.timing.upload.end = end_time
                    stats.timing.upload.duration = end_time - start_time
                upload_duration = end_time - start_time
                if upload_duration > 0 and self.layer_idx is not None:
                    bytes_uploaded = tensor_num_bytes(tensor)
                    if bytes_uploaded > 0:
                        S3_UPLOAD_SPEED_BYTES_PER_SEC.labels(layer_idx=self.layer_idx).set(
                            bytes_uploaded / upload_duration
                        )

            async with TimerLoggerMiner(
                name="submit_activation",
                metadata={"activation_id": activation_id, "direction": direction},
                hotkey=self._miner_api_client.hotkey.ss58_address[:8],
            ):
                start_time
                activation_stats = None
                if self._stats_tracker is not None:
                    activation_stats = self._stats_tracker.get_activation_stats_payload(activation_id)
                await self._miner_api_client.submit_activation_request(
                    submit_activation_request=SubmitActivationRequest(
                        activation_id=activation_id,
                        activation_path=upload_response.object_path,
                        direction=direction,
                        activation_stats=activation_stats,
                        attestation=attestation_payload,
                    ),
                )
                logger.success(f"✅ Successfully published activation {activation_id} direction {direction}")
                end_time = time.time()
                if self._stats_tracker is not None:
                    stats = self._stats_tracker.ensure_activation_stats(activation_id, direction=direction)
                    stats.timing.publish.start = start_time
                    stats.timing.publish.end = end_time
                    stats.timing.publish.duration = end_time - start_time
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
