from __future__ import annotations

import asyncio
import io
import json
from dataclasses import dataclass, field
from typing import Literal
from common.utils.verify_enclave_signature import payload_base64_from_obj
from loguru import logger
from miner.utils.timer_logger import TimerLoggerMiner
import torch
import time
from common.models.api_models import (
    AttestationChallengeResponse,
    LossReportRequest,
    MinerAttestationPayload,
    MountedAttestationPayload,
    SubmitActivationRequest,
)
from common.models.run_flags import RunFlags, RUN_FLAGS
from common.utils.exceptions import LayerStateException, MinerNotRegisteredException
from miner.utils.attestation_utils import AttestationUnavailableError, collect_attestation_payload
from miner.utils.activation_hash import compute_activation_hash
from common.models.activation_push import ActivationPushMessage
from miner.p2p import SenderUnavailableError
from iota_sdk.p2p import ActivationPushNackError, P2POperationTimings
from subnet.miner_api_client import MinerAPIClient
from common.models.compute_node import ComputeNode
from miner.sync_v2.elastic_device_mesh import ElasticDeviceMesh
from miner.training.peer_selection import select_random
from miner import settings as miner_settings

from miner.utils.stats import StatsTracker

if False:
    from miner.new_miner import Miner


def _peer_matches_target_layer(node: ComputeNode, target_layer: int) -> bool:
    """Return True if ``node`` is considered to belong to ``target_layer``."""
    return f"layer-{target_layer}" in node.groups


def _format_phases(timings: P2POperationTimings) -> str:
    """Compact ``conn=… open=… send=… recv=…`` (ms) for a single log line.

    A ``-`` for any field means that phase did not complete — useful for
    pinpointing which iroh phase a stalled send was sitting in.
    """

    def _fmt(v: float | None) -> str:
        return f"{v * 1000:.0f}" if v is not None else "-"

    return (
        f"conn={_fmt(timings.connection_duration)} "
        f"open={_fmt(timings.stream_open_duration)} "
        f"send={_fmt(timings.send_duration)} "
        f"recv={_fmt(timings.receive_duration)}"
    )


@dataclass
class _OutboundItem:
    """An activation message waiting to be sent to a peer."""

    msg: ActivationPushMessage
    target_p2p_node_ids: list[str] | None  # Known target (backward)
    enqueued_at: float
    next_layer: int | None  # For forward routing (needs peer selection)
    # Metadata carried so the orchestrator can be notified after P2P completes.
    activation_path: str | None = None
    attestation_payload: MinerAttestationPayload | MountedAttestationPayload | None = None
    input_hash: str | None = None
    output_hash: str = ""
    # Errors accumulate across publisher-level retries; each new send produces fresh
    # per-phase durations so only the last attempt's transport timing is preserved.
    p2p_errors: list[str] = field(default_factory=list)
    tries: int = 0

    # Priority + tiebreaker used by PriorityQueue: backward=0, forward=1
    _priority: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        self._priority = 0 if self.target_p2p_node_ids is not None else 1

    def __lt__(self, other: _OutboundItem) -> bool:
        """PriorityQueue comparison: lower priority first, then FIFO."""
        if self._priority != other._priority:
            return self._priority < other._priority
        return self.enqueued_at < other.enqueued_at


class ActivationPublisher:
    def __init__(
        self,
        miner_api_client: MinerAPIClient,
        miner: Miner,  # ty: ignore[invalid-type-form]
        run_flags: RunFlags | None = None,
        node_registry: ElasticDeviceMesh | None = None,
        peer_selector=None,
    ):
        self._miner_api_client = miner_api_client
        self._publishing_tasks: list[asyncio.Task] = []
        self._stats_tracker: StatsTracker | None = None
        self._run_flags: RunFlags = run_flags or RUN_FLAGS
        self.miner = miner
        self.layer_idx: str | None = None
        self._node_registry = node_registry
        self._peer_selector = peer_selector or select_random

        # Outbound send queue — drained concurrently by N worker tasks.
        self._outbound: asyncio.PriorityQueue[_OutboundItem] = asyncio.PriorityQueue()
        self._send_loop_tasks: list[asyncio.Task] = []
        # Forward routing: track empty peer lookups per layer to warn on empty → non-empty flapping.
        self._forward_peer_lookup_was_empty: dict[int, bool] = {}

    def attach_stats_tracker(self, tracker: StatsTracker | None) -> None:
        """Attach a stats tracker for dashboard metrics."""
        self._stats_tracker = tracker

    async def _report_miner_not_registered(self, exc: MinerNotRegisteredException, *, source: str) -> None:
        if not self.miner:
            return

        report = getattr(self.miner, "report_miner_not_registered_if_active", None)
        if report is None:
            return

        try:
            await report(exc, source=source)
        except Exception as report_exc:
            logger.warning(f"Failed to report miner kicked event from {source}: {report_exc}")

    # ── Send-loop lifecycle ──────────────────────────────────────────────────

    def start_send_loop(self) -> None:
        """Spawn ACTIVATION_SEND_CONCURRENCY worker tasks that drain the outbound queue."""
        if any(not t.done() for t in self._send_loop_tasks):
            return
        n = max(1, miner_settings.ACTIVATION_SEND_CONCURRENCY)
        self._send_loop_tasks = [asyncio.create_task(self._drain_outbound_worker(worker_id=i)) for i in range(n)]
        logger.info(f"Outbound send loop started with {n} worker(s)")

    def stop_send_loop(self) -> None:
        """Cancel all background send workers."""
        cancelled = 0
        for task in self._send_loop_tasks:
            if not task.done():
                task.cancel()
                cancelled += 1
        if cancelled:
            logger.info(f"Outbound send loop stopped ({cancelled} worker(s) cancelled)")
        self._send_loop_tasks = []

    # ── Public API ───────────────────────────────────────────────────────────

    def publish_activation(
        self,
        tensor: torch.Tensor,
        activation_id: str,
        direction: str,
        upload_url: list[str] | None,
        activation_path: str | None,
        source_p2p_node_ids: list[str] | None = None,
        sample_path: str | None = None,
    ):
        """Push the activation to the next peer and notify the orchestrator async."""
        task = asyncio.create_task(
            self._publish_activation(
                tensor=tensor,
                activation_id=activation_id,
                direction=direction,
                upload_url=upload_url,
                activation_path=activation_path,
                source_p2p_node_ids=source_p2p_node_ids,
                sample_path=sample_path,
            )
        )
        self._publishing_tasks.append(task)

    def publish_loss(self, loss: float, activation_id: str, layer_idx: int | None = None):
        """Publish a loss to the orchestrator."""
        task = asyncio.create_task(self._publish_loss(loss=loss, activation_id=activation_id, layer_idx=layer_idx))
        self._publishing_tasks.append(task)

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _publish_activation(
        self,
        tensor: torch.Tensor,
        activation_id: str,
        direction: str,
        upload_url: list[str] | None,
        activation_path: str | None,
        source_p2p_node_ids: list[str] | None = None,
        sample_path: str | None = None,
    ):
        """Push activation to the next peer and notify the orchestrator async."""
        try:
            publish_start = time.time()

            # Serialize tensor to bytes
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            tensor_bytes = buffer.getvalue()

            output_hash = compute_activation_hash(tensor_bytes)

            # Cache locally (still useful for any pull-based fallback)
            if self.miner:
                await self.miner.cache_activation(activation_id, tensor_bytes)
                logger.debug(f"Cached activation {activation_id}, hash={output_hash[:16]}...")

            input_hash = None
            if self.miner:
                input_hash = self.miner.get_input_hash(activation_id)

            # Spot-check upload to S3 (if orchestrator selected this activation)
            if upload_url is not None:
                logger.info(f"Spot-check: uploading activation {activation_id} to S3 ({len(tensor_bytes)} bytes)")
                try:
                    await MinerAPIClient.upload_to_s3(urls=upload_url, data=tensor_bytes, upload_id=None)
                    logger.success(f"Spot-check upload complete for {activation_id}")
                except Exception as e:
                    logger.error(f"Spot-check upload failed for {activation_id}: {e}")

            async with TimerLoggerMiner(
                name="upload_activation",  # Name kept for metrics compatibility (no actual S3 upload)
                metadata={
                    "activation_id": activation_id,
                    "direction": direction,
                },
                hotkey=self._miner_api_client.hotkey.ss58_address[:8],
            ):
                attestation_payload: MinerAttestationPayload | MountedAttestationPayload | None = None
                if self._run_flags.attest.isOn():
                    try:
                        challenge_response = await self._miner_api_client.request_attestation_challenge(
                            action="submit_activation",
                        )
                        if challenge_response is None:
                            raise AttestationUnavailableError(
                                "Orchestrator did not issue an attestation challenge for submit_activation"
                            )
                        challenge = AttestationChallengeResponse(
                            challenge_blob=challenge_response.attestation_challenge_blob,
                            self_checks=challenge_response.self_checks,
                            crypto=challenge_response.crypto,
                        )
                        challenge_id = json.loads(challenge_response.attestation_challenge_blob)["challenge_id"]
                        if self.miner and self.miner.is_mounted:
                            challenge_base64 = payload_base64_from_obj(challenge)
                            try:
                                attestation_payload = await self.miner.collect_mounted_attestation(
                                    challenge_base64=challenge_base64,
                                    challenge_id=challenge_id,
                                )
                            except Exception as mounted_exc:
                                logger.warning(
                                    f"Mounted attestation collection failed for activation {activation_id}; falling back to enclave signature: {mounted_exc}"
                                )
                                attestation_payload = await self.miner.enclave_sign_with_purpose(
                                    purpose="attestation",
                                    payload=challenge_base64,
                                    challenge_id=challenge_id,
                                )
                        else:
                            attestation_payload = await asyncio.to_thread(collect_attestation_payload, challenge)
                            logger.info(f"Collected attestation payload for activation {activation_id}")
                    except AttestationUnavailableError as exc:
                        error_code = getattr(exc, "error_code", None)
                        code_suffix = f" (error_code={error_code})" if error_code is not None else ""
                        logger.error(
                            f"Attestation unavailable while submitting activation {activation_id} {code_suffix}: {exc}"
                        )
                    except Exception as exc:
                        logger.exception(f"Error collecting attestation for activation {activation_id}: {exc}")

            publish_end = time.time()
            if self._stats_tracker is not None:
                stats = self._stats_tracker.ensure_activation_stats(activation_id, direction=direction)
                stats.timing.publish.start = publish_start
                stats.timing.publish.end = publish_end
                stats.timing.publish.duration = publish_end - publish_start

            # Orchestrator notification is deferred: ``_enqueue_for_send`` returns the
            # outbound item when a P2P push is scheduled, so notify fires from
            # ``_process_outbound_item`` once we have real ``timing.p2p`` data. When no
            # P2P send is needed (last-layer forward, layer-0 backward) we notify now.
            enqueued = self._enqueue_for_send(
                tensor_bytes=tensor_bytes,
                activation_id=activation_id,
                direction=direction,
                source_p2p_node_ids=source_p2p_node_ids,
                sample_path=sample_path,
                activation_path=activation_path,
                attestation_payload=attestation_payload,
                input_hash=input_hash,
                output_hash=output_hash,
            )
            if not enqueued:
                activation_stats = None
                if self._stats_tracker is not None:
                    activation_stats = self._stats_tracker.get_activation_stats_payload(activation_id)
                asyncio.create_task(
                    self._notify_orchestrator(
                        activation_id=activation_id,
                        activation_path=activation_path,
                        direction=direction,
                        activation_stats=activation_stats,
                        attestation_payload=attestation_payload,
                        input_hash=input_hash,
                        output_hash=output_hash,
                    )
                )

            logger.success(f"Activation {activation_id} enqueued for send ({direction})")

            if self.miner:
                await self.miner.clear_input_hash(activation_id)

        except (LayerStateException, MinerNotRegisteredException) as e:
            if isinstance(e, MinerNotRegisteredException):
                await self._report_miner_not_registered(e, source="activation_publish")
            logger.warning(f"Anticipated exception while publishing activation (swallowed): {e}")
        except Exception as e:
            logger.exception(f"Failed to publish activation: {e}")
            raise

    def _enqueue_for_send(
        self,
        tensor_bytes: bytes,
        activation_id: str,
        direction: str,
        source_p2p_node_ids: list[str] | None,
        sample_path: str | None = None,
        activation_path: str | None = None,
        attestation_payload: MinerAttestationPayload | MountedAttestationPayload | None = None,
        input_hash: str | None = None,
        output_hash: str = "",
    ) -> bool:
        """Build an ActivationPushMessage and place it on the outbound queue.

        Returns ``True`` when an item was enqueued (P2P send pending), ``False`` when
        no send is needed (last-layer forward, layer-0 backward, missing P2P, etc.).
        Caller uses the return value to decide whether to fire an immediate orchestrator
        notification or defer it to ``_process_outbound_item``.
        """
        if not self.miner or not self.miner.p2p:
            logger.warning(f"P2P not available — skipping enqueue for {activation_id}")
            return False

        my_p2p_node_ids: list[str] = self.miner.p2p.node_ids
        my_hotkey: str = self._miner_api_client.hotkey.ss58_address
        current_layer = int(self.layer_idx or 0)

        if direction == "forward":
            n_splits = (
                self.miner.model_manager.model_metadata.get("n_splits", 1)
                if self.miner and self.miner.model_manager.model_metadata
                else 1
            )
            next_layer = current_layer + 1
            if next_layer >= n_splits:
                # Last layer — no forward push
                return False
            if self._node_registry is None:
                logger.warning(f"No node_registry — cannot route forward activation {activation_id}")
                return False
            msg = ActivationPushMessage(
                activation_id=activation_id,
                direction=direction,
                source_hotkey=my_hotkey,
                source_p2p_node_ids=my_p2p_node_ids,
                tensor_bytes=tensor_bytes,
                sample_path=sample_path,
                source_layer=current_layer,
                target_layer=next_layer,
            )
            item = _OutboundItem(
                msg=msg,
                target_p2p_node_ids=None,  # Needs peer selection at drain time
                enqueued_at=time.time(),
                next_layer=next_layer,
                activation_path=activation_path,
                attestation_payload=attestation_payload,
                input_hash=input_hash,
                output_hash=output_hash,
            )
        elif direction == "backward":
            if not source_p2p_node_ids:
                return False
            target_layer = max(0, current_layer - 1)
            msg = ActivationPushMessage(
                activation_id=activation_id,
                direction=direction,
                source_hotkey=my_hotkey,
                source_p2p_node_ids=my_p2p_node_ids,
                tensor_bytes=tensor_bytes,
                sample_path=sample_path,
                source_layer=current_layer,
                target_layer=target_layer,
            )
            item = _OutboundItem(
                msg=msg,
                target_p2p_node_ids=source_p2p_node_ids,
                enqueued_at=time.time(),
                next_layer=None,
                activation_path=activation_path,
                attestation_payload=attestation_payload,
                input_hash=input_hash,
                output_hash=output_hash,
            )
        else:
            return False

        self._outbound.put_nowait(item)
        logger.debug(f"Enqueued {direction} activation {activation_id} (queue size: {self._outbound.qsize()})")
        return True

    async def _drain_outbound_worker(self, worker_id: int) -> None:
        """Worker task: pull items from the priority queue and send them.

        Multiple workers run concurrently so a single slow ``push_activation``
        cannot block other queued sends.
        """
        while True:
            try:
                item = await self._outbound.get()
                await self._process_outbound_item(item)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(f"Unexpected error in outbound send worker {worker_id}")
                await asyncio.sleep(1.0)

    @staticmethod
    def _retry_backoff_seconds(tries: int) -> float:
        """Exponential back-off keyed off attempt count: 0.25, 0.5, 1, 2, 4, capped at 5 s."""
        return min(0.25 * (2 ** max(0, tries - 1)), 5.0)

    async def _process_outbound_item(self, item: _OutboundItem) -> None:
        """Send one outbound item; re-enqueue on retryable failures.

        Owns the full per-item lifecycle: tries check, target resolution
        (backward = known target, forward = peer selection from registry),
        the send call, and exception handling.  Exceptions raised here
        propagate to the worker's outer ``except`` which logs + sleeps.
        """
        deadline_sec = miner_settings.ACTIVATION_SEND_DEADLINE_SECONDS
        if deadline_sec > 0:
            elapsed = time.time() - item.enqueued_at
            if elapsed > deadline_sec:
                logger.warning(
                    f"Dropping {item.msg.direction} activation {item.msg.activation_id} "
                    f"after {elapsed:.0f}s without successful ACK (deadline={deadline_sec:.0f}s)"
                )
                if self.miner is not None:
                    try:
                        await self.miner.cache.remove(item.msg.activation_id)
                    except Exception as exc:
                        logger.debug(f"Cache remove after send deadline (non-fatal): {exc}")
                return

        max_tries = miner_settings.ACTIVATION_SEND_MAX_TRIES
        item.tries += 1
        if item.tries > max_tries:
            logger.warning(
                f"Dropping {item.msg.direction} activation {item.msg.activation_id} "
                f"after {item.tries - 1} tries (max={max_tries})"
            )
            return

        # Resolve target
        target_ctx = ""
        if item.target_p2p_node_ids is not None:
            # Backward: target already known (from activation cache)
            target = item.target_p2p_node_ids
            target_ctx = f"peer=backward[{len(target)}] layer={item.msg.target_layer}"
            if self._node_registry is not None and item.msg.target_layer is not None:
                nodes = self._node_registry.get_group(f"layer-{item.msg.target_layer}")
                p2p_set = set(item.target_p2p_node_ids)
                if not any(n.p2p_node_ids and p2p_set & set(n.p2p_node_ids) for n in nodes):
                    logger.warning(
                        f"Backward targets {item.target_p2p_node_ids} not found among "
                        f"layer-{item.msg.target_layer} peers in registry "
                        f"(activation {item.msg.activation_id}) — sending anyway"
                    )
        else:
            # Forward: select peer from registry
            if self._node_registry is None:
                logger.error(f"No node_registry — cannot forward-send activation {item.msg.activation_id}")
                return
            layer_key = item.next_layer
            assert layer_key is not None
            nodes = self._node_registry.get_group(f"layer-{layer_key}")
            # A peer is eligible only if it has BOTH a p2p node id and at least
            # one address hint (relay URL or direct sockaddr). Iroh discovery is
            # disabled, so a peer with only a node_id is not dialable — skip it
            # until its address hints land in the registry.
            eligible = [n for n in nodes if n.p2p_node_ids and (n.iroh_relay_url or n.iroh_direct_addresses)]
            if not eligible:
                self._forward_peer_lookup_was_empty[layer_key] = True
                all_nodes = self._node_registry.all_nodes()
                node_summary = (
                    "; ".join(
                        f"{n.node_id[:12]}… groups={n.groups} p2p={len(n.p2p_node_ids)} "
                        f"relay={'y' if n.iroh_relay_url else 'n'} "
                        f"direct={len(n.iroh_direct_addresses)}"
                        for n in all_nodes
                    )
                    or "(empty registry)"
                )
                logger.warning(
                    f"No routable peers for layer-{layer_key} (activation {item.msg.activation_id}): "
                    f"{len(nodes)} node(s) matched that layer, {len(all_nodes)} total in registry, "
                    f"none with both p2p_node_ids and address hints — retrying. "
                    f"Registry contents: [{node_summary}]"
                )
                await asyncio.sleep(1.0)
                self._outbound.put_nowait(item)
                return
            if self._forward_peer_lookup_was_empty.pop(layer_key, False):
                logger.warning(
                    f"Registry flapping: peers became available for forward routing to layer {layer_key} "
                    f"(activation {item.msg.activation_id}) after prior empty lookup"
                )
            chosen = self._peer_selector(eligible)
            if not _peer_matches_target_layer(chosen, layer_key):
                logger.error(
                    f"Peer selection mismatch: chosen node {chosen.node_id} is not on layer-{layer_key} "
                    f"(activation {item.msg.activation_id}) — retrying"
                )
                await asyncio.sleep(1.0)
                self._outbound.put_nowait(item)
                return
            target = chosen.p2p_node_ids
            target_ctx = (
                f"peer={chosen.node_id[:8]} layer={layer_key} "
                f"relay={'y' if chosen.iroh_relay_url else 'n'} "
                f"direct={len(chosen.iroh_direct_addresses or [])}"
            )

        # Send. Per-attempt timings get hydrated by the sender; errors and the final
        # attempt count accumulate on the OutboundItem across publisher-level retries.
        attempt_timings = P2POperationTimings()
        try:
            logger.info(
                f"Activation push OUTBOUND | P2P send | id={item.msg.activation_id} "
                f"dir={item.msg.direction} attempt={item.tries}/{max_tries} "
                f"candidate_peers={len(target)} tensor_bytes={len(item.msg.tensor_bytes)}"
            )
            await self.miner.push_activation(target_p2p_node_ids=target, msg=item.msg, timings=attempt_timings)
            logger.info(
                f"Activation push OUTBOUND | P2P send finished | id={item.msg.activation_id} "
                f"dir={item.msg.direction} attempt={item.tries}"
            )
            logger.debug(f"Sent {item.msg.direction} activation {item.msg.activation_id}")
            attempt_timings.errors = list(item.p2p_errors) + list(attempt_timings.errors)
            attempt_timings.attempt_count = item.tries
            attempt_timings.retry_count = max(0, item.tries - 1)
            await self._record_and_notify(item, attempt_timings)
            return
        except SenderUnavailableError as send_exc:
            item.p2p_errors.append(f"SenderUnavailable: {send_exc}")
            # Sender subprocess restarting — re-enqueue with same target
            if item.tries < max_tries:
                backoff = self._retry_backoff_seconds(item.tries)
                logger.warning(
                    f"Sender unavailable for {item.msg.direction} activation {item.msg.activation_id} "
                    f"({send_exc}), re-enqueuing in {backoff:.2f}s (tries={item.tries}/{max_tries}) "
                    f"| {target_ctx} | req={attempt_timings.req_id[:8] if attempt_timings.req_id else '-'} | {_format_phases(attempt_timings)}"
                )
                await asyncio.sleep(backoff)
                self._outbound.put_nowait(item)
                return
            logger.error(
                f"Sender unavailable for {item.msg.direction} activation {item.msg.activation_id} "
                f"({send_exc}), dropping (tries exhausted) "
                f"| {target_ctx} | req={attempt_timings.req_id[:8] if attempt_timings.req_id else '-'} | {_format_phases(attempt_timings)}"
            )
        except ActivationPushNackError as nack_exc:
            item.p2p_errors.append(f"NACK[{nack_exc.status.name}]: {nack_exc}")
            # Receiver explicitly rejected the push (e.g. queue full)
            if item.tries < max_tries:
                if item.msg.direction == "forward":
                    item.target_p2p_node_ids = None  # re-select peer
                backoff = self._retry_backoff_seconds(item.tries)
                logger.warning(
                    f"Push NACK for {item.msg.direction} {item.msg.activation_id} "
                    f"({nack_exc.status.name}), re-enqueuing in {backoff:.2f}s (tries={item.tries}/{max_tries}) "
                    f"| {target_ctx} | req={attempt_timings.req_id[:8] if attempt_timings.req_id else '-'} | {_format_phases(attempt_timings)}"
                )
                await asyncio.sleep(backoff)
                self._outbound.put_nowait(item)
                return
            logger.error(
                f"Push NACK for {item.msg.activation_id}, dropping (tries exhausted) "
                f"| {target_ctx} | req={attempt_timings.req_id[:8] if attempt_timings.req_id else '-'} | {_format_phases(attempt_timings)}"
            )
        except Exception as send_exc:
            item.p2p_errors.append(f"{type(send_exc).__name__}: {send_exc}")
            # Peer unreachable / network error — re-enqueue, pick a different peer for forward sends
            if item.tries < max_tries:
                backoff = self._retry_backoff_seconds(item.tries)
                if item.msg.direction == "forward":
                    # Clear the resolved target so peer selection runs again
                    item.target_p2p_node_ids = None
                    logger.warning(
                        f"Peer unreachable for forward activation {item.msg.activation_id} "
                        f"({send_exc}), re-enqueuing for different peer in {backoff:.2f}s "
                        f"(tries={item.tries}/{max_tries}) "
                        f"| {target_ctx} | req={attempt_timings.req_id[:8] if attempt_timings.req_id else '-'} | {_format_phases(attempt_timings)}"
                    )
                else:
                    logger.warning(
                        f"Peer unreachable for backward activation {item.msg.activation_id} "
                        f"({send_exc}), re-enqueuing same target in {backoff:.2f}s "
                        f"(tries={item.tries}/{max_tries}) "
                        f"| {target_ctx} | req={attempt_timings.req_id[:8] if attempt_timings.req_id else '-'} | {_format_phases(attempt_timings)}"
                    )
                await asyncio.sleep(backoff)
                self._outbound.put_nowait(item)
                return
            logger.error(
                f"Failed to send {item.msg.direction} activation {item.msg.activation_id} "
                f"({send_exc}), dropping (tries exhausted) "
                f"| {target_ctx} | req={attempt_timings.req_id[:8] if attempt_timings.req_id else '-'} | {_format_phases(attempt_timings)}"
            )
            # remove activation if send failed
            try:
                await self.miner.cache.remove(item.msg.activation_id)
            except Exception as exc:
                logger.debug(f"Cache remove after send failure (non-fatal): {exc}")

        # Terminal failure: record the final attempt's timings + accumulated errors
        # and notify the orchestrator so the funnel reflects what actually happened.
        attempt_timings.errors = list(item.p2p_errors)
        attempt_timings.attempt_count = item.tries
        attempt_timings.retry_count = max(0, item.tries - 1)
        await self._record_and_notify(item, attempt_timings)

    async def _record_and_notify(self, item: _OutboundItem, timings: P2POperationTimings) -> None:
        """Persist P2P timings to stats and fire the deferred orchestrator notification."""
        if self._stats_tracker is not None:
            self._stats_tracker.record_p2p_operation(item.msg.activation_id, timings, direction=item.msg.direction)
        activation_stats = None
        if self._stats_tracker is not None:
            activation_stats = self._stats_tracker.get_activation_stats_payload(item.msg.activation_id)
        await self._notify_orchestrator(
            activation_id=item.msg.activation_id,
            activation_path=item.activation_path,
            direction=item.msg.direction,  # type: ignore[arg-type]
            activation_stats=activation_stats,
            attestation_payload=item.attestation_payload,
            input_hash=item.input_hash,
            output_hash=item.output_hash,
        )

    async def _notify_orchestrator(
        self,
        activation_id: str,
        activation_path: str | None,
        direction: Literal["forward", "backward"],
        activation_stats: dict | None,
        attestation_payload: MinerAttestationPayload | MountedAttestationPayload | None,
        input_hash: str | None,
        output_hash: str,
    ) -> None:
        """Submit activation metadata to orchestrator for miner_submissions tracking.

        This runs fully asynchronously and never blocks the activation pipeline.
        """
        try:
            async with TimerLoggerMiner(
                name="submit_activation",
                metadata={"activation_id": activation_id, "direction": direction},
                hotkey=self._miner_api_client.hotkey.ss58_address[:8],
            ):
                await self._miner_api_client.submit_activation_request(
                    submit_activation_request=SubmitActivationRequest(
                        activation_id=activation_id,
                        activation_path=activation_path,
                        direction=direction,
                        activation_stats=activation_stats,
                        attestation=attestation_payload,
                        input_activation_hash=input_hash,
                        output_activation_hash=output_hash,
                    ),
                )
                logger.debug(f"Notified orchestrator of activation {activation_id}")
        except (LayerStateException, MinerNotRegisteredException) as e:
            if isinstance(e, MinerNotRegisteredException):
                await self._report_miner_not_registered(e, source="activation_notify_orchestrator")
            logger.warning(f"Anticipated exception while notifying orchestrator (swallowed): {e}")
        except Exception as e:
            logger.error(f"Failed to notify orchestrator of activation {activation_id}: {e}")

    async def _publish_loss(self, loss: float, activation_id: str, layer_idx: int | None = None):
        """Report a loss to the orchestrator."""
        try:
            async with TimerLoggerMiner(
                name="publish_loss",
                metadata={"activation_id": activation_id},
                hotkey=self._miner_api_client.hotkey.ss58_address[:8],
            ):
                await self._miner_api_client.report_loss(
                    loss_report=LossReportRequest(activation_id=activation_id, loss=loss, layer_idx=layer_idx),
                )
                logger.success(f"Successfully published loss for activation {activation_id}")

        except (LayerStateException, MinerNotRegisteredException) as e:
            # Swallow expected exceptions
            if isinstance(e, MinerNotRegisteredException):
                await self._report_miner_not_registered(e, source="publish_loss")
            logger.warning(f"Anticipated exception has occurred while publishing loss (swallowed): {e}")
            pass
        except Exception as e:
            logger.error(f"Failed to publish loss to orchestrator: {e}")
            raise

    async def reset(self):
        """Cancel any in-progress publishing tasks and drain the outbound queue."""
        # Stop the send loop
        self.stop_send_loop()

        # Drain outbound queue
        dropped = 0
        while not self._outbound.empty():
            try:
                self._outbound.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        if dropped:
            logger.info(f"Dropped {dropped} queued outbound activations on reset")
        self._forward_peer_lookup_was_empty.clear()

        # Restart the send loop so new activations are drained after reset
        self.start_send_loop()

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
