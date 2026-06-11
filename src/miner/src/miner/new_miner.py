import aiohttp
import httpx
import pprint
import asyncio
import copy
import gc
import json
import ctypes
import multiprocessing
import os
import psutil
import sys
import random
import threading
import time
import webbrowser
import torch
from loguru import logger

from bittensor import Wallet

from common.snowpipe.messages.lifecycle_events import LifecycleEvent
from common.utils.location_utils import resolve_node_location
from common.utils.verify_enclave_signature import payload_base64_from_obj
from common.utils.epistula import sign_p2p_request
from miner.sync_v2.utils import sync_run_sync_prefix
from miner.sync_v2.elastic_device_mesh import ElasticDeviceMesh
from common.models.compute_node import ComputeNode
from miner.sync_v2.variable_manager import VariableManager
from common import settings as common_settings
from miner.utils.node_control_mixin import NodeControlMixin
from miner.utils.miner_dashboard_api import start_visualization_server
from miner.utils.partition_merging import download_previous_optimizer_state_for_partition_batch, merge_partition_batch
from miner.utils.partition_merging import get_partition_batch
from miner.utils.partition_merging import download_pseudograds_for_partition_batch
from miner.utils.partition_merging import upload_partition_batch
from miner.p2p import SenderUnavailableError
from subnet.utils.partition_utils import save_model_weights_and_optimizer_state
from subnet.utils.vector_utils import reconstruct_optimizer_state, get_optimizer_tensor_shapes
from miner.utils.timer_logger import TimerLoggerMiner
from miner.telemetry import TelemetryBufferService
from miner.telemetry.resource_metrics import disk_paths_from_env
from miner.utils.stats import StatsTracker
from subnet.common_api_client import CommonAPIClient
from miner.health_server import HealthServerMixin
from miner.utils.partition_merging import (
    get_weight_partition_info,
)
from miner import settings as miner_settings
from miner.state_manager import StateManager
from miner.utils.utils import (
    collect_system_data,
    upload_weights_blob,
    wait_for_state,
)
from miner.utils.run_utils import identify_best_run
from miner.utils.attestation_utils import collect_attestation_payload, AttestationUnavailableError
from iota_sdk.p2p import (
    P2PAuthFields,
    P2PExpiredError,
    P2PNotFoundError,
    P2PRequestError,
    P2PResponseStatus,
    P2PUnauthorizedError,
    encode_activation_request,
    decode_activation_response,
    P2POperationTimings,
    decode_push_ack,
    ActivationPushNackError,
)

from miner.p2p import P2PStack
from common.models.activation_push import ActivationPushMessage
from common.models.peer_status import PeerStatusBroadcast
from common.models.api_models import (
    AttestationChallengeResponse,
    EnclaveSignResponse,
    MinerAttestationPayload,
    MinerRegistrationResponse,
    MountedAttestationPayload,
    NodeLocation,
    RegisterMinerRequest,
    SubmittedWeightsAndOptimizerPresigned,
    WeightSubmitResponse,
    WeightUpdate,
)
from subnet.utils.vector_utils import flatten_optimizer_state
from common.utils.exceptions import (
    APIException,
    LayerStateException,
    RateLimitException,
    MinerNotRegisteredException,
    MinerResetException,
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
from subnet.base.base_neuron import BaseNeuron
from subnet.miner_api_client import MinerAPIClient
from subnet.model.utils import _clean_gpu_memory, log_gpu_memory_usage
from subnet.utils.partition_utils import (
    MergingPartition,
    delete_saved_model_weights_and_optimizer_state,
    load_model_weights,
    load_model_weights_and_optimizer_state,
)
from subnet.utils.vector_utils import check_for_nans_and_infs
from subnet.model import gpu_device
from subnet.utils.s3_torch import download_tensor

from miner.training import TrainingPhase
from common.settings import P2P_MAX_SENDER_CONNECTIONS


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

        # Initialize own identity and node registry before referencing them elsewhere
        self._own_node_id: str = self.wallet.hotkey.ss58_address
        self._own_compute_node = ComputeNode(node_id=self._own_node_id)
        self.elastic_device_mesh: ElasticDeviceMesh | None = None
        self._bridge_manager = VariableManager(url=common_settings.BRIDGE_V2_URL)

        self.stats_tracker = StatsTracker()

        # To be created on registration.
        self.training_phase: TrainingPhase | None = None

        self._latest_attestation_payloads: dict[
            str, MinerAttestationPayload | MountedAttestationPayload | EnclaveSignResponse
        ] = {}
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
                disk_paths=disk_paths_from_env(),
                # Hotkey name (e.g. "miner-52"), NOT wallet.name (which is the
                # coldkey/wallet identifier — often shared across miners).
                hotkey_name=getattr(self.wallet, "hotkey_str", None),
            )

        self.node_control_port = node_control_port or 8010
        self.node_control_process: multiprocessing.Process | None = None
        self.is_mounted = miner_settings.IS_MOUNTED

        # Node location — resolved once at startup via IP geolocation
        self._node_location: NodeLocation | None = None

        # P2P lifecycle manager (receiver subprocess + sender)
        self.p2p: P2PStack | None = None

        # Queue populated by the bridge task that reads from the P2PStack's
        # multiprocessing push_queue and forwards to this asyncio queue.
        # ActivationQueue reads from this instead of polling the orchestrator.
        self._p2p_push_queue: asyncio.Queue[ActivationPushMessage] = asyncio.Queue()
        self._push_bridge_task: asyncio.Task | None = None

        # Track miner start time for uptime reporting in peer status broadcasts.
        self._start_time: float = time.time()

        # Input hash tracking: activation_id -> hash of input we received
        self.input_activation_hashes: dict[str, str] = {}
        self.input_hash_lock = asyncio.Lock()

        # Per-peer semaphores to limit concurrent P2P requests to any single peer
        self._peer_semaphores: dict[str, asyncio.Semaphore] = {}
        self._peer_semaphores_lock = asyncio.Lock()
        self._max_concurrent_per_peer = 2  # Based on benchmark: BI degrades at concurrency >= 5

    async def _collect_attestation_payload(
        self, action: str
    ) -> MinerAttestationPayload | MountedAttestationPayload | EnclaveSignResponse | None:
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
                challenge_base64 = payload_base64_from_obj(challenge)
                try:
                    payload = await self.collect_mounted_attestation(
                        challenge_base64=challenge_base64,
                        challenge_id=challenge_id,
                    )
                    logger.debug(f"Collecting mounted attestation challenge {challenge_id} for action {action}")
                except Exception as mounted_exc:
                    logger.warning(
                        f"Mounted attestation collection failed for challenge {challenge_id}; falling back to enclave signature: {mounted_exc}"
                    )
                    payload = await self.enclave_sign_with_purpose(
                        purpose="attestation",
                        payload=challenge_base64,
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

    async def _start_p2p(self, timeout: float = 30.0) -> None:
        """Initialize and start the P2P stack (receiver subprocess + sender)."""
        self.p2p = P2PStack(
            cache_ttl=float(miner_settings.P2P_ACTIVATION_CACHE_TTL),
            max_cache_size=miner_settings.MAX_ACTIVATION_CACHE_SIZE,
            max_sender_connections=P2P_MAX_SENDER_CONNECTIONS,
        )
        # The set_on_sender_restarted / _on_sender_restarted callback hook
        # that lived here under the subprocess-RPC architecture is gone:
        # the iota_sdk-backed Sender is native Rust with no subprocess to
        # crash and no restart event to fire. Peer-address-book sync still
        # happens via _sync_peer_addrs_to_sender on every node-registry
        # tick, which is sufficient now that there's no subprocess to lose
        # the address book in the first place.
        seed = f"iota-miner-{self.wallet.hotkey.ss58_address}"
        await self.p2p.start(seed=seed, timeout=timeout)

    async def _initialize_node_registry(self) -> None:
        # Bridge: forward activation push messages from the multiprocessing
        # queue (written by the receiver subprocess) into the asyncio queue
        # consumed by ActivationQueue._drain_push_queue().
        self._push_bridge_task = asyncio.create_task(
            self._bridge_push_queue(),
            name="PushBridge",
        )

        # Advertise P2PStack receiver IDs so peers can push activations
        # and route status broadcasts to us.
        if self.elastic_device_mesh is not None:
            await self.elastic_device_mesh.initialize(node=self._own_compute_node)
        logger.info(f"P2P node IDs: {self.p2p.node_ids}")

    async def _bridge_push_queue(self) -> None:
        """Forward activation pushes from the P2PStack queue to ActivationQueue's queue.

        Originally bridged a multiprocessing.Queue from the receiver
        subprocess onto an asyncio.Queue here. The in-process P2PStack
        already exposes its push queue as an asyncio.Queue, so this is now
        a straightforward asyncio→asyncio forwarder. Kept as a separate
        task so ActivationQueue stays decoupled from the P2P plumbing.
        """
        while True:
            try:
                msg = await self.p2p.push_queue.get()
                logger.info(
                    f"Activation push RECV | id={msg.activation_id} "
                    f"dir={msg.direction} src_layer={msg.source_layer} tgt_layer={msg.target_layer} "
                    f"tensor_bytes={len(msg.tensor_bytes)}"
                )
                await self._p2p_push_queue.put(msg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"_bridge_push_queue forwarding error: {exc}")
                await asyncio.sleep(0.1)

    async def _broadcast_peer_status(self) -> None:
        """Send current metrics to all miners on adjacent layers via routed UNI.

        Fans out per-peer with an individual timeout so a single unreachable
        peer cannot stall the loop. Per-peer failures are aggregated into one
        summary log line.
        """
        if self.p2p is None or self.p2p.sender is None:
            return
        layer = self.state_manager.layer
        if layer is None:
            return

        queue = self.training_phase._queue
        msg = PeerStatusBroadcast(
            source_hotkey=self.hotkey,
            forward_queue_size=len(queue._forward_queue),
            backward_queue_size=len(queue._backward_queue),
            cache_size=len(self.training_phase._cache),
            cache_capacity=miner_settings.MAX_ACTIVATION_CACHE_SIZE,
            uptime_seconds=time.time() - self._start_time,
            layer_phase=self.miner_api_client.layer_state.value,
        )

        # Collect (nid, peer_ctx) pairs so failure logs can include relay/direct hints.
        target_node_ids: list[str] = []
        peer_ctx: dict[str, tuple[str, bool, int]] = {}
        for adj in [layer - 1, layer + 1]:
            if adj < 0:
                continue
            peers = self.elastic_device_mesh.get_group(f"layer-{adj}")
            for peer in peers:
                if peer.p2p_node_ids:
                    for nid in peer.p2p_node_ids:
                        target_node_ids.append(nid)
                        peer_ctx[nid] = (
                            peer.node_id[:8],
                            bool(peer.iroh_relay_url),
                            len(peer.iroh_direct_addresses or []),
                        )
                else:
                    logger.debug(f"Peer {peer.node_id[:8]}... in layer-{adj} has no p2p_node_ids")

        if not target_node_ids:
            logger.debug(f"No p2p_node_ids found on any adjacent-layer peer (layer={layer})")
            return

        per_peer_timeout = 5.0

        async def _send_one(nid: str) -> float:
            t0 = time.monotonic()
            await asyncio.wait_for(
                # iota_sdk Sender.send_routed signature is (target, route, model);
                # the auto-merged staging code used the old common.iroh order
                # (route, node_id, msg) which made "/peer/status" get treated
                # as the target node-id and threw PeerAddressUnknownError.
                self.p2p.sender.send_routed(nid, "/peer/status", msg),
                timeout=per_peer_timeout,
            )
            return time.monotonic() - t0

        send_t0 = time.monotonic()
        results = await asyncio.gather(
            *[_send_one(nid) for nid in target_node_ids],
            return_exceptions=True,
        )
        gather_dur = time.monotonic() - send_t0

        total = len(target_node_ids)
        failed_pairs = [(nid, r) for nid, r in zip(target_node_ids, results) if isinstance(r, BaseException)]
        failed = len(failed_pairs)
        if not failed:
            logger.debug(f"Broadcast peer status to {total} node(s) in {gather_dur * 1000:.0f}ms")
            return

        # Group failures by exception type and surface a sample with peer hints.
        err_counts: dict[str, int] = {}
        for _, exc in failed_pairs:
            err_counts[type(exc).__name__] = err_counts.get(type(exc).__name__, 0) + 1
        err_summary = ", ".join(f"{name}={count}" for name, count in sorted(err_counts.items()))

        sample_lines: list[str] = []
        for nid, exc in failed_pairs[:3]:
            short, has_relay, n_direct = peer_ctx.get(nid, ("?", False, 0))
            sample_lines.append(
                f"{short} (relay={'y' if has_relay else 'n'}, direct={n_direct}) " f"-> {type(exc).__name__}: {exc}"
            )

        logger.warning(
            f"Broadcast peer status: {total - failed}/{total} ok, {failed} unreachable in "
            f"{gather_dur * 1000:.0f}ms | errors: {err_summary} | "
            f"sample: [{'; '.join(sample_lines)}]"
        )

    async def _peer_status_broadcast_loop(self) -> None:
        """Background task: broadcast peer status to adjacent-layer miners
        and sync received status into the node registry.

        Stale node eviction: entries linger until each peer's own push overwrites them.
        """
        broadcast_interval = miner_settings.PEER_STATUS_BROADCAST_INTERVAL_SECONDS
        while True:
            try:
                await self._broadcast_peer_status()
            except Exception as exc:
                logger.debug(f"Peer status broadcast error: {exc}")

            try:
                if self.elastic_device_mesh is not None:
                    self.elastic_device_mesh.sync_valid_hotkeys()
            except Exception as exc:
                logger.debug(f"Valid hotkeys sync error: {exc}")

            try:
                if self.elastic_device_mesh is not None:
                    self.elastic_device_mesh.sync_peer_status_into_registry()
            except Exception as exc:
                logger.debug(f"Peer status sync error: {exc}")

            if self.p2p and self.p2p.peer_status_dict:
                n = len(self.p2p.peer_status_dict)
                if n:
                    logger.debug(f"Peer status dict has {n} entries")

            await asyncio.sleep(broadcast_interval)

    async def _stop_p2p(self) -> None:
        """Shutdown the P2P stack."""
        if self._push_bridge_task is not None:
            self._push_bridge_task.cancel()
            try:
                await self._push_bridge_task
            except asyncio.CancelledError:
                pass
            self._push_bridge_task = None
        if self.p2p is not None:
            await self.p2p.stop()
            self.p2p = None
            if self.elastic_device_mesh is not None:
                self.elastic_device_mesh.p2p = None

    @property
    def p2p_node_id(self) -> str | None:
        """Get this miner's P2P node ID."""
        if self.p2p is not None:
            return self.p2p.node_id
        return None

    @property
    def cache(self):
        """Expose the training phase's activation cache for the activation publisher."""
        if self.training_phase is None:
            return None
        return self.training_phase._cache

    async def push_activation(
        self,
        target_p2p_node_ids: list[str],
        msg: ActivationPushMessage,
        timings: P2POperationTimings | None = None,
    ) -> None:
        """Push an activation to a peer via bidirectional P2P with ack.

        Sends to **one** randomly chosen receiver from *target_p2p_node_ids*
        and waits for a single-byte ack.  An ack from one receiver is sufficient.

        Args:
            target_p2p_node_ids: P2P node IDs of the target miner.
            msg:                 The activation push message to send.
            timings:             Optional mutable timing record; the sender hydrates
                                 per-phase durations onto it for stats reporting.

        Raises:
            SenderUnavailableError: If the sender subprocess is restarting or
                unavailable.  Callers should retry with the same target.
            ActivationPushNackError: If the receiver explicitly NACKs the push
                (e.g. queue full).  Callers may re-enqueue or pick a different peer.
            Exception: If the send fails for other reasons (peer unreachable,
                network error, etc.).  Callers may retry with a different peer.
        """
        if self.p2p is None or self.p2p.sender is None:
            raise SenderUnavailableError("Sender not available")

        target = random.choice(target_p2p_node_ids)
        t0 = time.monotonic()
        payload_n = len(msg.tensor_bytes)
        logger.info(
            f"Activation push SEND | start | id={msg.activation_id} dir={msg.direction} "
            f"src_layer={msg.source_layer} tgt_layer={msg.target_layer} "
            f"peer={target[:16]}… tensor_bytes={payload_n}"
        )
        semaphore = await self._get_peer_semaphore(target)
        async with semaphore:
            # iota_sdk.p2p.Sender.send_routed_bi_raw takes (target, route, request, ...).
            # Note arg order differs from common.iroh's old (route, node_id, msg).
            ack_bytes = await self.p2p.sender.send_routed_bi_raw(
                target,
                "/activation/push",
                msg,
                timeout=miner_settings.ACTIVATION_PUSH_TIMEOUT_SECONDS,
                timings=timings,
            )
        status = decode_push_ack(ack_bytes)
        elapsed = time.monotonic() - t0
        if status != P2PResponseStatus.SUCCESS:
            logger.warning(
                f"Activation push SEND | NACK | id={msg.activation_id} dir={msg.direction} "
                f"peer={target[:16]}… status={status.name} elapsed_s={elapsed:.2f}"
            )
            raise ActivationPushNackError(
                status,
                f"Push {msg.activation_id} NACK from {target[:16]}...: {status.name}",
            )
        logger.info(
            f"Activation push SEND | ACK OK | id={msg.activation_id} dir={msg.direction} "
            f"peer={target[:16]}… elapsed_s={elapsed:.2f}"
        )

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

            id_bytes = activation_id.encode("utf-8")
            timestamp_ms, ss58_address, signature = sign_p2p_request(self.wallet.hotkey, id_bytes)
            auth = P2PAuthFields(
                timestamp_ms=timestamp_ms,
                ss58_address=ss58_address,
                signature=signature,
            )
            request = encode_activation_request(activation_id, auth=auth)
            epistula_end = time.time()

            stats = self.stats_tracker.ensure_activation_stats(activation_id)
            stats.timing.epistula.start = epistula_start
            stats.timing.epistula.end = epistula_end
            stats.timing.epistula.duration = epistula_end - epistula_start

            # All retry + timeout logic lives inside Sender.send_message_bi().
            # iota_sdk.p2p.Sender.send_message_bi takes (target, payload, protocol_id=PROTOCOL_ID_BI, ...);
            # the receiver-side max-message-size cap is enforced at Receiver
            # construction (DEFAULT_MAX_MESSAGE_SIZE), not per-call.
            response = await self.p2p.sender.send_message_bi(
                source_node_id,
                request,
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
            logger.info(f"🔄 Node registry: {pprint.pformat(self.elastic_device_mesh.to_dict())}")

            if self.miner_api_client.layer_state == LayerPhase.TRAINING:
                if self.need_to_pull_weights:
                    weight_download_tries = 3
                    weight_download_success = False
                    for i in range(weight_download_tries):
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
                                weight_download_success = True
                                break
                        except (
                            MinerNotRegisteredException,
                            MinerInitializingException,
                            MinerFrozenException,
                            MinerBlockedException,
                        ):
                            # Re-raise so the outer training_loop handler applies the
                            # correct back-off (60s) and status update.
                            raise
                        except torch.cuda.OutOfMemoryError as e:
                            torch.cuda.empty_cache()
                            logger.error(
                                f"Miner {self.hotkey[:8]} CUDA OOM downloading weights "
                                f"(attempt {i + 1}/{weight_download_tries}): {e}"
                            )
                            await asyncio.sleep(5)
                            continue

                        except Exception as e:
                            logger.debug(
                                f"Miner {self.hotkey[:8]} will NOT train until global weights "
                                f"are downloaded successfully... Retrying "
                                f"(attempt {i + 1}/{weight_download_tries})"
                            )
                            logger.error(f"Unexpected error during weight download: {e}")
                            await asyncio.sleep(1)
                            continue

                    if not weight_download_success:
                        logger.error(
                            f"Miner {self.hotkey[:8]} hit {weight_download_tries} "
                            f"consecutive failures — resetting to re-register"
                        )
                        raise MinerResetException(
                            "Error: Unexpected persistent errors during weight download. Resetting miner state."
                        )

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

                # Safety net: drop any per-activation stats left over from the
                # previous epoch (timeouts, error paths). The per-activation
                # eviction in training.py handles the happy path.
                self.stats_tracker.activation_stats.clear()

                # Make sure that the epoch counter increases every time we start training.
                # This is to avoid the edge case where nodes fail to get to the END of merging_partitions.
                self.model_manager.epoch_counter += 1
                logger.info(
                    f"🔄 Miner {self.hotkey[:8]} incremented epoch counter to: {self.model_manager.epoch_counter}"
                )

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
                    logger.info(
                        f"🔄 Miner got weight partition info: {len(partitions)} partitions assigned, "
                        f"{len(weight_path_per_layer)} miners submitted weights"
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

                await self._clear_stale_p2p_state()
                await self.training_phase.epoch_reset()
                return

            await asyncio.sleep(1.1)

    async def training_loop(self):
        """Main training loop delegating to tick with existing error handling."""
        broadcast_task = asyncio.create_task(self._peer_status_broadcast_loop())
        try:
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
                except MinerResetException as e:
                    logger.warning(f"🔄 Miner {self.hotkey[:8]} needs reset: {e}")
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
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        f"🔄 Miner {self.hotkey[:8]} HTTP status error "
                        f"({e.response.status_code} from {e.request.url}): {e}. Retrying..."
                    )
                    await asyncio.sleep(1)
                    continue
                except (aiohttp.ClientConnectorDNSError, aiohttp.ClientConnectorError) as e:
                    logger.warning(f"🔄 Miner {self.hotkey[:8]} Connection error (DNS/network): {e}. Retrying...")
                    await asyncio.sleep(5)
                    continue
                except (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadError,
                    httpx.ReadTimeout,
                    httpx.WriteError,
                    httpx.RemoteProtocolError,
                    httpx.PoolTimeout,
                ) as e:
                    logger.warning(
                        f"🔄 Miner {self.hotkey[:8]} httpx transient error ({type(e).__name__}): {e}. Retrying..."
                    )
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
                    logger.warning(
                        f"🔄 Miner {self.hotkey[:8]} has been temporarily blocked and cannot perform work: {e}"
                    )
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
        finally:
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass

    async def run(self):
        self._start_visualization_server_process(port=self.visualization_port)
        if self.visualization_auto_open:
            self._open_visualization_tab(f"http://localhost:{self.visualization_port}/vis.html")

        await self.report_training_state(state="resetting")
        await self.reset_miner_state()

        await self._initialize_node_registry()

        logger.info(f"🚀 Starting miner {self.hotkey[:8]} on layer {self.layer} | Timeout: {miner_settings.TIMEOUT}s")

        try:
            await self.report_training_state(
                state="waiting_training", run_id=self.state_manager.run_id, layer=self.state_manager.layer
            )
            await wait_for_state(
                state=LayerPhase.TRAINING, miner_api_client=self.miner_api_client, raise_bad_sync=False
            )
            await self.report_training_state(
                state="training", run_id=self.state_manager.run_id, layer=self.state_manager.layer
            )
            await self.training_loop()
        finally:
            if self.elastic_device_mesh is not None:
                await self.elastic_device_mesh.stop()

    async def _create_node_registry(self, run_id: str, layer: int, p2p: P2PStack) -> None:
        if self.elastic_device_mesh is not None:
            await self.elastic_device_mesh.stop()
        self.elastic_device_mesh = ElasticDeviceMesh(
            run_id=run_id,
            own_node_id=self._own_node_id,
            groups=["all", f"layer-{layer}"],
            p2p=p2p,
        )
        await self.elastic_device_mesh.start_background_sync(self._bridge_manager)

    async def _apply_registration_response(self, response: MinerRegistrationResponse):
        """
        Validate and apply a registration response to miner state.

        Returns:
            (assigned_layer, current_epoch)
        """
        if response.layer is None:
            raise Exception(f"Miner {self.hotkey[:8]} registered with no layer assigned, this should not happen")
        if response.num_partitions is None:
            raise Exception(f"Number of partitions is None for miner {self.hotkey[:8]}")

        model_config = response.model_cfg.model_dump()
        model_metadata = response.model_metadata.model_dump()

        old_run_id = self.state_manager.run_id
        assigned_layer = int(response.layer)
        current_epoch = int(response.current_epoch)

        logger.debug(f"Number of partitions for miner {self.hotkey[:8]}: {response.num_partitions}")
        self.num_partitions = int(response.num_partitions)

        # TODO: clean these up
        self.layer = assigned_layer
        self.state_manager.layer = assigned_layer
        self.state_manager.training_epoch_when_registered = current_epoch
        self.state_manager.run_id = response.run_id
        self.run_id = response.run_id

        self.model_manager.num_partitions = int(response.num_partitions)
        self.model_manager.model_config = model_config
        self.model_manager.model_metadata = model_metadata
        self.model_manager.epoch_on_registration = current_epoch

        # Local merge-cycle count is per assignment; re-register / new run must not inherit the old value.
        self.model_manager.epoch_counter = 0

        layer_groups = ["all", f"layer-{assigned_layer}"]
        self._own_compute_node = self._own_compute_node.model_copy(update={"groups": layer_groups})

        self._update_run_flags(response.run_flags)

        # Create a clean stats tracker.
        self.stats_tracker = StatsTracker(
            current_layer=assigned_layer,
            remote_epoch=current_epoch,
            run_id=self.run_id,
        )

        await self._create_node_registry(run_id=response.run_id, layer=assigned_layer, p2p=self.p2p)
        await self._setup_training_phase()

        if sync_run_sync_prefix(old_run_id) != sync_run_sync_prefix(response.run_id):
            # Pull existing registry from the bridge so we merge with peers
            # that already registered under this run, rather than overwriting
            # with a single-node snapshot.
            try:
                await self.elastic_device_mesh.pull()
            except Exception as exc:
                logger.warning(f"Failed to pull node_registry after run change (proceeding with local): {exc}")

        logger.success(
            f"✅ Miner {self.hotkey[:8]} registered successfully in layer {assigned_layer} on training epoch {current_epoch}"
        )
        logger.debug(f"Run flags for miner {self.hotkey[:8]}: {self.run_flags}")

        try:
            await self.elastic_device_mesh.initialize(self._own_compute_node)
        except Exception as e:
            logger.error(f"Error pushing node registry: {e}")
            raise
        logger.info(f"Registered with P2P node ID: {(self.p2p.node_id or 'n/a')[:16]}...")
        logger.debug(f"Updated node registry groups: {layer_groups}")

    async def _clear_stale_p2p_state(self) -> None:
        """Drain P2P queues, clear shared-memory cache, and reset input hashes.

        Called when rebuilding the training phase so that stale activations from
        a previous epoch / run are never processed with the wrong model weights
        or served to peers.
        """
        # 1. Drain the multiprocessing push queue (receiver -> bridge).
        if self.p2p is not None:
            dropped = self.p2p.drain_push_queue()
            if dropped:
                logger.info(f"Drained {dropped} stale message(s) from P2P push queue")

            # 2. Clear the SharedMemory activation cache.
            cleared = self.p2p.clear_activation_cache()
            if cleared:
                logger.info(f"Cleared {cleared} stale activation(s) from P2P shared-memory cache")

        # 3. Drain the asyncio-side bridge queue.
        bridge_dropped = 0
        while not self._p2p_push_queue.empty():
            try:
                self._p2p_push_queue.get_nowait()
                bridge_dropped += 1
            except asyncio.QueueEmpty:
                break
        if bridge_dropped:
            logger.info(f"Drained {bridge_dropped} stale message(s) from asyncio P2P bridge queue")

        # 4. Clear input hash tracking.
        async with self.input_hash_lock:
            count = len(self.input_activation_hashes)
            self.input_activation_hashes.clear()
            if count:
                logger.info(f"Cleared {count} stale input activation hash(es)")

    async def _setup_training_phase(self) -> None:
        """Tear down any existing ``TrainingPhase`` and setup a fresh one.

        Used on (re-)registration: the new instance
        starts with a clean cache/queue/publisher/LR-scheduler state, and the
        old instance's background tasks (publisher send loop, activation fetcher,
        distributed backward counter) are cleanly stopped so they don't linger.
        """
        existing: TrainingPhase | None = getattr(self, "training_phase", None)
        if existing is not None:
            try:
                await existing.shutdown()
            except Exception as e:
                logger.error(f"Failed to shut down previous TrainingPhase: {e}")

        await self._clear_stale_p2p_state()

        self.training_phase = TrainingPhase(
            miner_api_client=self.miner_api_client,
            state_manager=self.state_manager,
            model_manager=self.model_manager,
            device=self.device,
            run_flags=self.run_flags,
            mock=self.mock,
            is_mounted=miner_settings.IS_MOUNTED,
            node_registry=self.elastic_device_mesh,
            miner=self,
        )
        self.training_phase.attach_stats_tracker(self.stats_tracker)
        # The publisher's outbound send loop is per-instance; start it so activations
        # produced by this TrainingPhase actually get drained to peers.
        self.training_phase._publisher.start_send_loop()

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
                f"P2P node ID not available for miner {self.hotkey[:8]}. P2P must be initialized before registration."
            )

        logger.info(f"Collecting system data (includes speedtest) for miner {self.hotkey[:8]}...")
        system_data = collect_system_data()
        if system_data:
            try:
                payload = json.loads(system_data)
                if payload.get("bandwidth"):
                    logger.info(f"Speedtest complete for miner {self.hotkey[:8]}: {payload['bandwidth']}")
                else:
                    logger.warning(
                        f"No bandwidth in system data for miner {self.hotkey[:8]} (speedtest may have failed)"
                    )
            except json.JSONDecodeError:
                pass
        else:
            logger.warning(f"Failed to collect system data for miner {self.hotkey[:8]}")

        register_request = RegisterMinerRequest(
            run_id=best_run.run_id,
            register_as_metagraph_miner=True,
            p2p_node_id=self.p2p_node_id,
            system_data=system_data,
            location=self._node_location,
        )
        response: MinerRegistrationResponse = await self.miner_api_client.register_miner_request(
            register_miner_request=register_request
        )
        logger.info(f"Registered with P2P node ID: {self.p2p_node_id[:16]}...")

        await self._apply_registration_response(response)
        await self.register_set_status(status="registered")

        if self.telemetry_service:
            try:
                await self.telemetry_service.start()
            except Exception as e:
                logger.error(f"Failed to start telemetry service, continuing without telemetry: {e}")
                self.telemetry_service = None

        # Emit a location lifecycle event so geographic data is persisted to Snowflake
        if self._node_location is not None and self.telemetry_service is not None:
            location_event = LifecycleEvent(
                type="node_location",
                source_service="miner",
                miner_hotkey=self.hotkey,
                run_id=self.run_id,
                event_category="node_location",
                details=self._node_location.model_dump(),
            )
            self.telemetry_service.log(location_event)
            logger.debug(
                f"📍 Location event queued for telemetry: {self._node_location.city}, {self._node_location.country}"
            )

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

    def _log_resources(self, tag: str) -> None:
        vm = psutil.virtual_memory()
        ram_used = vm.used / 1024**3
        ram_total = vm.total / 1024**3
        disk = psutil.disk_usage("/")
        disk_free = disk.free / 1024**3
        disk_total = disk.total / 1024**3
        vram_msg = ""
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / 1024**3
            vram_res = torch.cuda.memory_reserved() / 1024**3
            vram_msg = f" | VRAM alloc={vram_alloc:.2f}GB res={vram_res:.2f}GB"
        logger.debug(
            f"[resources:{tag}] RAM {ram_used:.2f}/{ram_total:.2f}GB{vram_msg} | disk free {disk_free:.2f}/{disk_total:.2f}GB"
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
            # Baseline RAM before any of submit_weights' large CPU allocations.
            # Compare across epochs to confirm/refute the glibc-retention theory:
            # epoch 2's baseline here should match epoch 1's if the heap returns
            # cleanly between epochs, and be noticeably higher if it doesn't.
            self._log_resources("submit_weights_start")

            if self.training_phase.backwards_since_reset == 0:
                logger.warning(f"Backwards since reset for miner {self.hotkey[:8]} is 0, skipping")
                return

            # Pull the epoch-boundary cleanup forward.
            #
            # The full cleanup (_clear_stale_p2p_state + training_phase.epoch_reset)
            # normally only runs AFTER the layer transitions back to TRAINING
            # (see training_loop_tick around line 981). During the
            # weights_uploading phase — where this method runs — none of
            # that stale state is cleared yet, so it stacks on top of the
            # ~12 GB weight buffer and ~24 GB optimizer-state buffer we are
            # about to allocate. On 2026-05-24 this combination pushed
            # ~30 miners past the 96 GB host-RAM ceiling on Paperspace and
            # triggered fleet-wide OOM kills.
            #
            # Each piece below is safe to drop here:
            #   - activation cache  : backward passes for these forwards
            #                         won't run, the epoch is over
            #   - forward/backward
            #     queues            : same
            #   - publisher state   : peers have already transitioned phase,
            #                         queued sends are moot. Comment on
            #                         publisher.reset() notes this state
            #                         "accumulates host RAM across the whole run"
            #   - p2p shared-memory
            #     cache + queues    : stale activations from the prior epoch
            log_gpu_memory_usage(note="submit_weights entry, before cleanup")
            try:
                await self._clear_stale_p2p_state()
            except Exception as e:
                logger.warning(f"submit_weights: failed to clear stale p2p state: {e}")
            try:
                await self.training_phase._cache.reset()
                self.training_phase._queue._forward_queue.clear()
                self.training_phase._queue._backward_queue.clear()
                await self.training_phase._publisher.reset()
            except Exception as e:
                logger.warning(f"submit_weights: failed to drop stale training state: {e}")
            gc.collect()
            log_gpu_memory_usage(note="submit_weights, after cleanup")

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

            # Iterate over contiguous chunks to fill pseudo_gradients.
            #
            # Previously this used strided slicing ``previous_weights[i :: B]``,
            # which produces a non-contiguous view. The subsequent ``.to(fp32)``
            # then has to allocate a fresh contiguous fp32 buffer AND walk the
            # strided source with cache-hostile reads — empirically this caused
            # ~8× the expected allocation footprint (e.g. 800 MB of resident
            # growth for a ~95 MB fp32 result tensor) and the freed pages did
            # not get returned to the OS cleanly, inflating RSS heading into
            # the S3 upload.
            #
            # Contiguous chunking gives identical math, makes ``.to(fp32)`` a
            # cache-friendly sequential read, and lets PyTorch's CPU allocator
            # reuse the same chunk-sized pool every iteration without growth.
            total_elems = previous_weights.numel()
            chunk_size = -(-total_elems // miner_settings.PSEUDO_GRADIENTS_BATCH_SIZE)  # ceil div
            for start in range(0, total_elems, chunk_size):
                end = min(start + chunk_size, total_elems)
                logger.debug(f"Getting pseudo gradients for chunk [{start}:{end}]")
                previous_weights_batch = previous_weights[start:end]
                current_weights_batch = current_weights[start:end]
                pseudo_gradients_batch = previous_weights_batch.to(torch.float32) - current_weights_batch.to(
                    torch.float32
                )
                pseudo_gradients[start:end] = pseudo_gradients_batch.to(torch.bfloat16)

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

            # Free the two ~Nx bf16 buffers now that pseudo_gradients is fully materialized.
            # On a 6.5B-param stage these are ~13 GB each; holding them through S3 upload
            # has previously caused host OOM kills (no swap on Paperspace hosts).
            del previous_weights_batch, current_weights_batch, pseudo_gradients_batch
            del previous_weights, current_weights
            gc.collect()

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
                self._log_resources("after_collect_attestation")

                check_for_nans_and_infs(
                    tensor=pseudo_gradients,
                    name=f"pseudo gradients for miner {self.hotkey[:8]}",
                    exception_type=NanInfException,
                )
                self._log_resources("after_check_nan_inf")

                logger.info(
                    f"submit_weights: uploading self-describing weights blob "
                    f"for {self.num_partitions} partitions (1 S3 upload + 1 API submit call)"
                )
                path = await upload_weights_blob(
                    tensor=pseudo_gradients,
                    num_sections=self.num_partitions,
                    file_type="weights",
                    hotkey=self.wallet.hotkey,
                    miner_api_client=self.miner_api_client,
                    local_optimization_steps=self.training_phase.local_optimization_steps,
                    run_flags=self.run_flags,
                )

                response: WeightSubmitResponse = await self.miner_api_client.submit_weights(
                    weight_update=WeightUpdate(
                        weights_path=path.object_path,
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

                        # Upload optimizer state as a self-describing single-section blob
                        optimizer_state_upload = await upload_weights_blob(
                            tensor=flat_optimizer_state,
                            num_sections=1,
                            file_type="optimizer_state",
                            hotkey=self.wallet.hotkey,
                            miner_api_client=self.miner_api_client,
                            local_optimization_steps=self.training_phase.local_optimization_steps,
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

            finally:
                # End-of-submit_weights cleanup: drop the large CPU buffers that
                # were alive during upload (pseudo_gradients ~11 GB, and
                # flat_optimizer_state ~22 GB if this miner was selected to
                # upload optimizer state). Python would free these on frame
                # unwind anyway, but doing it explicitly + gc.collect() before
                # malloc_trim() gives us a defined ordering.
                #
                # Then ask glibc to actually return free pages to the OS so
                # epoch 2 doesn't inherit an inflated RSS baseline from epoch
                # 1's transient allocations. This is the direct counter to the
                # heap-retention symptom (RSS stays high after large bf16
                # allocations are freed internally but not returned to the
                # kernel).
                self._log_resources("submit_weights_end_before_cleanup")
                try:
                    del pseudo_gradients
                except NameError:
                    pass
                try:
                    del flat_optimizer_state
                except NameError:
                    pass
                gc.collect()
                # malloc_trim is glibc-specific. Best-effort: skip silently on
                # macOS, musl-libc Alpine, etc. The call is safe on glibc and
                # typically costs <100 ms.
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
                self._log_resources("submit_weights_end_after_cleanup")

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
            # Resolve node location from public IP (best-effort, non-blocking on failure)
            self._node_location = await resolve_node_location()

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

                # Stop outbound send loop before P2P
                try:
                    if self.training_phase is not None:
                        self.training_phase._publisher.stop_send_loop()
                except Exception as e:
                    logger.error(f"Failed to stop outbound send loop: {e}")

                # Stop P2P communication
                try:
                    await self._stop_p2p()
                    logger.info("P2P stopped")
                except Exception as e:
                    logger.error(f"Failed to stop P2P: {e}")

                # Stop the telemetry service
                try:
                    if self.telemetry_service:
                        await self.telemetry_service.stop()
                        logger.info("Telemetry service stopped")
                except Exception as e:
                    logger.error(f"Failed to stop telemetry service: {e}")

                try:
                    if self.elastic_device_mesh is not None:
                        await self.elastic_device_mesh.stop()
                except Exception as e:
                    logger.error(f"Failed to stop node registry: {e}")

                try:
                    await self._bridge_manager.stop()
                    logger.info("Bridge v2 variable manager stopped")
                except Exception as e:
                    logger.error(f"Failed to stop bridge variable manager: {e}")

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

        if self.elastic_device_mesh is not None:
            await self.elastic_device_mesh.stop()

        # Stop the telemetry service
        try:
            if self.telemetry_service:
                await self.telemetry_service.stop()
                logger.info("Telemetry service stopped")
        except Exception as e:
            logger.error(f"Failed to stop telemetry service: {e}")

        old_run_id = self.state_manager.run_id
        old_layer = self.state_manager.layer

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
        else:
            delete_saved_model_weights_and_optimizer_state(hotkey=self.hotkey)
            logger.info(f"Deleted stale weight files from previous run/layer (old run={old_run_id}, layer={old_layer})")

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
            n_batches = min(miner_settings.N_PARTITION_BATCHES, len(partitions))
            total_pseudograd_downloads = len(partitions) * len(weight_path_per_layer)
            total_optimizer_downloads = len(partitions)
            total_s3_uploads = len(partitions)
            logger.info(
                f"merge_partitions: {len(partitions)} partitions across {n_batches} batches | "
                f"{len(weight_path_per_layer)} miners contributing | "
                f"expected requests: {len(weight_path_per_layer)} blob-trailer (cached) + "
                f"{total_pseudograd_downloads} pseudograd + "
                f"{total_optimizer_downloads} optimizer downloads + "
                f"{total_s3_uploads} blob S3 uploads + {n_batches} API submit calls"
            )

            async def submit_batch(final_partitions: list[MinerPartition]) -> None:
                attestation_payload = await self._collect_attestation_payload(action="merged_partitions")
                await self.miner_api_client.submit_merged_partitions(
                    merged_partitions=final_partitions,
                    attestation=attestation_payload,
                )
                logger.debug(f"{len(final_partitions)} batch partitions submitted")

            submission_tasks: list[asyncio.Task] = []
            # Grab a batch of partitions to download the weights for
            for batch in range(n_batches):
                logger.debug(f"Merging batch {batch} of {n_batches}")
                # Baseline RAM at start of this batch. With the end-of-batch trim
                # below, this should stay roughly flat across iterations.
                self._log_resources(f"merge_batch_{batch}_start")

                # Grab a batch of partitions to merge (no downloading yet)
                batch_partitions: list[MergingPartition] = get_partition_batch(batch_index=batch, partitions=partitions)
                logger.debug(f"{len(batch_partitions)} batch partitions grabbed")

                # Download the weights for the batch. Returns the (filtered) source
                # miner list whose trailer fetch succeeded; this is positionally
                # aligned with each partition.pseudograds, and is reused below for
                # weighted averaging.
                merging_partitions, valid_sources = await download_pseudograds_for_partition_batch(
                    batch_partitions=batch_partitions,
                    submitted_weights_list=weight_path_per_layer,
                    run_flags=self.run_flags,
                )
                logger.debug(f"{len(merging_partitions)} batch partitions downloaded successfully")

                # Gets the old partition for the batch (which point us to the previous optimizer state)
                merging_partitions = await self.get_old_partition_for_partition_batch(merging_partitions)
                logger.debug(f"{len(merging_partitions)} batch partitions got old partition")

                # Download the previous optimizer state for the batch (fills partitions.old_optimizer_state with the previous optimizer state)
                merging_partitions = await download_previous_optimizer_state_for_partition_batch(merging_partitions)
                logger.debug(f"{len(merging_partitions)} batch partitions downloaded previous optimizer state")

                # Determine if we have enough memory in the GPU to merge the partitions on GPU or CPU.
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
                    else:
                        logger.debug(
                            "Merging partitions on GPU"
                            f" - needed {need_to_merge_on_gpu / 1024**3:.2f}GB, available {avail_memory / 1024**3:.2f}GB"
                            f" ({len(merging_partitions)} partition(s))"
                        )

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

                # TODO: Epoch counter starts at 0 but we increment at the START of the epoch.
                # For the "current global epoch" to be correct, we just need to subtract 1 from the epoch counter.
                # However, this is dumb but we are going to re-write the miner code soon.
                current_global_epoch = self.model_manager.epoch_on_registration + self.model_manager.epoch_counter - 1

                merged_partitions = await merge_partition_batch(
                    partition_batch=merging_partitions,
                    submitted_weights_list=valid_sources,
                    old_model=old_model,
                    weights_length=weights_length,
                    num_partitions=self.num_partitions,
                    device=device,
                    run_flags=self.run_flags,
                    epoch=current_global_epoch,
                )
                logger.debug(f"{len(merged_partitions)} batch partitions merged")
                log_gpu_memory_usage(note=f"after merging partitions on {device}")

                # Upload the merged partitions to S3 and fire off submission in the background
                final_partitions = await upload_partition_batch(
                    merged_partitions=merged_partitions,
                    hotkey=self.wallet.hotkey,
                    miner_api_client=self.miner_api_client,
                    run_flags=self.run_flags,
                )
                logger.debug(f"{len(final_partitions)} batch partitions uploaded")
                submission_tasks.append(asyncio.create_task(submit_batch(final_partitions)))

                self.model_manager.model = self.model_manager.model.to(self.device)

                del old_model
                del merged_partitions  # TODO: @cassova: do a better job of cleaning this up
                del final_partitions

                # End-of-batch cleanup: explicitly drop the partition-batch holders
                # (each MergingPartition pins ~340 MB pseudograds + ~340 MB old/new
                # optimizer state + ~340 MB weights). Without these dels the prior
                # iteration's tensors stay alive in the function frame until the
                # next iteration's downloads rebind the variable — i.e. peak
                # memory is briefly ~2× during the rebind.
                del batch_partitions, merging_partitions

                # Trim glibc's heap so pages from this batch's transient fp32
                # buffers (per-partition averaging, optimizer-state slicing,
                # serialization) get returned to the OS before the next batch.
                # Without this the heap inflated ~50 GB across 32 batches and
                # OOM-killed the process when the post-merge download tried to
                # allocate its target tensor. Mirrors the cleanup in
                # submit_weights's `finally` block.
                gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass  # macOS / musl / non-glibc — skip silently

                log_gpu_memory_usage(note="after merging partitions")
                self._log_resources(f"merge_batch_{batch}_end_after_trim")

            # Wait for all background submission tasks to complete
            await asyncio.gather(*submission_tasks)
            logger.debug(f"All {len(submission_tasks)} batch submission tasks completed")
