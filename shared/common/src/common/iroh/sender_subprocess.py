"""SenderSubprocess: runs a PooledSender in a child subprocess for fault isolation.

When the Iroh sender's Rust QUIC/DERP stack becomes poisoned (e.g. after a node
reset), ``Iroh.memory_with_options()`` blocks indefinitely. By running the sender
in a subprocess we can kill it and respawn — just like ReceiverProcess.

Architecture:
  - ``_sender_worker()`` — subprocess entry point, creates a PooledSender
  - ``SenderSubprocess(IrohSubprocess)`` — parent-side lifecycle manager
  - ``SenderProxy`` — parent-side API with same interface as PooledSender

IPC protocol (via multiprocessing queues):
  Request:  {"id": uuid, "method": str, "args": {...}}
  Response: {"id": uuid, "result": bytes|None, "error": str|None, "timings": dict|None}
"""

from __future__ import annotations

import asyncio
import os
import signal
import uuid
from typing import Any, Callable, TypeVar, overload

from loguru import logger
from pydantic import BaseModel

from common.iroh.iroh_subprocess import IrohSubprocess, _mp_ctx
from common.iroh.timings import P2POperationTimings

ModelT = TypeVar("ModelT", bound=BaseModel)


class SenderUnavailableError(Exception):
    """Raised when the sender subprocess is restarting or not available.

    Callers can catch this to distinguish transient sender unavailability
    (worth retrying) from permanent send failures (peer unreachable, etc.).
    """


# Sentinel for shutdown
_SHUTDOWN = "__shutdown__"


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------


def _sender_worker(
    request_queue: "Any",
    response_queue: "Any",
    status_queue: "Any",
    max_connections: int,
    retry_policy_dict: dict,
    timeouts_dict: dict,
    health_check_interval: float,
) -> None:
    """Entry point for the sender subprocess.

    Creates a PooledSender inside a fresh process with a clean Rust runtime.
    Reads requests from request_queue, dispatches to the sender, writes
    responses to response_queue.
    """
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    try:
        asyncio.run(
            _run_sender(
                request_queue,
                response_queue,
                status_queue,
                max_connections,
                retry_policy_dict,
                timeouts_dict,
                health_check_interval,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        try:
            status_queue.put(("error", str(exc)))
        except Exception:
            pass


async def _run_sender(
    request_queue: "Any",
    response_queue: "Any",
    status_queue: "Any",
    max_connections: int,
    retry_policy_dict: dict,
    timeouts_dict: dict,
    health_check_interval: float,
) -> None:
    """Async core of the sender subprocess."""
    from common.iroh.sender import PooledSender
    from common.iroh.retry import P2PRetryPolicy, P2PTimeouts

    retry_policy = P2PRetryPolicy(
        max_retries=retry_policy_dict["max_retries"],
        base_delay=retry_policy_dict["base_delay"],
        max_delay=retry_policy_dict["max_delay"],
        backoff_factor=retry_policy_dict["backoff_factor"],
        invalidate_on_timeout=retry_policy_dict["invalidate_on_timeout"],
        invalidate_on_error=retry_policy_dict["invalidate_on_error"],
    )
    timeouts = P2PTimeouts(**timeouts_dict)

    sender = PooledSender(
        max_connections=max_connections,
        retry_policy=retry_policy,
        timeouts=timeouts,
        health_check_interval=health_check_interval,
    )

    # Report ready
    status_queue.put(("started", "sender"))

    loop = asyncio.get_running_loop()

    async def on_unhealthy_forwarded(monitored: Any, result: Any) -> None:
        """Forward sender health events to parent."""
        try:
            status_queue.put(("unhealthy", result.health.value))
        except Exception:
            pass

    # Override the sender's unhealthy callback to forward to parent
    sender._monitored_node._on_unhealthy = on_unhealthy_forwarded

    async def _process_request(req: dict) -> None:
        """Process a single request from the parent."""
        req_id = req["id"]
        method = req["method"]
        args = req["args"]

        try:
            result = None
            timings_dict = None

            if method == "send_message":
                node_id = args["node_id"]
                message = args["message"]
                timings = P2POperationTimings() if args.get("has_timings") else None
                await sender.send_message(node_id, message, timings=timings)
                if timings is not None:
                    timings_dict = timings.model_dump()

            elif method == "send_message_bi":
                node_id = args["node_id"]
                message = args["message"]
                max_message_size = args["max_message_size"]
                timings = P2POperationTimings() if args.get("has_timings") else None
                result = await sender.send_message_bi(
                    node_id,
                    message,
                    max_message_size,
                    timings=timings,
                )
                if timings is not None:
                    timings_dict = timings.model_dump()

            elif method == "force_destroy_node":
                await sender.force_destroy_node()

            elif method == "shutdown":
                await sender.shutdown()

            else:
                raise ValueError(f"Unknown method: {method}")

            response_queue.put(
                {
                    "id": req_id,
                    "result": result,
                    "error": None,
                    "timings": timings_dict,
                }
            )

        except Exception as exc:
            response_queue.put(
                {
                    "id": req_id,
                    "result": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timings": None,
                }
            )

    # Main loop: read requests from queue, dispatch as tasks
    while True:
        try:
            req = await loop.run_in_executor(None, request_queue.get, True, 1.0)
        except Exception:
            continue

        if req == _SHUTDOWN:
            # Graceful shutdown
            try:
                await sender.force_destroy_node()
            except Exception:
                pass
            break

        asyncio.create_task(_process_request(req))


# ---------------------------------------------------------------------------
# Parent-side subprocess manager
# ---------------------------------------------------------------------------


class SenderSubprocess(IrohSubprocess):
    """Manages a PooledSender running in a child subprocess."""

    def __init__(
        self,
        max_connections: int = 5,
        retry_policy: "Any | None" = None,
        timeouts: "Any | None" = None,
        health_check_interval: float = 30.0,
    ):
        super().__init__(process_name="P2PSender")

        from common.iroh.retry import P2PRetryPolicy, P2PTimeouts

        self._max_connections = max_connections
        self._retry_policy = retry_policy or P2PRetryPolicy()
        self._timeouts = timeouts or P2PTimeouts()
        self._health_check_interval = health_check_interval

        self._request_queue: Any = _mp_ctx.Queue()
        self._response_queue: Any = _mp_ctx.Queue()
        self._proxy: SenderProxy | None = None
        self._dispatcher_task: asyncio.Task | None = None

    @property
    def proxy(self) -> SenderProxy:
        """Parent-side proxy with the same API as PooledSender."""
        if self._proxy is None:
            raise RuntimeError("SenderSubprocess not started")
        return self._proxy

    def _worker_target(self) -> Callable[..., Any]:
        return _sender_worker

    def _build_process_args(self) -> tuple:
        import dataclasses

        # Serialize retry_policy without retryable_exceptions (not picklable)
        rp = self._retry_policy
        retry_dict = {
            "max_retries": rp.max_retries,
            "base_delay": rp.base_delay,
            "max_delay": rp.max_delay,
            "backoff_factor": rp.backoff_factor,
            "invalidate_on_timeout": rp.invalidate_on_timeout,
            "invalidate_on_error": rp.invalidate_on_error,
        }
        timeouts_dict = dataclasses.asdict(self._timeouts)

        return (
            self._request_queue,
            self._response_queue,
            self._status_queue,
            self._max_connections,
            retry_dict,
            timeouts_dict,
            self._health_check_interval,
        )

    async def start(self, timeout: float = 5.0) -> str:
        """Start the sender subprocess and create the proxy."""
        # Recreate queues in case we're restarting
        self._request_queue = _mp_ctx.Queue()
        self._response_queue = _mp_ctx.Queue()

        node_id = await super().start(timeout=timeout)

        self._proxy = SenderProxy(self._request_queue, self._response_queue)
        self._dispatcher_task = asyncio.create_task(
            self._proxy._dispatch_responses(),
            name="SenderResponseDispatcher",
        )

        return node_id

    async def stop(self) -> None:
        """Send shutdown to subprocess, cancel dispatcher, kill process."""
        # Try graceful shutdown first
        try:
            self._request_queue.put(_SHUTDOWN)
        except Exception:
            pass

        # Cancel the response dispatcher
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            try:
                await asyncio.wait_for(self._dispatcher_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._dispatcher_task = None

        # Fail any pending futures
        if self._proxy is not None:
            self._proxy._fail_all_pending("SenderSubprocess stopped")
            self._proxy = None

        await super().stop()

    async def restart(self, timeout: float = 5.0) -> str:
        """Kill subprocess, fail pending futures, recreate queues, respawn."""
        # Cancel dispatcher
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            try:
                await asyncio.wait_for(self._dispatcher_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._dispatcher_task = None

        # Fail pending futures and null out the proxy so callers see None
        # during the restart window (prevents enqueuing on stale queues)
        if self._proxy is not None:
            self._proxy._fail_all_pending("SenderSubprocess restarting")
            self._proxy = None

        # Kill old process
        self._kill_subprocess()
        self._drain_queue()

        # Recreate queues (old queues may have stale data)
        self._request_queue = _mp_ctx.Queue()
        self._response_queue = _mp_ctx.Queue()

        # Spawn new process
        self._process = _mp_ctx.Process(
            target=self._worker_target(),
            args=self._build_process_args(),
            daemon=True,
            name=self._process_name,
        )
        self._process.start()
        logger.info(f"{self._process_name} respawned (pid={self._process.pid})")

        node_id = await self._wait_for_started(timeout=timeout)
        self._node_id = node_id
        logger.info(f"{self._process_name} restarted (node_id={node_id})")

        # Recreate proxy with new queues
        self._proxy = SenderProxy(self._request_queue, self._response_queue)
        self._dispatcher_task = asyncio.create_task(
            self._proxy._dispatch_responses(),
            name="SenderResponseDispatcher",
        )

        return node_id


# ---------------------------------------------------------------------------
# Parent-side proxy (same API as PooledSender)
# ---------------------------------------------------------------------------


class SenderProxy:
    """Parent-side API matching PooledSender's interface.

    Serializes requests onto request_queue, waits for responses via
    UUID-matched asyncio.Futures resolved by a background dispatcher.
    """

    _SEND_TIMEOUT: float = 60.0  # Max wait for any individual send

    def __init__(self, request_queue: Any, response_queue: Any):
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._pending: dict[str, asyncio.Future] = {}

    # ── send (unidirectional) ────────────────────────────────────────

    async def send_message(
        self,
        node_id: str | list[str],
        message: bytes,
        timings: P2POperationTimings | None = None,
    ) -> None:
        """Send a unidirectional message via the subprocess sender."""
        if isinstance(node_id, list):
            await asyncio.gather(*[self.send_message(nid, message) for nid in node_id])
            return

        response = await self._send_and_wait(
            "send_message",
            {"node_id": node_id, "message": message, "has_timings": timings is not None},
        )
        if timings is not None and response.get("timings"):
            self._hydrate_timings(timings, response["timings"])

    async def send_routed(
        self,
        route: str,
        node_id: str | list[str],
        model: BaseModel,
        serializer: Any = None,
        timings: P2POperationTimings | None = None,
    ) -> None:
        """Send a typed UNI message with routed envelope via subprocess."""
        from common.iroh.router import wrap_routed_envelope

        payload = wrap_routed_envelope(route, model, serializer)
        await self.send_message(node_id, payload, timings=timings)

    # ── send + receive (bidirectional) ───────────────────────────────

    @overload
    async def send_message_bi(
        self,
        node_id: str,
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = ...,
        timings: P2POperationTimings | None = ...,
    ) -> bytes:
        ...

    @overload
    async def send_message_bi(
        self,
        node_id: list[str],
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = ...,
        timings: P2POperationTimings | None = ...,
    ) -> list[bytes]:
        ...

    async def send_message_bi(
        self,
        node_id: str | list[str],
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = None,
        timings: P2POperationTimings | None = None,
    ) -> bytes | list[bytes]:
        """Send a message and wait for response via subprocess sender."""
        if isinstance(node_id, list):
            return list(
                await asyncio.gather(
                    *[self.send_message_bi(nid, message, max_message_size, callback) for nid in node_id]
                )
            )

        response = await self._send_and_wait(
            "send_message_bi",
            {
                "node_id": node_id,
                "message": message,
                "max_message_size": max_message_size,
                "has_timings": timings is not None,
            },
        )

        if timings is not None and response.get("timings"):
            self._hydrate_timings(timings, response["timings"])

        result = response["result"]
        if callback and result is not None:
            return callback(result)
        return result

    async def send_routed_bi_raw(
        self,
        route: str,
        node_id: str,
        model: BaseModel,
        max_message_size: int,
        serializer: Any = None,
        timings: P2POperationTimings | None = None,
    ) -> bytes:
        """Send a typed BI request with routed envelope, return raw response bytes.

        Unlike ``send_routed_bi`` on PooledSender, this does **not** unwrap the
        response as a routed envelope — it returns the raw bytes from the peer.
        Useful when the response is a simple status byte (e.g. activation push ack).
        """
        from common.iroh.router import wrap_routed_envelope

        payload = wrap_routed_envelope(route, model, serializer)
        return await self.send_message_bi(node_id, payload, max_message_size, timings=timings)

    # ── lifecycle methods (forwarded to subprocess) ──────────────────

    async def force_destroy_node(self, timeout: float = 5.0) -> None:
        """Tell subprocess to force-destroy its Iroh node."""
        try:
            await asyncio.wait_for(
                self._send_and_wait("force_destroy_node", {}),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning(f"SenderProxy force_destroy_node failed: {exc}")

    async def shutdown(self) -> None:
        """Tell subprocess sender to shut down."""
        try:
            await asyncio.wait_for(
                self._send_and_wait("shutdown", {}),
                timeout=5.0,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning(f"SenderProxy shutdown failed: {exc}")

    # ── internal ─────────────────────────────────────────────────────

    async def _send_and_wait(self, method: str, args: dict) -> dict:
        """Queue a request and wait for the matching response."""
        req_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[req_id] = future

        request = {"id": req_id, "method": method, "args": args}

        try:
            self._request_queue.put(request)
        except Exception as exc:
            self._pending.pop(req_id, None)
            raise SenderUnavailableError(f"Failed to enqueue sender request: {exc}") from exc

        try:
            response = await asyncio.wait_for(future, timeout=self._SEND_TIMEOUT)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise asyncio.TimeoutError(f"SenderProxy.{method} timed out after {self._SEND_TIMEOUT}s")

        if response.get("error"):
            raise RuntimeError(f"Sender subprocess error: {response['error']}")

        return response

    async def _dispatch_responses(self) -> None:
        """Background task: read response_queue and resolve matching Futures."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                response = await loop.run_in_executor(None, self._response_queue.get, True, 1.0)
            except Exception:
                await asyncio.sleep(0)
                continue

            req_id = response.get("id")
            if req_id is None:
                continue

            future = self._pending.pop(req_id, None)
            if future is not None and not future.done():
                future.set_result(response)

    def _fail_all_pending(self, reason: str) -> None:
        """Fail all pending futures (called on stop/restart)."""
        for req_id, future in self._pending.items():
            if not future.done():
                future.set_exception(SenderUnavailableError(reason))
        self._pending.clear()

    @staticmethod
    def _hydrate_timings(target: P2POperationTimings, source: dict) -> None:
        """Copy timing values from subprocess response dict into caller's object."""
        for field in (
            "connection_duration",
            "stream_open_duration",
            "send_duration",
            "receive_duration",
            "total_start",
            "total_end",
            "total_duration",
            "attempt_count",
            "retry_count",
            "total_backoff_time",
            "bytes_sent",
            "bytes_received",
        ):
            val = source.get(field)
            if val is not None:
                setattr(target, field, val)
        errors = source.get("errors")
        if errors:
            target.errors = errors
