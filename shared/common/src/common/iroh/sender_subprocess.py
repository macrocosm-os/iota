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
            {"method": "cancel", "args": {"target_id": "<original request id>"}, "id": ...}
  Response: {"id": uuid, "result": bytes|None, "error": str|None, "timings": dict|None}
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import signal
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, TypeVar, overload

from loguru import logger
from pydantic import BaseModel

from common.iroh.iroh_subprocess import IrohSubprocess, _mp_ctx
from common.iroh.timings import P2POperationTimings
from common.iroh.sender import PooledSender
from common.iroh.retry import P2PRetryPolicy, P2PSendCancelledError, P2PTimeouts
from common.iroh.router import wrap_routed_envelope

ModelT = TypeVar("ModelT", bound=BaseModel)


class SenderUnavailableError(Exception):
    """Raised when the sender subprocess is restarting or not available.

    Callers can catch this to distinguish transient sender unavailability
    (worth retrying) from permanent send failures (peer unreachable, etc.).
    """


# Sentinel for shutdown
_SHUTDOWN = "__shutdown__"


def _format_phases(timings: P2POperationTimings | None) -> str:
    """Format the four iroh phase durations for a single-line log entry.

    Missing phases are rendered as ``-`` so a stalled call is obvious from
    *which* field is unfilled — e.g. ``conn=12 open=4 send=- recv=-`` means
    the call hung inside ``write_all``/``finish``.
    """
    if timings is None:
        return "-"

    def _fmt(v: float | None) -> str:
        return f"{v * 1000:.0f}" if v is not None else "-"

    return (
        f"conn={_fmt(timings.connection_duration)} "
        f"open={_fmt(timings.stream_open_duration)} "
        f"send={_fmt(timings.send_duration)} "
        f"recv={_fmt(timings.receive_duration)}"
    )


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

    cancel_registry: dict[str, asyncio.Event] = {}
    # Bounded LRU of pre-arrival cancels. Cancels can race ahead of their target
    # request reaching the dispatcher loop, so we stash them here and apply at
    # task creation. Bounded to avoid an unbounded leak when cancels arrive AFTER
    # a request has already completed (post-completion cancels match no future
    # request, but UUIDs are unique so dropping old entries is safe).
    pending_cancel: "OrderedDict[str, None]" = OrderedDict()
    _PENDING_CANCEL_MAX = 1024
    inflight_tasks: dict[str, asyncio.Task] = {}

    def _signal_cancel(target_id: str) -> None:
        tid = str(target_id)
        logger.debug(f"[sender] cancel signal target: req_id={tid[:8]} (event set and/or task.cancel)")
        ev = cancel_registry.get(target_id)
        if ev is not None:
            ev.set()
        else:
            pending_cancel[target_id] = None
            pending_cancel.move_to_end(target_id)
            while len(pending_cancel) > _PENDING_CANCEL_MAX:
                pending_cancel.popitem(last=False)
        task = inflight_tasks.get(target_id)
        if task is not None and not task.done():
            task.cancel()

    async def _process_request(req: dict, cancel_ev: asyncio.Event) -> None:
        """Process a single request from the parent."""
        req_id = req["id"]
        method = req["method"]
        args = req["args"]
        # Note: ``ts`` is wall-clock (``time.time()``) so it is comparable
        # across the parent and subprocess; ``time.monotonic()`` is only
        # guaranteed comparable within a single process per the docs.
        enqueued_ts = req.get("ts")
        queue_dwell = (time.time() - enqueued_ts) if enqueued_ts is not None else None

        # Always populate timings for diagnostics, regardless of caller request.
        timings = P2POperationTimings()
        # Resolve a "<iroh:16> hk=<hotkey:8>" label for log lines so each send
        # can be cross-referenced against the node registry by either identity.
        node_id_short = ""
        hk_book: dict[str, str] = getattr(sender, "_peer_hotkeys", {})
        nid_arg = args.get("node_id")

        def _label(nid: str) -> str:
            hk = (hk_book.get(nid) or "?")[:8]
            return f"{nid[:16]} hk={hk}"

        if isinstance(nid_arg, str):
            node_id_short = _label(nid_arg)
        elif isinstance(nid_arg, list) and nid_arg:
            node_id_short = f"{_label(nid_arg[0])}+{len(nid_arg) - 1}"

        if queue_dwell is not None and queue_dwell > 0.5:
            logger.warning(
                f"[sender] req_id={req_id[:8]} method={method} node={node_id_short} "
                f"queue_dwell={queue_dwell * 1000:.0f}ms (slow)"
            )
        else:
            logger.debug(
                f"[sender] req_id={req_id[:8]} method={method} node={node_id_short} "
                f"queue_dwell={queue_dwell * 1000:.0f}ms"
                if queue_dwell is not None
                else f"[sender] req_id={req_id[:8]} method={method} node={node_id_short} start"
            )

        iroh_t0 = time.monotonic()
        try:
            result = None
            timings_dict = None

            if method == "send_message":
                node_id = args["node_id"]
                message = args["message"]
                await sender.send_message(
                    node_id,
                    message,
                    timings=timings,
                    cancellation_event=cancel_ev,
                )
                if args.get("has_timings"):
                    timings_dict = timings.model_dump()

            elif method == "send_message_bi":
                node_id = args["node_id"]
                message = args["message"]
                max_message_size = args["max_message_size"]
                result = await sender.send_message_bi(
                    node_id,
                    message,
                    max_message_size,
                    timings=timings,
                    cancellation_event=cancel_ev,
                )
                if args.get("has_timings"):
                    timings_dict = timings.model_dump()

            elif method == "register_peer":
                sender.register_peer(
                    args["node_id"],
                    args.get("relay_url"),
                    list(args.get("direct_addresses") or []),
                    hotkey=args.get("hotkey"),
                )

            elif method == "force_destroy_node":
                await sender.force_destroy_node()

            elif method == "shutdown":
                await sender.shutdown()

            else:
                raise ValueError(f"Unknown method: {method}")

            iroh_dur = time.monotonic() - iroh_t0
            if method in ("send_message", "send_message_bi") and iroh_dur > 1.0:
                logger.info(
                    f"[sender] req_id={req_id[:8]} method={method} node={node_id_short} "
                    f"ok dur={iroh_dur * 1000:.0f}ms phases={_format_phases(timings)}"
                )
            else:
                logger.debug(
                    f"[sender] req_id={req_id[:8]} method={method} node={node_id_short} ok dur={iroh_dur * 1000:.0f}ms"
                )

            response_queue.put(
                {
                    "id": req_id,
                    "result": result,
                    "error": None,
                    "timings": timings_dict,
                }
            )

        except asyncio.CancelledError:
            iroh_dur = time.monotonic() - iroh_t0
            logger.warning(
                f"[sender] req_id={req_id[:8]} method={method} node={node_id_short} "
                f"CANCELLED asyncio task.cancel dur={iroh_dur * 1000:.0f}ms "
                f"phases={_format_phases(timings)}"
            )
            response_queue.put(
                {
                    "id": req_id,
                    "result": None,
                    "error": "CancelledError: sender request task cancelled",
                    "timings": timings.model_dump(),
                }
            )

        except P2PSendCancelledError as exc:
            iroh_dur = time.monotonic() - iroh_t0
            logger.warning(
                f"[sender] req_id={req_id[:8]} method={method} node={node_id_short} "
                f"CANCELLED cooperative (cancel_event) dur={iroh_dur * 1000:.0f}ms "
                f"phases={_format_phases(timings)}"
            )
            response_queue.put(
                {
                    "id": req_id,
                    "result": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timings": timings.model_dump(),
                }
            )

        except Exception as exc:
            iroh_dur = time.monotonic() - iroh_t0
            logger.warning(
                f"[sender] req_id={req_id[:8]} method={method} node={node_id_short} "
                f"FAILED dur={iroh_dur * 1000:.0f}ms phases={_format_phases(timings)} "
                f"err={type(exc).__name__}: {exc}"
            )
            # Always return partial timings on error so callers can attribute the stall.
            response_queue.put(
                {
                    "id": req_id,
                    "result": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timings": timings.model_dump(),
                }
            )

    # Main loop: read requests from queue, dispatch as tasks
    last_qsize_log = 0.0
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

        req_method = req.get("method")
        if req_method == "cancel":
            target = req.get("args", {}).get("target_id")
            if target is not None:
                _signal_cancel(target)
            else:
                logger.warning("[sender] cancel IPC missing target_id")
            continue

        # Periodic backlog visibility: log queue depth at most every 5s, or
        # immediately on backlog spike (>=4 pending requests behind this one).
        now = time.monotonic()
        try:
            qsize = request_queue.qsize()
        except (NotImplementedError, OSError):
            qsize = -1
        if qsize >= 4 or (now - last_qsize_log) >= 5.0:
            in_flight = sum(1 for t in asyncio.all_tasks() if t.get_name().startswith("sender_req_"))
            logger.debug(f"[sender] loop: qsize_after_pop={qsize} in_flight_tasks={in_flight}")
            last_qsize_log = now

        req_id = str(req["id"])
        cancel_ev = asyncio.Event()
        cancel_registry[req_id] = cancel_ev
        if req_id in pending_cancel:
            cancel_ev.set()
            pending_cancel.pop(req_id, None)

        async def _wrapped(
            bound_req: dict = req,
            bound_id: str = req_id,
            bound_ev: asyncio.Event = cancel_ev,
        ) -> None:
            try:
                await _process_request(bound_req, bound_ev)
            finally:
                cancel_registry.pop(bound_id, None)
                inflight_tasks.pop(bound_id, None)
                pending_cancel.pop(bound_id, None)

        t = asyncio.create_task(_wrapped(), name=f"sender_req_{req_id[:8]}")
        inflight_tasks[req_id] = t


# ---------------------------------------------------------------------------
# Parent-side subprocess manager
# ---------------------------------------------------------------------------


class SenderSubprocess(IrohSubprocess):
    """Manages a PooledSender running in a child subprocess."""

    def __init__(
        self,
        max_connections: int = 32,
        retry_policy: "Any | None" = None,
        timeouts: "Any | None" = None,
        health_check_interval: float = 30.0,
    ):
        super().__init__(process_name="P2PSender")

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
            await asyncio.gather(*[self.send_message(nid, message, timings=timings) for nid in node_id])
            return

        response = await self._send_and_wait(
            "send_message",
            {"node_id": node_id, "message": message, "has_timings": timings is not None},
            timings=timings,
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
        timeout: float | None = ...,
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
        timeout: float | None = ...,
    ) -> list[bytes]:
        ...

    async def send_message_bi(
        self,
        node_id: str | list[str],
        message: bytes,
        max_message_size: int,
        callback: Callable[[bytes], bytes] | None = None,
        timings: P2POperationTimings | None = None,
        timeout: float | None = None,
    ) -> bytes | list[bytes]:
        """Send a message and wait for response via subprocess sender.

        ``timeout`` overrides the default ``SenderProxy._SEND_TIMEOUT`` (60 s)
        for this call only.
        """
        if isinstance(node_id, list):
            return list(
                await asyncio.gather(
                    *[
                        self.send_message_bi(
                            nid,
                            message,
                            max_message_size,
                            callback,
                            timings=timings,
                            timeout=timeout,
                        )
                        for nid in node_id
                    ]
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
            timeout=timeout,
            timings=timings,
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
        timeout: float | None = None,
    ) -> bytes:
        """Send a typed BI request with routed envelope, return raw response bytes.

        Unlike ``send_routed_bi`` on PooledSender, this does **not** unwrap the
        response as a routed envelope — it returns the raw bytes from the peer.
        Useful when the response is a simple status byte (e.g. activation push ack).

        ``timeout`` overrides the default ``SenderProxy._SEND_TIMEOUT`` (60 s)
        for this call only.
        """

        payload = wrap_routed_envelope(route, model, serializer)
        return await self.send_message_bi(node_id, payload, max_message_size, timings=timings, timeout=timeout)

    # ── peer address book ────────────────────────────────────────────

    async def register_peer(
        self,
        node_id: str,
        relay_url: str | None,
        direct_addresses: list[str],
        hotkey: str | None = None,
    ) -> None:
        """Cache a peer's relay + direct addresses in the sender's address book.

        With iroh's discovery service disabled, a peer's hints MUST be registered
        before any dial — otherwise iroh has no addressing info and fails with
        ``No addressing information for NodeId(...)``.

        ``hotkey`` (peer's SS58) is forwarded for log-labelling so the
        subprocess's ``[sender]`` log lines can show ``hk=<short>`` next to
        the iroh node id.  Optional — call sites that don't have it can omit.
        """
        try:
            await asyncio.wait_for(
                self._send_and_wait(
                    "register_peer",
                    {
                        "node_id": node_id,
                        "relay_url": relay_url,
                        "direct_addresses": list(direct_addresses or []),
                        "hotkey": hotkey,
                    },
                ),
                timeout=2.0,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning(f"SenderProxy register_peer({node_id[:16]}...) failed: {exc}")

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

    async def _send_and_wait(
        self,
        method: str,
        args: dict,
        timeout: float | None = None,
        timings: P2POperationTimings | None = None,
    ) -> dict:
        """Queue a request and wait for the matching response.

        ``timeout`` overrides ``_SEND_TIMEOUT`` for this single call.  Useful
        when the caller wants to fail fast (e.g. activation pushes that prefer
        re-routing to a different peer over waiting on a stuck one).

        If ``timings`` is provided, ``timings.req_id`` is set to the subprocess
        request ID before awaiting so callers can correlate a parent-side
        timeout (where the subprocess hasn't replied yet) against the
        subprocess's own per-request log line.
        """
        req_id = str(uuid.uuid4())
        if timings is not None:
            timings.req_id = req_id

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[req_id] = future

        # Stamp with wall-clock ts so the subprocess can compute queue dwell
        # time (monotonic is per-process per docs; wall-clock is comparable
        # across processes on the same host).
        enqueued_ts = time.time()
        request = {"id": req_id, "method": method, "args": args, "ts": enqueued_ts}

        try:
            self._request_queue.put(request)
        except Exception as exc:
            self._pending.pop(req_id, None)
            raise SenderUnavailableError(f"Failed to enqueue sender request: {exc}") from exc

        logger.debug(f"SenderProxy request enqueued req_id={req_id[:8]} method={method}")

        effective_timeout = timeout if timeout is not None else self._SEND_TIMEOUT
        timed_out = False
        try:
            try:
                response = await asyncio.wait_for(future, timeout=effective_timeout)
            except asyncio.TimeoutError:
                timed_out = True
                waited = time.time() - enqueued_ts
                logger.warning(
                    f"SenderProxy.{method} req_id={req_id[:8]} timed out after {effective_timeout}s "
                    f"(waited {waited:.2f}s, pending={len(self._pending)})"
                )
                raise asyncio.TimeoutError(f"SenderProxy.{method} timed out after {effective_timeout}s")
        finally:
            # Cleanup runs on every exit path — including outer CancelledError
            # (e.g. caller-side ``asyncio.wait_for`` firing). Without this, the
            # future stays in ``_pending`` and the subprocess task continues
            # competing for ``_conn_lock`` until it eventually completes,
            # accumulating ``in_flight_tasks`` until the subprocess saturates.
            #
            # We send the cancel IPC iff we (the parent) beat the dispatcher to
            # popping ``_pending``. The dispatcher pops on a successful reply;
            # if our pop returns the future, the subprocess hasn't replied yet —
            # it's still working and needs to be told to stop. (We can't gate
            # on ``future.done()`` here: ``asyncio.wait_for`` cancels its inner
            # future on timeout, so the future is always "done" by the time we
            # reach this finally on the timeout/cancel paths.)
            if self._pending.pop(req_id, None) is not None:
                try:
                    self._request_queue.put(
                        {
                            "id": str(uuid.uuid4()),
                            "method": "cancel",
                            "args": {"target_id": req_id},
                            "ts": time.time(),
                        }
                    )
                    if timed_out:
                        logger.info(f"SenderProxy cancel enqueued target: req_id={req_id[:8]}")
                    else:
                        logger.debug(f"SenderProxy cancel enqueued (caller-cancel): req_id={req_id[:8]}")
                except Exception as exc:
                    logger.warning(f"SenderProxy failed to enqueue subprocess cancel for req_id={req_id[:8]}: {exc}")

        # Hydrate partial timings (if any) before raising so the caller's
        # log lines can show how far iroh got — otherwise a subprocess-side
        # iroh exception would surface as `phases=conn=- open=- send=- recv=-`
        # even though the subprocess captured the actual phase progress.
        if timings is not None and response.get("timings"):
            self._hydrate_timings(timings, response["timings"])

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
                err = response.get("error")
                logger.debug(f"SenderProxy response received req_id={req_id[:8]} ok={err is None} error={err!r}")
                future.set_result(response)
            elif future is None:
                logger.debug(
                    f"SenderProxy orphan response req_id={req_id[:8]} "
                    f"(no pending future, likely parent timed out) error={response.get('error')!r}"
                )

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
