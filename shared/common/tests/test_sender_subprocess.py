"""Unit tests for SenderSubprocess IPC layer (parent ↔ subprocess timeout semantics).

These tests exercise the parent-side ``SenderProxy`` against a *fake* subprocess
that mimics ``_run_sender``'s task-per-request dispatch pattern. They do not
spawn real OS processes or use iroh — the goal is to isolate the IPC contract
between parent timeout and subprocess execution.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import time

import pytest

from common.iroh.sender_subprocess import SenderProxy


# ---------------------------------------------------------------------------
# Fake subprocess loop
# ---------------------------------------------------------------------------


class FakeSubprocess:
    """In-process stand-in for ``_run_sender`` that mirrors its dispatch pattern.

    Reads requests from ``request_queue``, spawns a task per request that
    "performs an iroh send" (just an ``asyncio.sleep`` for ``send_latency_s``),
    then writes a response to ``response_queue``. Tracks every send that
    actually completed so tests can assert on subprocess-side behavior
    independently of parent-side timeouts.
    """

    _STOP = "__STOP__"

    def __init__(self, request_queue, response_queue, send_latency_s: float):
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._send_latency = send_latency_s
        self.delivered: list[str] = []  # req_ids that ran to completion
        self.cancelled: list[str] = []  # req_ids whose task was cancelled
        self._loop_task: asyncio.Task | None = None
        self._inflight: list[asyncio.Task] = []

    async def _process_request(self, req: dict, cancel_ev: asyncio.Event) -> None:
        req_id = req["id"]
        try:
            sleep_t = asyncio.create_task(asyncio.sleep(self._send_latency))
            cancel_t = asyncio.create_task(cancel_ev.wait())
            done, _ = await asyncio.wait(
                {sleep_t, cancel_t},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_t in done:
                sleep_t.cancel()
                try:
                    await sleep_t
                except asyncio.CancelledError:
                    pass
                self.cancelled.append(req_id)
                self._response_queue.put(
                    {
                        "id": req_id,
                        "result": None,
                        "error": "P2PSendCancelledError: fake send aborted",
                        "timings": None,
                    }
                )
                return
            cancel_t.cancel()
            try:
                await cancel_t
            except asyncio.CancelledError:
                pass
            await sleep_t
            self.delivered.append(req_id)
            self._response_queue.put({"id": req_id, "result": b"ack", "error": None, "timings": None})
        except asyncio.CancelledError:
            if req_id not in self.cancelled:
                self.cancelled.append(req_id)
            self._response_queue.put(
                {
                    "id": req_id,
                    "result": None,
                    "error": "CancelledError: sender request task cancelled",
                    "timings": None,
                }
            )

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        cancel_registry: dict[str, asyncio.Event] = {}
        pending_cancel: set[str] = set()
        inflight_tasks: dict[str, asyncio.Task] = {}

        def signal_cancel(target_id: str) -> None:
            ev = cancel_registry.get(target_id)
            if ev is not None:
                ev.set()
            else:
                pending_cancel.add(target_id)
            t = inflight_tasks.get(target_id)
            if t is not None and not t.done():
                t.cancel()

        while True:
            try:
                req = await loop.run_in_executor(None, self._request_queue.get, True, 0.05)
            except Exception:
                continue
            if req == self._STOP:
                break
            if req.get("method") == "cancel":
                tid = req.get("args", {}).get("target_id")
                if tid is not None:
                    signal_cancel(tid)
                continue

            req_id = str(req["id"])
            cancel_ev = asyncio.Event()
            cancel_registry[req_id] = cancel_ev
            if req_id in pending_cancel:
                cancel_ev.set()
                pending_cancel.discard(req_id)

            async def _wrapped(
                bound_req: dict = req,
                bound_id: str = req_id,
                bound_ev: asyncio.Event = cancel_ev,
            ) -> None:
                try:
                    await self._process_request(bound_req, bound_ev)
                finally:
                    cancel_registry.pop(bound_id, None)
                    inflight_tasks.pop(bound_id, None)
                    pending_cancel.discard(bound_id)

            t = asyncio.create_task(_wrapped())
            self._inflight.append(t)
            inflight_tasks[req_id] = t

    def start(self) -> None:
        self._loop_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._request_queue.put(self._STOP)
        if self._loop_task is not None:
            try:
                await asyncio.wait_for(self._loop_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._loop_task.cancel()
        # Drain any in-flight tasks so the test sees their final state
        for t in self._inflight:
            if not t.done():
                try:
                    await asyncio.wait_for(t, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass


# ---------------------------------------------------------------------------
# Bug reproduction
# ---------------------------------------------------------------------------


class TestParentTimeoutCancelsSubprocessSend:
    """Parent-side timeout MUST also abort the subprocess-side send.

    Bug observed in production: when ``SenderProxy.send_message_bi`` times out
    on the parent, the subprocess task that's actually performing the iroh send
    is *not* cancelled — it keeps running and eventually delivers the message.
    The publisher's retry path then re-sends to a *different* peer, so the same
    activation lands at multiple receivers.

    The publisher attempts up to ``ACTIVATION_SEND_MAX_TRIES=5`` retries; this
    test reproduces the smaller variant of three retries that the production
    logs showed (one initial send + 2 visible duplicate deliveries).
    """

    @pytest.fixture
    def queues(self):
        ctx = multiprocessing.get_context("spawn")
        rq = ctx.Queue()
        sq = ctx.Queue()
        yield rq, sq
        # Best-effort drain; mp.Queue cleanup is implicit on GC.

    async def test_subprocess_send_aborted_when_parent_times_out(self, queues):
        request_queue, response_queue = queues

        # Tight timings so the test runs in ~1 second:
        # - Parent timeout: 50 ms (sender gives up fast)
        # - Subprocess send latency: 200 ms (would have completed had we waited)
        parent_timeout_s = 0.05
        send_latency_s = 0.20

        fake_sub = FakeSubprocess(request_queue, response_queue, send_latency_s=send_latency_s)
        fake_sub.start()

        proxy = SenderProxy(request_queue, response_queue)
        dispatcher = asyncio.create_task(proxy._dispatch_responses())

        try:
            timeouts_seen = 0
            t0 = time.monotonic()
            for attempt in range(3):
                try:
                    await proxy.send_message_bi(
                        node_id=f"peer-{attempt}",
                        message=b"hello",
                        max_message_size=1024,
                        timeout=parent_timeout_s,
                    )
                except asyncio.TimeoutError:
                    timeouts_seen += 1
            parent_elapsed = time.monotonic() - t0

            # Sanity: parent really did time out on every attempt.
            assert timeouts_seen == 3, f"Parent should have timed out on all 3 attempts, " f"saw {timeouts_seen}"
            # Sanity: parent didn't accidentally wait for subprocess sends.
            assert parent_elapsed < 3 * send_latency_s, (
                f"Parent should have given up well before sends completed; "
                f"elapsed={parent_elapsed:.3f}s vs 3×latency={3 * send_latency_s:.3f}s"
            )

            # Wait long enough for any unfinished subprocess sends to complete.
            await asyncio.sleep(send_latency_s * 2)

            # Subprocess must not complete sends after parent gave up; cooperative
            # cancel via IPC should abort before the fake latency elapses.
            assert len(fake_sub.delivered) == 0, (
                f"Expected no subprocess deliveries after parent timeouts, " f"got delivered={fake_sub.delivered}"
            )
            assert len(fake_sub.cancelled) == 3, (
                f"Expected 3 cooperative cancellations, got {len(fake_sub.cancelled)} "
                f"cancelled={fake_sub.cancelled}"
            )
        finally:
            dispatcher.cancel()
            try:
                await dispatcher
            except asyncio.CancelledError:
                pass
            await fake_sub.stop()
