"""Unit tests for P2P retry policy, executor, and operation timings."""

import asyncio

import pytest

from common.iroh.retry import P2PRetry, P2PRetryPolicy, P2PTimeouts, _is_connection_scoped_iroh_error
from common.iroh.timings import P2POperationTimings


# ---------------------------------------------------------------------------
# P2PTimeouts
# ---------------------------------------------------------------------------


class TestP2PTimeouts:
    def test_defaults(self):
        t = P2PTimeouts()
        assert t.connection == 5.0
        assert t.stream_open == 5.0
        assert t.send == 30.0
        assert t.receive == 15.0

    def test_custom_values(self):
        t = P2PTimeouts(connection=1.0, stream_open=2.0, send=3.0, receive=4.0)
        assert t.connection == 1.0
        assert t.stream_open == 2.0
        assert t.send == 3.0
        assert t.receive == 4.0

    def test_immutable(self):
        t = P2PTimeouts()
        with pytest.raises(AttributeError):
            t.connection = 99.0


# ---------------------------------------------------------------------------
# P2PRetryPolicy
# ---------------------------------------------------------------------------


class TestP2PRetryPolicy:
    def test_defaults(self):
        p = P2PRetryPolicy()
        assert p.max_retries == 2
        assert p.total_attempts == 3
        assert p.base_delay == 0.25
        assert p.max_delay == 5.0
        assert p.backoff_factor == 2.0

    def test_delay_for_attempt_exponential(self):
        p = P2PRetryPolicy(base_delay=1.0, backoff_factor=2.0, max_delay=10.0)
        assert p.delay_for_attempt(0) == 1.0  # 1 * 2^0
        assert p.delay_for_attempt(1) == 2.0  # 1 * 2^1
        assert p.delay_for_attempt(2) == 4.0  # 1 * 2^2
        assert p.delay_for_attempt(3) == 8.0  # 1 * 2^3

    def test_delay_capped_at_max(self):
        p = P2PRetryPolicy(base_delay=1.0, backoff_factor=10.0, max_delay=5.0)
        assert p.delay_for_attempt(0) == 1.0
        assert p.delay_for_attempt(1) == 5.0  # 10.0 capped to 5.0
        assert p.delay_for_attempt(5) == 5.0

    def test_no_retries(self):
        p = P2PRetryPolicy(max_retries=0)
        assert p.total_attempts == 1

    def test_immutable(self):
        p = P2PRetryPolicy()
        with pytest.raises(AttributeError):
            p.max_retries = 99


# ---------------------------------------------------------------------------
# P2PRetry — happy path
# ---------------------------------------------------------------------------


class TestP2PRetryExecute:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        policy = P2PRetryPolicy(max_retries=2)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        call_count = 0

        async def success():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry.execute(success)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        policy = P2PRetryPolicy(max_retries=2, base_delay=0.01)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "recovered"

        result = await retry.execute(fail_then_succeed)
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("persistent")

        with pytest.raises(RuntimeError, match="persistent"):
            await retry.execute(always_fail)
        assert call_count == 2  # 1 + 1 retry


# ---------------------------------------------------------------------------
# P2PRetry — timeout behaviour
# ---------------------------------------------------------------------------


class TestP2PRetryTimeout:
    @pytest.mark.asyncio
    async def test_timeout_retried_with_invalidation(self):
        """TimeoutError should be retried, invalidating the connection each time."""
        policy = P2PRetryPolicy(max_retries=2, base_delay=0.01)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        call_count = 0
        invalidate_count = 0

        async def on_inv():
            nonlocal invalidate_count
            invalidate_count += 1

        async def slow():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(10)

        with pytest.raises(asyncio.TimeoutError):
            await retry.execute(slow, timeout=0.01, on_invalidate=on_inv)
        assert call_count == 3  # 1 initial + 2 retries
        assert invalidate_count == 3  # invalidated on every timeout

    @pytest.mark.asyncio
    async def test_timeout_recovers_on_retry(self):
        """If the connection recovers after invalidation, retry should succeed."""
        policy = P2PRetryPolicy(max_retries=2, base_delay=0.01)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        call_count = 0

        async def on_inv():
            pass

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError()
            return "recovered"

        result = await retry.execute(fail_then_succeed, on_invalidate=on_inv)
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_calls_invalidate(self):
        """on_invalidate should be called when a timeout occurs."""
        policy = P2PRetryPolicy(max_retries=0, invalidate_on_timeout=True)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        invalidated = False

        async def on_inv():
            nonlocal invalidated
            invalidated = True

        async def slow():
            await asyncio.sleep(10)

        with pytest.raises(asyncio.TimeoutError):
            await retry.execute(slow, timeout=0.01, on_invalidate=on_inv)
        assert invalidated

    @pytest.mark.asyncio
    async def test_timeout_no_invalidate_when_disabled(self):
        """on_invalidate should NOT be called when invalidate_on_timeout is False."""
        policy = P2PRetryPolicy(max_retries=0, invalidate_on_timeout=False)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        invalidated = False

        async def on_inv():
            nonlocal invalidated
            invalidated = True

        async def slow():
            await asyncio.sleep(10)

        with pytest.raises(asyncio.TimeoutError):
            await retry.execute(slow, timeout=0.01, on_invalidate=on_inv)
        assert not invalidated


# ---------------------------------------------------------------------------
# P2PRetry — invalidation on error
# ---------------------------------------------------------------------------


class TestP2PRetryInvalidation:
    @pytest.mark.asyncio
    async def test_invalidate_called_on_retryable_error(self):
        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01, invalidate_on_error=True)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        invalidate_count = 0

        async def on_inv():
            nonlocal invalidate_count
            invalidate_count += 1

        async def always_fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await retry.execute(always_fail, on_invalidate=on_inv)
        # Called once on the retry backoff and once on final failure
        assert invalidate_count == 2

    @pytest.mark.asyncio
    async def test_no_invalidate_when_disabled(self):
        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01, invalidate_on_error=False)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        invalidate_count = 0

        async def on_inv():
            nonlocal invalidate_count
            invalidate_count += 1

        async def always_fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await retry.execute(always_fail, on_invalidate=on_inv)
        assert invalidate_count == 0


# ---------------------------------------------------------------------------
# P2PRetry — retryable exception filtering
# ---------------------------------------------------------------------------


class TestP2PRetryExceptionFilter:
    @pytest.mark.asyncio
    async def test_non_retryable_exception_not_retried(self):
        """Exceptions not in retryable_exceptions should propagate immediately."""
        policy = P2PRetryPolicy(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        call_count = 0

        async def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await retry.execute(raise_value_error)
        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_retryable_exception_is_retried(self):
        """Exceptions in retryable_exceptions should be retried."""
        policy = P2PRetryPolicy(
            max_retries=2,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        result = await retry.execute(fail_then_succeed)
        assert result == "ok"
        assert call_count == 3


# ---------------------------------------------------------------------------
# P2PRetry — cancellation
# ---------------------------------------------------------------------------


class TestP2PRetryCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        """CancelledError should never be swallowed or retried."""
        policy = P2PRetryPolicy(max_retries=3, base_delay=0.01)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)

        call_count = 0

        async def raise_cancel():
            nonlocal call_count
            call_count += 1
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await retry.execute(raise_cancel)
        assert call_count == 1


# ---------------------------------------------------------------------------
# P2POperationTimings
# ---------------------------------------------------------------------------


class TestP2POperationTimings:
    def test_defaults(self):
        t = P2POperationTimings()
        assert t.connection_duration is None
        assert t.stream_open_duration is None
        assert t.send_duration is None
        assert t.receive_duration is None
        assert t.total_start is None
        assert t.total_end is None
        assert t.total_duration is None
        assert t.attempt_count == 0
        assert t.retry_count == 0
        assert t.total_backoff_time == 0.0
        assert t.errors == []
        assert t.bytes_sent is None
        assert t.bytes_received is None

    def test_mutable(self):
        t = P2POperationTimings()
        t.connection_duration = 1.5
        t.bytes_sent = 1024
        assert t.connection_duration == 1.5
        assert t.bytes_sent == 1024

    def test_model_dump(self):
        t = P2POperationTimings(
            connection_duration=0.1,
            send_duration=0.2,
            receive_duration=0.3,
            total_start=100.0,
            total_end=100.6,
            total_duration=0.6,
            attempt_count=2,
            retry_count=1,
            total_backoff_time=0.25,
            errors=["RuntimeError: boom (attempt 1)"],
            bytes_sent=64,
            bytes_received=4096,
        )
        d = t.model_dump()
        assert d["connection_duration"] == 0.1
        assert d["send_duration"] == 0.2
        assert d["receive_duration"] == 0.3
        assert d["total_duration"] == 0.6
        assert d["attempt_count"] == 2
        assert d["retry_count"] == 1
        assert d["total_backoff_time"] == 0.25
        assert d["errors"] == ["RuntimeError: boom (attempt 1)"]
        assert d["bytes_sent"] == 64
        assert d["bytes_received"] == 4096
        # Ensure stream_open_duration (None) is present
        assert "stream_open_duration" in d
        assert d["stream_open_duration"] is None

    def test_errors_list_independence(self):
        """Ensure model_dump returns a copy of errors, not a reference."""
        t = P2POperationTimings(errors=["e1"])
        d = t.model_dump()
        d["errors"].append("e2")
        assert len(t.errors) == 1


# ---------------------------------------------------------------------------
# P2PRetry — timings integration
# ---------------------------------------------------------------------------


class TestP2PRetryTimings:
    @pytest.mark.asyncio
    async def test_timings_populated_on_success(self):
        policy = P2PRetryPolicy(max_retries=0)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)
        timings = P2POperationTimings()

        async def success():
            return "ok"

        result = await retry.execute(success, timings=timings)
        assert result == "ok"
        assert timings.attempt_count == 1
        assert timings.retry_count == 0
        assert timings.total_backoff_time == 0.0
        assert timings.errors == []

    @pytest.mark.asyncio
    async def test_timings_populated_after_retries(self):
        policy = P2PRetryPolicy(max_retries=2, base_delay=0.01)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)
        timings = P2POperationTimings()

        call_count = 0

        async def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "ok"

        await retry.execute(fail_twice, timings=timings)
        assert timings.attempt_count == 3
        assert timings.retry_count == 2
        assert timings.total_backoff_time > 0
        assert len(timings.errors) == 2
        assert "RuntimeError" in timings.errors[0]

    @pytest.mark.asyncio
    async def test_timings_populated_on_exhausted_retries(self):
        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)
        timings = P2POperationTimings()

        async def always_fail():
            raise RuntimeError("persistent")

        with pytest.raises(RuntimeError):
            await retry.execute(always_fail, timings=timings)
        assert timings.attempt_count == 2
        assert timings.retry_count == 1
        assert len(timings.errors) == 2

    @pytest.mark.asyncio
    async def test_timings_populated_on_timeout(self):
        policy = P2PRetryPolicy(max_retries=2, base_delay=0.01)
        timeouts = P2PTimeouts()
        retry = P2PRetry(policy, timeouts)
        timings = P2POperationTimings()

        async def slow():
            await asyncio.sleep(10)

        with pytest.raises(asyncio.TimeoutError):
            await retry.execute(slow, timeout=0.01, timings=timings)
        assert timings.attempt_count == 3  # retried on timeout
        assert len(timings.errors) == 3
        assert all("TimeoutError" in e for e in timings.errors)


# ---------------------------------------------------------------------------
# P2PRetry — IrohError node reset
# ---------------------------------------------------------------------------


class _FakeIrohError(Exception):
    """Stand-in for iroh.IrohError so tests don't need the real iroh package."""


class TestP2PRetryNodeReset:
    @pytest.mark.asyncio
    async def test_node_scoped_iroh_error_triggers_node_reset(self, monkeypatch):
        """When a node-scoped IrohError occurs, on_node_reset should be called."""
        import common.iroh.retry as retry_mod

        monkeypatch.setattr(retry_mod, "IrohError", _FakeIrohError)

        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01)
        retry = P2PRetry(policy, P2PTimeouts())

        reset_count = 0
        invalidate_count = 0

        async def on_reset():
            nonlocal reset_count
            reset_count += 1

        async def on_inv():
            nonlocal invalidate_count
            invalidate_count += 1

        async def raise_iroh():
            raise _FakeIrohError("endpoint bind failed")

        with pytest.raises(_FakeIrohError):
            await retry.execute(raise_iroh, on_invalidate=on_inv, on_node_reset=on_reset)
        # on_node_reset called on every attempt (retry + final)
        assert reset_count == 2
        # on_invalidate should NOT be called for node-scoped IrohError
        assert invalidate_count == 0

    @pytest.mark.asyncio
    async def test_iroh_error_recovery_after_reset(self, monkeypatch):
        """Node reset allows recovery: first attempt raises IrohError, second succeeds."""
        import common.iroh.retry as retry_mod

        monkeypatch.setattr(retry_mod, "IrohError", _FakeIrohError)

        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01)
        retry = P2PRetry(policy, P2PTimeouts())

        call_count = 0
        reset_count = 0

        async def on_reset():
            nonlocal reset_count
            reset_count += 1

        async def fail_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeIrohError("transient")
            return "recovered"

        result = await retry.execute(fail_then_ok, on_node_reset=on_reset)
        assert result == "recovered"
        assert call_count == 2
        assert reset_count == 1

    @pytest.mark.asyncio
    async def test_non_iroh_error_skips_node_reset(self, monkeypatch):
        """Regular exceptions should NOT trigger on_node_reset."""
        import common.iroh.retry as retry_mod

        monkeypatch.setattr(retry_mod, "IrohError", _FakeIrohError)

        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01)
        retry = P2PRetry(policy, P2PTimeouts())

        reset_count = 0
        invalidate_count = 0

        async def on_reset():
            nonlocal reset_count
            reset_count += 1

        async def on_inv():
            nonlocal invalidate_count
            invalidate_count += 1

        async def raise_runtime():
            raise RuntimeError("not an iroh error")

        with pytest.raises(RuntimeError):
            await retry.execute(raise_runtime, on_invalidate=on_inv, on_node_reset=on_reset)
        assert reset_count == 0
        # on_invalidate should still be called for non-IrohError exceptions
        assert invalidate_count == 2

    @pytest.mark.asyncio
    async def test_no_node_reset_callback_is_fine(self, monkeypatch):
        """IrohError without on_node_reset should still retry normally."""
        import common.iroh.retry as retry_mod

        monkeypatch.setattr(retry_mod, "IrohError", _FakeIrohError)

        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01)
        retry = P2PRetry(policy, P2PTimeouts())

        call_count = 0

        async def raise_iroh():
            nonlocal call_count
            call_count += 1
            raise _FakeIrohError("boom")

        with pytest.raises(_FakeIrohError):
            await retry.execute(raise_iroh)
        assert call_count == 2  # still retried

    @pytest.mark.asyncio
    async def test_iroh_error_none_fallback(self):
        """When IrohError is None (import failed), node reset is never triggered."""
        import common.iroh.retry as retry_mod

        original = retry_mod.IrohError
        retry_mod.IrohError = None  # type: ignore
        try:
            policy = P2PRetryPolicy(max_retries=1, base_delay=0.01)
            retry = P2PRetry(policy, P2PTimeouts())

            reset_count = 0
            invalidate_count = 0

            async def on_reset():
                nonlocal reset_count
                reset_count += 1

            async def on_inv():
                nonlocal invalidate_count
                invalidate_count += 1

            async def raise_runtime():
                raise RuntimeError("some error")

            with pytest.raises(RuntimeError):
                await retry.execute(raise_runtime, on_invalidate=on_inv, on_node_reset=on_reset)
            # Node reset should not be triggered when IrohError is None
            assert reset_count == 0
            # Regular invalidation should still happen
            assert invalidate_count == 2
        finally:
            retry_mod.IrohError = original


# ---------------------------------------------------------------------------
# _is_connection_scoped_iroh_error classification
# ---------------------------------------------------------------------------


class TestIsConnectionScopedIrohError:
    def test_connection_lost(self):
        assert _is_connection_scoped_iroh_error(Exception("connection lost")) is True

    def test_closed_by_peer(self):
        assert _is_connection_scoped_iroh_error(Exception("closed by peer: 0")) is True

    def test_timed_out(self):
        assert _is_connection_scoped_iroh_error(Exception("timed out")) is True

    def test_stream_closed(self):
        assert _is_connection_scoped_iroh_error(Exception("stream closed")) is True

    def test_reset_by_peer(self):
        assert _is_connection_scoped_iroh_error(Exception("reset by peer")) is True

    def test_case_insensitive(self):
        assert _is_connection_scoped_iroh_error(Exception("Connection Lost")) is True

    def test_unknown_error_is_not_connection_scoped(self):
        assert _is_connection_scoped_iroh_error(Exception("endpoint bind failed")) is False

    def test_empty_message_is_not_connection_scoped(self):
        assert _is_connection_scoped_iroh_error(Exception("")) is False


# ---------------------------------------------------------------------------
# P2PRetry — connection-scoped IrohError handling
# ---------------------------------------------------------------------------


class TestP2PRetryConnectionScopedIrohError:
    @pytest.mark.asyncio
    async def test_connection_lost_calls_invalidate_not_reset(self, monkeypatch):
        """'connection lost' is connection-scoped: invalidate, don't reset node."""
        import common.iroh.retry as retry_mod

        monkeypatch.setattr(retry_mod, "IrohError", _FakeIrohError)

        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01)
        retry = P2PRetry(policy, P2PTimeouts())

        reset_count = 0
        invalidate_count = 0

        async def on_reset():
            nonlocal reset_count
            reset_count += 1

        async def on_inv():
            nonlocal invalidate_count
            invalidate_count += 1

        async def raise_conn_lost():
            raise _FakeIrohError("connection lost")

        with pytest.raises(_FakeIrohError):
            await retry.execute(raise_conn_lost, on_invalidate=on_inv, on_node_reset=on_reset)
        assert reset_count == 0
        # on_invalidate called once in the IrohError branch per attempt
        assert invalidate_count == 2

    @pytest.mark.asyncio
    async def test_closed_by_peer_calls_invalidate_not_reset(self, monkeypatch):
        """'closed by peer' is connection-scoped: invalidate, don't reset node."""
        import common.iroh.retry as retry_mod

        monkeypatch.setattr(retry_mod, "IrohError", _FakeIrohError)

        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01)
        retry = P2PRetry(policy, P2PTimeouts())

        reset_count = 0
        invalidate_count = 0

        async def on_reset():
            nonlocal reset_count
            reset_count += 1

        async def on_inv():
            nonlocal invalidate_count
            invalidate_count += 1

        async def raise_closed():
            raise _FakeIrohError("closed by peer: 0")

        with pytest.raises(_FakeIrohError):
            await retry.execute(raise_closed, on_invalidate=on_inv, on_node_reset=on_reset)
        assert reset_count == 0
        assert invalidate_count == 2

    @pytest.mark.asyncio
    async def test_unknown_iroh_error_still_resets_node(self, monkeypatch):
        """Unknown IrohError messages should still trigger full node reset."""
        import common.iroh.retry as retry_mod

        monkeypatch.setattr(retry_mod, "IrohError", _FakeIrohError)

        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01)
        retry = P2PRetry(policy, P2PTimeouts())

        reset_count = 0
        invalidate_count = 0

        async def on_reset():
            nonlocal reset_count
            reset_count += 1

        async def on_inv():
            nonlocal invalidate_count
            invalidate_count += 1

        async def raise_unknown():
            raise _FakeIrohError("endpoint bind failed")

        with pytest.raises(_FakeIrohError):
            await retry.execute(raise_unknown, on_invalidate=on_inv, on_node_reset=on_reset)
        assert reset_count == 2
        assert invalidate_count == 0

    @pytest.mark.asyncio
    async def test_connection_scoped_recovery(self, monkeypatch):
        """Connection-scoped IrohError on first attempt, recovery on second."""
        import common.iroh.retry as retry_mod

        monkeypatch.setattr(retry_mod, "IrohError", _FakeIrohError)

        policy = P2PRetryPolicy(max_retries=1, base_delay=0.01)
        retry = P2PRetry(policy, P2PTimeouts())

        call_count = 0
        invalidate_count = 0
        reset_count = 0

        async def on_inv():
            nonlocal invalidate_count
            invalidate_count += 1

        async def on_reset():
            nonlocal reset_count
            reset_count += 1

        async def fail_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeIrohError("connection lost")
            return "recovered"

        result = await retry.execute(fail_then_ok, on_invalidate=on_inv, on_node_reset=on_reset)
        assert result == "recovered"
        assert call_count == 2
        assert invalidate_count == 1  # invalidated the dead connection
        assert reset_count == 0  # node was NOT reset
