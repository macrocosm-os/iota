"""Tests for SyncedVariable on_update callback."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from miner.sync.collections import SyncedDict, SyncedList
from miner.sync.variable import PollingLoop, SyncedVariable


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_sv(default=None, on_update=None, **kwargs) -> SyncedVariable:
    """Create a SyncedVariable with a stub PollingLoop (no real HTTP)."""
    loop = MagicMock(spec=PollingLoop)
    return SyncedVariable(
        variable_id="test_var",
        default=default,
        polling_loop=loop,
        on_update=on_update,
        **kwargs,
    )


# ── Basic callback invocation ──────────────────────────────────────────────────


def test_callback_called_on_scalar_update():
    called_with = []
    sv = _make_sv(default=0, on_update=lambda v: called_with.append(v))

    sv._apply(42)

    assert called_with == [42]
    assert sv.value == 42


def test_callback_called_with_updated_value_after_dict_apply():
    called_with = []
    sv = _make_sv(
        default=SyncedDict({"a": 1}),
        on_update=lambda v: called_with.append(dict(v)),
    )

    sv._apply({"a": 2, "b": 3})

    assert called_with == [{"a": 2, "b": 3}]


def test_callback_called_with_updated_value_after_list_apply():
    called_with = []
    sv = _make_sv(
        default=SyncedList([1, 2]),
        on_update=lambda v: called_with.append(list(v)),
    )

    sv._apply([10, 20, 30])

    assert called_with == [[10, 20, 30]]


def test_no_callback_when_on_update_is_none():
    """Ensure _apply works fine when no callback is set."""
    sv = _make_sv(default=0)
    sv._apply(99)
    assert sv.value == 99


def test_callback_called_multiple_times():
    calls = []
    sv = _make_sv(default="", on_update=calls.append)

    sv._apply("first")
    sv._apply("second")

    assert calls == ["first", "second"]


# ── Async callback ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_callback_is_scheduled_as_task():
    results = []

    async def async_cb(value):
        results.append(value)

    sv = _make_sv(default=0, on_update=async_cb)

    sv._apply(7)
    # Let the scheduled coroutine run
    await asyncio.sleep(0)

    assert results == [7]


@pytest.mark.asyncio
async def test_async_callback_receives_correct_value():
    received = []

    async def async_cb(value):
        received.append(value)

    sv = _make_sv(default=None, on_update=async_cb)

    sv._apply({"key": "value"})
    await asyncio.sleep(0)

    assert received == [{"key": "value"}]


# ── Error resilience ───────────────────────────────────────────────────────────


def test_callback_exception_does_not_propagate():
    def bad_cb(v):
        raise RuntimeError("boom")

    sv = _make_sv(default=0, on_update=bad_cb)
    # Must not raise
    sv._apply(1)
    assert sv.value == 1


def test_callback_exception_is_logged(caplog):
    def bad_cb(v):
        raise ValueError("oops")

    sv = _make_sv(default=0, on_update=bad_cb)

    with patch("miner.sync.variable.logger") as mock_logger:
        sv._apply(5)
        mock_logger.exception.assert_called_once()
        assert "on_update error" in mock_logger.exception.call_args[0][0]


# ── Callback not triggered by local .value setter ─────────────────────────────


def test_callback_not_triggered_by_value_setter():
    calls = []
    sv = _make_sv(default=0, on_update=calls.append, push_on_set=False)

    sv.value = 100  # local write — callback should NOT fire

    assert calls == []
    assert sv.value == 100


# ── Callback reference can be changed after construction ──────────────────────


def test_callback_can_be_replaced():
    first_calls = []
    second_calls = []

    sv = _make_sv(default=0, on_update=first_calls.append)
    sv._apply(1)

    sv._on_update = second_calls.append
    sv._apply(2)

    assert first_calls == [1]
    assert second_calls == [2]


def test_callback_can_be_removed():
    calls = []
    sv = _make_sv(default=0, on_update=calls.append)
    sv._apply(1)

    sv._on_update = None
    sv._apply(2)

    assert calls == [1]  # second apply produced no call
