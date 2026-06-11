"""Tests for VariableManager."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx

from miner.sync_v2.synced_variable import SyncedVariableV2
from miner.sync_v2.variable_manager import VariableManager, _CollectionPollEntry


# ── Helpers ────────────────────────────────────────────────────────────────────


def _resp(data: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data, request=httpx.Request("POST", "http://test/"))


def _make_var(name: str = "v", run_id: str = "run", write_rule: str = "LWW", **kwargs) -> SyncedVariableV2:
    bridge_registered = kwargs.pop("bridge_registered", True)
    var = SyncedVariableV2(run_id=run_id, name=name, var_type="dict", write_rule=write_rule, default={}, **kwargs)
    var._client = AsyncMock()
    var._bridge_registered = bridge_registered
    return var


def _make_manager() -> VariableManager:
    mgr = VariableManager(url="http://test")
    mgr._http = AsyncMock()
    return mgr


# ── get_min_interval ──────────────────────────────────────────────────────────


def test_get_min_interval_no_vars_returns_default():
    mgr = _make_manager()
    assert mgr.get_min_interval() == 1.0


def test_get_min_interval_returns_smallest():
    mgr = _make_manager()
    a, b = _make_var("a"), _make_var("b")
    a._pull_frequency = 10.0
    a._push_frequency = 8.0
    b._pull_frequency = 3.0
    b._push_frequency = 6.0
    mgr.registered_vars = {"a": a, "b": b}
    assert mgr.get_min_interval() == 3.0


# ── get_vars_to_sync ──────────────────────────────────────────────────────────


async def test_get_vars_to_sync_pull_overdue():
    mgr = _make_manager()
    var = _make_var()
    var._pull_frequency = 1.0
    var._last_pulled = time.time() - 10  # 10s ago → overdue
    mgr.registered_vars = {"v": var}
    push, pull = await mgr.get_vars_to_sync()
    assert var in pull
    assert var not in push


async def test_get_vars_to_sync_push_dirty_and_overdue():
    mgr = _make_manager()
    var = _make_var()
    var._push_frequency = 1.0
    var._needs_push = True
    var._last_pushed = time.time() - 10
    mgr.registered_vars = {"v": var}
    push, pull = await mgr.get_vars_to_sync()
    assert var in push


async def test_get_vars_to_sync_push_not_dirty():
    mgr = _make_manager()
    var = _make_var()
    var._push_frequency = 1.0
    var._needs_push = False
    var._last_pushed = time.time() - 10
    mgr.registered_vars = {"v": var}
    push, _ = await mgr.get_vars_to_sync()
    assert var not in push


async def test_get_vars_to_sync_within_frequency_not_returned():
    mgr = _make_manager()
    var = _make_var()
    var._pull_frequency = 60.0
    var._last_pulled = time.time() - 1  # pulled 1s ago, frequency=60s → not overdue
    mgr.registered_vars = {"v": var}
    _, pull = await mgr.get_vars_to_sync()
    assert var not in pull


# ── _batch_pull ────────────────────────────────────────────────────────────────


async def test_batch_pull_updates_vars():
    mgr = _make_manager()
    var = _make_var(name="x", run_id="r")
    mgr._http.post = AsyncMock(
        return_value=_resp(
            {
                "variables": [
                    {
                        "var_id": "r/x",
                        "value": {"a": 1},
                        "version": 5,
                        "metadata": {"var_type": "dict", "write_rule": "LWW", "created_at": "", "updated_at": "now"},
                        "lock": {"locked": False},
                    }
                ],
                "errors": [],
            }
        )
    )
    await mgr._batch_pull([var])
    assert var._cached_value == {"a": 1}
    assert var.version == 5
    assert var.updated_at == "now"
    assert var._has_fetched is True


async def test_batch_pull_logs_errors(caplog):
    mgr = _make_manager()
    var = _make_var(name="x", run_id="r")
    mgr._http.post = AsyncMock(
        return_value=_resp({"variables": [], "errors": [{"var_id": "r/x", "error": "VariableNotFound"}]})
    )
    await mgr._batch_pull([var])  # should not raise
    assert var._cached_value is None  # unchanged


async def test_batch_pull_http_error_is_swallowed():
    mgr = _make_manager()
    var = _make_var()
    mgr._http.post = AsyncMock(return_value=httpx.Response(503, request=httpx.Request("POST", "http://test/")))
    await mgr._batch_pull([var])  # should not raise


# ── _batch_push ────────────────────────────────────────────────────────────────


async def test_batch_push_lww():
    mgr = _make_manager()
    var = _make_var(write_rule="LWW")
    var._cached_value = {"score": 9}
    var._needs_push = True
    var.version = 2
    mgr._http.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run/v", "status": "updated", "version": 3}]})
    )
    await mgr._batch_push([var])
    assert var.version == 3
    assert var._needs_push is False
    body = mgr._http.post.call_args.kwargs["json"]
    update = body["updates"][0]
    assert update["value"] == {"score": 9}
    assert "current_version" not in update


async def test_batch_push_cas_includes_version():
    mgr = _make_manager()
    var = _make_var(write_rule="CAS")
    var._cached_value = 42
    var._needs_push = True
    var.version = 7
    mgr._http.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run/v", "status": "updated", "version": 8}]})
    )
    await mgr._batch_push([var])
    body = mgr._http.post.call_args.kwargs["json"]
    assert body["updates"][0]["current_version"] == 7


async def test_batch_push_lock_vars_skipped():
    mgr = _make_manager()
    var = _make_var(write_rule="LOCK")
    var._cached_value = "x"
    var._needs_push = True
    await mgr._batch_push([var])
    mgr._http.post.assert_not_awaited()


async def test_batch_push_server_error_logged():
    mgr = _make_manager()
    var = _make_var(write_rule="LWW")
    var._cached_value = 1
    var._needs_push = True
    var.version = 1
    mgr._http.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run/v", "status": "error", "error": "InvalidType"}]})
    )
    await mgr._batch_push([var])
    assert var._needs_push is True  # flag not cleared on error


async def test_batch_push_http_error_is_swallowed():
    mgr = _make_manager()
    var = _make_var(write_rule="LWW")
    var._cached_value = 1
    var._needs_push = True
    mgr._http.post = AsyncMock(return_value=httpx.Response(500, request=httpx.Request("POST", "http://test/")))
    await mgr._batch_push([var])  # should not raise


def test_get_min_interval_includes_collection():
    mgr = _make_manager()
    mgr._collection_entries["c"] = _CollectionPollEntry(
        key="c",
        collection=object(),
        wildcard_path="x/*",
        after_pull=None,
        pull_frequency=5.0,
        push_frequency=7.0,
    )
    assert mgr.get_min_interval() == 5.0


async def test_poll_collection_wildcard_and_sync_callback():
    mgr = _make_manager()
    col = AsyncMock()
    col.wildcard_fetch = AsyncMock(return_value={"k": "v"})
    col.push_dirty = AsyncMock()
    seen: list[Any] = []

    def cb(raw):
        seen.append(raw)

    mgr._collection_entries["reg"] = _CollectionPollEntry(
        key="reg",
        collection=col,
        wildcard_path="p/*",
        after_pull=cb,
        pull_frequency=0.0,
        push_frequency=None,
    )
    await mgr._poll_registered_collections()
    col.wildcard_fetch.assert_awaited_once_with("p/*")
    assert seen == [{"k": "v"}]
    col.push_dirty.assert_not_awaited()


async def test_unwatch_collection_stops_polling_entry():
    mgr = _make_manager()
    col = AsyncMock()
    col.wildcard_fetch = AsyncMock(return_value={})
    mgr._collection_entries["k"] = _CollectionPollEntry(
        key="k",
        collection=col,
        wildcard_path="a/*",
        after_pull=None,
        pull_frequency=0.0,
        push_frequency=None,
    )
    mgr.unwatch_collection("k")
    await mgr._poll_registered_collections()
    col.wildcard_fetch.assert_not_awaited()


# ── _ensure_started / stop ────────────────────────────────────────────────────


async def test_ensure_started_creates_task():
    mgr = _make_manager()

    async def fake_poll():
        await asyncio.sleep(9999)

    with patch.object(mgr, "_poll_loop", fake_poll):
        mgr._ensure_started()
        assert mgr._task is not None
        assert not mgr._task.done()
        await mgr.stop()


async def test_ensure_started_idempotent():
    mgr = _make_manager()

    async def fake_poll():
        await asyncio.sleep(9999)

    with patch.object(mgr, "_poll_loop", fake_poll):
        mgr._ensure_started()
        first_task = mgr._task
        mgr._ensure_started()
        assert mgr._task is first_task
        await mgr.stop()


async def test_stop_cancels_task():
    mgr = _make_manager()

    async def fake_poll():
        await asyncio.sleep(9999)

    with patch.object(mgr, "_poll_loop", fake_poll):
        mgr._ensure_started()
        await mgr.stop()
        assert mgr._task is None


async def test_stop_when_not_started_is_safe():
    mgr = _make_manager()
    await mgr.stop()  # must not raise
