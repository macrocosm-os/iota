"""Tests for SyncedVariableV2 and Lock."""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock

from miner.sync_v2.synced_variable import Lock, SyncedVariableV2


# ── Helpers ────────────────────────────────────────────────────────────────────


def _resp(data: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data, request=httpx.Request("POST", "http://test/"))


def _make_var(**kwargs) -> SyncedVariableV2:
    defaults = dict(run_id="run-abc", name="my_var", var_type="dict", default={})
    bridge_registered = kwargs.pop("bridge_registered", True)
    defaults.update(kwargs)
    var = SyncedVariableV2(**defaults)
    var._client = AsyncMock()
    var._bridge_registered = bridge_registered
    return var


def _make_lock(var_ids=None, tokens: dict | None = None) -> Lock:
    client = AsyncMock()
    lock = Lock(var_ids or "run-abc/my_var", client)
    if tokens:
        lock._tokens = tokens
    return lock


def _side_effect(*responses: httpx.Response):
    return iter(responses)


# ── var_id property ────────────────────────────────────────────────────────────


def test_var_id():
    var = _make_var(run_id="run-xyz", name="step")
    assert var.var_id == "run-xyz/step"


# ── Lock ──────────────────────────────────────────────────────────────────────


async def test_lock_aenter_success_single():
    lock = _make_lock("run-abc/x")
    lock._client.post = AsyncMock(
        return_value=_resp({"locks": [{"var_id": "run-abc/x", "status": "locked", "lock_token": "tok1"}]})
    )
    result = await lock.__aenter__()
    assert result is lock
    assert lock.lock_token == "tok1"
    assert lock._tokens == {"run-abc/x": "tok1"}


async def test_lock_aenter_multi_var():
    ids = ["run/a", "run/b"]
    lock = _make_lock(ids)
    lock._client.post = AsyncMock(
        return_value=_resp(
            {
                "locks": [
                    {"var_id": "run/a", "status": "locked", "lock_token": "tok-a"},
                    {"var_id": "run/b", "status": "locked", "lock_token": "tok-b"},
                ]
            }
        )
    )
    await lock.__aenter__()
    assert lock.token_for("run/a") == "tok-a"
    assert lock.token_for("run/b") == "tok-b"
    # lock_token is None when multiple vars are held
    assert lock.lock_token is None


async def test_lock_aenter_timeout_raises_and_releases_partial():
    ids = ["run/a", "run/b"]
    lock = _make_lock(ids)
    lock._client.post = AsyncMock(
        side_effect=_side_effect(
            _resp(
                {
                    "locks": [
                        {"var_id": "run/a", "status": "locked", "lock_token": "tok-a"},
                        {"var_id": "run/b", "status": "timeout", "error": "LockUnavailable"},
                    ]
                }
            ),
            _resp({}),  # unlock call
        )
    )
    with pytest.raises(RuntimeError, match="run/b"):
        await lock.__aenter__()
    # Ensure unlock was called for the already-acquired lock
    assert lock._client.post.call_count == 2


async def test_lock_aexit_releases_all():
    lock = _make_lock(["run/a", "run/b"], tokens={"run/a": "tok-a", "run/b": "tok-b"})
    lock._client.post = AsyncMock(return_value=_resp({}))
    await lock.__aexit__(None, None, None)
    lock._client.post.assert_awaited_once()
    payload = lock._client.post.call_args.kwargs["json"]["locks"]
    released = {item["var_id"]: item["lock_token"] for item in payload}
    assert released == {"run/a": "tok-a", "run/b": "tok-b"}
    assert lock._tokens == {}


async def test_lock_aexit_swallows_http_errors(caplog):
    lock = _make_lock("run/x", tokens={"run/x": "tok"})
    lock._client.post = AsyncMock(side_effect=httpx.ConnectError("down"))
    await lock.__aexit__(None, None, None)  # must not raise
    assert lock._tokens == {}


async def test_lock_context_manager_happy_path():
    lock = _make_lock("run-abc/node")
    lock._client.post = AsyncMock(
        side_effect=_side_effect(
            _resp({"locks": [{"var_id": "run-abc/node", "status": "locked", "lock_token": "secret"}]}),
            _resp({}),
        )
    )
    async with lock as acquired:
        assert acquired.lock_token == "secret"
    assert lock._tokens == {}


# ── register ──────────────────────────────────────────────────────────────────


async def test_register_success():
    var = _make_var()
    var._client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "created", "version": 1}]})
    )
    await var.register(default_value={})
    assert var.version == 1
    assert var._cached_value == {}
    body = var._client.post.call_args.kwargs["json"]
    assert body["variables"][0]["var_id"] == "run-abc/my_var"
    assert body["variables"][0]["var_type"] == "dict"


async def test_register_already_registered_updates_version():
    var = _make_var()
    var._client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "already_registered", "version": 5}]})
    )
    await var.register(default_value={})
    assert var.version == 5


async def test_register_error_raises():
    var = _make_var()
    var._client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "error", "error": "InvalidType: ..."}]})
    )
    with pytest.raises(RuntimeError, match="InvalidType"):
        await var.register(default_value="wrong")


async def test_register_http_error_propagates():
    var = _make_var()
    var._client.post = AsyncMock(return_value=httpx.Response(500, request=httpx.Request("POST", "http://test/")))
    with pytest.raises(httpx.HTTPStatusError):
        await var.register(default_value={})


async def test_create_success():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "created", "version": 1}]})
    )
    var = await SyncedVariableV2.create(
        client=mock_client, run_id="run-abc", name="my_var", var_type="dict", default={}
    )
    assert var._bridge_registered is True
    assert var.version == 1
    assert var._cached_value == {}


async def test_create_registration_error_raises():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "error", "error": "InvalidType"}]})
    )
    with pytest.raises(RuntimeError, match="InvalidType"):
        await SyncedVariableV2.create(client=mock_client, run_id="run-abc", name="my_var", var_type="dict", default={})


# ── fetch_value ────────────────────────────────────────────────────────────────


async def test_fetch_value_success():
    var = _make_var()
    var._client.post = AsyncMock(
        return_value=_resp(
            {
                "variables": [
                    {
                        "var_id": "run-abc/my_var",
                        "value": {"k": 1},
                        "version": 7,
                        "metadata": {"var_type": "dict", "write_rule": "LWW", "created_at": "", "updated_at": "t1"},
                        "lock": {"locked": False},
                    }
                ],
                "errors": [],
            }
        )
    )
    result = await var.fetch_value()
    assert result == {"k": 1}
    assert var._cached_value == {"k": 1}
    assert var.version == 7
    assert var.updated_at == "t1"
    assert var._has_fetched is True


async def test_fetch_value_not_found_raises():
    var = _make_var()
    var._client.post = AsyncMock(
        return_value=_resp({"variables": [], "errors": [{"var_id": "run-abc/my_var", "error": "VariableNotFound"}]})
    )
    with pytest.raises(RuntimeError, match="VariableNotFound"):
        await var.fetch_value()


async def test_fetch_value_empty_response_returns_cached():
    var = _make_var()
    var._cached_value = {"old": True}
    var._client.post = AsyncMock(return_value=_resp({"variables": [], "errors": []}))
    result = await var.fetch_value()
    assert result == {"old": True}


# ── set_value ─────────────────────────────────────────────────────────────────


async def test_set_value_lww():
    var = _make_var(write_rule="LWW")
    var.version = 3
    var._client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "updated", "version": 4}]})
    )
    await var.set_value({"new": "val"})
    assert var.version == 4
    assert var._cached_value == {"new": "val"}
    assert var._needs_push is False


async def test_set_value_cas_sends_current_version():
    var = _make_var(write_rule="CAS")
    var.version = 2
    var._client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "updated", "version": 3}]})
    )
    await var.set_value({"x": 1}, current_version=2)
    body = var._client.post.call_args.kwargs["json"]
    assert body["updates"][0]["current_version"] == 2


async def test_set_value_cas_missing_version_raises():
    var = _make_var(write_rule="CAS")
    with pytest.raises(ValueError, match="current_version"):
        await var.set_value({"x": 1})


async def test_set_value_lock_sends_token():
    var = _make_var(write_rule="LOCK")
    var.version = 1
    lock = _make_lock("run-abc/my_var", tokens={"run-abc/my_var": "the-token"})
    var._client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "updated", "version": 2}]})
    )
    await var.set_value({"x": 1}, lock=lock)
    body = var._client.post.call_args.kwargs["json"]
    assert body["updates"][0]["lock_token"] == "the-token"


async def test_set_value_lock_without_lock_raises():
    var = _make_var(write_rule="LOCK")
    with pytest.raises(ValueError, match="Lock"):
        await var.set_value({"x": 1})


async def test_set_value_error_response_raises():
    var = _make_var(write_rule="LWW")
    var._client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "error", "error": "InvalidType"}]})
    )
    with pytest.raises(RuntimeError, match="InvalidType"):
        await var.set_value("wrong")


# ── get_cached_value ──────────────────────────────────────────────────────────


def test_get_cached_value_require_fetch_false():
    var = _make_var(require_fetch=False)
    var._cached_value = 42
    assert var.get_cached_value() == 42


def test_get_cached_value_require_fetch_true_before_fetch_raises():
    var = _make_var(require_fetch=True)
    with pytest.raises(RuntimeError, match="fetch_value"):
        var.get_cached_value()


def test_get_cached_value_require_fetch_true_after_fetch_ok():
    var = _make_var(require_fetch=True)
    var._has_fetched = True
    var._cached_value = {"ready": True}
    assert var.get_cached_value() == {"ready": True}


# ── set / set_and_push ────────────────────────────────────────────────────────


def test_set_sets_value_and_flag():
    var = _make_var(write_rule="LWW")
    var.set({"updated": True})
    assert var._cached_value == {"updated": True}
    assert var._needs_push is True


def test_set_lock_raises():
    var = _make_var(write_rule="LOCK")
    with pytest.raises(ValueError, match="set"):
        var.set({"x": 1})


async def test_set_and_push_lww_sets_and_pushes():
    var = _make_var(write_rule="LWW")
    var.version = 3
    var._client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "updated", "version": 4}]})
    )
    await var.set_and_push({"new": "val"})
    assert var.version == 4
    assert var._cached_value == {"new": "val"}
    assert var._needs_push is False


async def test_set_and_push_cas_uses_cached_version():
    var = _make_var(write_rule="CAS")
    var.version = 2
    var._client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "updated", "version": 3}]})
    )
    await var.set_and_push({"x": 1})
    body = var._client.post.call_args.kwargs["json"]
    assert body["updates"][0]["current_version"] == 2


async def test_set_and_push_lock_sends_token():
    var = _make_var(write_rule="LOCK")
    lock = _make_lock("run-abc/my_var", tokens={"run-abc/my_var": "the-token"})
    var._client.post = AsyncMock(
        return_value=_resp({"results": [{"var_id": "run-abc/my_var", "status": "updated", "version": 2}]})
    )
    await var.set_and_push({"x": 1}, lock=lock)
    body = var._client.post.call_args.kwargs["json"]
    assert body["updates"][0]["lock_token"] == "the-token"


# ── Lock with SyncedVariableV2 targets ────────────────────────────────────────


async def test_lock_accepts_synced_variable():
    var = _make_var(run_id="r", name="n")
    lock = Lock(var)
    lock._client.post = AsyncMock(
        return_value=_resp({"locks": [{"var_id": "r/n", "status": "locked", "lock_token": "tok"}]})
    )
    async with lock as acquired:
        assert acquired.lock_token == "tok"
        assert acquired.var_ids == ["r/n"]


async def test_lock_accepts_multiple_synced_variables():
    a = _make_var(run_id="r", name="a")
    b = _make_var(run_id="r", name="b")
    client = AsyncMock()
    a._client = client
    b._client = client
    lock = Lock([a, b])
    lock._client.post = AsyncMock(
        return_value=_resp(
            {
                "locks": [
                    {"var_id": "r/a", "status": "locked", "lock_token": "tok-a"},
                    {"var_id": "r/b", "status": "locked", "lock_token": "tok-b"},
                ]
            }
        )
    )
    await lock.__aenter__()
    assert lock.token_for("r/a") == "tok-a"
    assert lock.token_for("r/b") == "tok-b"


def test_lock_string_target_without_client_raises():
    with pytest.raises(ValueError, match="client is required"):
        Lock("run/a")
