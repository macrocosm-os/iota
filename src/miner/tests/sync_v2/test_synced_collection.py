"""Tests for synced_collection module functions."""

from __future__ import annotations

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock

from miner.sync_v2.synced_collection import (
    delete,
    fetch_all,
    get_cached,
    push_dirty,
    register_many,
    set_many,
    set_value,
    wildcard_delete,
    wildcard_fetch,
)
from miner.sync_v2.synced_variable import Lock, SyncedVariableV2


def _resp(data: dict, status: int = 200, method: str = "POST") -> httpx.Response:
    return httpx.Response(status, json=data, request=httpx.Request(method, "http://test/"))


def _make_client() -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock()
    client.get = AsyncMock()
    return client


def _side_effect(*responses: httpx.Response):
    return iter(responses)


# ── register_many ─────────────────────────────────────────────────────────────


async def test_register_many_batch():
    client = _make_client()
    client.post = AsyncMock(
        return_value=_resp(
            {
                "results": [
                    {"var_id": "run-abc/a", "status": "created", "version": 1},
                    {"var_id": "run-abc/b", "status": "created", "version": 1},
                ]
            }
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(
        client,
        "run-abc",
        [
            {"name": "a", "default_value": 0, "var_type": "int"},
            {"name": "b", "default_value": 0.0, "var_type": "float"},
        ],
        variables=variables,
    )
    assert set(variables) == {"a", "b"}
    assert get_cached("a", variables) == 0
    assert get_cached("b", variables) == 0.0
    body = client.post.call_args.kwargs["json"]
    assert len(body["variables"]) == 2


async def test_register_many_empty_noop():
    client = _make_client()
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(client, "run-abc", [], variables=variables)
    client.post.assert_not_awaited()


# ── set_many / set_value ──────────────────────────────────────────────────────


async def test_set_many_updates_tracked():
    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp({"results": [{"var_id": "run-abc/x", "status": "created", "version": 1}]}),
            _resp({"results": [{"var_id": "run-abc/x", "status": "updated", "version": 2}]}),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(client, "run-abc", [{"name": "x", "default_value": 0, "var_type": "int"}], variables=variables)
    await set_many(client, "run-abc", variables, {"x": 42})
    assert variables["x"].get_cached_value() == 42
    assert variables["x"].version == 2


async def test_set_many_cas_uses_explicit_current_versions():
    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp(
                {
                    "results": [
                        {"var_id": "run-abc/a", "status": "created", "version": 1},
                        {"var_id": "run-abc/b", "status": "created", "version": 1},
                    ]
                }
            ),
            _resp(
                {
                    "results": [
                        {"var_id": "run-abc/a", "status": "updated", "version": 11},
                        {"var_id": "run-abc/b", "status": "updated", "version": 21},
                    ]
                }
            ),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(
        client,
        "run-abc",
        [
            {"name": "a", "default_value": 0, "var_type": "int"},
            {"name": "b", "default_value": 0, "var_type": "int"},
        ],
        rule="CAS",
        variables=variables,
    )
    variables["a"].version = 10
    variables["b"].version = 20
    await set_many(client, "run-abc", variables, {"a": 1, "b": 2}, current_versions={"a": 9, "b": 19})
    body = client.post.call_args.kwargs["json"]
    by_id = {u["var_id"]: u for u in body["updates"]}
    assert by_id["run-abc/a"]["current_version"] == 9
    assert by_id["run-abc/b"]["current_version"] == 19


async def test_set_many_lock_token_per_var():
    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp({"results": [{"var_id": "run-abc/u", "status": "created", "version": 1}]}),
            _resp({"results": [{"var_id": "run-abc/u", "status": "updated", "version": 2}]}),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(
        client,
        "run-abc",
        [{"name": "u", "default_value": 0, "var_type": "int"}],
        rule="LOCK",
        variables=variables,
    )
    lock = Lock(["run-abc/u"], client)
    lock._tokens = {"run-abc/u": "tok-u"}
    await set_many(client, "run-abc", variables, {"u": 5}, lock=lock)
    body = client.post.call_args.kwargs["json"]
    assert body["updates"][0]["lock_token"] == "tok-u"


async def test_push_dirty():
    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp({"results": [{"var_id": "run-abc/x", "status": "created", "version": 1}]}),
            _resp({"results": [{"var_id": "run-abc/x", "status": "updated", "version": 2}]}),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(client, "run-abc", [{"name": "x", "default_value": 0, "var_type": "int"}], variables=variables)
    variables["x"].set(99)
    await push_dirty(client, "run-abc", variables)
    assert get_cached("x", variables) == 99
    assert variables["x"]._needs_push is False


async def test_push_dirty_cas_mismatch_refreshes_version_and_retries():
    # Local CAS version is stale (pulls missed this key). push_dirty must
    # refresh the version with a keyed /get and retry, keeping the local value.
    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp({"results": [{"var_id": "run-abc/x", "status": "created", "version": 1}]}),
            _resp({"results": [{"var_id": "run-abc/x", "status": "error", "error": "CASVersionMismatch"}]}),
            _resp({"variables": [{"var_id": "run-abc/x", "value": 7, "version": 41}]}),
            _resp({"results": [{"var_id": "run-abc/x", "status": "updated", "version": 42}]}),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(
        client,
        "run-abc",
        [{"name": "x", "default_value": 0, "var_type": "int"}],
        rule="CAS",
        variables=variables,
    )
    variables["x"].set(99)
    await push_dirty(client, "run-abc", variables)

    retry_body = client.post.call_args.kwargs["json"]
    assert retry_body["updates"][0]["current_version"] == 41
    assert get_cached("x", variables) == 99  # local dirty value pushed, not the bridge's 7
    assert variables["x"].version == 42
    assert variables["x"]._needs_push is False


async def test_push_dirty_cas_mismatch_key_gone_joins_reregister():
    # Mismatch, but the keyed /get says the variable vanished — must fall
    # through to the re-register path instead of retrying the CAS write.
    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp({"results": [{"var_id": "run-abc/x", "status": "created", "version": 1}]}),
            _resp({"results": [{"var_id": "run-abc/x", "status": "error", "error": "CASVersionMismatch"}]}),
            _resp({"variables": [], "errors": [{"var_id": "run-abc/x", "error": "VariableNotFound"}]}),
            _resp({"results": [{"var_id": "run-abc/x", "status": "created", "version": 1}]}),
            _resp({"results": [{"var_id": "run-abc/x", "status": "updated", "version": 2}]}),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(
        client,
        "run-abc",
        [{"name": "x", "default_value": 0, "var_type": "int"}],
        rule="CAS",
        variables=variables,
    )
    variables["x"].set(99)
    await push_dirty(client, "run-abc", variables)
    assert get_cached("x", variables) == 99
    assert variables["x"]._needs_push is False


async def test_push_dirty_bulk_loss_reregisters_full_entry():
    # Only `a` is dirty, but the bridge lost the whole entry — re-register must
    # cover both leaves (a and b), not just the dirty miss, then converge.
    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp(
                {
                    "results": [
                        {"var_id": "run-abc/a", "status": "created", "version": 1},
                        {"var_id": "run-abc/b", "status": "created", "version": 1},
                    ]
                }
            ),
            _resp({"results": [{"var_id": "run-abc/a", "status": "error", "error": "VariableNotFound"}]}),
            _resp(
                {
                    "results": [
                        {"var_id": "run-abc/a", "status": "created", "version": 1},
                        {"var_id": "run-abc/b", "status": "created", "version": 1},
                    ]
                }
            ),
            _resp({"results": [{"var_id": "run-abc/a", "status": "updated", "version": 2}]}),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(
        client,
        "run-abc",
        [
            {"name": "a", "default_value": 0, "var_type": "int"},
            {"name": "b", "default_value": 0, "var_type": "int"},
        ],
        variables=variables,
    )
    variables["a"].set(99)
    await push_dirty(client, "run-abc", variables)

    # The re-register call (3rd POST) must carry the FULL entry, both leaves.
    register_calls = [c for c in client.post.call_args_list if "variables" in c.kwargs["json"]]
    reregister = register_calls[-1].kwargs["json"]["variables"]
    var_ids = {v["var_id"] for v in reregister}
    assert var_ids == {"run-abc/a", "run-abc/b"}
    assert get_cached("a", variables) == 99
    assert variables["a"]._needs_push is False


async def test_push_dirty_persistent_failure_escalates(monkeypatch):
    import miner.sync_v2.synced_collection as sc

    monkeypatch.setattr(sc.asyncio, "sleep", AsyncMock())
    err = MagicMock()
    monkeypatch.setattr(sc.logger, "error", err)

    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp({"results": [{"var_id": "run-abc/a", "status": "created", "version": 1}]}),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(client, "run-abc", [{"name": "a", "default_value": 0, "var_type": "int"}], variables=variables)
    variables["a"].set(7)

    async def post_fn(path, **kwargs):
        if path == "/set":
            return _resp({"results": [{"var_id": "run-abc/a", "status": "error", "error": "VariableNotFound"}]})
        if path == "/register":
            return _resp({"detail": "bad"}, status=400)  # re-register keeps failing
        raise AssertionError(f"unexpected path {path}")

    client.post = AsyncMock(side_effect=post_fn)

    # Must not raise, must stop at the cap, must escalate (not swallow silently).
    await push_dirty(client, "run-abc", variables, max_retries=2)
    assert err.called


async def test_push_dirty_recovers_on_later_retry(monkeypatch):
    # Re-register succeeds but the leaves don't persist on the first retry
    # (slots still healing); the loop must keep retrying and converge on a
    # later attempt — the "keep retrying until it sticks" case.
    import miner.sync_v2.synced_collection as sc

    monkeypatch.setattr(sc.asyncio, "sleep", AsyncMock())

    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp(
                {
                    "results": [
                        {"var_id": "run-abc/a", "status": "created", "version": 1},
                        {"var_id": "run-abc/b", "status": "created", "version": 1},
                    ]
                }
            ),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(
        client,
        "run-abc",
        [
            {"name": "a", "default_value": 0, "var_type": "int"},
            {"name": "b", "default_value": 0, "var_type": "int"},
        ],
        variables=variables,
    )
    variables["a"].set(11)
    variables["b"].set(22)

    set_calls = {"n": 0}

    async def post_fn(path, **kwargs):
        if path == "/register":  # re-register always succeeds
            return _resp(
                {
                    "results": [
                        {"var_id": "run-abc/a", "status": "created", "version": 1},
                        {"var_id": "run-abc/b", "status": "created", "version": 1},
                    ]
                }
            )
        if path == "/set":
            set_calls["n"] += 1
            # calls 1 (initial) and 2 (first retry) still miss; call 3 sticks.
            if set_calls["n"] < 3:
                return _resp(
                    {
                        "results": [
                            {"var_id": "run-abc/a", "status": "error", "error": "VariableNotFound"},
                            {"var_id": "run-abc/b", "status": "error", "error": "VariableNotFound"},
                        ]
                    }
                )
            return _resp(
                {
                    "results": [
                        {"var_id": "run-abc/a", "status": "updated", "version": 2},
                        {"var_id": "run-abc/b", "status": "updated", "version": 2},
                    ]
                }
            )
        raise AssertionError(f"unexpected path {path}")

    client.post = AsyncMock(side_effect=post_fn)

    await push_dirty(client, "run-abc", variables, max_retries=3)

    # Converged after a second re-register + retry.
    assert get_cached("a", variables) == 11
    assert get_cached("b", variables) == 22
    assert variables["a"]._needs_push is False
    assert variables["b"]._needs_push is False
    register_calls = [c for c in client.post.call_args_list if "variables" in c.kwargs["json"]]
    assert len(register_calls) == 2  # first retry missed, so it re-registered twice
    assert set_calls["n"] == 3


async def test_set_value_single():
    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp({"results": [{"var_id": "run-abc/z", "status": "created", "version": 1}]}),
            _resp({"results": [{"var_id": "run-abc/z", "status": "updated", "version": 2}]}),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(
        client, "run-abc", [{"name": "z", "default_value": "hi", "var_type": "str"}], variables=variables
    )
    await set_value("z", "bye", variables=variables)
    assert get_cached("z", variables) == "bye"


# ── wildcard_fetch ────────────────────────────────────────────────────────────


async def test_wildcard_fetch_lists_then_gets():
    client = _make_client()
    client.get = AsyncMock(
        return_value=httpx.Response(
            200,
            json=[
                {"var_id": "run-abc/pfx/n1", "metadata": {}},
                {"var_id": "run-abc/pfx/n2", "metadata": {}},
            ],
            request=httpx.Request("GET", "http://test/vars"),
        )
    )
    client.post = AsyncMock(
        return_value=_resp(
            {
                "variables": [
                    {
                        "var_id": "run-abc/pfx/n1",
                        "value": 1,
                        "version": 1,
                        "metadata": {"updated_at": "t"},
                    }
                ]
            }
        )
    )
    out = await wildcard_fetch(client, "run-abc", "pfx/*")
    assert "pfx/n1" in out
    client.get.assert_awaited_once()


async def test_wildcard_fetch_updates_tracked_cache():
    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp({"results": [{"var_id": "run-abc/a/b", "status": "created", "version": 1}]}),
            _resp(
                {
                    "variables": [
                        {
                            "var_id": "run-abc/a/b",
                            "value": {"k": 9},
                            "version": 3,
                            "metadata": {"updated_at": "t"},
                        }
                    ]
                }
            ),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(
        client, "run-abc", [{"name": "a/b", "default_value": {}, "var_type": "dict"}], variables=variables
    )
    client.get = AsyncMock(
        return_value=httpx.Response(
            200,
            json=[{"var_id": "run-abc/a/b"}],
            request=httpx.Request("GET", "http://test/vars"),
        )
    )
    await wildcard_fetch(client, "run-abc", "a/*", variables=variables)
    assert get_cached("a/b", variables) == {"k": 9}


async def test_wildcard_fetch_no_match_short_circuits():
    client = _make_client()
    client.get = AsyncMock(return_value=httpx.Response(200, json=[], request=httpx.Request("GET", "http://test/vars")))
    out = await wildcard_fetch(client, "run-abc", "missing/*")
    assert out == {}
    client.post.assert_not_awaited()


# ── fetch_all / delete / wildcard_delete ──────────────────────────────────────


async def test_fetch_all_batch_get():
    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp({"results": [{"var_id": "run-abc/p", "status": "created", "version": 1}]}),
            _resp(
                {
                    "variables": [
                        {
                            "var_id": "run-abc/p",
                            "value": 3,
                            "version": 2,
                            "metadata": {"updated_at": "t"},
                        }
                    ]
                }
            ),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(client, "run-abc", [{"name": "p", "default_value": 0, "var_type": "int"}], variables=variables)
    data = await fetch_all(client, "run-abc", variables)
    assert data["p"] == 3


async def test_delete_untracked_uses_computed_var_id():
    client = _make_client()
    client.post = AsyncMock(return_value=_resp({}))
    variables: dict[str, SyncedVariableV2] = {}
    await delete(client, "run-abc", "orphan", variables=variables)
    body = client.post.call_args.kwargs["json"]
    assert body["var_ids"] == ["run-abc/orphan"]


async def test_wildcard_delete_parses_results():
    client = _make_client()
    client.post = AsyncMock(
        side_effect=_side_effect(
            _resp({"results": [{"var_id": "run-abc/old/x", "status": "created", "version": 1}]}),
            _resp({"results": [{"var_id": "run-abc/old/x", "status": "deleted"}]}),
        )
    )
    variables: dict[str, SyncedVariableV2] = {}
    await register_many(
        client, "run-abc", [{"name": "old/x", "default_value": {}, "var_type": "dict"}], variables=variables
    )
    client.get = AsyncMock(
        return_value=httpx.Response(
            200,
            json=[{"var_id": "run-abc/old/x"}],
            request=httpx.Request("GET", "http://test/vars"),
        )
    )
    deleted = await wildcard_delete(client, "run-abc", "old/*", variables=variables)
    assert deleted == ["old/x"]
    assert "old/x" not in variables


# ── introspection ─────────────────────────────────────────────────────────────


def test_get_cached_raises_for_unknown():
    variables: dict[str, SyncedVariableV2] = {}
    with pytest.raises(KeyError, match="not tracked"):
        get_cached("nope", variables)
