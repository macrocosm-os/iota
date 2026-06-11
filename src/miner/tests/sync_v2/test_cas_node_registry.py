"""Tests for CASNodeRegistry — entry mapping and CAS bridge updates."""

from __future__ import annotations

import inspect
import time
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx

from common.models.compute_node import ComputeNode
from miner.sync_v2.elastic_device_mesh import _STALE_KEEPALIVE_SECONDS, ElasticDeviceMesh
from miner.sync_v2.utils import sync_run_sync_prefix
from miner.sync_v2.variable_manager import VariableManager

RUN_ID = "test-run"
OWN_NODE = "miner1"
RUN_KEY = sync_run_sync_prefix(RUN_ID)


def _resp(data: dict, status: int = 200, method: str = "POST") -> httpx.Response:
    return httpx.Response(status, json=data, request=httpx.Request(method, "http://test/"))


class _FakeBridgeStore:
    """Minimal in-memory bridge backing store with CAS version checks."""

    def __init__(self) -> None:
        self.vars: dict[str, dict[str, Any]] = {}

    def register(self, variables: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for spec in variables:
            vid = spec["var_id"]
            if vid in self.vars:
                results.append({"var_id": vid, "status": "already_registered", "version": self.vars[vid]["version"]})
                continue
            self.vars[vid] = {
                "value": spec["default_value"],
                "version": 1,
                "write_rule": spec.get("write_rule", "CAS"),
                "var_type": spec.get("var_type"),
            }
            results.append({"var_id": vid, "status": "created", "version": 1})
        return results

    def set(self, updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for update in updates:
            vid = update["var_id"]
            entry = self.vars.get(vid)
            if entry is None:
                results.append({"var_id": vid, "status": "error", "error": "VariableNotFound"})
                continue
            if entry["write_rule"] == "CAS":
                expected = update.get("current_version")
                if expected != entry["version"]:
                    results.append({"var_id": vid, "status": "error", "error": "version mismatch"})
                    continue
            entry["value"] = update["value"]
            entry["version"] += 1
            results.append({"var_id": vid, "status": "updated", "version": entry["version"]})
        return results

    def get(self, var_ids: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for vid in var_ids:
            entry = self.vars.get(vid)
            if entry is None:
                continue
            out.append(
                {
                    "var_id": vid,
                    "value": entry["value"],
                    "version": entry["version"],
                    "metadata": {"updated_at": "t"},
                }
            )
        return out

    def list_prefix(self, prefix: str) -> list[dict[str, str]]:
        return [{"var_id": vid} for vid in self.vars if vid.startswith(prefix)]

    def delete(self, var_ids: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for vid in var_ids:
            if vid in self.vars:
                del self.vars[vid]
                results.append({"var_id": vid, "status": "deleted"})
            else:
                results.append({"var_id": vid, "status": "error", "error": "not found"})
        return results


def _make_fake_bridge_client(store: _FakeBridgeStore) -> AsyncMock:
    client = AsyncMock()

    async def post(path: str, **kwargs: Any) -> httpx.Response:
        body = kwargs.get("json", {})
        if path == "/register":
            return _resp({"results": store.register(body.get("variables", []))})
        if path == "/set":
            return _resp({"results": store.set(body.get("updates", []))})
        if path == "/get":
            return _resp({"variables": store.get(body.get("var_ids", []))})
        if path == "/delete":
            return _resp({"results": store.delete(body.get("var_ids", []))})
        return _resp({})

    async def get(path: str, **kwargs: Any) -> httpx.Response:
        if path == "/vars":
            prefix = kwargs.get("params", {}).get("prefix", "")
            return httpx.Response(
                200,
                json=store.list_prefix(prefix),
                request=httpx.Request("GET", "http://test/vars"),
            )
        return httpx.Response(404, request=httpx.Request("GET", "http://test/"))

    client.post = AsyncMock(side_effect=post)
    client.get = AsyncMock(side_effect=get)
    return client


async def _started_registry(
    *,
    own_node_id: str = OWN_NODE,
    run_id: str = RUN_ID,
) -> tuple[ElasticDeviceMesh, VariableManager, _FakeBridgeStore]:
    store = _FakeBridgeStore()
    mgr = VariableManager(url="http://test")
    mgr._http = _make_fake_bridge_client(store)
    reg = ElasticDeviceMesh(run_id=run_id, own_node_id=own_node_id)
    await reg.start_background_sync(mgr)
    return reg, mgr, store


def _leaf_var_id(slug: str, field: str) -> str:
    return f"{RUN_KEY}/node_registry/{slug}/{field}"


def _post_calls(mgr: VariableManager):
    return cast(AsyncMock, mgr._http.post).call_args_list


def _seed_peer(
    store: _FakeBridgeStore,
    slug: str,
    *,
    groups: list[str] | None = None,
    last_keepalive: float | None = None,
) -> None:
    ka = time.time() if last_keepalive is None else last_keepalive
    for field, value in (
        ("node_id", slug),
        ("groups", groups or ["layer-1"]),
        ("last_keepalive", ka),
    ):
        store.vars[_leaf_var_id(slug, field)] = {
            "value": value,
            "version": 1,
            "write_rule": "CAS",
        }


def _registry_with_dict_values(own_node_id: str, slug_entries: dict) -> ElasticDeviceMesh:
    """Create a registry backed by a mock SyncedDictionary with given slug→entry values."""
    reg = ElasticDeviceMesh(run_id="test-run", own_node_id=own_node_id)
    mock_dict = MagicMock()
    mock_dict.values.return_value = slug_entries.values()
    reg._dict = mock_dict
    return reg


# ── Entry mapping (no bridge) ──────────────────────────────────────────────────


def test_initialize_is_async():
    reg = _registry_with_dict_values(
        own_node_id="5FH6jq",
        slug_entries={"5FH6jq": {"node_id": "5FH6jq", "groups": ["all"]}},
    )
    coro = reg.initialize(ComputeNode(node_id="5FH6jq"))
    assert inspect.iscoroutine(coro)
    coro.close()  # avoid unawaited-coroutine warning


def test_entries_uses_node_id_from_payload():
    reg = _registry_with_dict_values(
        own_node_id="5FH6jq",
        slug_entries={
            "5FH6jq": {"node_id": "5FH6jq", "groups": ["all"]},
            "other": {"node_id": "other_key", "groups": ["all"]},
        },
    )
    assert set(reg.keys()) == {"5FH6jq", "other_key"}
    assert reg["other_key"]["node_id"] == "other_key"


def test_entries_skips_payloads_without_node_id():
    reg = _registry_with_dict_values(
        own_node_id="a",
        slug_entries={"bad": {"groups": ["all"]}},
    )
    assert "bad" not in reg
    assert len(reg) == 0


# ── CAS bridge integration ─────────────────────────────────────────────────────


async def test_start_registers_leaves_with_cas_rule():
    reg, _, store = await _started_registry()
    try:
        cas_vars = [v for v in store.vars.values() if v["write_rule"] == "CAS"]
        assert cas_vars, "expected CAS-registered node_registry leaves"
        assert all(vid.startswith(f"{RUN_KEY}/node_registry/{OWN_NODE}/") for vid in store.vars)
    finally:
        await reg.stop()


async def test_start_registers_default_groups():
    reg, _, store = await _started_registry()
    try:
        groups_vid = _leaf_var_id(OWN_NODE, "groups")
        assert store.vars[groups_vid]["value"] == ["all"]
    finally:
        await reg.stop()


async def test_register_pushes_cas_updates_with_current_version():
    reg, mgr, store = await _started_registry()
    try:
        node = ComputeNode(node_id=OWN_NODE, groups=["all", "layer-2", "shard-a"])
        await reg.register(node)

        set_calls = [c for c in _post_calls(mgr) if c.args and c.args[0] == "/set"]
        assert set_calls, "register() should POST /set after queuing fields"
        updates = set_calls[-1].kwargs["json"]["updates"]
        assert updates, "expected at least one CAS update"
        assert all("current_version" in u for u in updates), "CAS updates must include current_version"

        groups_vid = _leaf_var_id(OWN_NODE, "groups")
        assert store.vars[groups_vid]["value"] == ["all", "layer-2", "shard-a"]
        assert store.vars[groups_vid]["version"] >= 2

        nodes = reg.all_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == OWN_NODE
        assert nodes[0].groups == ["all", "layer-2", "shard-a"]
    finally:
        await reg.stop()


async def test_register_after_pull_does_not_push_peer_leaves():
    reg, _, store = await _started_registry()
    peer_slug = "peer2"
    try:
        _seed_peer(store, peer_slug, groups=["layer-1"], last_keepalive=time.time())
        await reg.pull()
        peer_versions_before = {
            vid: data["version"]
            for vid, data in store.vars.items()
            if vid.startswith(f"{RUN_KEY}/node_registry/{peer_slug}/")
        }

        await reg.register(ComputeNode(node_id=OWN_NODE, groups=["layer-2"]))

        assert peer_versions_before
        assert {
            vid: data["version"]
            for vid, data in store.vars.items()
            if vid.startswith(f"{RUN_KEY}/node_registry/{peer_slug}/")
        } == peer_versions_before
    finally:
        await reg.stop()


async def test_second_update_succeeds_when_version_matches():
    reg, _, store = await _started_registry()
    try:
        await reg.initialize(ComputeNode(node_id=OWN_NODE, groups=["layer-1"]))
        groups_vid = _leaf_var_id(OWN_NODE, "groups")
        version_after_first = store.vars[groups_vid]["version"]

        reg._require_started()[OWN_NODE]["groups"].set(["layer-3", "backup"])
        await reg.push()

        assert store.vars[groups_vid]["value"] == ["layer-3", "backup"]
        assert store.vars[groups_vid]["version"] == version_after_first + 1
        assert reg.all_nodes()[0].groups == ["layer-3", "backup"]
    finally:
        await reg.stop()


async def test_cas_version_mismatch_does_not_update_bridge():
    reg, _, store = await _started_registry()
    try:
        await reg.register(ComputeNode(node_id=OWN_NODE, groups=["layer-1"]))
        groups_vid = _leaf_var_id(OWN_NODE, "groups")

        # Another writer updated the bridge while our local cache is still stale.
        concurrent_value = ["layer-99"]
        store.vars[groups_vid]["version"] += 1
        store.vars[groups_vid]["value"] = concurrent_value
        version_on_bridge = store.vars[groups_vid]["version"]

        d = reg._require_started()
        d[OWN_NODE]["groups"].set(["layer-5"])
        await reg.push()

        assert store.vars[groups_vid]["value"] == concurrent_value
        assert store.vars[groups_vid]["version"] == version_on_bridge
    finally:
        await reg.stop()


async def test_initialize_publishes_keepalive_to_bridge():
    reg, _, store = await _started_registry()
    try:
        await reg.initialize(ComputeNode(node_id=OWN_NODE))

        ka_vid = _leaf_var_id(OWN_NODE, "last_keepalive")
        assert ka_vid in store.vars
        assert store.vars[ka_vid]["value"] > 0
        assert store.vars[ka_vid]["version"] >= 2
    finally:
        await reg.stop()


async def test_pull_merges_peer_registry_entry():
    reg, _, store = await _started_registry()
    peer_slug = "peer2"
    try:
        _seed_peer(store, peer_slug, last_keepalive=time.time())

        await reg.pull()

        node_ids = {n.node_id for n in reg.all_nodes()}
        assert OWN_NODE in node_ids
        assert peer_slug in node_ids
        assert reg.get_group("layer-1")[0].node_id == peer_slug
    finally:
        await reg.stop()


async def test_pull_evicts_peer_with_stale_keepalive():
    reg, mgr, store = await _started_registry()
    peer_slug = "stale_peer"
    try:
        _seed_peer(store, peer_slug, last_keepalive=time.time() - _STALE_KEEPALIVE_SECONDS - 60)

        await reg.pull()

        node_ids = {n.node_id for n in reg.all_nodes()}
        assert peer_slug not in node_ids
        assert not any(vid.startswith(f"{RUN_KEY}/node_registry/{peer_slug}/") for vid in store.vars)
        delete_calls = [c for c in _post_calls(mgr) if c.args and c.args[0] == "/delete"]
        assert delete_calls, "stale peer leaves should be deleted from the bridge"
    finally:
        await reg.stop()


async def test_pull_keeps_peer_with_recent_keepalive():
    reg, mgr, store = await _started_registry()
    peer_slug = "live_peer"
    try:
        _seed_peer(store, peer_slug, last_keepalive=time.time() - 10)

        await reg.pull()

        assert peer_slug in {n.node_id for n in reg.all_nodes()}
        delete_calls = [c for c in _post_calls(mgr) if c.args and c.args[0] == "/delete"]
        assert not delete_calls
    finally:
        await reg.stop()


async def test_repulls_full_entry_after_eviction_by_peer():
    """A live miner whose slug was evicted re-publishes its full record, not just last_keepalive."""
    reg, _, store = await _started_registry()
    try:
        node = ComputeNode(node_id=OWN_NODE, p2p_node_ids=["abc123"], groups=["all", "layer-2"])
        await reg.initialize(node)

        # A peer evicts this miner: every leaf under its slug is deleted on the bridge.
        for vid in [v for v in store.vars if v.startswith(f"{RUN_KEY}/node_registry/{OWN_NODE}/")]:
            del store.vars[vid]

        # The keepalive tick recreates only the dirty last_keepalive leaf.
        d = reg._require_started()
        d[OWN_NODE]["last_keepalive"].set(time.time())
        await d.push_dirty()
        assert set(store.vars) == {_leaf_var_id(OWN_NODE, "last_keepalive")}

        # The next background pull sees the stripped entry and re-publishes the full record.
        await d._after_pull(await d.wildcard_fetch("node_registry/*"))
        await d.push_dirty()

        assert store.vars[_leaf_var_id(OWN_NODE, "p2p_node_ids")]["value"] == ["abc123"]
        assert store.vars[_leaf_var_id(OWN_NODE, "groups")]["value"] == ["all", "layer-2"]
        assert store.vars[_leaf_var_id(OWN_NODE, "node_id")]["value"] == OWN_NODE
        assert store.vars[_leaf_var_id(OWN_NODE, "joined_at")]["value"] == node.joined_at
        assert OWN_NODE in {n.node_id for n in reg.all_nodes()}
    finally:
        await reg.stop()


async def test_backfills_leaves_lost_to_partial_delete():
    """Leaves missing from an otherwise-visible own entry are restored from the stored record."""
    reg, _, store = await _started_registry()
    try:
        node = ComputeNode(node_id=OWN_NODE, p2p_node_ids=["abc123"])
        await reg.initialize(node)

        # A partial delete stripped joined_at, p2p_node_ids, and last_keepalive but left node_id.
        del store.vars[_leaf_var_id(OWN_NODE, "joined_at")]
        del store.vars[_leaf_var_id(OWN_NODE, "p2p_node_ids")]
        del store.vars[_leaf_var_id(OWN_NODE, "last_keepalive")]

        before_repair = time.time()
        d = reg._require_started()
        await d._after_pull(await d.wildcard_fetch("node_registry/*"))
        await d.push_dirty()

        assert store.vars[_leaf_var_id(OWN_NODE, "joined_at")]["value"] == node.joined_at
        assert store.vars[_leaf_var_id(OWN_NODE, "p2p_node_ids")]["value"] == ["abc123"]
        # Restored keepalive must be stamped fresh, not taken from the stored snapshot.
        assert store.vars[_leaf_var_id(OWN_NODE, "last_keepalive")]["value"] >= before_repair
    finally:
        await reg.stop()


async def test_pull_does_not_evict_own_entry_when_stale():
    reg, _, store = await _started_registry()
    try:
        ka_vid = _leaf_var_id(OWN_NODE, "last_keepalive")
        store.vars[ka_vid]["value"] = time.time() - _STALE_KEEPALIVE_SECONDS - 60

        await reg.pull()

        assert OWN_NODE in {n.node_id for n in reg.all_nodes()}
        assert ka_vid in store.vars
    finally:
        await reg.stop()


# ── Leader election ────────────────────────────────────────────────────────────


def test_get_leader_returns_oldest_node():
    t = time.time()
    reg = _registry_with_dict_values(
        own_node_id="node-a",
        slug_entries={
            "node-a": {"node_id": "node-a", "groups": ["all"], "joined_at": t + 200},
            "node-b": {"node_id": "node-b", "groups": ["all"], "joined_at": t + 100},
            "node-c": {"node_id": "node-c", "groups": ["all"], "joined_at": t + 50},
        },
    )
    leader = reg.get_leader("all")
    assert leader is not None
    assert leader.node_id == "node-c"


def test_get_leader_returns_none_for_empty_group():
    reg = _registry_with_dict_values(
        own_node_id="node-a",
        slug_entries={"node-a": {"node_id": "node-a", "groups": ["all"], "joined_at": 100.0}},
    )
    assert reg.get_leader("nonexistent-group") is None


def test_get_leader_tiebreak_by_node_id():
    t = 100.0
    reg = _registry_with_dict_values(
        own_node_id="node-zzz",
        slug_entries={
            "node-zzz": {"node_id": "node-zzz", "groups": ["all"], "joined_at": t},
            "node-aaa": {"node_id": "node-aaa", "groups": ["all"], "joined_at": t},
        },
    )
    leader = reg.get_leader("all")
    assert leader is not None
    assert leader.node_id == "node-aaa"


def test_own_is_leader_when_oldest():
    t = time.time()
    reg = _registry_with_dict_values(
        own_node_id="node-a",
        slug_entries={
            "node-a": {"node_id": "node-a", "groups": ["all"], "joined_at": t + 50},
            "node-b": {"node_id": "node-b", "groups": ["all"], "joined_at": t + 200},
        },
    )
    assert reg.is_leader("all") is True
    assert reg.is_leader("layer-1") is False


def test_own_is_leader_false_when_not_oldest():
    t = time.time()
    reg = _registry_with_dict_values(
        own_node_id="node-b",
        slug_entries={
            "node-a": {"node_id": "node-a", "groups": ["all"], "joined_at": t + 50},
            "node-b": {"node_id": "node-b", "groups": ["all"], "joined_at": t + 200},
        },
    )
    assert reg.is_leader("all") is False


async def test_get_leader_after_eviction():
    """After the oldest peer is evicted (stale keepalive), own node becomes leader."""
    reg, _, store = await _started_registry(own_node_id="node-newer")
    try:
        # Seed older peer with a stale keepalive so it gets evicted on pull
        _seed_peer(store, "node-older", groups=["all"], last_keepalive=time.time() - _STALE_KEEPALIVE_SECONDS - 60)
        store.vars[_leaf_var_id("node-older", "joined_at")] = {"value": 50.0, "version": 1, "write_rule": "CAS"}
        # Own node joined later
        store.vars[_leaf_var_id("node-newer", "joined_at")] = {"value": 200.0, "version": 1, "write_rule": "CAS"}
        store.vars[_leaf_var_id("node-newer", "groups")] = {"value": ["all"], "version": 1, "write_rule": "CAS"}

        await reg.pull()

        assert "node-older" not in {n.node_id for n in reg.all_nodes()}
        assert reg.is_leader("all") is True
    finally:
        await reg.stop()
