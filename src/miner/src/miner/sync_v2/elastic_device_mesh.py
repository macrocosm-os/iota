"""Run-scoped node registry using bridge v2 with one CAS variable per node field."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable, Iterator

from loguru import logger

from common.models.compute_node import ComputeNode
from miner.sync_v2.synced_dictionary import SyncedDictionary
from miner.sync_v2.utils import sync_run_sync_prefix
from miner.sync_v2.variable_manager import VariableManager

if TYPE_CHECKING:
    from miner.p2p import P2PStack

_REL_PREFIX = "node_registry"
_KEEPALIVE_INTERVAL = 5.0
_FRESH_PULL_WINDOW_SECONDS = 5.0
_STALE_KEEPALIVE_SECONDS = 30.0


def _bridge_key_segment(node_id: str) -> str:
    """Make *node_id* safe as a single path segment under ``node_registry/``."""
    return "".join((c if c.isalnum() or c in "-_" else "_") for c in str(node_id))


def _keepalive_age(entry: dict[str, Any], *, now: float) -> float:
    """Seconds since *entry*'s ``last_keepalive``; treat missing/invalid as infinitely stale."""
    raw = entry.get("last_keepalive")
    if raw is None:
        return float("inf")
    try:
        return now - float(raw)
    except (TypeError, ValueError):
        return float("inf")


class ElasticDeviceMesh:
    """Node discovery state backed by bridge v2: one CAS leaf per ComputeNode field.

    A lightweight wrapper around a :class:`SyncedDictionary`. Each miner
    writes only its own ``{run_id}/node_registry/{slug(node_id)}/…`` leaves (CAS);
    the dict pulls all entries automatically via :class:`VariableManager`.

    On each pull, any peer whose ``last_keepalive`` is older than
    :data:`_STALE_KEEPALIVE_SECONDS` may be evicted by any miner (including this
    one), provided the registry snapshot was fetched within the last
    :data:`_FRESH_PULL_WINDOW_SECONDS`.

    Call :meth:`start_background_sync` to attach to a :class:`VariableManager`.
    """

    def __init__(
        self,
        run_id: str | None,
        own_node_id: str,
        p2p: P2PStack | None = None,
        groups: list[str] = [],
        *,
        on_update: Callable[[ElasticDeviceMesh], Any] | None = None,
    ) -> None:
        self._own_node_id = own_node_id
        self._own_slug = _bridge_key_segment(own_node_id)
        self._on_update = on_update
        self._run_key = sync_run_sync_prefix(run_id)
        self.p2p: P2PStack | None = p2p
        self._dict: SyncedDictionary | None = None
        self._manager: VariableManager | None = None
        self._local_overrides: dict[str, Any] = {}  # peer-local state not pushed to bridge
        self._keepalive_task: asyncio.Task | None = None
        self._last_pulled_at: float = 0.0
        # P2P integration — set by the miner after the P2P stack starts
        self._registered_peer_addrs: dict[str, tuple[str | None, tuple[str, ...]]] = {}
        self._groups = groups

    # ── Internal: slug→entry → node_id→entry ──────────────────────────────────

    def _entries(self) -> dict[str, Any]:
        """Map node_id → entry, merging synced dict with local overrides."""
        result: dict[str, Any] = {}
        if self._dict is not None:
            for val in self._dict.values():
                if isinstance(val, dict) and (nid := val.get("node_id")) is not None:
                    result[str(nid)] = val
        result.update(self._local_overrides)
        return result

    async def _on_dict_pull(self, _d: SyncedDictionary) -> None:
        self._local_overrides.clear()
        self._last_pulled_at = time.time()
        await self._evict_stale_entries()
        self._on_miner_update()
        if self._on_update is not None:
            self._on_update(self)

    async def _evict_stale_entries(self) -> None:
        """Remove peer entries whose keepalive is stale, using a fresh pull snapshot."""
        if self._last_pulled_at <= 0:
            return
        if time.time() - self._last_pulled_at >= _FRESH_PULL_WINDOW_SECONDS:
            return

        d = self._require_started()
        now = time.time()
        for slug, entry in list(d.to_dict().items()):
            if slug == self._own_slug or not isinstance(entry, dict):
                continue
            node_id = entry.get("node_id")
            if node_id is not None and str(node_id) == self._own_node_id:
                continue
            if _keepalive_age(entry, now=now) <= _STALE_KEEPALIVE_SECONDS:
                continue
            try:
                await d.delete_slug(slug)
                if node_id is not None:
                    self._local_overrides.pop(str(node_id), None)
                logger.info(
                    f"[ElasticDeviceMesh] Evicted stale node {node_id!r} "
                    f"(slug={slug!r}, keepalive age {_keepalive_age(entry, now=now):.0f}s)"
                )
            except Exception as exc:
                logger.warning(f"[ElasticDeviceMesh] Failed to evict stale node {node_id!r}: {exc}")

    def _require_started(self) -> SyncedDictionary:
        if self._dict is None:
            raise RuntimeError("ElasticDeviceMesh.start_background_sync() must be called first")
        return self._dict

    # ── dict-like interface ────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        return self._entries()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == self._own_node_id:
            d = self._require_started()
            if isinstance(value, dict):
                for field, v in value.items():
                    d[self._own_slug][field].set(v)
            else:
                d[self._own_slug].set(value)
        else:
            self._local_overrides[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._entries()

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries())

    def __len__(self) -> int:
        return len(self._entries())

    def __repr__(self) -> str:
        return f"ElasticDeviceMesh({self._entries()!r})"

    def get(self, key: str, default: Any = None) -> Any:
        return self._entries().get(key, default)

    def keys(self):
        return self._entries().keys()

    def values(self):
        return self._entries().values()

    def items(self):
        return self._entries().items()

    def to_dict(self) -> dict[str, Any]:
        return dict(self._entries())

    # ── registry write helpers ─────────────────────────────────────────────────

    async def initialize(self, node: ComputeNode) -> None:
        """Publish own entry to the registry for the first time (or after re-registration).

        Writes all ComputeNode fields (including layer groups) plus a fresh
        keepalive timestamp, then pushes to the bridge.  Call this once after
        registration is complete and the compute node is fully configured.
        """
        d = self._require_started()
        entry = node.model_dump()
        entry["last_keepalive"] = time.time()
        for field, value in entry.items():
            d[self._own_slug][field].set(value)
        await d.push_dirty()

    async def register(self, node: ComputeNode) -> None:
        """Add or update any *node* — writes each ComputeNode field as its own leaf."""
        d = self._require_started()
        slug = _bridge_key_segment(node.node_id)
        for field, value in node.model_dump().items():
            d[slug][field].set(value)
        await d.push_dirty()

    # ── registry query helpers ─────────────────────────────────────────────────

    def all_nodes(self) -> list[ComputeNode]:
        return [ComputeNode(**entry) for entry in self._entries().values() if isinstance(entry, dict)]

    def get_group(self, group: str) -> list[ComputeNode]:
        out: list[ComputeNode] = []
        for raw in self._entries().values():
            if not isinstance(raw, dict):
                continue
            groups = raw.get("groups") or []
            if group in groups:
                out.append(ComputeNode(**raw))
        return out

    def get_leader(self, group: str = "all") -> ComputeNode | None:
        """Return the leader of *group*: the node with the earliest joined_at.

        Tie-breaks by node_id lexicographically. Returns None if the group is empty.
        """
        members = self.get_group(group)
        if not members:
            return None
        return min(members, key=lambda n: (n.joined_at, n.node_id))

    def is_leader(self, group: str = "all") -> bool:
        """Return True if this node is currently the leader of *group*."""
        leader = self.get_leader(group)
        return leader is not None and leader.node_id == self._own_node_id

    # ── bridge push / pull ─────────────────────────────────────────────────────

    async def push(self) -> None:
        await self._require_started().push_dirty()

    async def pull(self) -> None:
        d = self._require_started()
        await d.pull()
        self._last_pulled_at = time.time()
        await self._evict_stale_entries()

    # ── P2P integration ────────────────────────────────────────────────────────

    def on_sender_restarted(self) -> None:
        """Clear the address-book cache and re-push all peer hints to the new sender subprocess."""
        self._registered_peer_addrs.clear()
        self.sync_peer_addrs_to_sender()

    def sync_peer_addrs_to_sender(self) -> None:
        """Push every peer's iroh address hints into the sender subprocess.

        Skips own node and peers whose hints haven't changed, so safe to call
        on every registry update.
        """
        sender = self.p2p.sender if self.p2p is not None else None
        if sender is None:
            return
        for raw in self.values():
            if not isinstance(raw, dict):
                continue
            peer_hotkey = raw.get("node_id")
            if peer_hotkey == self._own_node_id:
                continue
            relay_url = raw.get("iroh_relay_url")
            direct_addresses = tuple(raw.get("iroh_direct_addresses") or [])
            if relay_url is None and not direct_addresses:
                continue
            for p2p_node_id in raw.get("p2p_node_ids") or []:
                key: tuple[str | None, tuple[str, ...]] = (relay_url, direct_addresses)
                if self._registered_peer_addrs.get(p2p_node_id) == key:
                    continue
                self._registered_peer_addrs[p2p_node_id] = key
                asyncio.create_task(
                    sender.register_peer(p2p_node_id, relay_url, list(direct_addresses), hotkey=peer_hotkey),
                    name=f"register-peer-{p2p_node_id[:8]}",
                )

    def sync_valid_hotkeys(self) -> None:
        """Publish currently-registered hotkeys into the receiver's run-scope filter.

        ``/peer/status`` from any hotkey not in this set is dropped at the receiver
        subprocess, keeping cross-run noise out of peer_status_dict.
        """
        if self.p2p is None:
            return
        valid_dict = self.p2p.valid_hotkeys_dict
        if valid_dict is None:
            return
        registered = {node.node_id for node in self.all_nodes()}
        try:
            existing = set(valid_dict.keys())
        except Exception:
            return
        for hotkey in registered - existing:
            valid_dict[hotkey] = True
        for hotkey in existing - registered:
            valid_dict.pop(hotkey, None)

    def sync_peer_status_into_registry(self) -> None:
        """Read received peer status from the P2P shared dict and merge into registry entries."""
        if self.p2p is None:
            return
        peer_dict = self.p2p.peer_status_dict
        if peer_dict is None:
            return
        try:
            snapshot = dict(peer_dict)
        except Exception:
            return
        hotkey_to_node_id: dict[str, str] = {node.node_id: node.node_id for node in self.all_nodes()}
        for source_hotkey, (status_dict, received_at) in snapshot.items():
            node_id = hotkey_to_node_id.get(source_hotkey)
            if node_id is None:
                continue
            entry = self.get(node_id)
            if entry is None:
                continue
            entry = dict(entry)
            status_dict["last_status_received"] = received_at
            entry["runtime_metrics"] = status_dict
            self[node_id] = entry

    def _stamp_own_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Ensure all miner-owned fields are present on the own registry entry.

        Called after each pull to restore fields this miner is authoritative for
        (p2p_node_ids, routing groups) if a remote fetch carried stale values.
        """
        if self.p2p is not None:
            entry["p2p_node_ids"] = self.p2p.node_ids
            entry["iroh_relay_url"] = self.p2p.relay_url
            entry["iroh_direct_addresses"] = self.p2p.direct_addresses
        if self._groups:
            entry["groups"] = list(set(["all", *self._groups]))
        return entry

    def _on_miner_update(self) -> None:
        """Re-stamp own entry and sync peer addresses after each pull."""
        entry = self.get(self._own_node_id)
        if entry is not None:
            self._stamp_own_entry(entry)
            self[self._own_node_id] = entry
        self.sync_peer_addrs_to_sender()

    # ── keepalive loop ─────────────────────────────────────────────────────────

    async def _keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(_KEEPALIVE_INTERVAL)
            try:
                if self._dict is None:
                    break
                self._dict[self._own_slug]["last_keepalive"].set(time.time())
                await self._dict.push_dirty()
            except Exception as exc:
                logger.warning(f"[ElasticDeviceMesh] Keepalive error: {exc}")

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def start_background_sync(self, manager: VariableManager) -> None:
        """Attach to *manager* for background pull/push and start keepalive loop."""
        self._manager = manager
        self._dict = manager.synced_dict(
            run_id=self._run_key,
            name=_REL_PREFIX,
            rule="CAS",
            pull_frequency=2.0,
            on_pull=self._on_dict_pull,
        )
        await self._dict.register({self._own_slug: ComputeNode(node_id=self._own_node_id).model_dump()})
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def stop(self) -> None:
        """Unregister from the VariableManager and cancel the keepalive loop."""
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        if self._manager is not None:
            self._manager.unwatch_collection(f"{self._run_key}/{_REL_PREFIX}")
        self._manager = None
        self._dict = None
        self._local_overrides.clear()
