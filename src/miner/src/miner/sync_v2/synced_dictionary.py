"""SyncedDictionary — nested dict-like wrapper using synced_collection helpers.

Each leaf key path is its own bridge variable.  All mutations go through ``.set()``
on a node returned by bracket access::

    d["step"].set(1)
    d["config"]["beta1"].set(0.9)

Created via ``manager.synced_dict(run_id, name, ...)``.
"""

from __future__ import annotations

import copy
import inspect
from typing import TYPE_CHECKING, Any, Iterator

import httpx
from loguru import logger

from miner.sync_v2.synced_collection import (
    delete,
    push_dirty,
    register_many,
    set_many,
    wildcard_delete,
    wildcard_fetch,
)
from miner.sync_v2.synced_variable import Lock, SyncedVariableV2

if TYPE_CHECKING:
    from miner.sync_v2.variable_manager import VariableManager

# ── Sentinel for "key not present" ────────────────────────────────────────────

_MISSING = object()

# ── Bridge type inference ──────────────────────────────────────────────────────

_PYTHON_TO_BRIDGE_TYPE: dict[type, str] = {
    bool: "bool",  # before int — bool is a subclass of int
    int: "int",
    float: "float",
    str: "str",
    dict: "dict",
    list: "list",
}


def _infer_type(v: Any) -> str:
    return _PYTHON_TO_BRIDGE_TYPE.get(type(v), "str")


# ── Flatten / unflatten ───────────────────────────────────────────────────────


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten ``{"config": {"beta1": 0.9}}`` → ``{"config/beta1": 0.9}``."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        path = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten(v, path))
        else:
            result[path] = v
    return result


def _unflatten(flat: dict[str, Any]) -> dict:
    """Rebuild ``{"config/beta1": 0.9, "step": 1}`` → ``{"config": {"beta1": 0.9}, "step": 1}``."""
    result: dict = {}
    for path, value in flat.items():
        parts = path.split("/")
        d = result
        for part in parts[:-1]:
            if not isinstance(d.get(part), dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def _deep_update(base: dict, updates: dict) -> None:
    """Recursively merge *updates* into *base* in-place."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _mark_tree_dirty(variables: dict[str, SyncedVariableV2], var_name: str, value: Any) -> None:
    """Queue registered leaves after a set() call.

    If *var_name* is registered (even as a dict-typed leaf), queues it and
    stops.  Otherwise recurses into dict values to find registered sub-paths.
    """
    var = variables.get(var_name)
    if var is not None:
        if var.write_rule != "LOCK":
            var.set(value)
        return
    if isinstance(value, dict):
        for subpath, leaf_val in _flatten(value).items():
            _mark_tree_dirty(variables, f"{var_name}/{subpath}", leaf_val)


def _flatten_aware(local: dict, name: str, variables: dict[str, SyncedVariableV2]) -> dict[str, Any]:
    """Flatten like :func:`_flatten` but treat registered variable paths as atomic leaves.

    If a path is already tracked in *variables*, its value is kept intact even
    if it is a ``dict`` — it will not be further split into sub-keys.
    """
    result: dict[str, Any] = {}

    def _recurse(d: dict, prefix: str) -> None:
        for k, v in d.items():
            path = f"{prefix}/{k}" if prefix else k
            if f"{name}/{path}" in variables or not isinstance(v, dict):
                result[path] = v
            else:
                _recurse(v, path)

    _recurse(local, "")
    return result


# ── _Node ──────────────────────────────────────────────────────────────────────


class _Node:
    """A lazily-evaluated path within a :class:`SyncedDictionary`.

    Returned by bracket access on the dict or on another node::

        d["step"]                  # _Node at path ("step",)
        d["config"]["beta1"]       # _Node at path ("config", "beta1")

    **Writing** (sync — marks dirty for background push)::

        d["step"].set(1)
        d["config"]["beta1"].set(0.9)
        d["config"].set({"beta1": 0.9, "beta2": 0.999})  # sets entire sub-tree

    **Reading** (sync, from local cache)::

        val = d["step"].get()          # returns None if absent
        val = d["step"].get(default=0)
        val = d["step"].value          # raises KeyError if absent

    **Deleting** (async — removes from bridge)::

        await d["config"]["old_key"].delete()

    **Iterating** (treats the node as a namespace, reads from local cache)::

        for key in d["config"]: ...
        d["config"].keys()
        "beta1" in d["config"]
    """

    __slots__ = ("_root", "_path")

    def __init__(self, root: SyncedDictionary, path: tuple[str, ...]) -> None:
        self._root = root
        self._path = path

    def _var_name(self) -> str:
        return f"{self._root._name}/{'/'.join(self._path)}"

    def _local_val(self) -> Any:
        """Navigate to the value at this path in ``_local`` without mutating it."""
        d = self._root._local
        try:
            for part in self._path:
                d = d[part]
            return d
        except (KeyError, TypeError):
            return _MISSING

    # ── Chained access ────────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> _Node:
        return _Node(self._root, self._path + (key,))

    # ── Write ─────────────────────────────────────────────────────────────────

    def set(self, value: Any) -> None:
        """Update local cache and mark dirty for background push.

        This is synchronous — the value is queued for the bridge.  Call
        :meth:`SyncedDictionary.push` to flush immediately, or rely on
        the background loop when ``push_frequency`` is configured.

        If *value* is a ``dict``, the entire sub-tree is updated in-place.
        Registration of new paths on the bridge happens lazily on the next
        :meth:`SyncedDictionary.push` call.

        For LOCK variables the local cache is updated but dirty marking is
        skipped — hold a lock and call ``push(lock=lock)`` to flush.
        """
        d = self._root._local
        for part in self._path[:-1]:
            d = d.setdefault(part, {})
        key = self._path[-1]
        var_name = self._var_name()
        if isinstance(value, dict) and var_name not in self._root._variables:
            # Unregistered path: merge-update so sub-tree semantics work naturally.
            # Registered dict-typed leaves (e.g. register_leaf) are replaced entirely.
            _deep_update(d.setdefault(key, {}), value)
        else:
            d[key] = value
        _mark_tree_dirty(self._root._variables, var_name, value)

    async def set_and_push(self, value: Any, *, lock: Lock | None = None) -> None:
        """Update local cache and immediately push this dictionary to the bridge."""
        self.set(value)
        await self._root.push(lock=lock)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, default: Any = None) -> Any:
        """Return the cached value at this path, or *default* if absent."""
        val = self._local_val()
        return default if val is _MISSING else val

    @property
    def value(self) -> Any:
        """Return the cached value; raises ``KeyError`` if absent."""
        val = self._local_val()
        if val is _MISSING:
            raise KeyError("/".join(self._path))
        return val

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete(self) -> None:
        """Delete this leaf from the bridge and remove it from local cache."""
        var_name = self._var_name()
        root = self._root
        if var_name in root._variables:
            await delete(root._http, root._run_id, var_name, variables=root._variables)
        else:
            logger.warning(f"SyncedDictionary: {var_name!r} not in registered variables, skipping bridge delete")
        d = self._root._local
        try:
            for part in self._path[:-1]:
                d = d[part]
            d.pop(self._path[-1], None)
        except (KeyError, TypeError):
            pass

    # ── Namespace-like iteration ───────────────────────────────────────────────

    def keys(self):
        v = self._local_val()
        return v.keys() if isinstance(v, dict) else {}.keys()

    def values(self):
        v = self._local_val()
        return v.values() if isinstance(v, dict) else {}.values()

    def items(self):
        v = self._local_val()
        return v.items() if isinstance(v, dict) else {}.items()

    def __contains__(self, key: str) -> bool:
        v = self._local_val()
        return isinstance(v, dict) and key in v

    def __len__(self) -> int:
        v = self._local_val()
        return len(v) if isinstance(v, dict) else 0

    def __iter__(self) -> Iterator:
        v = self._local_val()
        return iter(v) if isinstance(v, dict) else iter([])

    def __repr__(self) -> str:
        val = self._local_val()
        cached = repr(val) if val is not _MISSING else "<absent>"
        return f"_Node({'/'.join(self._path)!r}, {cached})"


# ── SyncedDictionary ──────────────────────────────────────────────────────────


class SyncedDictionary:
    """Nested dict-like wrapper backed by synced_collection module functions.

    Each leaf key path is its own bridge variable.  All mutations go through
    :meth:`_Node.set`; reading goes through :meth:`_Node.get` or
    :attr:`_Node.value`.  Assignment with ``=`` is intentionally not supported
    — use ``.set()`` instead.

    **Create via the factory** (mirrors ``manager.synced_dict()``)::

        d = manager.synced_dict(run_id="my-run", name="optimizer", rule="LWW")

    **Register a schema** (async, idempotent; infers bridge types from Python types)::

        await d.register({
            "step": 0,
            "lr": 1e-3,
            "config": {"beta1": 0.9, "beta2": 0.999},
        })

    **Write** (sync, marks dirty — flushed by background loop or explicit push)::

        d["step"].set(1)
        d["config"]["beta1"].set(0.85)
        d["config"].set({"beta1": 0.85, "beta2": 0.998})  # update sub-tree

    **Read** (sync, from local cache)::

        val = d["step"].get()
        val = d["step"].get(default=0)
        val = d["config"]["beta1"].value  # raises KeyError if absent

    **Delete** (async, removes from bridge)::

        await d["config"]["old_key"].delete()

    **Network ops**::

        await d.pull()           # wildcard fetch → unflatten → local cache
        await d.push()           # lazy-register new paths + set_many
        await d.push(lock=lock)  # for LOCK rule

    **Background sync** (enrolled in VariableManager poll loop)::

        d = manager.synced_dict("my-run", "state", pull_frequency=5.0, push_frequency=2.0)

    **Snapshot / raw access**::

        d.to_dict()          # deep copy of local cache as a plain nested dict
        d.var("config/lr")   # underlying SyncedVariableV2 (for locking, version)
    """

    def __init__(
        self,
        run_id: str,
        name: str,
        *,
        manager: VariableManager,
        rule: str = "LWW",
        pull_frequency: float | None = None,
        push_frequency: float | None = None,
        on_pull: Any = None,
    ) -> None:
        self._run_id = run_id
        self._name = name
        self._rule = rule
        self._http: httpx.AsyncClient = manager._http
        self._local: dict = {}
        self._variables: dict[str, SyncedVariableV2] = {}
        self._on_pull = on_pull

        if pull_frequency is not None or push_frequency is not None:
            manager.register_collection(
                f"{run_id}/{name}",
                self,
                wildcard_path=f"{name}/*",
                after_pull=self._after_pull,
                pull_frequency=pull_frequency or 1e10,
                push_frequency=push_frequency,
            )

    # ── Schema registration ───────────────────────────────────────────────────

    async def register(self, schema: dict, *, types: dict[str, str] | None = None) -> None:
        """Batch-register all leaves in *schema* as bridge variables.

        *schema* is a nested dict; each leaf becomes a separate bridge variable.
        Bridge types are inferred from Python types.  Idempotent — already-
        registered variables are silently accepted.  Use *types* to override
        inferred types for paths whose default value is ``None``.  Also seeds
        the local cache with the schema's default values.
        """
        flat = _flatten(schema)
        if not flat:
            return
        entries = [
            {
                "name": f"{self._name}/{path}",
                "default_value": value,
                "var_type": (types or {}).get(path, _infer_type(value)),
            }
            for path, value in flat.items()
        ]
        await register_many(self._http, self._run_id, entries, rule=self._rule, variables=self._variables)
        _deep_update(self._local, schema)

    # ── Network ops ───────────────────────────────────────────────────────────

    async def pull(self) -> None:
        """Fetch all variables under this dict's prefix and update local cache."""
        raw = await wildcard_fetch(self._http, self._run_id, f"{self._name}/*", variables=self._variables)
        prefix = f"{self._name}/"
        stripped = {k.removeprefix(prefix): v for k, v in raw.items() if k.startswith(prefix)}
        self._local = _unflatten(stripped)

    async def register_leaf(self, path: str, default_value: Any, *, var_type: str | None = None) -> None:
        """Register a single variable at *path* as an atomic leaf.

        Unlike :meth:`register`, a dict *default_value* is stored as a single
        ``dict``-typed bridge variable rather than being split into sub-fields.
        Idempotent locally — bridge registration is skipped if already tracked.
        """
        full_name = f"{self._name}/{path}"
        if full_name not in self._variables:
            await register_many(
                self._http,
                self._run_id,
                [
                    {
                        "name": full_name,
                        "default_value": default_value,
                        "var_type": var_type or _infer_type(default_value),
                    }
                ],
                rule=self._rule,
                variables=self._variables,
            )
        parts = path.split("/")
        d = self._local
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = default_value

    async def push(self, *, lock: Any = None) -> None:
        """Flush local cache to the bridge.

        Lazily registers any keys present locally but not yet on the bridge.
        Variables registered via :meth:`register_leaf` (including dict-typed
        atomic leaves) are pushed as a single value, not split into sub-fields.
        For LOCK rule: pass an active *lock* acquired via ``manager.lock(d.var(...))``.
        For CAS rule: call :meth:`pull` first — stored versions are used automatically.
        """
        flat = _flatten_aware(self._local, self._name, self._variables)
        if not flat:
            return

        new_entries = [
            {
                "name": f"{self._name}/{path}",
                "default_value": value,
                "var_type": _infer_type(value),
            }
            for path, value in flat.items()
            if f"{self._name}/{path}" not in self._variables
        ]
        if new_entries:
            await register_many(self._http, self._run_id, new_entries, rule=self._rule, variables=self._variables)

        updates = {f"{self._name}/{path}": value for path, value in flat.items()}
        await set_many(self._http, self._run_id, self._variables, updates, lock=lock)

    async def delete_slug(self, slug: str) -> None:
        """Delete all bridge leaves under ``{name}/{slug}/`` and drop the local entry."""
        await wildcard_delete(self._http, self._run_id, f"{self._name}/{slug}/*", variables=self._variables)
        self._local.pop(slug, None)

    # ── Background sync interface (called by VariableManager poll loop) ────────

    async def wildcard_fetch(self, path: str) -> dict[str, Any]:
        """Fetch bridge variables matching *path* — called by the VariableManager loop."""
        return await wildcard_fetch(self._http, self._run_id, path, variables=self._variables)

    async def push_dirty(self) -> None:
        """Push all dirty variables — called by the VariableManager background loop."""
        await push_dirty(self._http, self._run_id, self._variables)

    # ── After-pull hook (called by VariableManager background loop) ────────────

    async def _after_pull(self, raw: dict[str, Any]) -> None:
        prefix = f"{self._name}/"
        stripped = {k.removeprefix(prefix): v for k, v in raw.items() if k.startswith(prefix)}
        if stripped:
            self._local = _unflatten(stripped)
        if self._on_pull is not None:
            res = self._on_pull(self)
            if inspect.isawaitable(res):
                await res

    # ── Dict-level access ─────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> _Node:
        return _Node(self, (key,))

    def __contains__(self, key: str) -> bool:
        return key in self._local

    def __len__(self) -> int:
        return len(self._local)

    def __iter__(self) -> Iterator:
        return iter(self._local)

    def __repr__(self) -> str:
        return f"SyncedDictionary({self._name!r}, {self._local!r})"

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached value for *key*, or *default* if absent."""
        return self._local.get(key, default)

    def keys(self):
        return self._local.keys()

    def values(self):
        return self._local.values()

    def items(self):
        return self._local.items()

    # ── Snapshot / raw access ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a deep copy of the local cache as a plain nested Python dict."""
        return copy.deepcopy(self._local)

    def var(self, path: str) -> SyncedVariableV2:
        """Return the underlying ``SyncedVariableV2`` for a flat leaf path.

        Example: ``d.var("config/beta1")`` returns the var for ``{name}/config/beta1``.
        """
        full_name = f"{self._name}/{path}"
        if full_name not in self._variables:
            raise KeyError(f"{full_name!r} is not registered; call register() first")
        return self._variables[full_name]
