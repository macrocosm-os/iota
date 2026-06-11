# Bridge v2 Sync Client (`sync_v2`)

The `sync_v2` module lets miners read and write shared state through the bridge's `/bridge` HTTP
API.  It replaces the older `/sync` polling protocol with an explicit register/get/set/lock model.

---

## Table of contents

1. [Concepts](#concepts)
2. [SyncedVariableV2 — single variable](#syncedvariablev2--single-variable)
   - [LWW — last write wins](#lww--last-write-wins)
   - [CAS — compare and swap](#cas--compare-and-swap)
   - [LOCK — exclusive write](#lock--exclusive-write)
3. [SyncedCollection — path-indexed groups](#synccollection--path-indexed-groups)
4. [VariableManager — background polling](#variablemanager--background-polling)
5. [Edge cases](#edge-cases)
   - [require_fetch guard](#require_fetch-guard)
   - [set + VariableManager](#set--variablemanager)
   - [CAS conflict retry loop](#cas-conflict-retry-loop)
   - [Lock timeout and partial acquire cleanup](#lock-timeout-and-partial-acquire-cleanup)
   - [Multi-variable lock with SyncedCollection](#multi-variable-lock-with-synccollection)
   - [LOCK variables are skipped by VariableManager push](#lock-variables-are-skipped-by-variablemanager-push)
   - [Empty or partial fetch response](#empty-or-partial-fetch-response)
   - [Idempotent registration](#idempotent-registration)

---

## Concepts

| Term | Meaning |
|---|---|
| `var_id` | Globally unique bridge key: `{run_id}/{name}` |
| `write_rule` | Concurrency policy for writes: `LWW`, `CAS`, or `LOCK` |
| `version` | Monotonically increasing integer bumped on every successful write |
| `lock_token` | Opaque string issued by the bridge; must accompany writes when `write_rule=LOCK` |

**Write rules at a glance:**

- **LWW** (Last-Write-Wins) — no ordering guarantee; whichever write arrives last wins.
  Use for metrics, heartbeats, and anything where a slightly stale overwrite is fine.
- **CAS** (Compare-And-Swap) — you supply the version you last saw; the bridge rejects the write
  if the version has already moved on.  Use for counters or ranked scores where you must not
  overwrite a newer value with an older one.
- **LOCK** — you must hold the distributed lock before writing.  Only one holder at a time.
  Use for configuration blobs or any state that must be read-modify-written atomically.

---

## SyncedVariableV2 — single variable

```python
from miner.sync_v2.synced_variable import SyncedVariableV2
```

Construction requires ``default`` (the value sent in the first ``POST /register``). Use
:meth:`~SyncedVariableV2.create` to construct and register in one step; registration failures
raise immediately. :meth:`~SyncedVariableV2.register` remains available for idempotent
re-registration.

### LWW — last write wins

```python
async def lww_example(run_id: str) -> None:
    score = await SyncedVariableV2.create(
        run_id=run_id,
        name="miner_score",
        var_type="float",
        write_rule="LWW",
        default=0.0,
    )

    # Read the current value from the bridge.
    value = await score.fetch_value()
    print(f"score = {value}")   # e.g. 0.0

    # Write unconditionally.
    await score.set_value(0.87)

    # Local cache reflects the last fetch or set — no network call.
    cached = score.get_cached_value()
    print(f"cached score = {cached}")  # 0.87
```

### CAS — compare and swap

CAS writes are rejected by the bridge if `current_version` does not match the server's version.
Always supply `current_version=var.version` (the version you last fetched or successfully wrote).

```python
async def cas_example(run_id: str) -> None:
    best = await SyncedVariableV2.create(
        run_id=run_id,
        name="best_loss",
        var_type="float",
        write_rule="CAS",
        default=1e9,
    )

    # Fetch before a CAS write to get the current version.
    current = await best.fetch_value()

    new_loss = 0.042
    if new_loss < current:
        # Pass the version we observed — the bridge rejects stale writes.
        await best.set_value(new_loss, current_version=best.version)
```

### LOCK — exclusive write

A `Lock` is an async context manager.  The bridge blocks concurrent holders up to `wait_ms`
before raising `RuntimeError`.

```python
from miner.sync_v2.synced_variable import Lock, SyncedVariableV2

async def lock_example(run_id: str) -> None:
    config = await SyncedVariableV2.create(
        run_id=run_id,
        name="hyperparams",
        var_type="dict",
        write_rule="LOCK",
        default={"lr": 1e-3, "warmup_steps": 100},
    )

    async with Lock(config) as lock:
        current = await config.fetch_value()
        updated = {**current, "lr": 5e-4}
        await config.set_value(updated, lock=lock)
```

---

## SyncedCollection — path-indexed groups

`SyncedCollection` manages **multiple** bridge variables that share one `httpx.AsyncClient` and are
addressed by **slash-separated names** relative to `run_id`. The full bridge key for a tracked
variable is always `{run_id}/{name}` where `name` is whatever string you passed to `add()` or
`register_many()` (for example `node_registry/miner1/iroh_nodes`).

Typical uses: ad-hoc registries, wildcard listing with `wildcard_fetch` / `wildcard_delete`, and
batch APIs (`register_many`, `set_many`, `fetch_all`). The same APIs cover structured “typed dict”
workloads when each logical field is its own tracked path under a shared prefix (see below).

**Miners — shared node registry:** Production miners use ``CASNodeRegistry`` (``miner.sync_v2.cas_node_registry``),
which wraps ``SyncedDictionary`` with ``write_rule="CAS"``. Each miner writes only
``{run_id}/node_registry/{slug(node_id)}/…`` (one leaf per ComputeNode field). Periodically the client runs
``wildcard_fetch("node_registry/*")`` and merges each payload's ``node_id`` field into the local
``NodeRegistry`` used for P2P routing (same logical map as the legacy single-variable JSON-patch
registry, without whole-dict writes).

```python
from miner.sync_v2.synced_collection import SyncedCollection

async def collection_example(run_id: str) -> None:
    col = SyncedCollection(run_id=run_id, name="node_registry", write_rule="LWW")

    await col.add("node_registry/miner1/iroh_nodes", default_value={}, var_type="dict")
    await col.set_value("node_registry/miner1/iroh_nodes", {"addr": "..."})

    all_nodes = await col.wildcard_fetch("node_registry/*/iroh_nodes")
    # {"node_registry/miner1/iroh_nodes": {...}, ...}

    await col.delete("node_registry/miner1/iroh_nodes")
```

**Batch helpers**

| Method | Role |
|--------|------|
| `register_many(entries)` | One `POST /register` body with many variables; on success, seeds each variable's local cache from `default_value`. |
| `set_many(values, lock=..., current_versions=...)` | One `POST /set` for many tracked names (LOCK tokens and CAS versions per variable as in single-var writes). |
| `fetch_all()` | One `POST /get` for every **tracked** variable. |
| `wildcard_fetch(path)` | `GET /vars?prefix=...` to list ids, then `POST /get` for matches. Updates cache for variables already tracked; others appear only in the returned dict. |

The `name` field on the `SyncedCollection` model is informational/metadata for your app; tracked
keys are the string names stored in `_variables` (shown in `col.names`).

### Structured multi-field state

Give each logical field its own tracked name such as ``optimizer_state/step`` and ``optimizer_state/lr``.
Use ``register_many`` for one batched ``POST /register``, ``wildcard_fetch("optimizer_state/*")`` or
``fetch_all()`` for reads (when those paths are exactly what you track), and ``set_many`` for writes.

```python
from miner.sync_v2.synced_collection import SyncedCollection

async def optimizer_state_example(run_id: str) -> None:
    prefix = "optimizer_state"
    col = SyncedCollection(run_id=run_id, name="optimizer", write_rule="LWW")

    await col.register_many(
        [
            {"name": f"{prefix}/step", "var_type": "int", "default_value": 0},
            {"name": f"{prefix}/loss", "var_type": "float", "default_value": 1e9},
            {"name": f"{prefix}/lr", "var_type": "float", "default_value": 1e-3},
        ]
    )

    await col.wildcard_fetch(f"{prefix}/*")

    await col.set_many({f"{prefix}/step": 1, f"{prefix}/loss": 0.73})

    flat = await col.fetch_all()
    lr_only = flat[f"{prefix}/lr"]
```

Nested structure is just longer paths, e.g. ``optimizer_state/metrics/loss`` as a single bridge variable.

### Partial updates

``set_many`` only sends the keys you include; other tracked variables are untouched on the bridge and keep their cached values locally.

### Multi-variable locking

Collect :class:`~miner.sync_v2.synced_variable.SyncedVariableV2` instances or ``var_id`` strings and
pass them to :class:`~miner.sync_v2.synced_variable.Lock`. When passing variables, the HTTP client
is taken from them; when passing ids only, supply ``client`` explicitly.
Pass the same ``lock`` into ``set_many`` so per-variable tokens attach to each update:

```python
from miner.sync_v2.synced_variable import Lock

async with Lock([col.variable("opt_state/step"), col.variable("opt_state/lr")]) as lock:
    flat = await col.fetch_all()
    await col.set_many(
        {
            "opt_state/step": flat["opt_state/step"] + 1,
            "opt_state/lr": flat["opt_state/lr"] * 0.99,
        },
        lock=lock,
    )
```

Or with raw ids:

```python
var_ids = [col.variable(n).var_id for n in ("opt_state/step", "opt_state/lr")]
async with Lock(var_ids, col._client) as lock:
    flat = await col.fetch_all()
    await col.set_many(
        {
            "opt_state/step": flat["opt_state/step"] + 1,
            "opt_state/lr": flat["opt_state/lr"] * 0.99,
        },
        lock=lock,
    )
```

Lock only the names you need when you want concurrent writers on different fields.

### CAS batch writes

Each variable in ``set_many`` compares versions independently. Override versions with the ``current_versions`` mapping when your snapshot predates the cached versions.

If you need to enforce “all keys present” for one logical snapshot, validate ``updates.keys()`` in application code before calling ``set_many``.

> **Bridge semantics:** `/set` applies each update separately in order. If you need multiple CAS
> fields to succeed or fail together as one atomic transaction, that requires backend support —
> this client batch is still separate compares per variable.

---

## VariableManager — background polling

`VariableManager` runs a background loop that automatically pulls stale variables and pushes
dirty ones at configurable intervals.  Your code only calls `set` to queue a write;
the manager handles batching and timing.

```python
from miner.sync_v2.synced_variable import SyncedVariableV2
from miner.sync_v2.variable_manager import VariableManager

async def manager_example(run_id: str) -> None:
    score = await SyncedVariableV2.create(
        run_id=run_id,
        name="score",
        var_type="float",
        write_rule="LWW",
        default=0.0,
    )

    manager = VariableManager()
    await manager.setup_background_polling(
        score,
        pull_frequency=5.0,   # refresh from bridge every 5 s
        push_frequency=2.0,   # flush dirty writes every 2 s
    )
    await manager.start()

    # Somewhere in your training loop — no await needed.
    score.set(0.91)

    # The background loop will push 0.91 within the next push_frequency seconds.
    # Read locally without a network call.
    cached = score.get_cached_value()
    print(f"local value = {cached}")  # 0.91 immediately

    # Shutdown: flushes pending work and closes the HTTP client.
    await manager.stop()
```

### Multiple variables

```python
async def multi_var_manager(run_id: str) -> None:
    score = await SyncedVariableV2.create(
        run_id=run_id,
        name="score",
        var_type="float",
        write_rule="LWW",
        default=None,
    )
    state = await SyncedVariableV2.create(
        run_id=run_id,
        name="status",
        var_type="str",
        write_rule="LWW",
        default=None,
    )

    manager = VariableManager()
    await manager.setup_background_polling(score,  pull_frequency=5.0, push_frequency=2.0)
    await manager.setup_background_polling(state, pull_frequency=10.0, push_frequency=1.0)
    await manager.start()

    # The manager's loop interval is the minimum across all registered frequencies (1.0 s here).
    print(manager.get_min_interval())  # 1.0
```

### ``SyncedCollection`` (wildcard registry)

The same ``VariableManager`` can poll a ``SyncedCollection``:

- Pass ``wildcard_path="foo/*"`` to run ``wildcard_fetch`` on each pull (when variable names are not known up-front). Use ``wildcard_path=None`` to call ``fetch_all()`` for **tracked** names only.
- Pass an optional ``after_pull(collection, raw_dict)`` callback (sync or async) to merge ``raw_dict`` into app state.
- Set ``push_frequency=None`` to disable background push for that collection. Otherwise the loop calls ``push_dirty_tracked()``, which batches tracked variables queued with ``set``.

Unregister with ``manager.unregister_collection(key)`` when the collection is torn down.

```python
manager = VariableManager()
manager.setup_background_polling_collection(
    col,
    key="node_registry_ns",
    wildcard_path="node_registry/*",
    after_pull=my_merge_fn,
    pull_frequency=2.0,
    push_frequency=None,
)
await manager.start()
```

---

## Edge cases

### require_fetch guard

By default `get_cached_value()` reads the local cache synchronously (no `await`) and returns whatever was last written or the default (`None` if you
never called `fetch` or `register`).  Setting `require_fetch=True` makes the variable refuse to serve
cached values until at least one network fetch has completed — useful when a stale default
would be silently dangerous.

```python
async def require_fetch_example(run_id: str) -> None:
    var = await SyncedVariableV2.create(
        run_id=run_id,
        name="weights_hash",
        var_type="str",
        require_fetch=True,
        default="",
    )

    # Before any fetch — raises RuntimeError.
    try:
        val = var.get_cached_value()
    except RuntimeError as e:
        print(e)  # "Must call fetch_value() before get_cached_value() ..."

    # After a successful fetch — returns the real value.
    await var.fetch_value()
    val = var.get_cached_value()  # safe
    print(val)
```

`SyncedCollection` passes ``require_fetch`` through to every tracked variable from ``add()`` or ``register_many()``:

```python
from miner.sync_v2.synced_collection import SyncedCollection

col = SyncedCollection(run_id=run_id, name="training", require_fetch=True)
await col.register_many(
    [{"name": "training_state/step", "var_type": "int", "default_value": 0}]
)

try:
    col.get_cached("training_state/step")
except RuntimeError:
    pass

await col.fetch_all()
assert col.get_cached("training_state/step") == 0
```

---

### set + VariableManager

`set` updates the local cache **and** sets an internal flag so the `VariableManager`
loop will push the value on its next cycle.  It is purely local and synchronous — no `await`.

```python
# Fast path: write locally, let the background loop handle the network call.
score.set(0.99)

# Immediate local read — returns 0.99 without any network call.
local = score.get_cached_value()  # 0.99

# Once the manager loop fires, _needs_push is cleared and version is bumped.
```

> **Important:** `set` queues a value for the next push. For standalone variables,
> call `set_and_push()` or `set_value()` to write to the bridge immediately.

> **Important:** Variables with `write_rule="LOCK"` are **never** auto-pushed by the manager.
> You must call `set_value(value, lock=lock)` explicitly while holding the lock.

---

### CAS conflict retry loop

When two writers race, one CAS write will fail with a `RuntimeError` (the bridge rejects the
write because the version moved on).  The correct pattern is to re-fetch and retry.

```python
import asyncio

async def cas_retry(run_id: str, new_value: float, max_retries: int = 5) -> None:
    var = await SyncedVariableV2.create(
        run_id=run_id,
        name="best_score",
        var_type="float",
        write_rule="CAS",
        default=0.0,
    )

    for attempt in range(max_retries):
        current = await var.fetch_value()   # refresh version
        try:
            await var.set_value(new_value, current_version=var.version)
            return  # success
        except RuntimeError as exc:
            if "VersionConflict" in str(exc) and attempt < max_retries - 1:
                await asyncio.sleep(0.1 * (2 ** attempt))  # back off
                continue
            raise
```

---

### Lock timeout and partial acquire cleanup

`Lock.__aenter__` requests all var_ids in a single call.  If any lock comes back with status
`timeout` (i.e. another holder didn't release within `wait_ms`), the `Lock` **automatically
releases** whichever locks it did acquire before raising `RuntimeError`.  You never need to
clean up manually.

```python
async def lock_timeout_example(run_id: str) -> None:
    var = await SyncedVariableV2.create(
        run_id=run_id,
        name="exclusive_config",
        var_type="dict",
        write_rule="LOCK",
        default={},
    )

    try:
        # wait_ms=500 — give up after 500 ms if someone else holds it.
        async with Lock(var, ttl_ms=10_000, wait_ms=500) as lock:
            current = await var.fetch_value()
            await var.set_value({**current, "patched": True}, lock=lock)
    except RuntimeError as exc:
        # "Could not acquire lock for 'run/exclusive_config': timeout"
        print(f"Lock unavailable: {exc}")
        # Partial locks (if any) were already released by __aenter__.
```

---

### Multi-variable lock with SyncedCollection

Build ``Lock`` from the :class:`~miner.sync_v2.synced_variable.SyncedVariableV2` instances or
``var_id`` list for every name you need (all tracked names for a full “dict” lock,
or a subset). ``set_many(..., lock=lock)`` attaches per-variable tokens the same way ``SyncedVariableV2.set_value`` does.

```python
from miner.sync_v2.synced_collection import SyncedCollection
from miner.sync_v2.synced_variable import Lock

async def dict_lock_example(run_id: str) -> None:
    col = SyncedCollection(run_id=run_id, name="opt", write_rule="LOCK")
    await col.register_many(
        [
            {"name": "opt_state/step", "var_type": "int", "default_value": 0},
            {"name": "opt_state/lr", "var_type": "float", "default_value": 1e-3},
        ]
    )

    async with Lock([col.variable(n) for n in col.names]) as lock:
        flat = await col.fetch_all()
        await col.set_many(
            {
                "opt_state/step": flat["opt_state/step"] + 1,
                "opt_state/lr": flat["opt_state/lr"] * 0.99,
            },
            lock=lock,
        )
```

While holding that lock you can also call ``col.variable("opt_state/step").set_value(...)`` and pass the same ``lock`` if you need the single-variable API.

---

### LOCK variables are skipped by VariableManager push

`VariableManager._batch_push` intentionally skips any variable with `write_rule="LOCK"` —
a background loop cannot safely acquire a lock on your behalf.  Calling `set` on a LOCK
variable raises `ValueError`; use `set_and_push(..., lock=lock)` or `set_value(...)` while
holding the lock.

```python
# Bad: set on a LOCK variable — no lock is available for the push.
locked_var.set(new_value)   # raises ValueError

# Correct: acquire the lock and write explicitly.
async with Lock(locked_var) as lock:
    await locked_var.set_and_push(new_value, lock=lock)
```

---

### Empty or partial fetch response

When the bridge returns an empty `variables` list (e.g. the variable was never registered or
was deleted), `SyncedVariableV2.fetch_value()` raises `RuntimeError`. `SyncedCollection.wildcard_fetch()`
returns only bridge keys present in the batch response map. Tracked variables missing from that response
keep their prior local caches (typically the registration default).

```python
from miner.sync_v2.synced_collection import SyncedCollection

col = SyncedCollection(run_id=run_id, name="st", write_rule="LWW")
await col.register_many(
    [
        {"name": "state/a", "var_type": "int", "default_value": 0},
        {"name": "state/b", "var_type": "float", "default_value": 0.0},
    ]
)

# Suppose "state/b" was missing from the last wildcard GET payload.
partial = await col.wildcard_fetch("state/*")
# partial may list only whatever the bridge returned (e.g. {"state/a": 7})

# Per-field caches: read each tracked name with col.get_cached("state/b").
b_cached = col.get_cached("state/b")
```

Snapshots you build yourself (dict comprehensions over ``col.names``, etc.) are ordinary Python dicts —
mutating a snapshot does not change the synced caches until you ``set_many``.

---

### Idempotent registration

:meth:`~SyncedVariableV2.create` registers on construction and raises if registration fails.
:meth:`~SyncedVariableV2.register` is idempotent for variables already on the bridge: it updates
the local ``version`` from the bridge response, and for a newly ``created`` variable seeds the
local cache from the given default.

```python
# Preferred: register at construction time.
score = await SyncedVariableV2.create(run_id=run_id, name="score", var_type="float", default=0.0)

# Or register explicitly after constructing (e.g. in tests).
var = SyncedVariableV2(run_id=run_id, name="score", var_type="float", default=0.0)
await var.register(default_value=0.0)
```

The bridge does **not** overwrite an existing stored value when you register again; the default applies only when the variable is created.

> **Note:** `SyncedCollection.register_many([])` is a no-op (no HTTP). `fetch_all()` on an empty
> collection also returns immediately without a request.
