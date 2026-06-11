"""Example 07: SyncedDictionary — nested dict-like wrapper over a SyncedCollection.

Each leaf key is its own bridge variable.  All mutations go through .set();
reads through .get() or .value.  Assignment with = is not supported.

    d["step"].set(1)                    # update a leaf (sync, marks dirty)
    d["config"]["beta1"].set(0.9)       # nested leaf (sync)
    d["config"]["new_key"].set("hello") # adds new key; registered on next push()
    await d["config"]["old_key"].delete()  # remove from bridge (async)

Use SyncedDictionary when you want:
  - Python dict-like syntax for shared distributed state.
  - Per-leaf bridge variables (versioning, partial updates, per-key locking).
  - Background push/pull without managing the poll loop yourself.
"""

import asyncio

from miner.sync_v2.variable_manager import VariableManager, get_bridge_url

RUN_ID = "demo-run"


# ── A. Basic LWW usage ────────────────────────────────────────────────────────


async def section_a_lww_basic() -> None:
    """Register a schema, set values, push to bridge, pull back."""
    print("\n── A. LWW basic ──")
    manager = VariableManager(url=get_bridge_url())

    d = manager.synced_dict(run_id=RUN_ID, name="optimizer_a", rule="LWW")
    await d.register(
        {
            "step": 0,
            "lr": 1e-3,
            "config": {"beta1": 0.9, "beta2": 0.999},
        }
    )

    d["step"].set(1)
    d["lr"].set(1e-4)
    d["config"]["beta1"].set(0.85)
    d["config"].set({"beta2": 0.998})  # update entire sub-tree

    await d.push()
    print(f"  pushed: {d.to_dict()}")

    await d.pull()
    assert d["step"].value == 1
    assert d["config"]["beta1"].value == 0.85
    print(f"  pulled: step={d['step'].get()}, beta1={d['config']['beta1'].get()}")

    await manager.stop()


# ── B. Dynamic key add / remove ───────────────────────────────────────────────


async def section_b_dynamic_keys() -> None:
    """Add new keys via .set() (lazy registration on push) and delete via .delete()."""
    print("\n── B. Dynamic keys ──")
    manager = VariableManager(url=get_bridge_url())

    d = manager.synced_dict(run_id=RUN_ID, name="optimizer_b", rule="LWW")
    await d.register({"step": 0})

    # New keys — not yet on the bridge; registered lazily on push()
    d["config"]["warmup_steps"].set(100)
    d["config"]["scheduler"].set("cosine")
    d["grad_clip"].set(1.0)

    await d.push()  # registers + sets all three new leaves
    print(f"  after add: {d.to_dict()}")

    await d.pull()
    assert d["grad_clip"].value == 1.0
    assert d["config"]["warmup_steps"].value == 100

    # Remove keys from bridge (async)
    await d["config"]["scheduler"].delete()
    await d["grad_clip"].delete()

    await d.pull()
    print(f"  after remove: {d.to_dict()}")
    assert "grad_clip" not in d
    assert "scheduler" not in d["config"]

    await manager.stop()


# ── C. Background sync ────────────────────────────────────────────────────────


async def section_c_background_sync() -> None:
    """Background push/pull: .set() marks dirty; manager flushes within push_frequency seconds."""
    print("\n── C. Background sync ──")
    manager = VariableManager(url=get_bridge_url())

    d = manager.synced_dict(
        run_id=RUN_ID,
        name="optimizer_c",
        rule="LWW",
        pull_frequency=5.0,
        push_frequency=2.0,
    )
    await d.register({"step": 0, "loss": 1.0})

    for step in range(3):
        d["step"].set(step)
        d["loss"].set(round(1.0 - step * 0.1, 2))
        print(f"  step {step}: local={d.to_dict()}")
        await asyncio.sleep(0.05)

    await asyncio.sleep(2.5)  # give background loop time to flush
    print("  background push completed")

    await manager.stop()


# ── D. LOCK atomic read-modify-write ──────────────────────────────────────────


async def section_d_lock_atomic() -> None:
    """LOCK rule: .set() is local-only; must push(lock=lock) to write to bridge."""
    print("\n── D. LOCK atomic update ──")
    manager = VariableManager(url=get_bridge_url())

    d = manager.synced_dict(run_id=RUN_ID, name="optimizer_d", rule="LOCK")
    await d.register({"lr": 1e-3, "momentum": 0.9})

    async with manager.lock(d.var("lr"), ttl_ms=10_000, wait_ms=2_000) as lock:
        await d.pull()
        d["lr"].set(d["lr"].value * 0.9)  # local only — LOCK skips dirty marking
        await d.push(lock=lock)

    print(f"  after lock update: {d.to_dict()}")
    assert abs(d["lr"].value - 9e-4) < 1e-9

    await manager.stop()


# ── E. CAS optimistic retry ───────────────────────────────────────────────────


async def section_e_cas_update() -> None:
    """CAS rule: pull() fetches current version; push() uses it; retry on conflict."""
    print("\n── E. CAS optimistic ──")
    manager = VariableManager(url=get_bridge_url())

    d = manager.synced_dict(run_id=RUN_ID, name="metrics_e", rule="CAS")
    await d.register({"best_loss": 1.0, "best_step": 0})

    new_loss = 0.42
    for attempt in range(5):
        await d.pull()
        if new_loss < d["best_loss"].value:
            d["best_loss"].set(new_loss)
            d["best_step"].set(10)
            try:
                await d.push()  # uses versions stored during pull()
                print(f"  CAS ok on attempt {attempt + 1}: {d.to_dict()}")
                break
            except RuntimeError as exc:
                print(f"  CAS conflict: {exc} — retrying")
                await asyncio.sleep(0.05 * 2**attempt)
        else:
            print("  no improvement, skipping")
            break

    await manager.stop()


# ── F. Edge cases ─────────────────────────────────────────────────────────────


async def section_f_edge_cases() -> None:
    """Pre-register buffering, absent-key reads, namespace iteration, to_dict copy."""
    print("\n── F. Edge cases ──")
    manager = VariableManager(url=get_bridge_url())

    d = manager.synced_dict(run_id=RUN_ID, name="edge_f")

    # .set() before register() buffers locally; bridge registration happens on push()
    d["step"].set(0)
    d["config"]["beta1"].set(0.9)
    assert d["step"].get() == 0
    assert d["config"]["beta1"].get() == 0.9
    print(f"  pre-register local: {d.to_dict()}")

    # Absent keys return None from .get() (with optional default)
    assert d["nonexistent"].get() is None
    assert d["nonexistent"].get(default=99) == 99

    # Chained proxy for deeply nested absent paths
    assert d["a"]["b"]["c"].get() is None

    # Namespace iteration via _Node
    assert "beta1" in d["config"]
    assert len(d["config"]) == 1
    assert list(d["config"].keys()) == ["beta1"]

    # register() + push() flushes the pre-set state
    await d.register({"step": 0, "config": {"beta1": 0.9}})
    d["step"].set(1)
    await d.push()

    await d.pull()
    assert d["step"].value == 1

    # to_dict() is a deep copy — mutations don't affect local cache
    snapshot = d.to_dict()
    snapshot["step"] = 999
    assert d["step"].value == 1

    await manager.stop()
    print("  edge cases ok")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await section_a_lww_basic()
    await section_b_dynamic_keys()
    await section_c_background_sync()
    await section_d_lock_atomic()
    await section_e_cas_update()
    await section_f_edge_cases()
    print("\nAll SyncedDictionary examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
