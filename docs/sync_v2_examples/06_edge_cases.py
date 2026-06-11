"""Example 06: Edge cases.

Covers:
  A. require_fetch guard
  B. set on LOCK variable — raises ValueError
  C. CAS conflict — version mismatch, retry loop
  D. Lock timeout — wait_ms exceeded, partial locks auto-released
  E. Multi-variable lock via SyncedCollection + Lock
  F. Partial fetch error — some fields fail, others succeed
  G. Empty register_many — no-op
  H. Lock TTL auto-expiry — crash-safe, no manual cleanup needed
"""

import asyncio

from miner.sync_v2.synced_variable import Lock
from miner.sync_v2.variable_manager import VariableManager, get_bridge_url


RUN_ID = "demo-run"


# ── A. require_fetch guard ─────────────────────────────────────────────────────


async def edge_a_require_fetch() -> None:
    """get_cached_value() raises until at least one fetch_value() has succeeded."""
    print("\n── A. require_fetch ──")
    manager = VariableManager(url=get_bridge_url())
    var = await manager.create_var(
        run_id=RUN_ID,
        name="weights_hash",
        var_type="str",
        default="",
        require_fetch=True,  # <── guard enabled
    )

    # Without a prior fetch — raises.
    try:
        var.get_cached_value()
        assert False, "should have raised"
    except RuntimeError as exc:
        print(f"  blocked before fetch: {exc}")

    # After fetch — allowed.
    await var.fetch_value()
    val = var.get_cached_value()
    print(f"  allowed after fetch: {val!r}")

    await manager.stop()


# ── B. set on LOCK variable — raises ValueError ───────────────────────────────


async def edge_b_lock_set_raises() -> None:
    """set() raises ValueError for LOCK variables."""
    print("\n── B. LOCK + set ──")

    manager = VariableManager(url=get_bridge_url())
    locked_var = await manager.create_var(
        run_id=RUN_ID,
        name="exclusive_cfg",
        var_type="dict",
        write_rule="LOCK",
        default={},
    )

    try:
        locked_var.set({"patched": True})
        assert False, "should have raised"
    except ValueError as exc:
        print(f"  confirmed: set raises for LOCK: {exc}")

    # Correct pattern: hold the lock, then write explicitly.
    async with manager.lock(locked_var, wait_ms=2_000) as lock:
        await locked_var.set_value({"patched": True}, lock=lock)
    assert locked_var._needs_push is False
    print("  explicit set_value with lock: ok")

    await manager.stop()


# ── C. CAS conflict — retry loop ──────────────────────────────────────────────


async def edge_c_cas_conflict_retry() -> None:
    """Demonstrate the retry loop when a CAS write finds the version has moved."""
    print("\n── C. CAS retry ──")

    manager = VariableManager(url=get_bridge_url())
    counter = await manager.create_var(run_id=RUN_ID, name="cas_counter", var_type="int", default=0, write_rule="CAS")

    max_retries = 5
    for attempt in range(max_retries):
        current = await counter.fetch_value()
        try:
            await counter.set_value(current + 1, current_version=counter.version)
            print(f"  incremented to {current + 1} on attempt {attempt + 1}")
            break
        except RuntimeError as exc:
            if attempt < max_retries - 1:
                print(f"  conflict on attempt {attempt + 1}: {exc} — retrying")
                await asyncio.sleep(0.05 * (2**attempt))
            else:
                raise

    await manager.stop()


# ── D. Lock timeout — partial acquire cleanup ──────────────────────────────────


async def edge_d_lock_timeout() -> None:
    """If wait_ms is exceeded, any partially-acquired locks are released automatically."""
    print("\n── D. Lock timeout ──")

    manager = VariableManager(url=get_bridge_url())
    var = await manager.create_var(
        run_id=RUN_ID, name="slow_resource", var_type="str", default="init", write_rule="LOCK"
    )

    # Simulate a very short wait window — likely to time out in a real system
    # under contention.  In this example it may succeed if nothing holds the lock.
    try:
        async with manager.lock(var, ttl_ms=5_000, wait_ms=100) as lock:
            await var.set_value("updated", lock=lock)
            print("  lock acquired and released cleanly")
    except RuntimeError as exc:
        # __aenter__ releases any partial locks before raising — no cleanup needed.
        print(f"  lock timed out (expected under contention): {exc}")

    await manager.stop()


# ── E. Multi-variable lock via SyncedCollection ───────────────────────────────


async def edge_e_multi_var_lock() -> None:
    """Lock every tracked field under a prefix, then batch-set with set_many."""
    print("\n── E. Multi-variable lock ──")

    manager = VariableManager(url=get_bridge_url())
    col = manager.collection(run_id=RUN_ID, name="atomic", rule="LOCK")
    await col.register_many(
        [
            {"name": "atomic_state/step", "var_type": "int", "default_value": 0},
            {"name": "atomic_state/lr", "var_type": "float", "default_value": 1e-3},
        ]
    )
    async with Lock([col.var(n) for n in col.names], ttl_ms=10_000, wait_ms=3_000) as lock:
        assert lock.lock_token is None

        flat = await col.fetch_all()
        print(f"  locked state: {flat}")

        await col.set_many(
            {
                "atomic_state/step": flat["atomic_state/step"] + 1,
                "atomic_state/lr": flat["atomic_state/lr"] * 0.99,
            },
            lock=lock,
        )
        print("  atomic update written")

    await manager.stop()


# ── F. Partial fetch error ─────────────────────────────────────────────────────


async def edge_f_partial_fetch() -> None:
    """Fields that error on fetch keep their last cached value; others update."""
    print("\n── F. Partial fetch error ──")

    manager = VariableManager(url=get_bridge_url())
    col = manager.collection(run_id=RUN_ID, name="partial", rule="LWW")
    await col.register_many(
        [
            {"name": "partial_state/good", "var_type": "int", "default_value": 42},
            {"name": "partial_state/maybe", "var_type": "float", "default_value": 0.0},
        ]
    )

    raw = await col.wildcard_fetch("partial_state/*")
    print(f"  initial fetch: {raw}")

    # If the bridge later returns an error for "maybe" (e.g. it was deleted),
    # wildcard_fetch leaves untracked keys absent while tracked caches retain defaults.
    print("  (partial-error path covered in test_bridge_v2.py)")

    await manager.stop()


# ── G. Empty register_many ────────────────────────────────────────────────────


async def edge_g_empty_dict() -> None:
    """register_many([]) and fetch_all() skip HTTP when nothing is tracked."""
    print("\n── G. Empty collection ──")

    manager = VariableManager(url=get_bridge_url())
    empty = manager.collection(run_id=RUN_ID, name="empty", rule="LWW")
    await empty.register_many([])
    result = await empty.fetch_all()
    assert result == {}
    print("  empty collection: register_many([]) and fetch_all are no-ops")

    await manager.stop()


# ── H. Lock TTL auto-expiry (crash-safe) ──────────────────────────────────────


async def edge_h_ttl_safety() -> None:
    """ttl_ms ensures the bridge auto-releases the lock if the holder crashes."""
    print("\n── H. TTL auto-expiry ──")
    print(
        "  If the holder process crashes after __aenter__ but before __aexit__,\n"
        "  the bridge releases the lock automatically after ttl_ms milliseconds.\n"
        "  Always set ttl_ms to slightly longer than your expected critical section,\n"
        "  and wait_ms to how long other callers should queue before giving up.\n"
        "  Example: ttl_ms=30_000 (30 s), wait_ms=5_000 (5 s)."
    )


# ── Main ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    await edge_a_require_fetch()
    await edge_b_lock_set_raises()
    await edge_c_cas_conflict_retry()
    await edge_d_lock_timeout()
    await edge_e_multi_var_lock()
    await edge_f_partial_fetch()
    await edge_g_empty_dict()
    await edge_h_ttl_safety()
    print("\nAll edge-case examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
