"""Example 03: LOCK — exclusive read-modify-write.

A single miner holds the bridge lock, reads the config, patches it, and
writes it back.  No other writer can touch the variable until the lock is
released.

Use LOCK when:
  - You need a read-modify-write that must appear atomic to other readers.
  - The value is complex (e.g. a nested dict) and partial-update semantics
    (like CAS field-by-field) would be fragile.
  - Contention is low enough that serialised writes are acceptable.

Lock parameters:
  - ttl_ms  — the bridge auto-releases the lock after this many milliseconds,
               preventing deadlocks if the holder crashes mid-operation.
  - wait_ms — how long to wait for the lock to become free before giving up.
               RuntimeError is raised if the lock cannot be acquired in time.
"""

import asyncio

from miner.sync_v2.variable_manager import VariableManager, get_bridge_url


RUN_ID = "demo-run"


async def patch_hyperparams(new_lr: float) -> None:
    manager = VariableManager(url=get_bridge_url())
    config = await manager.create_var(
        run_id=RUN_ID,
        name="hyperparams",
        var_type="dict",
        default={"lr": 1e-3, "warmup_steps": 100, "batch_size": 32},
        write_rule="LOCK",
    )

    try:
        # Acquire the lock.  wait_ms=2000 means we give up after 2 seconds.
        async with manager.lock(config, ttl_ms=10_000, wait_ms=2_000) as lock:
            # Fetch the current value while holding the lock.
            current = await config.fetch_value()
            print(f"current config: {current}")

            # Modify and write back — must pass the lock.
            updated = {**current, "lr": new_lr}
            await config.set_value(updated, lock=lock)
            print(f"updated config: {updated}")
        # Lock is automatically released here even if set_value raised.

    except RuntimeError as exc:
        # Could not acquire the lock in 2 s — another holder is stuck or slow.
        print(f"Could not acquire lock: {exc}")
    finally:
        await manager.stop()


async def main() -> None:
    await patch_hyperparams(new_lr=5e-4)


if __name__ == "__main__":
    asyncio.run(main())
