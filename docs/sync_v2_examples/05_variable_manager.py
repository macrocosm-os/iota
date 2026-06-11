"""Example 05: VariableManager — background push/pull.

VariableManager runs an asyncio task that:
  1. Pulls any variable whose last fetch is older than pull_frequency seconds.
  2. Pushes any variable that has been set() and whose last push is
     older than push_frequency seconds.

Your training loop only calls set(new_value) — no await needed.
The manager batches all pulls and pushes into single HTTP calls per cycle.

LOCK variables are silently skipped during background push.  Use set_value()
directly while holding the lock for those.
"""

import asyncio

from miner.sync_v2.variable_manager import VariableManager, get_bridge_url


RUN_ID = "demo-run"


async def main() -> None:
    manager = VariableManager(url=get_bridge_url())

    # Pull score every 5 s; push any dirty score within 2 s.
    score = await manager.create_var(
        run_id=RUN_ID,
        name="score",
        var_type="float",
        default=None,
        write_rule="LWW",
        require_fetch=False,
        pull_frequency=5.0,
        push_frequency=2.0,
    )
    # Pull status every 10 s; push dirty status within 1 s.
    status = await manager.create_var(
        run_id=RUN_ID,
        name="status",
        var_type="str",
        default=None,
        write_rule="LWW",
        require_fetch=False,
        pull_frequency=10.0,
        push_frequency=1.0,
    )

    # The polling loop interval is the minimum across all registered frequencies.
    print(f"poll interval: {manager.get_min_interval()} s")  # 1.0 s
    print("background polling started")

    # ── Training loop ──────────────────────────────────────────────────────────
    for step in range(3):
        await asyncio.sleep(0.1)  # simulate forward pass

        # Synchronous — updates local cache and sets _needs_push flag.
        score.set(round(0.5 - step * 0.1, 2))
        status.set(f"step-{step}")

        # Immediate local read — no network call.
        local_score = score.get_cached_value()
        local_status = status.get_cached_value()
        print(f"step {step}: local score={local_score}, status={local_status!r}")

    # Give the background loop a moment to flush.
    await asyncio.sleep(2.5)

    # ── Shutdown ───────────────────────────────────────────────────────────────
    # stop() cancels the task and closes the HTTP client.
    await manager.stop()
    print("stopped")


if __name__ == "__main__":
    asyncio.run(main())
