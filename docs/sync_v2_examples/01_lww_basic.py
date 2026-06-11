"""Example 01: LWW (Last-Write-Wins) — simplest use case.

A miner publishes its latest evaluation score after each forward pass.
There is no ordering constraint: whichever write arrives at the bridge last wins.

Run with:
    python -m docs.sync_v2_examples.01_lww_basic
(from the repo root, with BRIDGE_URL set in the environment)
"""

import asyncio

from miner.sync_v2.variable_manager import VariableManager, get_bridge_url


RUN_ID = "demo-run"


async def main() -> None:
    manager = VariableManager(url=get_bridge_url())

    score = await manager.create_var(run_id=RUN_ID, name="eval_score", var_type="float", default=0.0, write_rule="LWW")

    # fetch_value reads the current value from the bridge.
    current = await score.fetch_value()
    print(f"fetched value: {current}  (version {score.version})")

    # Simulate a training step that produces a new score.
    new_score = 0.87
    await score.set_value(new_score)
    print(f"wrote {new_score}  — now at version {score.version}")

    # The local cache is updated synchronously after set_value.
    # get_cached_value() never hits the network (synchronous — no await).
    cached = score.get_cached_value()
    assert cached == new_score, "cache should reflect the last write"
    print(f"cached value: {cached}")

    await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
