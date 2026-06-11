"""Example 02: CAS (Compare-And-Swap) — optimistic concurrency.

Two miners race to update a shared "best loss" tracker.  Only the one whose
version matches the current server version succeeds; the other must re-fetch
and retry.  This prevents a stale write from overwriting a fresher value.

CAS is the right pattern when:
  - Multiple writers may compete for the same variable.
  - You must never overwrite a newer value with an older one.
  - You are OK with a retry loop on conflict.
"""

import asyncio
import random

from miner.sync_v2.variable_manager import VariableManager, get_bridge_url


RUN_ID = "demo-run"
MAX_RETRIES = 5


async def try_update_best_loss(miner_id: str, candidate_loss: float) -> bool:
    """Try to update best_loss if candidate is lower.  Returns True on success."""
    manager = VariableManager(url=get_bridge_url())
    var = await manager.create_var(run_id=RUN_ID, name="best_loss", var_type="float", default=1e9, write_rule="CAS")

    try:
        for attempt in range(MAX_RETRIES):
            current_loss = await var.fetch_value()

            if candidate_loss >= current_loss:
                print(f"[{miner_id}] candidate {candidate_loss:.4f} is not better than {current_loss:.4f}, skipping")
                return False

            try:
                # Pass the version we observed — bridge rejects if version changed.
                await var.set_value(candidate_loss, current_version=var.version)
                print(f"[{miner_id}] updated best_loss → {candidate_loss:.4f} (attempt {attempt + 1})")
                return True
            except RuntimeError as exc:
                if "VersionConflict" in str(exc) or "version" in str(exc).lower():
                    # Another writer raced us — back off and retry.
                    backoff = 0.05 * (2**attempt) + random.uniform(0, 0.02)
                    print(f"[{miner_id}] version conflict on attempt {attempt + 1}, retrying in {backoff:.2f}s …")
                    await asyncio.sleep(backoff)
                    continue
                raise  # unexpected error — propagate

        print(f"[{miner_id}] gave up after {MAX_RETRIES} retries")
        return False
    finally:
        await manager.stop()


async def main() -> None:
    # Two miners running concurrently, each with a different candidate loss.
    miner_a = try_update_best_loss("miner-A", candidate_loss=0.312)
    miner_b = try_update_best_loss("miner-B", candidate_loss=0.287)
    results = await asyncio.gather(miner_a, miner_b, return_exceptions=True)
    print(f"results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
