"""Example 04: Structured optimizer state with SyncedCollection.

SyncedCollection is a static utility class — all methods take an HTTP client
and run_id explicitly.  The caller owns the *variables* dict that tracks
registered :class:`~miner.sync_v2.synced_variable.SyncedVariableV2` objects.

Each field is its own bridge variable under a path prefix, e.g.
``{run_id}/optimizer_state/step``.  Register with :meth:`register_many`, load
with :meth:`wildcard_fetch` or :meth:`fetch_all`, write subsets with
:meth:`set_many`.

When to split a dict into per-field variables instead of one ``dict`` variable:
  - Fields have different bridge types (int, float, str).
  - You want partial updates without rewriting the whole blob.
  - You want per-field or multi-field locking via :class:`~miner.sync_v2.synced_variable.Lock`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from miner.sync_v2.synced_collection import SyncedCollection
from miner.sync_v2.synced_variable import SyncedVariableV2
from miner.sync_v2.variable_manager import VariableManager, get_bridge_url

RUN_ID = "demo-run"
PREFIX = "optimizer_state"


def strip_prefix(flat: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Turn ``{"optimizer_state/step": 1}`` into ``{"step": 1}`` for ergonomics."""
    p = prefix + "/"
    return {k.removeprefix(p): v for k, v in flat.items() if k.startswith(p)}


async def main() -> None:
    manager = VariableManager(url=get_bridge_url())
    client = manager._http

    # Each caller owns a variables dict — SyncedCollection populates it.
    variables: dict[str, SyncedVariableV2] = {}

    await SyncedCollection.register_many(
        client,
        RUN_ID,
        [
            {"name": f"{PREFIX}/step", "var_type": "int", "default_value": 0},
            {"name": f"{PREFIX}/loss", "var_type": "float", "default_value": 1e9},
            {"name": f"{PREFIX}/lr", "var_type": "float", "default_value": 1e-3},
            {"name": f"{PREFIX}/grad_norm", "var_type": "float", "default_value": 0.0},
        ],
        variables=variables,
    )
    print("registered all fields")

    raw = await SyncedCollection.wildcard_fetch(client, RUN_ID, f"{PREFIX}/*", variables=variables)
    values = strip_prefix(raw, PREFIX)
    print(f"fetched: {values}")

    step = values["step"] + 1
    new_loss = values["loss"] * 0.95

    await SyncedCollection.set_many(
        client,
        RUN_ID,
        variables,
        {
            f"{PREFIX}/step": step,
            f"{PREFIX}/loss": new_loss,
        },
    )

    full = strip_prefix(await SyncedCollection.fetch_all(client, RUN_ID, variables), PREFIX)
    print(f"after step {step}: {full}")

    full["extra"] = True
    cached_again = strip_prefix({n: SyncedCollection.get_cached(n, variables) for n in variables}, PREFIX)
    assert "extra" not in cached_again, "mutation did not leak into cache"

    try:
        await SyncedCollection.set_many(client, RUN_ID, variables, {f"{PREFIX}/nonexistent": 1})
    except KeyError as exc:
        print(f"expected error: {exc}")

    await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
