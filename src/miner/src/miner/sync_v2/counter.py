"""SyncedCounter — atomic integer counter backed by bridge v2 (CAS-based)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from miner.sync_v2.synced_variable import SyncedVariableV2

if TYPE_CHECKING:
    from miner.sync_v2.variable_manager import VariableManager


class SyncedCounter:
    """Atomic integer counter using CAS on a bridge v2 variable.

    Uses read-increment-write with Compare-And-Swap for atomicity.
    The counter variable is registered on first :meth:`start` call.
    """

    def __init__(self, name: str, namespace: str, manager: VariableManager) -> None:
        self._name = name
        self._namespace = namespace
        self._manager = manager
        self._var: SyncedVariableV2 | None = None

    @property
    def value(self) -> int:
        if self._var is None:
            return 0
        return self._var._cached_value or 0

    async def start(self) -> None:
        self._var = await self._manager.create_var(
            run_id=self._namespace,
            name=self._name,
            var_type="int",
            default=0,
            write_rule="CAS",
        )

    async def stop(self) -> None:
        pass  # client is owned by the VariableManager

    async def increment(self, delta: int = 1) -> int:
        if self._var is None:
            raise RuntimeError("SyncedCounter not started — call start() first")
        max_retries = 10
        for attempt in range(max_retries):
            current = await self._var.fetch_value() or 0
            new_val = current + delta
            try:
                await self._var.set_value(new_val, current_version=self._var.version)
                return new_val
            except RuntimeError:
                if attempt < max_retries - 1:
                    logger.debug(f"SyncedCounter CAS conflict on attempt {attempt + 1}, retrying")
                    continue
        raise RuntimeError(f"SyncedCounter: too many CAS retries for {self._namespace}/{self._name}")
