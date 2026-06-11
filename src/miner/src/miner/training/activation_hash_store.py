from __future__ import annotations

import asyncio


class ActivationHashStore:
    """Thread-safe store for input activation hashes, keyed by activation_id."""

    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def store(self, activation_id: str, input_hash: str) -> None:
        async with self._lock:
            self._hashes[activation_id] = input_hash

    def get(self, activation_id: str) -> str | None:
        return self._hashes.get(activation_id)

    async def remove(self, activation_id: str) -> None:
        async with self._lock:
            self._hashes.pop(activation_id, None)

    async def clear_all(self) -> int:
        async with self._lock:
            count = len(self._hashes)
            self._hashes.clear()
            return count
