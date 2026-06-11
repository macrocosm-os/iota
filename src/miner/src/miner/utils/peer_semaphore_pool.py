from __future__ import annotations

import asyncio


class PeerSemaphorePool:
    """Get-or-create semaphores keyed by peer node_id to cap concurrent requests per peer."""

    def __init__(self, max_concurrent: int = 2) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()
        self._max_concurrent = max_concurrent

    async def get(self, node_id: str) -> asyncio.Semaphore:
        async with self._lock:
            if node_id not in self._semaphores:
                self._semaphores[node_id] = asyncio.Semaphore(self._max_concurrent)
            return self._semaphores[node_id]
