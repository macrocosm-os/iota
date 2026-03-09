"""
Mutable timing record for P2P operations.

Created by callers, passed into Sender methods, and progressively
populated as each phase (connection, stream-open, send, receive)
completes.  After the call returns (or raises) the caller can inspect
the record to feed into a stats tracker or logging system.

Deliberately kept separate from retry logic so that either module can
be used independently.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

TimingsPhaseField = Literal[
    "connection_duration",
    "stream_open_duration",
    "send_duration",
    "receive_duration",
]


class P2POperationTimings(BaseModel):
    """Mutable timing record populated by Sender during a P2P operation.

    All timestamps use ``time.time()`` (wall-clock) so they align with
    the rest of the miner's timing infrastructure.
    """

    # ── per-phase durations (seconds, filled by Sender) ──────────
    connection_duration: float | None = None
    stream_open_duration: float | None = None
    send_duration: float | None = None
    receive_duration: float | None = None

    # ── overall span (filled by Sender) ──────────────────────────
    total_start: float | None = None
    total_end: float | None = None
    total_duration: float | None = None

    # ── retry metadata (filled by P2PRetry) ──────────────────────
    attempt_count: int = 0
    retry_count: int = 0
    total_backoff_time: float = 0.0
    errors: List[str] = Field(default_factory=list)

    # ── payload metadata (filled by Sender) ──────────────────────
    bytes_sent: int | None = None
    bytes_received: int | None = None
