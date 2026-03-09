from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

_GIT_SHA = os.getenv("COMMIT_HASH", "unknown")


class SnowpipeMessage(BaseModel):
    """Base class for all Snowpipe streaming messages.

    Subclasses must set a ``type`` field identifying the message type.
    For simple messages (e.g. TestEvent), set a default on ``type``.
    For wide-table messages, callers set ``type`` at construction time.
    """

    type: str
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_service: str
    git_sha: str = Field(default=_GIT_SHA)

    def to_row(self) -> dict[str, Any]:
        """Serialize to a flat dict for Snowflake ingestion."""
        return self.model_dump(mode="json")


class SnowpipeIndexedMessage(SnowpipeMessage):
    """Extended base for production tables with common indexing columns.

    Used by TrainingMetric and LifecycleEvent. Adds optional fields
    for filtering/joining in Snowflake without needing VARIANT queries.
    """

    miner_hotkey: str | None = None
    miner_coldkey: str | None = None
    correlation_id: str | None = None
    activation_id: str | None = None
    run_id: str | None = None
    epoch: int | None = None
    layer_idx: int | None = None
    received_at: datetime | None = None
