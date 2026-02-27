from __future__ import annotations

from common.snowpipe.base import SnowpipeIndexedMessage


class LifecycleEvent(SnowpipeIndexedMessage):
    """Wide-table message for lifecycle events and state transitions.

    Callers set ``type`` to categorize the event (e.g. ``"miner_status_change"``,
    ``"phase_transition"``, ``"run_state_change"``, ``"phase_snapshot"``).

    Example::

        LifecycleEvent(
            type="miner_status_change", source_service="scheduler",
            event_category="miner", miner_hotkey="5F...",
            old_state={"status": "idle"}, new_state={"status": "frozen"},
            reason="No activations in 3 epochs",
        )
    """

    event_category: str
    old_state: dict | None = None
    new_state: dict | None = None
    reason: str | None = None
    details: dict | None = None
