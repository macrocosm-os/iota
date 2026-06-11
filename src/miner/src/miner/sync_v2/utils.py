"""Shared sync utilities: SyncError, sync_run_sync_prefix."""

from __future__ import annotations


class SyncError(Exception):
    """Raised when an explicit push/pull to the bridge fails."""

    pass


def sync_run_sync_prefix(run_id: str | None) -> str:
    """Return a sanitised run-scoped prefix for bridge keys.

    Pre-registration uses ``run-pending`` until a real ``run_id`` is assigned.
    """
    if not run_id:
        return "run-pending"
    safe = "".join((c if c.isalnum() or c in "-_" else "_") for c in str(run_id))
    return f"run-{safe}"
