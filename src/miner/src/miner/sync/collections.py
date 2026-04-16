"""Patch-tracked collection types: SyncedDict, SyncedList.

Uses JSON Patch (RFC 6902) via ``jsonpatch`` to compute and apply minimal diffs.
"""

from __future__ import annotations

import copy
from typing import Any

import jsonpatch


def compute_patch(old: Any, new: Any) -> list[dict]:
    """Return a JSON Patch (RFC 6902) from *old* to *new*."""
    return jsonpatch.make_patch(old, new).patch


def apply_patch(obj: Any, ops: list[dict]) -> Any:
    """Apply *ops* (a JSON Patch) to *obj* and return the new state."""
    return jsonpatch.apply_patch(obj, ops, in_place=False)


class SyncedDict(dict):
    """A ``dict`` subclass that tracks mutations for delta-based sync.

    Dirty detection and patch computation are done by comparing against a
    baseline snapshot taken at construction or after each successful push.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._baseline: dict = copy.deepcopy(dict(self))

    def is_dirty(self) -> bool:
        """Return ``True`` if any value has changed since the last commit."""
        return dict(self) != self._baseline

    def get_patch(self) -> list[dict]:
        """Compute the JSON Patch from the last-synced baseline to now."""
        return compute_patch(self._baseline, dict(self))

    def commit(self) -> None:
        """Record the current state as the new sync baseline."""
        self._baseline = copy.deepcopy(dict(self))

    def apply_incoming_patch(self, ops: list[dict]) -> None:
        """Apply a patch received from the server in-place, then commit."""
        new_state: dict = apply_patch(dict(self), ops)
        super().clear()
        super().update(new_state)
        self._baseline = copy.deepcopy(dict(self))

    def apply_full_value(self, value: dict) -> None:
        """Replace the entire dict with a full value received from the server."""
        super().clear()
        super().update(value)
        self._baseline = copy.deepcopy(dict(self))


class SyncedList(list):
    """A ``list`` subclass that tracks mutations for delta-based sync."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._baseline: list = copy.deepcopy(list(self))

    def is_dirty(self) -> bool:
        return list(self) != self._baseline

    def get_patch(self) -> list[dict]:
        return compute_patch(self._baseline, list(self))

    def commit(self) -> None:
        self._baseline = copy.deepcopy(list(self))

    def apply_incoming_patch(self, ops: list[dict]) -> None:
        new_state: list = apply_patch(list(self), ops)
        super().clear()
        super().extend(new_state)
        self._baseline = copy.deepcopy(list(self))

    def apply_full_value(self, value: list) -> None:
        super().clear()
        super().extend(value)
        self._baseline = copy.deepcopy(list(self))
