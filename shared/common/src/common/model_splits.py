"""Compute balanced pipeline stage layouts from a model's layer count.

"Balanced" here means the first and last stages get slightly fewer transformer
layers than the middle stages, because the first stage also hosts the embedding
and the last stage hosts the lm_head. The heuristic matches the existing
convention used in ``common/configs.py`` (e.g. for 1B / n_layers=16):

  - K = 3 -> [4, 8, 4]   -> [[-1, 4], [4, 12], [12, -1]]
  - K = 4 -> [3, 5, 5, 3] -> [[-1, 3], [3, 8], [8, 13], [13, -1]]

Rules:
  - K == 1: single stage hosts every layer.
  - K == 2: split layers in half (no end-light adjustment with no middle stage).
  - K >= 3: end stages get ``n_layers // K - 1`` layers when ``n_layers // K >= 3``,
            otherwise ``n_layers // K``; the remaining layers are distributed across
            the middle stages. Leftover from non-divisible division is packed into
            the leftmost middle stages.

If the end-light strategy would leave a middle stage with zero layers (very small
models with many stages), the function falls back to an as-even-as-possible split.
"""

from __future__ import annotations

__all__ = [
    "balanced_stage_sizes",
    "stage_sizes_to_splits",
    "balanced_model_splits",
]


def balanced_stage_sizes(n_layers: int, n_stages: int) -> list[int]:
    """Return a list of per-stage layer counts that sums to ``n_layers``."""
    if n_stages <= 0:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")
    if n_stages > n_layers:
        raise ValueError(f"Cannot split {n_layers} layers across {n_stages} stages")
    if n_stages == 1:
        return [n_layers]
    if n_stages == 2:
        first = n_layers // 2
        return [first, n_layers - first]

    base = n_layers // n_stages
    end_size = base - 1 if base >= 3 else max(1, base)

    middle_total = n_layers - 2 * end_size
    n_middle = n_stages - 2
    if middle_total < n_middle:
        # Not enough layers for the end-light strategy; fall back to as-even-as-possible.
        extra = n_layers % n_stages
        return [base + 1 if i < extra else base for i in range(n_stages)]

    middle_base, extra = divmod(middle_total, n_middle)
    middle = [middle_base + 1 if i < extra else middle_base for i in range(n_middle)]
    return [end_size] + middle + [end_size]


def stage_sizes_to_splits(stage_sizes: list[int]) -> list[list[int]]:
    """Convert per-stage sizes (e.g. ``[4, 8, 4]``) to boundary pairs.

    The first stage's start and the last stage's end are sentinels of ``-1``.
    Example: ``[4, 8, 4]`` -> ``[[-1, 4], [4, 12], [12, -1]]``.
    """
    if not stage_sizes:
        raise ValueError("stage_sizes must be non-empty")
    splits: list[list[int]] = []
    cursor = 0
    for i, size in enumerate(stage_sizes):
        start = -1 if i == 0 else cursor
        cursor += size
        end = -1 if i == len(stage_sizes) - 1 else cursor
        splits.append([start, end])
    return splits


def balanced_model_splits(n_layers: int, n_stages: int) -> list[list[int]]:
    """Convenience wrapper: return ``model_splits`` directly for ``(n_layers, n_stages)``."""
    return stage_sizes_to_splits(balanced_stage_sizes(n_layers, n_stages))
