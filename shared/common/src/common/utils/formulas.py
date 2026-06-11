import math
from common import settings as common_settings


def calculate_n_partitions(n_miners: int, n_splits: int) -> int:
    """Calculate the number of partitions for a given number of miners.

    Args:
        n_miners (int): The number of miners.

    Returns:
        int: The number of partitions.
    """
    return (n_miners // n_splits) * 2


def compute_multipart_layout(total_bytes: int) -> tuple[int, bool]:
    """Decide how many parts a blob of `total_bytes` should be split into.

    Uses MIN_PART_SIZE as the target part size (more parts → more parallelism)
    and clamps to MAX_NUM_PARTS. Returns `(num_parts, multipart)`.
    """
    assert total_bytes > 0, "total_bytes must be positive"
    num_parts = max(1, math.ceil(total_bytes / common_settings.MIN_PART_SIZE))
    num_parts = min(num_parts, common_settings.MAX_NUM_PARTS)
    return num_parts, num_parts > 1
