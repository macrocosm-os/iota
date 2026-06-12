from enum import Enum


class MinerStatus(str, Enum):
    """Enumeration of possible miner statuses."""

    FORWARD = "forward"
    BACKWARD = "backward"
    IDLE = "idle"
    UPLOADING_WEIGHTS = "uploading_weights"
    MERGING_PARTITIONS = "merging_partitions"
    INITIALIZING = "initializing"
    FROZEN = "frozen"
    KICKED = "kicked"
