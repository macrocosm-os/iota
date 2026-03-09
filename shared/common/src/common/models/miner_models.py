from enum import Enum

from loguru import logger
from pydantic import BaseModel, model_validator
from typing import Literal


class MinerStatus(str, Enum):
    """Enumeration of possible miner statuses."""

    FORWARD = "forward"
    BACKWARD = "backward"
    IDLE = "idle"
    UPLOADING_WEIGHTS = "uploading_weights"
    MERGING_PARTITIONS = "merging_partitions"
    INITIALIZING = "initializing"
    FROZEN = "frozen"


class ChunkMetadata(BaseModel):
    start_idx: int
    end_idx: int
    start_byte: int
    end_byte: int
    chunk_number: int
    weighting_factor: int | None = None
    tensor_path: str
    metadata_path: str
    chunk_dtype: Literal["bfloat16"]
    data_type: Literal["weights", "optimizer_state"]

    def compatible(self, other: "ChunkMetadata") -> bool:
        """Check if two chunk metadata objects are compatible."""
        if (
            self.start_idx == other.start_idx
            and self.end_idx == other.end_idx
            and self.start_byte == other.start_byte
            and self.end_byte == other.end_byte
        ):
            return True
        logger.warning(f"Metadata mismatch | chunk {self.chunk_number} | {self.data_type}: {self} != {other}")
        return False

    @model_validator(mode="after")
    def verify_type(self):
        if self.data_type == "weights":
            assert (
                "weight" in self.tensor_path
            ), f"Weights tensor path (path: {self.tensor_path}) does not contain 'weight'"
            assert (
                "weight" in self.metadata_path
            ), f"Weights metadata path (path: {self.metadata_path}) does not contain 'weight'"
        elif self.data_type == "optimizer_state":
            assert (
                "optimizer_state" in self.tensor_path
            ), f"Optimizer state tensor path (path: {self.tensor_path}) does not contain 'optimizer_state'"
            assert (
                "optimizer_state" in self.metadata_path
            ), f"Optimizer state metadata path (path: {self.metadata_path}) does not contain 'optimizer_state'"
        return self
