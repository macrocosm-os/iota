from typing import Optional

from common.utils.shared_states import LayerPhase
from pydantic import BaseModel, model_validator


class LayerStateError(BaseModel):
    layer: int
    expected_status: LayerPhase
    actual_status: LayerPhase
    message: Optional[str] = None

    @model_validator(mode="after")
    def make_message(self):
        self.message = (
            f"Layer {self.layer} is not in the expected state {self.expected_status}, actual: {self.actual_status}"
        )
        return self


class FileSizeError(Exception):
    def __init__(self, message: str = "File size validation failed", **kwargs):
        super().__init__(message)
        self.message = message


class FileTooOldError(Exception):
    def __init__(self, message: str = "File is too old", **kwargs):
        super().__init__(message)
        self.message = message


class S3FileNotFoundError(Exception):
    def __init__(self, message: str = "S3 file not found", **kwargs):
        super().__init__(message)
        self.message = message


class RunFullError(BaseModel):
    message: str = "Run is full"
    name: Optional[str] = None


class EntityNotRegisteredError(BaseModel):
    message: str = "Entity not registered"
    name: Optional[str] = None


class EntityOnMetagraphError(BaseModel):
    message: str = "Entity is on the metagraph but shouldn't be (pool miner)"
    name: Optional[str] = None


class EntityNotOnMetagraphError(BaseModel):
    message: str = "Entity is not on the metagraph but should be (non-pool miner)"
    name: Optional[str] = None


class MinerFrozenError(BaseModel):
    message: str = "Miner is frozen and cannot perform work until further notice"


class MinerInitializingError(BaseModel):
    message: str = "Miner is initializing and cannot perform work until the next epoch"


class BaseErrorModel(BaseModel):
    error_name: str | None = None
    error_dict: dict | None = None


class SpecVersionError(BaseModel):
    expected_version: int
    actual_version: str
    message: str = "Spec version mismatch"

    @model_validator(mode="after")
    def make_message(self):
        self.message = f"Spec version mismatch. Expected: {self.expected_version}, Received: {self.actual_version}"
        return self
