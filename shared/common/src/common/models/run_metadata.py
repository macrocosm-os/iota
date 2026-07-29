from enum import Enum, IntEnum
from typing import Any

from pydantic import BaseModel, Field


class KickPolicy(str, Enum):
    """Policy options for performance kick."""

    NO_KICK_POLICY = "no_kick_policy"
    BOTTOM_BY_SCORE = "bottom_by_score"
    SCORE_THRESHOLD = "score_threshold"


class BottomByScoreParams(BaseModel):
    """Parameters for the bottom_by_score kick policy."""

    n_kick: int = Field(ge=1)


class ScoreThresholdParams(BaseModel):
    """Parameters for the score_threshold kick policy."""

    score_threshold: float = Field(ge=0.0)


KICK_POLICY_PARAM_MODELS: dict[KickPolicy, type[BaseModel] | None] = {
    KickPolicy.NO_KICK_POLICY: None,  # No parameters for no kick policy
    KickPolicy.BOTTOM_BY_SCORE: BottomByScoreParams,
    KickPolicy.SCORE_THRESHOLD: ScoreThresholdParams,
}


class RunTier(IntEnum):
    """Known run tier options."""

    IRON = 0
    BRONZE = 1
    SILVER = 2
    GOLD = 3


class MinerTier(IntEnum):
    """Known miner tier options, including the miner-only banned state."""

    BANNED = -1
    IRON = RunTier.IRON.value
    BRONZE = RunTier.BRONZE.value
    SILVER = RunTier.SILVER.value
    GOLD = RunTier.GOLD.value


class RunMetadata(BaseModel):
    """Per-run operational metadata."""

    kick_policy: KickPolicy = Field(default=KickPolicy.NO_KICK_POLICY)
    kick_policy_params: dict[str, Any] = Field(default_factory=dict)
    is_public: bool = Field(default=False)
    tier: RunTier = Field(default=RunTier.IRON)

    def is_kick_active(self) -> bool:
        return self.kick_policy != KickPolicy.NO_KICK_POLICY

    def parsed_kick_params(self) -> BaseModel | None:
        if not self.is_kick_active():
            return None
        model_cls = KICK_POLICY_PARAM_MODELS.get(self.kick_policy)
        if model_cls is None:
            return None
        return model_cls.model_validate(self.kick_policy_params)


def create_run_metadata(run_metadata: RunMetadata | None = None) -> RunMetadata:
    """Build run_metadata for create/update; defaults to kick disabled."""
    if run_metadata is not None:
        return run_metadata.model_copy(deep=True)
    return RunMetadata(kick_policy=KickPolicy.NO_KICK_POLICY, kick_policy_params={})
