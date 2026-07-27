import pytest
from pydantic import ValidationError

from common.models.run_metadata import (
    BottomByScoreParams,
    KickPolicy,
    RunMetadata,
    ScoreThresholdParams,
    create_run_metadata,
)


def test_is_kick_active_when_bottom_by_score():
    meta = RunMetadata(
        kick_policy=KickPolicy.BOTTOM_BY_SCORE,
        kick_policy_params={"n_kick": 2},
    )
    assert meta.is_kick_active()
    params = meta.parsed_kick_params()
    assert isinstance(params, BottomByScoreParams)
    assert params.n_kick == 2


def test_is_kick_inactive_for_no_kick_policy():
    meta = RunMetadata()
    assert meta.kick_policy == KickPolicy.NO_KICK_POLICY
    assert not meta.is_kick_active()
    assert meta.parsed_kick_params() is None


def test_parsed_kick_params_requires_n_kick_for_bottom_by_score():
    meta = RunMetadata(
        kick_policy=KickPolicy.BOTTOM_BY_SCORE,
        kick_policy_params={},
    )
    with pytest.raises(ValidationError):
        meta.parsed_kick_params()


def test_parsed_kick_params_for_score_threshold():
    meta = RunMetadata(
        kick_policy=KickPolicy.SCORE_THRESHOLD,
        kick_policy_params={"score_threshold": 0.25},
    )
    params = meta.parsed_kick_params()
    assert isinstance(params, ScoreThresholdParams)
    assert params.score_threshold == 0.25


def test_parsed_kick_params_requires_score_threshold():
    meta = RunMetadata(
        kick_policy=KickPolicy.SCORE_THRESHOLD,
        kick_policy_params={},
    )
    with pytest.raises(ValidationError):
        meta.parsed_kick_params()


def test_create_run_metadata_none_uses_defaults():
    stored = create_run_metadata(None)
    assert stored.kick_policy == KickPolicy.NO_KICK_POLICY
    assert stored.kick_policy_params == {}
    assert not stored.is_kick_active()


def test_create_run_metadata_preserves_explicit_kick_policy():
    incoming = RunMetadata(
        kick_policy=KickPolicy.BOTTOM_BY_SCORE,
        kick_policy_params={"n_kick": 2},
    )
    stored = create_run_metadata(incoming)
    assert stored.kick_policy == KickPolicy.BOTTOM_BY_SCORE
    assert stored.kick_policy_params == {"n_kick": 2}


def test_create_run_metadata_fills_missing_kick_policy():
    stored = create_run_metadata(RunMetadata())
    assert stored.kick_policy == KickPolicy.NO_KICK_POLICY
    assert stored.kick_policy_params == {}


def test_is_public_defaults_to_false():
    assert RunMetadata().is_public is False


def test_create_run_metadata_preserves_is_public():
    stored = create_run_metadata(RunMetadata(is_public=False))
    assert stored.is_public is False

    stored = create_run_metadata(RunMetadata(is_public=True))
    assert stored.is_public is True
