"""The training announcement is gated on the orchestrator-assigned status.

A miner benched in the DB (status 'initializing'/'frozen', refreshed via the
heartbeat) must broadcast 'initializing' so fail-closed peer selection
excludes it as a push target — even when its layer is in the training phase.
"""

from unittest.mock import AsyncMock

import pytest

from miner.new_miner import Miner


def _make_miner(db_status: str | None, local_status: str) -> Miner:
    miner = Miner.__new__(Miner)
    miner.hotkey = "5TestHotkey"
    miner.miner_status = local_status
    miner.miner_api_client = type("StubClient", (), {"db_miner_status": db_status})()
    miner._transition_miner_status = AsyncMock()
    return miner


@pytest.mark.asyncio
async def test_benched_miner_broadcasts_initializing_not_training():
    miner = _make_miner(db_status="initializing", local_status="training")
    await miner._announce_training_status()
    miner._transition_miner_status.assert_awaited_once_with("initializing")


@pytest.mark.asyncio
async def test_benched_miner_does_not_flap_back_to_training():
    # After MinerInitializingException already set the local status, further
    # ticks must not re-announce anything while the bench holds.
    miner = _make_miner(db_status="initializing", local_status="initializing")
    await miner._announce_training_status()
    miner._transition_miner_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_frozen_miner_broadcasts_initializing():
    miner = _make_miner(db_status="frozen", local_status="training")
    await miner._announce_training_status()
    miner._transition_miner_status.assert_awaited_once_with("initializing")


@pytest.mark.asyncio
async def test_unbenched_miner_announces_training():
    # Heartbeat cleared the bench (epoch flip) -> announce training again.
    miner = _make_miner(db_status="idle", local_status="initializing")
    await miner._announce_training_status()
    miner._transition_miner_status.assert_awaited_once_with("training")


@pytest.mark.asyncio
async def test_missing_status_behaves_like_today():
    # Older orchestrator returns no status: gate nothing (backward compat).
    miner = _make_miner(db_status=None, local_status="initializing")
    await miner._announce_training_status()
    miner._transition_miner_status.assert_awaited_once_with("training")


@pytest.mark.asyncio
async def test_already_training_is_a_noop():
    miner = _make_miner(db_status="idle", local_status="training")
    await miner._announce_training_status()
    miner._transition_miner_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_heartbeat_caches_db_miner_status(monkeypatch):
    from subnet.miner_api_client import CommonAPIClient, MinerAPIClient

    client = MinerAPIClient(hotkey=None)
    assert client.db_miner_status is None
    payload = {"run_id": "run-x", "layer": 3, "epoch": 6, "phase": "training", "status": "initializing"}
    monkeypatch.setattr(CommonAPIClient, "orchestrator_request", AsyncMock(return_value=payload))
    response = await client.heartbeat()
    assert response.status == "initializing"
    assert client.db_miner_status == "initializing"


@pytest.mark.asyncio
async def test_registration_response_bench_transitions_immediately():
    # In-process re-registration lands INITIALIZING: gate without a heartbeat.
    miner = _make_miner(db_status=None, local_status="training")
    await miner._seed_status_from_registration("initializing")
    assert miner.miner_api_client.db_miner_status == "initializing"
    miner._transition_miner_status.assert_awaited_once_with("initializing")


@pytest.mark.asyncio
async def test_registration_response_working_status_seeds_without_transition():
    miner = _make_miner(db_status=None, local_status="training")
    await miner._seed_status_from_registration("idle")
    assert miner.miner_api_client.db_miner_status == "idle"
    miner._transition_miner_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_registration_response_missing_status_seeds_nothing():
    # Older orchestrator: no status field -> keep prior knowledge.
    miner = _make_miner(db_status="idle", local_status="training")
    await miner._seed_status_from_registration(None)
    assert miner.miner_api_client.db_miner_status == "idle"
    miner._transition_miner_status.assert_not_awaited()
