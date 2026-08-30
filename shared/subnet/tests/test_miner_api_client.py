from unittest.mock import AsyncMock

import pytest

from common.models.api_models import (
    MinerRegistrationQueueStatusResponse,
    MinerRegistrationResponse,
    RegisterMinerRequest,
)
from common.models.run_flags import RunFlags
from subnet import miner_api_client as miner_api_client_module
from subnet.common_api_client import CommonAPIClient
from subnet.miner_api_client import MinerAPIClient, registration_queue_poll_sleep_seconds


def test_registration_queue_poll_sleep_adds_bounded_jitter(monkeypatch):
    calls = []

    def fake_uniform(min_sleep: float, max_sleep: float) -> float:
        calls.append((min_sleep, max_sleep))
        return 5.5

    monkeypatch.setattr(miner_api_client_module.random, "uniform", fake_uniform)

    assert registration_queue_poll_sleep_seconds(5.0) == 5.5
    assert calls == [(4.0, 6.0)]


def test_registration_queue_poll_sleep_can_disable_jitter(monkeypatch):
    monkeypatch.setattr(
        miner_api_client_module.random,
        "uniform",
        lambda *_args: pytest.fail("random.uniform should not be called when jitter is disabled"),
    )

    assert registration_queue_poll_sleep_seconds(5.0, jitter_fraction=0.0) == 5.0
    assert registration_queue_poll_sleep_seconds(0.0) == 0.0


@pytest.mark.asyncio
async def test_wait_for_registration_queue_uses_jittered_poll_sleep(monkeypatch):
    sleeps = []
    status_requests = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def fake_status_request(queue_id: str) -> MinerRegistrationResponse:
        status_requests.append(queue_id)
        return MinerRegistrationResponse(run_id="run-1", run_flags=RunFlags(), num_partitions=2)

    monkeypatch.setattr(miner_api_client_module.random, "uniform", lambda _min_sleep, _max_sleep: 4.25)
    monkeypatch.setattr(miner_api_client_module.asyncio, "sleep", fake_sleep)

    client = MinerAPIClient()
    monkeypatch.setattr(client, "_registration_queue_status_request", fake_status_request)

    response = await client._wait_for_registration_queue(
        MinerRegistrationQueueStatusResponse(
            queue_id="queue-1",
            status="queued",
            poll_after_seconds=5.0,
        )
    )

    assert response.run_id == "run-1"
    assert sleeps == [4.25]
    assert status_requests == ["queue-1"]


@pytest.mark.asyncio
async def test_register_request_does_not_replay_single_use_attestation(monkeypatch):
    request = AsyncMock(
        return_value={
            "run_id": "run-1",
            "run_flags": {},
            "num_partitions": 2,
        }
    )
    monkeypatch.setattr(CommonAPIClient, "orchestrator_request", request)

    response = await MinerAPIClient().register_miner_request(RegisterMinerRequest(p2p_node_id="node-1"))

    assert response.run_id == "run-1"
    assert request.await_args.kwargs["path"] == "/miner/register"
    assert request.await_args.kwargs["max_attempts"] == 1
