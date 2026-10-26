from types import SimpleNamespace

import pytest
from common.models.api_models import (
    MinerRegistrationQueueStatusResponse,
    MinerRegistrationResponse,
    RegisterMinerRequest,
)
from common.models.run_flags import RunFlags
from subnet import miner_api_client as miner_api_client_module
from subnet.common_api_client import CommonAPIClient
from subnet.miner_api_client import MinerAPIClient
from subnet.utils import registration_queue_store

HOTKEY = "5HotkeyAddress"


def _client() -> MinerAPIClient:
    return MinerAPIClient(hotkey=SimpleNamespace(ss58_address=HOTKEY))


def _registered() -> MinerRegistrationResponse:
    return MinerRegistrationResponse(run_id="run-1", run_flags=RunFlags(), num_partitions=2)


@pytest.fixture
def no_sleep(monkeypatch):
    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(miner_api_client_module.asyncio, "sleep", fake_sleep)


@pytest.mark.asyncio
async def test_queue_id_is_saved_while_waiting_and_cleared_on_success(monkeypatch, no_sleep):
    client = _client()
    seen_saved = []

    async def fake_status_request(queue_id):
        seen_saved.append(registration_queue_store.load_queue_id(HOTKEY, "peer-1"))
        return _registered()

    monkeypatch.setattr(client, "_registration_queue_status_request", fake_status_request)

    await client._wait_for_registration_queue(
        MinerRegistrationQueueStatusResponse(queue_id="queue-1", status="queued", position=2781),
        p2p_node_id="peer-1",
    )

    assert seen_saved == ["queue-1"]
    assert registration_queue_store.load_queue_id(HOTKEY) is None


@pytest.mark.asyncio
async def test_register_resumes_saved_queue_entry_instead_of_rejoining(monkeypatch, no_sleep):
    registration_queue_store.save_queue_id(HOTKEY, "queue-old", "peer-1")
    client = _client()
    status_calls = []
    statuses = iter(
        [
            MinerRegistrationQueueStatusResponse(queue_id="queue-old", status="queued", position=12),
            _registered(),
        ]
    )

    async def fake_status_request(queue_id):
        status_calls.append(queue_id)
        return next(statuses)

    async def fail_post(**_kwargs):
        pytest.fail("should not POST /miner/register while a saved queue entry is alive")

    monkeypatch.setattr(client, "_registration_queue_status_request", fake_status_request)
    monkeypatch.setattr(CommonAPIClient, "orchestrator_request", fail_post)

    response = await client.register_miner_request(RegisterMinerRequest(p2p_node_id="peer-1"))

    assert response.run_id == "run-1"
    assert status_calls == ["queue-old", "queue-old"]
    assert registration_queue_store.load_queue_id(HOTKEY) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("dead", ["expired", "failed", "error"])
async def test_register_rejoins_when_saved_entry_is_gone(monkeypatch, no_sleep, dead):
    registration_queue_store.save_queue_id(HOTKEY, "queue-old", "peer-1")
    client = _client()

    async def fake_status_request(queue_id):
        if dead == "error":
            raise RuntimeError("404 queue entry not found")
        return MinerRegistrationQueueStatusResponse(queue_id=queue_id, status=dead)

    posts = []

    async def fake_post(**kwargs):
        posts.append(kwargs["path"])
        return {"run_id": "run-1", "run_flags": {}, "num_partitions": 2}

    monkeypatch.setattr(client, "_registration_queue_status_request", fake_status_request)
    monkeypatch.setattr(CommonAPIClient, "orchestrator_request", fake_post)
    monkeypatch.setattr(client, "parse_response", lambda response: response)

    response = await client.register_miner_request(RegisterMinerRequest(p2p_node_id="peer-1"))

    assert response.run_id == "run-1"
    assert posts == ["/miner/register"]
    assert registration_queue_store.load_queue_id(HOTKEY) is None


def test_saved_entry_ignored_for_a_different_p2p_node():
    registration_queue_store.save_queue_id(HOTKEY, "queue-old", "peer-1")

    assert registration_queue_store.load_queue_id(HOTKEY, "peer-2") is None
    assert registration_queue_store.load_queue_id(HOTKEY, "peer-1") is None  # cleared


def test_stale_entry_is_dropped(monkeypatch):
    registration_queue_store.save_queue_id(HOTKEY, "queue-old", "peer-1")
    monkeypatch.setattr(registration_queue_store, "QUEUE_ENTRY_MAX_AGE_SECONDS", -1)

    assert registration_queue_store.load_queue_id(HOTKEY, "peer-1") is None


def test_corrupt_store_is_ignored():
    registration_queue_store.store_path().write_text("{not json")

    assert registration_queue_store.load_queue_id(HOTKEY) is None
    registration_queue_store.save_queue_id(HOTKEY, "queue-1")
    assert registration_queue_store.load_queue_id(HOTKEY) == "queue-1"
