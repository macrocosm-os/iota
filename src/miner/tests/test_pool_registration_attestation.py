from types import SimpleNamespace

import pytest

from common.models.api_models import MountedAttestationPayload
from miner.pool.miner import Miner


@pytest.mark.asyncio
async def test_pool_registration_sends_mounted_attestation(monkeypatch):
    miner = object.__new__(Miner)
    miner._is_mounted = True
    miner.hotkey = "hotkey1234"
    miner.p2p = SimpleNamespace(node_id="peer-1")
    miner._selected_payout_coldkey = "coldkey-1"
    miner._node_location = None
    miner._apply_registration_response = lambda _response: _async_noop()
    miner.register_set_status = lambda **_kwargs: _async_noop()
    miner.register_set_queue_state = lambda *_args, **_kwargs: _async_noop()

    sent_requests = []
    mounted_payload = MountedAttestationPayload(
        schema_version=1,
        key_id="key-1",
        public_key_base64="pub",
        payload_base64="payload",
        signature_der_base64="sig",
        payload_sha256_base64="sha",
        alg="ES256",
        challenge_id="challenge-1",
    )

    async def fake_request_attestation_challenge(action, run_id=None):
        assert action == "registration"
        assert run_id is None
        return SimpleNamespace(
            attestation_challenge_blob='{"challenge_id":"challenge-1"}',
            self_checks=["self"],
            crypto="crypto",
        )

    async def fake_register_miner_request(
        register_miner_request,
        confirmation_attestation_factory=None,
        queue_state_callback=None,
    ):
        sent_requests.append(register_miner_request)
        assert confirmation_attestation_factory is not None
        assert queue_state_callback is miner.register_set_queue_state
        return SimpleNamespace(
            model_cfg=SimpleNamespace(model_dump=lambda: {}),
            model_metadata=SimpleNamespace(model_dump=lambda: {}),
        )

    async def fake_collect_mounted_attestation(*, challenge_base64, challenge_id):
        assert challenge_id == "challenge-1"
        assert challenge_base64
        return mounted_payload

    miner.miner_api_client = SimpleNamespace(
        request_attestation_challenge=fake_request_attestation_challenge,
        register_miner_request=fake_register_miner_request,
    )
    monkeypatch.setattr("miner.pool.miner.collect_system_data", lambda: '{"bandwidth":{}}')
    miner.collect_mounted_attestation = fake_collect_mounted_attestation

    await miner.register()

    assert sent_requests, "register request should be sent"
    request = sent_requests[0]
    assert request.attestation == mounted_payload
    assert request.coldkey == "coldkey-1"
    assert request.register_as_metagraph_miner is False
    assert request.enclave_payload is None
    assert request.p2p_node_id == "peer-1"


async def _async_noop(**_kwargs):
    return None
