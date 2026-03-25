from types import SimpleNamespace

import pytest

from common.models.api_models import MountedAttestationPayload
from common.models.run_flags import RunFlags
from miner.pool.miner import Miner


@pytest.mark.asyncio
async def test_pool_registration_sends_mounted_attestation(monkeypatch):
    miner = object.__new__(Miner)
    miner._is_mounted = True
    miner.hotkey = "hotkey1234"
    miner.p2p = SimpleNamespace(node_id="peer-1")
    miner._selected_payout_coldkey = "coldkey-1"
    miner._apply_registration_response = lambda _response: None
    miner.register_set_best_run = lambda **_kwargs: _async_noop()
    miner.register_set_status = lambda **_kwargs: _async_noop()

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

    async def fake_fetch_run_info_request():
        run_flags = RunFlags()
        run_flags.attest.enabled = True
        return [SimpleNamespace(run_id="run-123", run_flags=run_flags)]

    async def fake_request_attestation_challenge(action, run_id=None):
        assert action == "registration"
        assert run_id == "run-123"
        return SimpleNamespace(
            attestation_challenge_blob='{"challenge_id":"challenge-1"}',
            self_checks=["self"],
            crypto="crypto",
        )

    async def fake_register_miner_request(register_miner_request):
        sent_requests.append(register_miner_request)
        return SimpleNamespace(
            model_cfg=SimpleNamespace(model_dump=lambda: {}),
            model_metadata=SimpleNamespace(model_dump=lambda: {}),
        )

    async def fake_change_payout_coldkey_request(_payout_coldkey):
        return {}

    async def fake_collect_mounted_attestation(*, challenge_base64, challenge_id):
        assert challenge_id == "challenge-1"
        assert challenge_base64
        return mounted_payload

    miner.miner_api_client = SimpleNamespace(
        fetch_run_info_request=fake_fetch_run_info_request,
        request_attestation_challenge=fake_request_attestation_challenge,
        register_miner_request=fake_register_miner_request,
        change_payout_coldkey_request=fake_change_payout_coldkey_request,
    )
    monkeypatch.setattr("miner.pool.miner.get_miner_pool_run", lambda run_info_list: run_info_list[0])
    monkeypatch.setattr("miner.pool.miner.collect_system_data", lambda: '{"bandwidth":{}}')
    miner.collect_mounted_attestation = fake_collect_mounted_attestation

    await miner.register()

    assert sent_requests, "register request should be sent"
    request = sent_requests[0]
    assert request.attestation == mounted_payload
    assert request.enclave_payload is None
    assert request.run_id == "run-123"
    assert request.p2p_node_id == "peer-1"


async def _async_noop(**_kwargs):
    return None
