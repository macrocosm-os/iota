import time

import pytest
from fastapi import HTTPException
from substrateinterface import Keypair

from common.utils.epistula import EpistulaHeaders, create_message_body, generate_header


def test_epistula_valid_signature_passes():
    body = {"payout_coldkey": "coldkey-1"}
    body_bytes = create_message_body(body)
    generated_headers = generate_header(Keypair.create_from_uri("//Alice"), body_bytes)
    headers = EpistulaHeaders(
        timestamp=generated_headers["Epistula-Timestamp"],
        signed_by=generated_headers["Epistula-Signed-By"],
        request_signature=generated_headers["Epistula-Request-Signature"],
        request_id=generated_headers["X-Request-Id"],
        spec_version=generated_headers["X-Spec-Version"],
    )

    assert headers.verify_signature_v2(body=body_bytes, now=time.time(), timeout=60) is None


def test_epistula_malformed_signature_raises_http_exception():
    body = {"payout_coldkey": "coldkey-1"}
    headers = EpistulaHeaders(
        timestamp=str(round(time.time() * 1000)),
        signed_by=Keypair.create_from_uri("//Alice").ss58_address,
        request_signature="sample_signature",
        request_id="test-malformed-signature",
        spec_version="30017",
    )

    with pytest.raises(HTTPException) as exc_info:
        headers.verify_signature_v2(body=create_message_body(body), now=time.time(), timeout=60)

    assert exc_info.value.status_code == 400
    assert "Signature" in exc_info.value.detail["error"]
