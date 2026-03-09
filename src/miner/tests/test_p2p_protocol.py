"""Tests for P2P message encoding/decoding protocol."""

import struct
from unittest.mock import patch

from common.iroh.p2p_protocol import (
    P2PResponseStatus,
    P2PUnauthorizedError,
    encode_activation_request,
    decode_activation_request,
    encode_activation_response,
    decode_activation_response,
    _REQUEST_V1,
    _REQUEST_V2,
    _SS58_FIELD_LEN,
    _SIG_FIELD_LEN,
)


class TestActivationRequestV1:
    """Tests for v1 (unsigned) activation request encoding/decoding."""

    def test_encode_decode_roundtrip(self):
        """Request should survive encode/decode roundtrip."""
        activation_id = "test-activation-123"
        encoded = encode_activation_request(activation_id)
        decoded_id, auth = decode_activation_request(encoded)
        assert decoded_id == activation_id
        assert auth is None

    def test_encode_produces_bytes(self):
        """Encoding should produce bytes."""
        encoded = encode_activation_request("test-id")
        assert isinstance(encoded, bytes)

    def test_encode_format_v1(self):
        """Verify the v1 encoding format: [0x01][2 bytes len][id bytes]."""
        activation_id = "abc"
        encoded = encode_activation_request(activation_id)
        # 1 byte version + 2 bytes for length (big-endian) + 3 bytes for "abc"
        assert len(encoded) == 6
        assert encoded[0] == _REQUEST_V1
        # Length should be 3 (big-endian)
        assert encoded[1] == 0
        assert encoded[2] == 3
        # ID bytes
        assert encoded[3:] == b"abc"

    def test_handles_uuid_format(self):
        """Should handle UUID-style activation IDs."""
        activation_id = "550e8400-e29b-41d4-a716-446655440000"
        encoded = encode_activation_request(activation_id)
        decoded_id, auth = decode_activation_request(encoded)
        assert decoded_id == activation_id
        assert auth is None

    def test_handles_long_ids(self):
        """Should handle longer activation IDs."""
        activation_id = "a" * 1000
        encoded = encode_activation_request(activation_id)
        decoded_id, auth = decode_activation_request(encoded)
        assert decoded_id == activation_id

    def test_handles_unicode(self):
        """Should handle unicode characters in ID (edge case)."""
        activation_id = "test-\u00e9\u00e8\u00ea"
        encoded = encode_activation_request(activation_id)
        decoded_id, auth = decode_activation_request(encoded)
        assert decoded_id == activation_id

    def test_handles_empty_id(self):
        """Should handle empty activation ID (edge case)."""
        activation_id = ""
        encoded = encode_activation_request(activation_id)
        decoded_id, auth = decode_activation_request(encoded)
        assert decoded_id == activation_id

    def test_legacy_format_decoded_as_v1(self):
        """Old-style messages (no version byte) should still decode."""
        # Old format: [2B id_len][id_bytes] — first byte would be 0x00 for short IDs
        activation_id = "test-legacy"
        id_bytes = activation_id.encode("utf-8")
        legacy_msg = struct.pack(">H", len(id_bytes)) + id_bytes
        decoded_id, auth = decode_activation_request(legacy_msg)
        assert decoded_id == activation_id
        assert auth is None


class TestActivationRequestV2:
    """Tests for v2 (signed) activation request encoding/decoding."""

    def _make_v2_message(self, activation_id: str, timestamp_ms: int, ss58: str, sig: bytes) -> bytes:
        """Helper to manually construct a v2 message."""
        id_bytes = activation_id.encode("utf-8")
        ss58_padded = ss58.encode("utf-8").ljust(_SS58_FIELD_LEN, b"\x00")
        sig_padded = sig[:_SIG_FIELD_LEN].ljust(_SIG_FIELD_LEN, b"\x00")
        return (
            bytes([_REQUEST_V2])
            + struct.pack(">Q", timestamp_ms)
            + ss58_padded
            + sig_padded
            + struct.pack(">H", len(id_bytes))
            + id_bytes
        )

    def test_decode_v2_extracts_auth_fields(self):
        """v2 messages should return P2PAuthFields."""
        ts = 1700000000000
        ss58 = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        sig = b"\xab" * 64
        msg = self._make_v2_message("act-123", ts, ss58, sig)

        decoded_id, auth = decode_activation_request(msg)
        assert decoded_id == "act-123"
        assert auth is not None
        assert auth.timestamp_ms == ts
        assert auth.ss58_address == ss58
        assert auth.signature == sig

    def test_v2_overhead(self):
        """v2 overhead should be 1 + 8 + 48 + 64 + 2 = 123 bytes + id."""
        activation_id = "x"
        msg = self._make_v2_message(activation_id, 0, "addr", b"\x00" * 64)
        # 123 header + 1 byte id
        assert len(msg) == 124

    def test_encode_with_hotkey_produces_v2(self):
        """encode_activation_request with hotkey= should produce v2 format."""
        from unittest.mock import MagicMock

        fake_hotkey = MagicMock()
        fake_hotkey.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        fake_hotkey.sign.return_value = b"\xaa" * 64

        with patch("common.utils.epistula.time.time", return_value=1700000000.0):
            encoded = encode_activation_request("test-act", hotkey=fake_hotkey)

        assert encoded[0] == _REQUEST_V2
        decoded_id, auth = decode_activation_request(encoded)
        assert decoded_id == "test-act"
        assert auth is not None
        assert auth.ss58_address == fake_hotkey.ss58_address
        assert len(auth.signature) == _SIG_FIELD_LEN

    def test_v2_roundtrip_with_real_keypair(self):
        """Full sign → encode → decode → verify roundtrip with a real Keypair."""
        from substrateinterface import Keypair

        kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        activation_id = "activation-uuid-roundtrip"

        encoded = encode_activation_request(activation_id, hotkey=kp)
        decoded_id, auth = decode_activation_request(encoded)

        assert decoded_id == activation_id
        assert auth is not None
        assert auth.ss58_address == kp.ss58_address

        # Verify the signature
        from common.utils.epistula import verify_p2p_request

        id_bytes = activation_id.encode("utf-8")
        is_valid, reason = verify_p2p_request(
            body=id_bytes,
            timestamp_ms=auth.timestamp_ms,
            ss58_address=auth.ss58_address,
            signature=auth.signature,
            timeout_ms=30000,
        )
        assert is_valid, f"Signature verification failed: {reason}"


class TestActivationResponse:
    """Tests for activation response encoding/decoding."""

    def test_encode_decode_found_roundtrip(self):
        """Response with data should survive roundtrip."""
        tensor_bytes = b"tensor data here"
        encoded = encode_activation_response(tensor_bytes)
        decoded_bytes, status = decode_activation_response(encoded)
        assert decoded_bytes == tensor_bytes
        assert status == P2PResponseStatus.SUCCESS

    def test_encode_decode_not_found(self):
        """Response for not found should decode with NOT_FOUND status."""
        encoded = encode_activation_response(None, P2PResponseStatus.NOT_FOUND)
        decoded_bytes, status = decode_activation_response(encoded)
        assert decoded_bytes is None
        assert status == P2PResponseStatus.NOT_FOUND

    def test_encode_decode_expired(self):
        """Response for expired activation should decode with EXPIRED status."""
        encoded = encode_activation_response(None, P2PResponseStatus.EXPIRED)
        decoded_bytes, status = decode_activation_response(encoded)
        assert decoded_bytes is None
        assert status == P2PResponseStatus.EXPIRED

    def test_encode_decode_error(self):
        """Response for error should decode with ERROR status."""
        encoded = encode_activation_response(None, P2PResponseStatus.ERROR)
        decoded_bytes, status = decode_activation_response(encoded)
        assert decoded_bytes is None
        assert status == P2PResponseStatus.ERROR

    def test_encode_decode_unauthorized(self):
        """Response for unauthorized should decode with UNAUTHORIZED status."""
        encoded = encode_activation_response(None, P2PResponseStatus.UNAUTHORIZED)
        decoded_bytes, status = decode_activation_response(encoded)
        assert decoded_bytes is None
        assert status == P2PResponseStatus.UNAUTHORIZED

    def test_found_response_format(self):
        """Verify found response format: [0x01][tensor bytes]."""
        tensor_bytes = b"data"
        encoded = encode_activation_response(tensor_bytes)
        assert encoded[0] == P2PResponseStatus.SUCCESS
        assert encoded[1:] == tensor_bytes

    def test_not_found_response_format(self):
        """Verify not found response format: [0x00]."""
        encoded = encode_activation_response(None, P2PResponseStatus.NOT_FOUND)
        assert encoded == b"\x00"

    def test_expired_response_format(self):
        """Verify expired response format: [0x02]."""
        encoded = encode_activation_response(None, P2PResponseStatus.EXPIRED)
        assert encoded == b"\x02"

    def test_error_response_format(self):
        """Verify error response format: [0x03]."""
        encoded = encode_activation_response(None, P2PResponseStatus.ERROR)
        assert encoded == b"\x03"

    def test_unauthorized_response_format(self):
        """Verify unauthorized response format: [0x04]."""
        encoded = encode_activation_response(None, P2PResponseStatus.UNAUTHORIZED)
        assert encoded == b"\x04"

    def test_handles_large_tensor(self):
        """Should handle large tensor data."""
        tensor_bytes = b"X" * (1024 * 1024)  # 1MB
        encoded = encode_activation_response(tensor_bytes)
        decoded_bytes, status = decode_activation_response(encoded)
        assert decoded_bytes == tensor_bytes
        assert status == P2PResponseStatus.SUCCESS

    def test_handles_empty_tensor(self):
        """Should handle empty tensor bytes (distinct from not found)."""
        tensor_bytes = b""
        encoded = encode_activation_response(tensor_bytes)
        # Empty data still has success status byte
        assert encoded == b"\x01"
        decoded_bytes, status = decode_activation_response(encoded)
        assert decoded_bytes == b""
        assert status == P2PResponseStatus.SUCCESS

    def test_handles_binary_data(self):
        """Should handle arbitrary binary data."""
        tensor_bytes = bytes(range(256))  # All byte values
        encoded = encode_activation_response(tensor_bytes)
        decoded_bytes, status = decode_activation_response(encoded)
        assert decoded_bytes == tensor_bytes
        assert status == P2PResponseStatus.SUCCESS

    def test_empty_response_treated_as_not_found(self):
        """Empty response bytes should be treated as not found."""
        decoded_bytes, status = decode_activation_response(b"")
        assert decoded_bytes is None
        assert status == P2PResponseStatus.NOT_FOUND

    def test_unknown_status_code_treated_as_error(self):
        """Unknown status codes should be treated as ERROR."""
        decoded_bytes, status = decode_activation_response(b"\xff")
        assert decoded_bytes is None
        assert status == P2PResponseStatus.ERROR

    def test_status_enum_values(self):
        """Verify the enum has the expected integer values."""
        assert P2PResponseStatus.NOT_FOUND == 0x00
        assert P2PResponseStatus.SUCCESS == 0x01
        assert P2PResponseStatus.EXPIRED == 0x02
        assert P2PResponseStatus.ERROR == 0x03
        assert P2PResponseStatus.UNAUTHORIZED == 0x04


class TestProtocolIntegration:
    """Integration tests for the full request/response flow."""

    def test_full_flow_found(self):
        """Test complete request -> found response flow."""
        # Requester encodes request
        activation_id = "activation-uuid-123"
        request = encode_activation_request(activation_id)

        # Responder decodes request
        received_id, auth = decode_activation_request(request)
        assert received_id == activation_id

        # Responder looks up activation (simulated - found)
        tensor_data = b"serialized tensor bytes here"

        # Responder encodes response
        response = encode_activation_response(tensor_data)

        # Requester decodes response
        received_data, status = decode_activation_response(response)
        assert received_data == tensor_data
        assert status == P2PResponseStatus.SUCCESS

    def test_full_flow_not_found(self):
        """Test complete request -> not found response flow."""
        # Requester encodes request
        activation_id = "missing-activation"
        request = encode_activation_request(activation_id)

        # Responder decodes request
        received_id, auth = decode_activation_request(request)
        assert received_id == activation_id

        # Responder looks up activation (simulated - not found)
        # Responder encodes not found response
        response = encode_activation_response(None, P2PResponseStatus.NOT_FOUND)

        # Requester decodes response
        received_data, status = decode_activation_response(response)
        assert received_data is None
        assert status == P2PResponseStatus.NOT_FOUND

    def test_full_flow_expired(self):
        """Test complete request -> expired response flow."""
        activation_id = "expired-activation"
        request = encode_activation_request(activation_id)

        received_id, auth = decode_activation_request(request)
        assert received_id == activation_id

        response = encode_activation_response(None, P2PResponseStatus.EXPIRED)

        received_data, status = decode_activation_response(response)
        assert received_data is None
        assert status == P2PResponseStatus.EXPIRED

    def test_full_flow_error(self):
        """Test complete request -> error response flow."""
        activation_id = "error-activation"
        request = encode_activation_request(activation_id)

        received_id, auth = decode_activation_request(request)
        assert received_id == activation_id

        response = encode_activation_response(None, P2PResponseStatus.ERROR)

        received_data, status = decode_activation_response(response)
        assert received_data is None
        assert status == P2PResponseStatus.ERROR

    def test_full_flow_unauthorized(self):
        """Test complete request -> unauthorized response flow."""
        activation_id = "unauth-activation"
        request = encode_activation_request(activation_id)

        received_id, auth = decode_activation_request(request)
        assert received_id == activation_id

        response = encode_activation_response(None, P2PResponseStatus.UNAUTHORIZED)

        received_data, status = decode_activation_response(response)
        assert received_data is None
        assert status == P2PResponseStatus.UNAUTHORIZED


class TestP2PAuth:
    """Tests for P2P epistula sign/verify functions."""

    def test_sign_verify_roundtrip(self):
        """Signed request should verify successfully."""
        from substrateinterface import Keypair
        from common.utils.epistula import sign_p2p_request, verify_p2p_request

        kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        body = b"test-activation-id"

        timestamp_ms, ss58, sig = sign_p2p_request(kp, body)

        is_valid, reason = verify_p2p_request(body, timestamp_ms, ss58, sig, timeout_ms=30000)
        assert is_valid, f"Expected valid, got: {reason}"

    def test_stale_request_rejected(self):
        """Request older than timeout should be rejected."""
        from substrateinterface import Keypair
        from common.utils.epistula import sign_p2p_request, verify_p2p_request

        kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        body = b"test-body"

        timestamp_ms, ss58, sig = sign_p2p_request(kp, body)

        # Use a very short timeout and pretend time has passed
        is_valid, reason = verify_p2p_request(body, timestamp_ms - 60000, ss58, sig, timeout_ms=1000)
        assert not is_valid
        assert "stale" in reason

    def test_wrong_body_rejected(self):
        """Verification with different body should fail."""
        from substrateinterface import Keypair
        from common.utils.epistula import sign_p2p_request, verify_p2p_request

        kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        body = b"original-body"
        timestamp_ms, ss58, sig = sign_p2p_request(kp, body)

        is_valid, reason = verify_p2p_request(b"different-body", timestamp_ms, ss58, sig, timeout_ms=30000)
        assert not is_valid

    def test_wrong_signature_rejected(self):
        """Verification with corrupted signature should fail."""
        from substrateinterface import Keypair
        from common.utils.epistula import sign_p2p_request, verify_p2p_request

        kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        body = b"test-body"
        timestamp_ms, ss58, sig = sign_p2p_request(kp, body)

        bad_sig = bytes([b ^ 0xFF for b in sig])  # flip all bits
        is_valid, reason = verify_p2p_request(body, timestamp_ms, ss58, bad_sig, timeout_ms=30000)
        assert not is_valid

    def test_unauthorized_error_class(self):
        """P2PUnauthorizedError should have correct status."""
        err = P2PUnauthorizedError("auth failed")
        assert err.status == P2PResponseStatus.UNAUTHORIZED
        assert "auth failed" in str(err)

    def test_v1_unsigned_request_detected(self):
        """v1 request should return None auth_fields so receiver can reject."""
        encoded = encode_activation_request("some-id")
        _, auth = decode_activation_request(encoded)
        assert auth is None
