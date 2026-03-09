"""
P2P message encoding/decoding protocol for activation transfer.

With bidirectional communication, requests and responses travel on the same connection:
- Request: activation_id (v1) or signed activation_id (v2)
- Response: status byte + tensor bytes (if success)

Wire formats:
  v1: [0x01][2B id_len][id_bytes]
  v2: [0x02][8B timestamp_ms BE][48B ss58_address NUL-padded][64B signature][2B id_len][id_bytes]
"""

import struct
from dataclasses import dataclass
from enum import IntEnum

_REQUEST_V1 = 0x01
_REQUEST_V2 = 0x02
_SS58_FIELD_LEN = 48
_SIG_FIELD_LEN = 64


@dataclass(frozen=True)
class P2PAuthFields:
    """Authentication fields extracted from a v2 request."""

    timestamp_ms: int
    ss58_address: str
    signature: bytes


class P2PResponseStatus(IntEnum):
    """Status codes for P2P activation responses."""

    NOT_FOUND = 0x00  # Activation never existed in cache
    SUCCESS = 0x01  # Found, tensor data follows
    EXPIRED = 0x02  # Was in cache but TTL exceeded
    ERROR = 0x03  # Internal error processing request
    UNAUTHORIZED = 0x04  # Auth failure


class P2PRequestError(Exception):
    """Base exception for P2P activation request failures."""

    def __init__(self, message: str, status: P2PResponseStatus):
        super().__init__(message)
        self.status = status


class P2PNotFoundError(P2PRequestError):
    """Raised when the requested activation was never in the peer's cache."""

    def __init__(self, message: str):
        super().__init__(message, P2PResponseStatus.NOT_FOUND)


class P2PExpiredError(P2PRequestError):
    """Raised when the requested activation existed but TTL was exceeded."""

    def __init__(self, message: str):
        super().__init__(message, P2PResponseStatus.EXPIRED)


class P2PUnauthorizedError(P2PRequestError):
    """Raised when the request failed authentication."""

    def __init__(self, message: str):
        super().__init__(message, P2PResponseStatus.UNAUTHORIZED)


def encode_activation_request(activation_id: str, hotkey=None) -> bytes:
    """
    Encode request for an activation.

    When ``hotkey`` (a ``Keypair``) is provided, produces a signed v2 message.
    Otherwise falls back to the legacy v1 format (useful for tests).

    v1 format: [0x01][2B id_len][id_bytes]
    v2 format: [0x02][8B timestamp_ms BE][48B ss58 NUL-padded][64B sig][2B id_len][id_bytes]

    Args:
        activation_id: The activation ID to request.
        hotkey: Optional Keypair for signing (produces v2 when set).

    Returns:
        Encoded message bytes.
    """
    id_bytes = activation_id.encode("utf-8")

    if hotkey is None:
        return bytes([_REQUEST_V1]) + struct.pack(">H", len(id_bytes)) + id_bytes

    # v2: sign the id_bytes payload
    from common.utils.epistula import sign_p2p_request

    timestamp_ms, ss58_address, signature = sign_p2p_request(hotkey, id_bytes)

    ss58_padded = ss58_address.encode("utf-8").ljust(_SS58_FIELD_LEN, b"\x00")
    sig_padded = signature[:_SIG_FIELD_LEN].ljust(_SIG_FIELD_LEN, b"\x00")

    return (
        bytes([_REQUEST_V2])
        + struct.pack(">Q", timestamp_ms)
        + ss58_padded
        + sig_padded
        + struct.pack(">H", len(id_bytes))
        + id_bytes
    )


def decode_activation_request(message: bytes) -> tuple[str, P2PAuthFields | None]:
    """
    Decode activation request to get activation_id and optional auth fields.

    Args:
        message: Raw request bytes.

    Returns:
        Tuple of (activation_id, auth_fields).
        auth_fields is None for v1 (unsigned) requests.
    """
    version = message[0]

    if version == _REQUEST_V1:
        id_len = struct.unpack(">H", message[1:3])[0]
        activation_id = message[3 : 3 + id_len].decode("utf-8")
        return activation_id, None

    if version == _REQUEST_V2:
        offset = 1
        timestamp_ms = struct.unpack(">Q", message[offset : offset + 8])[0]
        offset += 8
        ss58_address = message[offset : offset + _SS58_FIELD_LEN].rstrip(b"\x00").decode("utf-8")
        offset += _SS58_FIELD_LEN
        signature = message[offset : offset + _SIG_FIELD_LEN]
        offset += _SIG_FIELD_LEN
        id_len = struct.unpack(">H", message[offset : offset + 2])[0]
        offset += 2
        activation_id = message[offset : offset + id_len].decode("utf-8")

        auth = P2PAuthFields(timestamp_ms=timestamp_ms, ss58_address=ss58_address, signature=signature)
        return activation_id, auth

    # Unknown version — treat as v1 for backward compat (old format had no version byte)
    id_len = struct.unpack(">H", message[:2])[0]
    activation_id = message[2 : 2 + id_len].decode("utf-8")
    return activation_id, None


def encode_activation_response(
    tensor_bytes: bytes | None,
    status: P2PResponseStatus = P2PResponseStatus.SUCCESS,
) -> bytes:
    """
    Encode activation response with status and optional tensor data.

    Format: [1 byte status][tensor_bytes if SUCCESS]

    Args:
        tensor_bytes: The serialized tensor data, or None for non-success statuses
        status: Response status code

    Returns:
        Encoded response bytes
    """
    if tensor_bytes is not None and status == P2PResponseStatus.SUCCESS:
        return bytes([status]) + tensor_bytes
    return bytes([status])


def decode_activation_response(message: bytes) -> tuple[bytes | None, P2PResponseStatus]:
    """
    Decode activation response.

    Args:
        message: Raw response bytes

    Returns:
        Tuple of (tensor_bytes or None, status code)
    """
    if len(message) == 0:
        return None, P2PResponseStatus.NOT_FOUND

    raw_status = message[0]
    try:
        status = P2PResponseStatus(raw_status)
    except ValueError:
        # Unknown status code — treat as error
        return None, P2PResponseStatus.ERROR

    if status == P2PResponseStatus.SUCCESS:
        return message[1:], status
    return None, status
