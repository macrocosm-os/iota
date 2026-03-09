"""iroh: A P2P communication framework built on Iroh."""

__version__ = "0.1.2"

import common.utils.patched_iroh  # noqa: F401

# Monkeypatch IrohError so tracebacks/logs show the actual Rust error message
# instead of the generic "IrohError" class name.
from iroh import IrohError

IrohError.__str__ = IrohError.message

from common.iroh.connection import PeerConnection
from common.iroh.sender import Sender, PooledSender
from common.iroh.receiver import Receiver
from common.iroh.protocol import (
    UniProtocol,
    BiProtocol,
    UniProtocolFactory,
    BiProtocolFactory,
    P2PCallback,
    ResponseLike,
    PROTOCOL_ID_UNI,
    PROTOCOL_ID_BI,
)
from common.iroh.serializer import (
    Serializer,
    SerializerType,
    JsonSerializer,
    MsgpackSerializer,
    get_serializer,
    wrap_envelope,
    unwrap_envelope,
)
from common.iroh.monitored_node import MonitoredNode, NodeHealth, HealthCheckResult
from common.iroh.timings import P2POperationTimings, TimingsPhaseField
from common.iroh.retry import P2PRetry, P2PRetryPolicy, P2PTimeouts
from common.iroh.settings import DEFAULT_MAX_MESSAGE_SIZE
from common.iroh.p2p_stack import P2PStack
from common.iroh.receiver_process import ReceiverProcess
from common.iroh.p2p_protocol import (
    P2PResponseStatus,
    P2PRequestError,
    P2PNotFoundError,
    P2PExpiredError,
    P2PUnauthorizedError,
    P2PAuthFields,
    encode_activation_request,
    decode_activation_request,
    encode_activation_response,
    decode_activation_response,
)

__all__ = [
    "PeerConnection",
    "Sender",
    "PooledSender",
    "Receiver",
    "UniProtocol",
    "BiProtocol",
    "UniProtocolFactory",
    "BiProtocolFactory",
    "P2PCallback",
    "ResponseLike",
    "PROTOCOL_ID_UNI",
    "PROTOCOL_ID_BI",
    "Serializer",
    "SerializerType",
    "JsonSerializer",
    "MsgpackSerializer",
    "get_serializer",
    "wrap_envelope",
    "unwrap_envelope",
    "P2POperationTimings",
    "TimingsPhaseField",
    "P2PRetry",
    "P2PRetryPolicy",
    "P2PTimeouts",
    "MonitoredNode",
    "NodeHealth",
    "HealthCheckResult",
    "DEFAULT_MAX_MESSAGE_SIZE",
    "P2PStack",
    "ReceiverProcess",
    "P2PResponseStatus",
    "P2PRequestError",
    "P2PNotFoundError",
    "P2PExpiredError",
    "P2PUnauthorizedError",
    "P2PAuthFields",
    "encode_activation_request",
    "decode_activation_request",
    "encode_activation_response",
    "decode_activation_response",
]
