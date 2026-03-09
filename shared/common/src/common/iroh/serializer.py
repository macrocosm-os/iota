"""
Serializer abstraction with JSON and Msgpack implementations.

WHY pluggable:
    - JSON: human-readable, great for debugging with Wireshark or log inspection,
      larger on the wire (~2x vs msgpack for typical payloads)
    - Msgpack: binary, ~30-50% smaller, faster parse/serialize, not human-readable

The serializer ID (1 byte) is the first byte of every wire envelope, so:
    - A peer using JSON can receive from a peer using Msgpack (and vice versa)
    - You can migrate a cluster from JSON to Msgpack incrementally
    - Debugging tools can detect the format without configuration

HOW TO ADD A NEW SERIALIZER:
    1. Implement the Serializer ABC
    2. Assign a unique 1-byte ID (0x03, 0x04, ...)
    3. Register it in SERIALIZER_REGISTRY
    4. Add it to SerializerType enum
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar

import msgpack
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class SerializerType(Enum):
    JSON = "json"
    MSGPACK = "msgpack"


class Serializer(ABC):
    """Interface for message body serialization. Implementations must be stateless and thread-safe."""

    @abstractmethod
    def serialize(self, model: BaseModel) -> bytes:
        ...

    @abstractmethod
    def deserialize(self, data: bytes, model_cls: type[ModelT]) -> ModelT:
        ...

    @property
    @abstractmethod
    def id(self) -> bytes:
        """Single byte identifying this serializer on the wire."""
        ...


class JsonSerializer(Serializer):
    def serialize(self, model: BaseModel) -> bytes:
        return model.model_dump_json().encode("utf-8")

    def deserialize(self, data: bytes, model_cls: type[ModelT]) -> ModelT:
        return model_cls.model_validate_json(data)

    @property
    def id(self) -> bytes:
        return b"\x01"


class MsgpackSerializer(Serializer):
    def serialize(self, model: BaseModel) -> bytes:
        return msgpack.packb(model.model_dump(), use_bin_type=True)

    def deserialize(self, data: bytes, model_cls: type[ModelT]) -> ModelT:
        raw = msgpack.unpackb(data, raw=False)
        return model_cls.model_validate(raw)

    @property
    def id(self) -> bytes:
        return b"\x02"


# Global registry: wire byte -> serializer instance
SERIALIZER_REGISTRY: dict[bytes, Serializer] = {
    b"\x01": JsonSerializer(),
    b"\x02": MsgpackSerializer(),
}


def get_serializer(t: SerializerType) -> Serializer:
    """Look up a serializer by enum. Used at IrohApp init time."""
    mapping = {SerializerType.JSON: b"\x01", SerializerType.MSGPACK: b"\x02"}
    return SERIALIZER_REGISTRY[mapping[t]]


def wrap_envelope(model: BaseModel, serializer: Serializer) -> bytes:
    """Create wire envelope: [1 byte serializer ID][serialized body]."""
    return serializer.id + serializer.serialize(model)


def unwrap_envelope(data: bytes, model_cls: type[ModelT]) -> ModelT:
    """Parse wire envelope, auto-detect serializer from first byte, and deserialize body."""
    serializer_id = data[:1]
    serializer = SERIALIZER_REGISTRY.get(serializer_id)
    if serializer is None:
        raise ValueError(f"Unknown serializer ID: {serializer_id!r}")
    return serializer.deserialize(data[1:], model_cls)
