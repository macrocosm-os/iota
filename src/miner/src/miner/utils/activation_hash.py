"""
BLAKE3 hash computation for activation integrity verification.

Hashes the full tensor bytes using BLAKE3 which is designed for
high-throughput data hashing (multi-threaded, SIMD-accelerated).
"""

import blake3


def compute_activation_hash(tensor_bytes: bytes) -> str:
    """
    Compute BLAKE3 hash of the full activation tensor bytes.

    Args:
        tensor_bytes: Raw bytes of the serialized tensor

    Returns:
        Hex-encoded BLAKE3 hash (64 chars)
    """
    return blake3.blake3(tensor_bytes).hexdigest()


def verify_activation_hash(tensor_bytes: bytes, expected_hash: str) -> bool:
    """
    Verify tensor bytes match expected hash.

    Args:
        tensor_bytes: Raw bytes of the serialized tensor
        expected_hash: Expected hex-encoded hash

    Returns:
        True if hash matches, False otherwise
    """
    return compute_activation_hash(tensor_bytes) == expected_hash
