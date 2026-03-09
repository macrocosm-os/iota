"""Tests for activation hash computation and verification."""


from miner.utils.activation_hash import (
    compute_activation_hash,
    verify_activation_hash,
)


class TestComputeActivationHash:
    """Tests for compute_activation_hash function."""

    def test_small_data(self):
        data = b"small test data"
        hash1 = compute_activation_hash(data)

        assert len(hash1) == 64
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_same_data_produces_same_hash(self):
        data = b"test data for hashing"
        hash1 = compute_activation_hash(data)
        hash2 = compute_activation_hash(data)
        assert hash1 == hash2

    def test_different_data_produces_different_hash(self):
        data1 = b"test data 1"
        data2 = b"test data 2"
        hash1 = compute_activation_hash(data1)
        hash2 = compute_activation_hash(data2)
        assert hash1 != hash2

    def test_large_data_hashes_full_content(self):
        """Modifying any part of large data should change the hash."""
        size = 256 * 1024  # 256KB
        data = b"A" * size
        hash1 = compute_activation_hash(data)

        data_modified_middle = bytearray(data)
        data_modified_middle[size // 2] = ord("X")
        hash2 = compute_activation_hash(bytes(data_modified_middle))
        assert hash1 != hash2

    def test_detects_start_changes(self):
        data = b"A" * 200_000
        hash1 = compute_activation_hash(data)

        data_modified = b"X" + b"A" * 199_999
        hash2 = compute_activation_hash(data_modified)
        assert hash1 != hash2

    def test_detects_end_changes(self):
        data = b"A" * 200_000
        hash1 = compute_activation_hash(data)

        data_modified = b"A" * 199_999 + b"X"
        hash2 = compute_activation_hash(data_modified)
        assert hash1 != hash2

    def test_detects_truncation(self):
        data = b"A" * 1000
        hash1 = compute_activation_hash(data)

        truncated = b"A" * 500
        hash2 = compute_activation_hash(truncated)
        assert hash1 != hash2

    def test_detects_extension(self):
        data = b"A" * 1000
        hash1 = compute_activation_hash(data)

        extended = b"A" * 1500
        hash2 = compute_activation_hash(extended)
        assert hash1 != hash2

    def test_empty_data(self):
        hash_result = compute_activation_hash(b"")
        assert len(hash_result) == 64


class TestVerifyActivationHash:
    """Tests for verify_activation_hash function."""

    def test_valid_hash_returns_true(self):
        data = b"test data for verification"
        expected_hash = compute_activation_hash(data)
        assert verify_activation_hash(data, expected_hash) is True

    def test_invalid_hash_returns_false(self):
        data = b"test data"
        wrong_hash = "a" * 64
        assert verify_activation_hash(data, wrong_hash) is False

    def test_corrupted_data_fails_verification(self):
        data = b"original data"
        expected_hash = compute_activation_hash(data)

        corrupted_data = b"corrupted data"
        assert verify_activation_hash(corrupted_data, expected_hash) is False

    def test_truncated_data_fails_verification(self):
        data = b"A" * 1000
        expected_hash = compute_activation_hash(data)

        truncated = data[:500]
        assert verify_activation_hash(truncated, expected_hash) is False

    def test_large_data_verification(self):
        large_data = b"X" * (256 * 1024)
        expected_hash = compute_activation_hash(large_data)
        assert verify_activation_hash(large_data, expected_hash) is True
