"""Unit tests for ReceiverProcess — subprocess isolation and SharedMemory cache."""

from __future__ import annotations

import time
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import MagicMock, patch

import pytest

from common.iroh.receiver_process import ReceiverProcess, _shm_name


# ---------------------------------------------------------------------------
# _shm_name helper
# ---------------------------------------------------------------------------


class TestShmName:
    def test_deterministic(self):
        assert _shm_name("abc") == _shm_name("abc")

    def test_different_ids_different_names(self):
        assert _shm_name("id-1") != _shm_name("id-2")

    def test_length_within_macos_limit(self):
        # macOS shm_open limit is 31 chars (including leading /)
        name = _shm_name("some-very-long-activation-id-that-might-be-problematic")
        assert len(name) <= 30  # 30 chars leaves room for OS-added /

    def test_prefix(self):
        name = _shm_name("test-id")
        assert name.startswith("iota_")


# ---------------------------------------------------------------------------
# ReceiverProcess — SharedMemory cache (no subprocess needed)
# ---------------------------------------------------------------------------


class TestReceiverProcessCache:
    """Test the SharedMemory cache operations without actually spawning a subprocess."""

    def _make_process(self, cache_ttl: float = 300.0, max_cache_size: int = 100) -> ReceiverProcess:
        """Create a ReceiverProcess and manually set up its Manager for testing."""
        import multiprocessing

        rp = ReceiverProcess(
            seed="test-seed",
            max_message_size=1024,
            cache_ttl=cache_ttl,
            max_cache_size=max_cache_size,
        )
        # Manually create manager and metadata dict (normally done in start())
        rp._manager = multiprocessing.Manager()
        rp._metadata_dict = rp._manager.dict()
        return rp

    def _cleanup(self, rp: ReceiverProcess) -> None:
        """Clean up SharedMemory and Manager."""
        rp._cleanup_all_shm()
        if rp._manager is not None:
            try:
                rp._manager.shutdown()
            except Exception:
                pass

    def test_cache_and_read_via_shm(self):
        rp = self._make_process()
        try:
            activation_id = "test-activation-1"
            data = b"hello world tensor data"

            rp.cache_activation(activation_id, data)

            # Verify metadata is set
            meta = rp._metadata_dict[activation_id]
            shm_name, size, ts = meta
            assert size == len(data)
            assert shm_name == _shm_name(activation_id)

            # Verify data can be read from SharedMemory
            shm = SharedMemory(name=shm_name, create=False)
            try:
                read_data = bytes(shm.buf[:size])
                assert read_data == data
            finally:
                shm.close()
        finally:
            self._cleanup(rp)

    def test_cache_overwrites_existing(self):
        rp = self._make_process()
        try:
            activation_id = "overwrite-test"
            rp.cache_activation(activation_id, b"original")
            rp.cache_activation(activation_id, b"updated")

            meta = rp._metadata_dict[activation_id]
            shm_name, size, _ = meta
            shm = SharedMemory(name=shm_name, create=False)
            try:
                assert bytes(shm.buf[:size]) == b"updated"
            finally:
                shm.close()
        finally:
            self._cleanup(rp)

    def test_eviction_on_capacity(self):
        rp = self._make_process(max_cache_size=3)
        try:
            for i in range(5):
                rp.cache_activation(f"act-{i}", f"data-{i}".encode())

            # Only the last 3 should remain
            assert len(rp._shm_blocks) == 3
            assert "act-0" not in rp._shm_blocks
            assert "act-1" not in rp._shm_blocks
            assert "act-2" in rp._shm_blocks
            assert "act-3" in rp._shm_blocks
            assert "act-4" in rp._shm_blocks
        finally:
            self._cleanup(rp)

    def test_eviction_on_ttl(self):
        rp = self._make_process(cache_ttl=0.1)
        try:
            rp.cache_activation("old", b"old-data")
            time.sleep(0.2)  # Wait for TTL to expire

            # Caching a new activation triggers eviction of expired entries
            rp.cache_activation("new", b"new-data")

            assert "old" not in rp._shm_blocks
            assert "old" not in rp._metadata_dict
            assert "new" in rp._shm_blocks
        finally:
            self._cleanup(rp)

    def test_cleanup_all_shm(self):
        rp = self._make_process()
        try:
            rp.cache_activation("a", b"data-a")
            rp.cache_activation("b", b"data-b")
            assert len(rp._shm_blocks) == 2

            rp._cleanup_all_shm()
            assert len(rp._shm_blocks) == 0
        finally:
            self._cleanup(rp)

    def test_cache_requires_started(self):
        """cache_activation should raise if the process was never started."""
        rp = ReceiverProcess(
            seed="test",
            max_message_size=1024,
            cache_ttl=300.0,
        )
        with pytest.raises(RuntimeError, match="not started"):
            rp.cache_activation("x", b"y")

    def test_remove_entry_missing_key_no_error(self):
        rp = self._make_process()
        try:
            # Should not raise
            rp._remove_entry("nonexistent")
        finally:
            self._cleanup(rp)

    def test_is_alive_no_process(self):
        rp = ReceiverProcess(seed="test", max_message_size=1024, cache_ttl=300.0)
        assert rp.is_alive() is False

    def test_node_id_none_initially(self):
        rp = ReceiverProcess(seed="test", max_message_size=1024, cache_ttl=300.0)
        assert rp.node_id is None


# ---------------------------------------------------------------------------
# ReceiverProcess — check_status_queue
# ---------------------------------------------------------------------------


class TestStatusQueue:
    def test_check_status_queue_empty(self):
        rp = ReceiverProcess(seed="test", max_message_size=1024, cache_ttl=300.0)
        assert rp.check_status_queue() == []

    def test_check_status_queue_drains(self):
        rp = ReceiverProcess(seed="test", max_message_size=1024, cache_ttl=300.0)
        rp._status_queue.put(("unhealthy", "degraded"))
        rp._status_queue.put(("started", "node123"))

        # multiprocessing.Queue put is async; wait for items to be available
        time.sleep(0.1)

        messages = rp.check_status_queue()
        assert len(messages) == 2
        assert messages[0] == ("unhealthy", "degraded")
        assert messages[1] == ("started", "node123")

        # Second call should be empty
        assert rp.check_status_queue() == []


# ---------------------------------------------------------------------------
# ReceiverProcess — lifecycle (mocked subprocess)
# ---------------------------------------------------------------------------


class TestReceiverProcessLifecycle:
    @pytest.mark.asyncio
    async def test_start_returns_node_id(self):
        """Test that start() waits for the subprocess to report a node_id."""
        rp = ReceiverProcess(seed="test-seed", max_message_size=1024, cache_ttl=300.0)

        # Mock Process so we don't actually spawn
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mock_proc.pid = 12345

        with patch("common.iroh.receiver_process._mp_ctx.Process", return_value=mock_proc):
            with patch("common.iroh.receiver_process._mp_ctx.Manager") as mock_mgr_cls:
                mock_mgr = MagicMock()
                mock_mgr.dict.return_value = {}
                mock_mgr_cls.return_value = mock_mgr

                # Simulate the subprocess sending "started" message
                rp._status_queue.put(("started", "test-node-id-abc123"))

                node_id = await rp.start()
                assert node_id == "test-node-id-abc123"
                assert rp.node_id == "test-node-id-abc123"
                mock_proc.start.assert_called_once()

        # Cleanup
        rp._process = None
        if rp._manager and rp._manager is not mock_mgr:
            rp._manager.shutdown()

    @pytest.mark.asyncio
    async def test_start_timeout_raises(self):
        """Test that start() raises TimeoutError if subprocess doesn't start."""
        rp = ReceiverProcess(seed="test-seed", max_message_size=1024, cache_ttl=300.0)

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mock_proc.pid = 12345

        with patch("common.iroh.receiver_process._mp_ctx.Process", return_value=mock_proc):
            with patch("common.iroh.receiver_process._mp_ctx.Manager") as mock_mgr_cls:
                mock_mgr = MagicMock()
                mock_mgr.dict.return_value = {}
                mock_mgr_cls.return_value = mock_mgr

                # Don't put any message — should timeout
                with pytest.raises(TimeoutError):
                    await rp._wait_for_started(timeout=0.5)

        rp._process = None

    @pytest.mark.asyncio
    async def test_start_subprocess_dies_raises(self):
        """Test that start() raises RuntimeError if subprocess exits early."""
        rp = ReceiverProcess(seed="test-seed", max_message_size=1024, cache_ttl=300.0)
        rp._process = MagicMock()
        rp._process.is_alive.return_value = False
        rp._process.exitcode = 1

        with pytest.raises(RuntimeError, match="died during startup"):
            await rp._wait_for_started(timeout=2.0)

    @pytest.mark.asyncio
    async def test_stop_kills_process_and_cleans_shm(self):
        """Test that stop() kills the subprocess and cleans up SharedMemory."""
        import multiprocessing

        rp = ReceiverProcess(seed="test-seed", max_message_size=1024, cache_ttl=300.0)
        rp._manager = multiprocessing.Manager()
        rp._metadata_dict = rp._manager.dict()

        # Cache something
        rp.cache_activation("stop-test", b"data")
        assert len(rp._shm_blocks) == 1

        # Mock the process
        rp._process = MagicMock()
        rp._process.is_alive.return_value = False
        rp._process.pid = None

        await rp.stop()

        assert len(rp._shm_blocks) == 0
        assert rp._manager is None
        assert rp._metadata_dict is None
        assert rp._node_id is None

    @pytest.mark.asyncio
    async def test_restart_preserves_cache(self):
        """Test that restart() preserves SharedMemory blocks and metadata."""
        import multiprocessing

        rp = ReceiverProcess(seed="test-seed", max_message_size=1024, cache_ttl=300.0)
        rp._manager = multiprocessing.Manager()
        rp._metadata_dict = rp._manager.dict()

        # Cache an activation
        rp.cache_activation("restart-test", b"preserved-data")

        # Set up a mock process
        rp._process = MagicMock()
        rp._process.is_alive.return_value = False
        rp._process.pid = None

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mock_proc.pid = 99999

        # Use side_effect on start() to inject the "started" message
        # after _drain_queue() has already run (which happens before Process.start)
        def on_start():
            rp._status_queue.put(("started", "new-node-id"))

        mock_proc.start.side_effect = on_start

        with patch("common.iroh.receiver_process._mp_ctx.Process", return_value=mock_proc):
            node_id = await rp.restart()

        assert node_id == "new-node-id"

        # Cache should still be there
        assert "restart-test" in rp._shm_blocks
        meta = rp._metadata_dict["restart-test"]
        shm_name, size, _ = meta
        shm = SharedMemory(name=shm_name, create=False)
        try:
            assert bytes(shm.buf[:size]) == b"preserved-data"
        finally:
            shm.close()

        # Cleanup
        rp._process = None
        rp._cleanup_all_shm()
        try:
            rp._manager.shutdown()
        except Exception:
            pass
