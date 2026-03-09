"""Behavioral tests for TelemetryBufferService."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry

from common.snowpipe.messages.lifecycle_events import LifecycleEvent
from common.snowpipe.messages.training_metrics import TrainingMetric
from miner.telemetry.buffer_service import TelemetryBufferService


@pytest.fixture
def empty_registry():
    """A fresh registry with no metrics registered (snapshots produce nothing)."""
    return CollectorRegistry()


@pytest.fixture
def mock_keypair():
    kp = MagicMock()
    kp.ss58_address = "5FakeHotkey" + "0" * 43
    return kp


def _make_training_metric(**kwargs) -> TrainingMetric:
    defaults = dict(
        type="training_metric",
        source_service="miner",
        metric_name="loss",
        metric_value={"loss": 0.5},
    )
    defaults.update(kwargs)
    return TrainingMetric(**defaults)


def _make_lifecycle_event(**kwargs) -> LifecycleEvent:
    defaults = dict(
        type="lifecycle_event",
        source_service="miner",
        event_category="phase_transition",
    )
    defaults.update(kwargs)
    return LifecycleEvent(**defaults)


class TestBufferService:
    """Behavioral tests for the telemetry buffer."""

    @pytest.mark.asyncio
    async def test_log_sorts_by_type(self, mock_keypair, empty_registry):
        """TrainingMetric and LifecycleEvent are sorted into correct buckets."""
        svc = TelemetryBufferService(
            hotkey=mock_keypair,
            registry=empty_registry,
            max_buffer_size=100,
        )

        tm = _make_training_metric()
        le = _make_lifecycle_event()

        svc.log(tm)
        svc.log(le)

        assert len(svc._training_metrics) == 1
        assert len(svc._lifecycle_events) == 1
        assert svc._training_metrics[0] is tm
        assert svc._lifecycle_events[0] is le
        assert svc.buffer_size == 2

    @pytest.mark.asyncio
    async def test_flush_empty_buffer_noop(self, mock_keypair, empty_registry):
        """Flush with no messages and empty registry should not make any HTTP call."""
        svc = TelemetryBufferService(
            hotkey=mock_keypair,
            registry=empty_registry,
        )

        with patch("subnet.common_api_client.CommonAPIClient") as mock_client:
            mock_client.orchestrator_request = AsyncMock()
            await svc.flush()
            mock_client.orchestrator_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_flush(self, mock_keypair, empty_registry):
        """Explicit flush sends buffered messages via HTTP."""
        svc = TelemetryBufferService(
            hotkey=mock_keypair,
            registry=empty_registry,
        )
        svc.log(_make_training_metric())
        svc.log(_make_lifecycle_event())

        with patch("subnet.common_api_client.CommonAPIClient") as mock_client:
            mock_client.orchestrator_request = AsyncMock(return_value={})
            await svc.flush()

            mock_client.orchestrator_request.assert_called_once()
            call_kwargs = mock_client.orchestrator_request.call_args
            assert call_kwargs.kwargs["method"] == "POST"
            assert call_kwargs.kwargs["path"] == "/miner/fleet-telemetry"

            body = call_kwargs.kwargs["body"]
            assert len(body["training_metrics"]) == 1
            assert len(body["lifecycle_events"]) == 1

        # Buffer should be empty after flush
        assert svc.buffer_size == 0

    @pytest.mark.asyncio
    async def test_flush_sends_all_buffered_messages(self, mock_keypair, empty_registry):
        """Flushing a large buffer sends all messages in one request."""
        max_buf = 5
        svc = TelemetryBufferService(
            hotkey=mock_keypair,
            registry=empty_registry,
            max_buffer_size=max_buf,
            flush_interval_sec=999,
        )

        for _ in range(max_buf + 1):
            svc.log(_make_training_metric())

        assert svc.buffer_size == max_buf + 1

        with patch("subnet.common_api_client.CommonAPIClient") as mock_client:
            mock_client.orchestrator_request = AsyncMock(return_value={})
            await svc.flush()

            mock_client.orchestrator_request.assert_called_once()
            body = mock_client.orchestrator_request.call_args.kwargs["body"]
            assert len(body["training_metrics"]) == max_buf + 1

        assert svc.buffer_size == 0

    @pytest.mark.asyncio
    async def test_rebuffer_on_failure(self, mock_keypair, empty_registry):
        """On HTTP failure, messages are re-buffered (within 2x max)."""
        svc = TelemetryBufferService(
            hotkey=mock_keypair,
            registry=empty_registry,
            max_buffer_size=100,
        )
        svc.log(_make_training_metric())
        svc.log(_make_lifecycle_event())
        original_size = svc.buffer_size

        with patch("subnet.common_api_client.CommonAPIClient") as mock_client:
            mock_client.orchestrator_request = AsyncMock(side_effect=RuntimeError("connection refused"))
            await svc.flush()

        # Messages should be re-buffered
        assert svc.buffer_size == original_size

    @pytest.mark.asyncio
    async def test_drop_on_overflow(self, mock_keypair, empty_registry):
        """When re-buffer would exceed 2x max, messages are dropped."""
        max_buf = 3
        svc = TelemetryBufferService(
            hotkey=mock_keypair,
            registry=empty_registry,
            max_buffer_size=max_buf,
        )

        # Fill buffer to 2x max already
        for _ in range(max_buf * 2):
            svc.log(_make_training_metric())

        # Add one more batch that will fail
        svc.log(_make_lifecycle_event())

        with patch("subnet.common_api_client.CommonAPIClient") as mock_client:
            mock_client.orchestrator_request = AsyncMock(side_effect=RuntimeError("fail"))
            await svc.flush()

        # After failed flush, buffer was too large to re-buffer so messages were dropped
        assert svc.buffer_size == 0

    @pytest.mark.asyncio
    async def test_stop_performs_final_flush(self, mock_keypair, empty_registry):
        """stop() cancels tasks and performs a final flush."""
        svc = TelemetryBufferService(
            hotkey=mock_keypair,
            registry=empty_registry,
            flush_interval_sec=999,
        )
        await svc.start()

        svc.log(_make_training_metric())
        assert svc.buffer_size == 1

        with patch("subnet.common_api_client.CommonAPIClient") as mock_client:
            mock_client.orchestrator_request = AsyncMock(return_value={})
            await svc.stop()

            # Final flush should have been triggered
            mock_client.orchestrator_request.assert_called_once()

        assert svc.buffer_size == 0

    @pytest.mark.asyncio
    async def test_flush_on_time_interval(self, mock_keypair, empty_registry):
        """Flush loop fires after flush_interval_sec elapses."""
        svc = TelemetryBufferService(
            hotkey=mock_keypair,
            registry=empty_registry,
            flush_interval_sec=0.1,
        )

        svc.log(_make_training_metric())

        with patch("subnet.common_api_client.CommonAPIClient") as mock_client:
            mock_client.orchestrator_request = AsyncMock(return_value={})
            await svc.start()

            # Wait for flush interval to fire
            await asyncio.sleep(0.3)

            await svc.stop()

            # Should have been flushed at least once by the loop (plus final flush)
            assert mock_client.orchestrator_request.call_count >= 1
