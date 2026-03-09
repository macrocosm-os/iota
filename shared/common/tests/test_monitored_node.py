"""Unit tests for MonitoredNode background health monitoring."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from common.iroh.monitored_node import (
    MonitoredNode,
    NodeHealth,
)


# ---------------------------------------------------------------------------
# Helpers — mock Iroh node
# ---------------------------------------------------------------------------


def _make_mock_node(
    status_side_effect=None,
    home_relay_return=None,
    home_relay_side_effect=None,
):
    """Create a mock Iroh node with configurable behaviour."""
    node = MagicMock()

    # node.node().status() -> awaitable
    status_mock = AsyncMock(side_effect=status_side_effect)
    node_inner = MagicMock()
    node_inner.status = status_mock
    node.node.return_value = node_inner

    # node.net().home_relay() -> awaitable
    if home_relay_side_effect is not None:
        relay_mock = AsyncMock(side_effect=home_relay_side_effect)
    else:
        relay_mock = AsyncMock(return_value=home_relay_return)
    net_inner = MagicMock()
    net_inner.home_relay = relay_mock
    node.net.return_value = net_inner

    return node


# ---------------------------------------------------------------------------
# Health check logic
# ---------------------------------------------------------------------------


class TestHealthCheckLogic:
    @pytest.mark.asyncio
    async def test_healthy_when_status_ok_and_relay_present(self):
        node = _make_mock_node(home_relay_return="https://relay.example.com")
        m = MonitoredNode(node=node, check_timeout=1.0)

        result = await m.check_health()

        assert result.health == NodeHealth.HEALTHY
        assert result.status_ok is True
        assert result.relay_connected is True
        assert result.home_relay == "https://relay.example.com"
        assert result.error is None
        assert m.is_healthy is True

    @pytest.mark.asyncio
    async def test_dead_when_status_times_out(self):
        async def slow_status():
            await asyncio.sleep(10)

        node = _make_mock_node(status_side_effect=slow_status)
        m = MonitoredNode(node=node, check_timeout=0.05)

        result = await m.check_health()

        assert result.health == NodeHealth.DEAD
        assert result.status_ok is False
        assert "status() failed" in result.error
        assert m.is_healthy is False

    @pytest.mark.asyncio
    async def test_dead_when_status_raises(self):
        node = _make_mock_node(status_side_effect=RuntimeError("node crashed"))
        m = MonitoredNode(node=node, check_timeout=1.0)

        result = await m.check_health()

        assert result.health == NodeHealth.DEAD
        assert "status() failed" in result.error
        assert "node crashed" in result.error

    @pytest.mark.asyncio
    async def test_degraded_when_relay_is_none(self):
        node = _make_mock_node(home_relay_return=None)
        m = MonitoredNode(node=node, check_timeout=1.0)

        result = await m.check_health()

        assert result.health == NodeHealth.DEGRADED
        assert result.status_ok is True
        assert result.relay_connected is False

    @pytest.mark.asyncio
    async def test_degraded_when_relay_times_out(self):
        async def slow_relay():
            await asyncio.sleep(10)

        node = _make_mock_node(home_relay_side_effect=slow_relay)
        m = MonitoredNode(node=node, check_timeout=0.05)

        result = await m.check_health()

        assert result.health == NodeHealth.DEGRADED
        assert result.status_ok is True
        assert result.relay_connected is False
        assert "home_relay() timed out" in result.error

    @pytest.mark.asyncio
    async def test_dead_when_no_node_set(self):
        m = MonitoredNode(node=None, check_timeout=1.0)

        result = await m.check_health()

        assert result.health == NodeHealth.DEAD
        assert result.error == "no node set"


# ---------------------------------------------------------------------------
# Monitoring lifecycle
# ---------------------------------------------------------------------------


class TestMonitoringLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_task_and_stop_cancels(self):
        m = MonitoredNode(check_interval=100.0)
        assert m.is_monitoring is False

        m.start_monitoring()
        assert m.is_monitoring is True

        await m.stop_monitoring()
        assert m.is_monitoring is False

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self):
        m = MonitoredNode(check_interval=100.0)
        m.start_monitoring()
        task1 = m._monitor_task

        m.start_monitoring()
        task2 = m._monitor_task

        assert task1 is task2
        await m.stop_monitoring()

    @pytest.mark.asyncio
    async def test_set_node_resets_counters(self):
        node1 = _make_mock_node(status_side_effect=RuntimeError("dead"))
        m = MonitoredNode(node=node1, check_timeout=1.0)

        # Drive up failures
        await m.check_health()
        await m.check_health()
        assert m.consecutive_failures == 2
        assert m.is_healthy is False

        # Set a new node — should reset
        node2 = _make_mock_node(home_relay_return="https://relay.example.com")
        m.set_node(node2)
        assert m.consecutive_failures == 0
        assert m.is_healthy is True
        assert m.last_check is None


# ---------------------------------------------------------------------------
# Callback integration
# ---------------------------------------------------------------------------


class TestCallbackIntegration:
    @pytest.mark.asyncio
    async def test_callback_fires_when_unhealthy(self):
        callback_results = []

        async def on_unhealthy(monitored, result):
            callback_results.append(result)

        node = _make_mock_node(status_side_effect=RuntimeError("dead"))
        m = MonitoredNode(
            node=node,
            on_unhealthy=on_unhealthy,
            check_interval=0.01,
            check_timeout=1.0,
        )

        m.start_monitoring()
        await asyncio.sleep(0.1)
        await m.stop_monitoring()

        assert len(callback_results) > 0
        assert all(r.health != NodeHealth.HEALTHY for r in callback_results)

    @pytest.mark.asyncio
    async def test_callback_does_not_fire_when_healthy(self):
        callback_results = []

        async def on_unhealthy(monitored, result):
            callback_results.append(result)

        node = _make_mock_node(home_relay_return="https://relay.example.com")
        m = MonitoredNode(
            node=node,
            on_unhealthy=on_unhealthy,
            check_interval=0.01,
            check_timeout=1.0,
        )

        m.start_monitoring()
        await asyncio.sleep(0.1)
        await m.stop_monitoring()

        assert len(callback_results) == 0

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_kill_monitor(self):
        call_count = 0

        async def bad_callback(monitored, result):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("callback exploded")

        node = _make_mock_node(status_side_effect=RuntimeError("dead"))
        m = MonitoredNode(
            node=node,
            on_unhealthy=bad_callback,
            check_interval=0.01,
            check_timeout=1.0,
        )

        m.start_monitoring()
        await asyncio.sleep(0.1)
        await m.stop_monitoring()

        # Monitor survived the exception and kept calling
        assert call_count >= 2
        assert m.is_monitoring is False  # stopped cleanly

    @pytest.mark.asyncio
    async def test_consecutive_failures_increment_and_reset(self):
        node = _make_mock_node(status_side_effect=RuntimeError("dead"))
        m = MonitoredNode(node=node, check_timeout=1.0)

        await m.check_health()
        assert m.consecutive_failures == 1
        await m.check_health()
        assert m.consecutive_failures == 2

        # Switch to a healthy node
        healthy_node = _make_mock_node(home_relay_return="https://relay.example.com")
        m._node = healthy_node
        await m.check_health()
        assert m.consecutive_failures == 0
        assert m.is_healthy is True
