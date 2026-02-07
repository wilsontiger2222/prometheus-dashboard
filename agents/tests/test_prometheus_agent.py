"""Tests for the PrometheusAgent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.prometheus.agent import PrometheusAgent


@pytest.fixture
def prometheus():
    with patch("agents.shared.base_agent.load_agent_config") as mock_config:
        mock_config.return_value = {
            "name": "prometheus",
            "response_timeout_seconds": 2,
            "keyword_shortcuts": {
                "/trade": "sentinel",
                "/status": "watchdog",
                "/research": "scout",
            },
            "agents": ["sentinel", "watchdog", "scout"],
            "heartbeat_stale_seconds": 90,
        }
        agent = PrometheusAgent(config_path="dummy.json")
        agent.bus = AsyncMock()
        agent._running = True
        return agent


class TestPrometheusAgent:
    def test_prometheus_name(self, prometheus):
        assert prometheus.name == "prometheus"

    def test_loads_shortcuts(self, prometheus):
        assert "/trade" in prometheus._router._shortcuts

    def test_loads_agents_list(self, prometheus):
        assert "sentinel" in prometheus._agents

    @pytest.mark.asyncio
    async def test_handle_message_keyword(self, prometheus):
        """Keyword shortcut dispatches to correct agent."""
        await prometheus.handle_message("/trade BTC long")
        prometheus.bus.publish.assert_called()
        call_args = prometheus.bus.publish.call_args_list[0]
        assert call_args[0][0] == "prometheus/dispatch"

    @pytest.mark.asyncio
    async def test_handle_message_unknown(self, prometheus):
        """Unknown message with no AI returns None."""
        prometheus._router.route_ai = AsyncMock(return_value=None)
        result = await prometheus.handle_message("asdfghjkl")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_heartbeat(self, prometheus):
        """Heartbeat messages update the tracker."""
        message = {"payload": {"agent": "sentinel", "status": "alive"}}
        await prometheus._on_heartbeat("agents/heartbeat", message)
        status = prometheus._heartbeats.get_status()
        assert "sentinel" in status
        assert status["sentinel"]["alive"] is True

    @pytest.mark.asyncio
    async def test_on_dispatch_status(self, prometheus):
        """Dispatch with task=status returns agent liveness."""
        prometheus._heartbeats.record("sentinel")
        prometheus._heartbeats.record("watchdog")
        message = {"payload": {"task": "status"}}
        await prometheus.on_dispatch(message)
        prometheus.bus.publish.assert_called()
