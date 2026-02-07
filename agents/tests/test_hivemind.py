"""Tests for the Hivemind launcher."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from agents.hivemind import Hivemind, AGENT_REGISTRY


@pytest.fixture
def mock_configs():
    """Patch all agent config loading."""
    with patch("agents.shared.base_agent.load_agent_config") as mock_config:
        mock_config.return_value = {"name": "test"}
        yield mock_config


class TestHivemind:
    def test_creates_all_agents_by_default(self, mock_configs):
        hive = Hivemind()
        assert len(hive._agents) == 6
        names = {a.name for a in hive._agents}
        assert names == {"prometheus", "sentinel", "watchdog", "scout", "forge", "herald"}

    def test_creates_subset_from_config(self, mock_configs):
        with patch("agents.hivemind.load_agent_config") as hive_cfg:
            hive_cfg.return_value = {
                "enabled_agents": ["prometheus", "sentinel"],
                "redis_url": "redis://localhost:6379",
            }
            hive = Hivemind(config_path="dummy.json")
            assert len(hive._agents) == 2

    def test_get_agent_returns_correct_agent(self, mock_configs):
        hive = Hivemind()
        prom = hive.get_agent("prometheus")
        assert prom is not None
        assert prom.name == "prometheus"

    def test_get_agent_returns_none_for_unknown(self, mock_configs):
        hive = Hivemind()
        assert hive.get_agent("nonexistent") is None

    @pytest.mark.asyncio
    async def test_stop_calls_agent_stop(self, mock_configs):
        hive = Hivemind()
        for agent in hive._agents:
            agent.stop = AsyncMock()
        await hive.stop()
        for agent in hive._agents:
            agent.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_handles_agent_exception(self, mock_configs):
        hive = Hivemind()
        hive._agents[0].stop = AsyncMock(side_effect=RuntimeError("fail"))
        for agent in hive._agents[1:]:
            agent.stop = AsyncMock()
        await hive.stop()
        # All other agents still get stopped
        for agent in hive._agents[1:]:
            agent.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_event_triggers_stop(self, mock_configs):
        hive = Hivemind()
        for agent in hive._agents:
            agent.start = AsyncMock(side_effect=asyncio.CancelledError)
            agent.stop = AsyncMock()
        # Set shutdown immediately
        hive._shutdown_event.set()
        await hive.start()
        for agent in hive._agents:
            agent.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_hivemind_starts_bridge(self, mock_configs):
        with patch("agents.hivemind.load_agent_config") as hive_cfg:
            hive_cfg.return_value = {
                "telegram_bridge_enabled": True,
                "gateway_url": "ws://localhost:18789",
            }
            hive = Hivemind(config_path="dummy.json")
            for agent in hive._agents:
                agent.start = AsyncMock(side_effect=asyncio.CancelledError)
                agent.stop = AsyncMock()
            with patch("agents.hivemind.TelegramBridge") as MockBridge:
                mock_bridge = AsyncMock()
                MockBridge.return_value = mock_bridge
                mock_bridge.start = AsyncMock(side_effect=asyncio.CancelledError)
                mock_bridge.stop = AsyncMock()
                hive._shutdown_event.set()
                await hive.start()
                MockBridge.assert_called_once()
