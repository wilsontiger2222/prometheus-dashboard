"""Integration tests: Redis bus + cross-agent MockBus tests."""

import asyncio
import pytest

try:
    import redis
    redis.Redis().ping()
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

from agents.shared.base_agent import BaseAgent
from agents.shared.bus import RedisBus


class EchoAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="echo", **kwargs)
        self.received = []

    async def on_dispatch(self, message):
        self.received.append(message)

    async def run(self):
        while self._running:
            await asyncio.sleep(0.1)


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.asyncio
async def test_agent_receives_dispatch_via_bus(tmp_path):
    """Full lifecycle: agent starts, receives dispatch, heartbeats."""
    config = tmp_path / "config.json"
    config.write_text('{"name": "echo"}')

    agent = EchoAgent(config_path=str(config))

    # Start agent in background
    task = asyncio.create_task(agent.start())
    await asyncio.sleep(0.5)  # Let it connect and subscribe

    # Send a dispatch from a separate bus connection
    dispatcher = RedisBus()
    await dispatcher.connect()
    await dispatcher.publish(
        "prometheus/dispatch",
        {"task": "greet", "target": "echo", "data": "hello"},
        sender="prometheus",
    )
    await asyncio.sleep(0.5)  # Let message propagate

    # Verify agent received the dispatch
    assert len(agent.received) == 1
    assert agent.received[0]["payload"]["task"] == "greet"

    # Clean up
    await agent.stop()
    await dispatcher.disconnect()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# --- Cross-agent integration tests (no Redis required) ---

from unittest.mock import patch, AsyncMock
from agents.tests.mock_bus import MockBus
from agents.herald.agent import HeraldAgent
from agents.herald.composer import Draft
from agents.prometheus.agent import PrometheusAgent
from agents.sentinel.agent import SentinelAgent
from agents.watchdog.agent import WatchdogAgent
from agents.hivemind import Hivemind


def _make_agent(cls, name, shared_bus):
    """Create an agent with mocked config and shared MockBus."""
    with patch("agents.shared.base_agent.load_agent_config") as mock_cfg:
        mock_cfg.return_value = {"name": name}
        agent = cls(config_path="dummy.json")
        agent.bus = shared_bus
        agent._running = True
        return agent


class TestCrossAgentIntegration:

    @pytest.mark.asyncio
    async def test_dispatch_reaches_target_agent(self):
        """Prometheus dispatch message reaches targeted agent via MockBus."""
        bus = MockBus()
        await bus.connect()
        sentinel = _make_agent(SentinelAgent, "sentinel", bus)
        await bus.subscribe("prometheus/dispatch", sentinel._handle_dispatch)

        await bus.publish(
            "prometheus/dispatch",
            {"target": "sentinel", "task": "status"},
            sender="prometheus",
        )
        # Sentinel should have published a response
        status_msgs = bus.get_published("sentinel/status")
        assert len(status_msgs) >= 1

    @pytest.mark.asyncio
    async def test_dispatch_ignores_wrong_target(self):
        """Agent ignores dispatch meant for different agent."""
        bus = MockBus()
        await bus.connect()
        sentinel = _make_agent(SentinelAgent, "sentinel", bus)
        await bus.subscribe("prometheus/dispatch", sentinel._handle_dispatch)

        await bus.publish(
            "prometheus/dispatch",
            {"target": "watchdog", "task": "status"},
            sender="prometheus",
        )
        status_msgs = bus.get_published("sentinel/status")
        assert len(status_msgs) == 0

    @pytest.mark.asyncio
    async def test_herald_trigger_from_sentinel_alert(self):
        """Herald auto-drafts content when Sentinel publishes take_profit alert."""
        bus = MockBus()
        await bus.connect()

        with patch("agents.shared.base_agent.load_agent_config") as mock_cfg:
            mock_cfg.return_value = {
                "name": "herald",
                "platforms": ["twitter"],
                "require_approval": False,
                "schedule_interval_seconds": 3600,
            }
            herald = HeraldAgent(config_path="dummy.json")
            herald.bus = bus
            herald._running = True
            herald._composer.compose = AsyncMock(return_value=Draft(
                content="BTC moon!", platform="twitter",
                trigger="sentinel", topic="BTC trade win",
            ))

        await bus.subscribe("sentinel/alerts", herald._on_sentinel_alert)

        await bus.publish(
            "sentinel/alerts",
            {"type": "take_profit", "symbol": "BTC", "pnl": 500},
            sender="sentinel",
        )
        assert len(herald._queue.get_pending()) == 1

    @pytest.mark.asyncio
    async def test_herald_trigger_from_forge_deploy(self):
        """Herald auto-drafts content when Forge publishes successful deploy."""
        bus = MockBus()
        await bus.connect()

        with patch("agents.shared.base_agent.load_agent_config") as mock_cfg:
            mock_cfg.return_value = {
                "name": "herald",
                "platforms": ["twitter"],
                "require_approval": False,
                "schedule_interval_seconds": 3600,
            }
            herald = HeraldAgent(config_path="dummy.json")
            herald.bus = bus
            herald._running = True
            herald._composer.compose = AsyncMock(return_value=Draft(
                content="Deploy done!", platform="twitter",
                trigger="forge", topic="repo deployment",
            ))

        await bus.subscribe("forge/deploys", herald._on_forge_deploy)

        await bus.publish(
            "forge/deploys",
            {"success": True, "repo": "prometheus-dashboard"},
            sender="forge",
        )
        assert len(herald._queue.get_pending()) == 1

    @pytest.mark.asyncio
    async def test_heartbeat_tracking_across_agents(self):
        """Prometheus tracks heartbeats from other agents via MockBus."""
        bus = MockBus()
        await bus.connect()

        with patch("agents.shared.base_agent.load_agent_config") as mock_cfg:
            mock_cfg.return_value = {
                "name": "prometheus",
                "keyword_shortcuts": {},
                "agents": ["sentinel"],
                "response_timeout_seconds": 5,
                "heartbeat_stale_seconds": 90,
            }
            prom = PrometheusAgent(config_path="dummy.json")
            prom.bus = bus
            prom._running = True

        await bus.subscribe("agents/heartbeat", prom._on_heartbeat)

        await bus.publish(
            "agents/heartbeat",
            {"agent": "sentinel", "status": "alive"},
            sender="sentinel",
        )
        assert prom._heartbeats.is_alive("sentinel")

    @pytest.mark.asyncio
    async def test_hivemind_stop_all_agents(self):
        """Hivemind.stop() shuts down all agents."""
        with patch("agents.shared.base_agent.load_agent_config") as mock_cfg:
            mock_cfg.return_value = {"name": "test"}
            hive = Hivemind()
            for agent in hive._agents:
                agent.stop = AsyncMock()
            await hive.stop()
            for agent in hive._agents:
                agent.stop.assert_called_once()
