import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.shared.base_agent import BaseAgent


class TestAgent(BaseAgent):
    """Concrete test agent."""

    def __init__(self, **kwargs):
        super().__init__(name="test_agent", **kwargs)
        self.dispatches = []
        self.started = False

    async def on_dispatch(self, message):
        self.dispatches.append(message)

    async def run(self):
        self.started = True


@pytest.fixture
def agent(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"name": "test_agent", "foo": "bar"}))
    return TestAgent(config_path=str(config_file), redis_url="redis://localhost:6379")


def test_agent_name(agent):
    assert agent.name == "test_agent"


def test_agent_loads_config(agent):
    assert agent.config["foo"] == "bar"


def test_agent_is_abstract():
    """BaseAgent requires run() and on_dispatch() to be implemented."""
    with pytest.raises(TypeError):
        BaseAgent(name="bad")


@pytest.mark.asyncio
async def test_agent_heartbeat(agent):
    agent.bus = AsyncMock()
    agent.bus.publish = AsyncMock()
    await agent.heartbeat()
    agent.bus.publish.assert_called_once()
    call_args = agent.bus.publish.call_args
    assert call_args[0][0] == "agents/heartbeat"
    assert call_args[0][1]["agent"] == "test_agent"
    assert call_args[0][1]["status"] == "alive"


@pytest.mark.asyncio
async def test_agent_dispatch_routing(agent):
    message = {
        "from": "prometheus",
        "channel": "prometheus/dispatch",
        "payload": {"task": "test", "target": "test_agent"},
    }
    await agent.on_dispatch(message)
    assert len(agent.dispatches) == 1
    assert agent.dispatches[0] == message
