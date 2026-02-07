"""Integration test: agent connects to Redis, heartbeats, receives dispatch."""

import asyncio
import pytest

try:
    import redis
    redis.Redis().ping()
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")

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
