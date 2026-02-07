"""Hivemind launcher — starts all agents in a single event loop."""

import asyncio
import signal
import sys
from pathlib import Path

from agents.shared.config import load_agent_config
from agents.shared.logger import get_agent_logger
from agents.prometheus.agent import PrometheusAgent
from agents.sentinel.agent import SentinelAgent
from agents.watchdog.agent import WatchdogAgent
from agents.scout.agent import ScoutAgent
from agents.forge.agent import ForgeAgent
from agents.herald.agent import HeraldAgent

logger = get_agent_logger("hivemind")

AGENT_REGISTRY = {
    "prometheus": (PrometheusAgent, str(Path(__file__).parent / "prometheus" / "config.json")),
    "sentinel":   (SentinelAgent,   str(Path(__file__).parent / "sentinel" / "config.json")),
    "watchdog":   (WatchdogAgent,   str(Path(__file__).parent / "watchdog" / "config.json")),
    "scout":      (ScoutAgent,      str(Path(__file__).parent / "scout" / "config.json")),
    "forge":      (ForgeAgent,      str(Path(__file__).parent / "forge" / "config.json")),
    "herald":     (HeraldAgent,     str(Path(__file__).parent / "herald" / "config.json")),
}


class Hivemind:
    """Manages all Hivemind agents in a single event loop."""

    def __init__(self, config_path: str | None = None):
        self._agents = []
        self._tasks = []
        self._shutdown_event = asyncio.Event()
        self._bridge = None

        hive_config = {}
        if config_path:
            hive_config = load_agent_config(config_path)
        self._hive_config = hive_config

        enabled = hive_config.get("enabled_agents", list(AGENT_REGISTRY.keys()))
        redis_url = hive_config.get("redis_url", "redis://localhost:6379")
        gateway_url = hive_config.get("gateway_url", "ws://127.0.0.1:18789")
        gateway_token = hive_config.get("gateway_token", "")

        for name in enabled:
            if name not in AGENT_REGISTRY:
                logger.warning(f"Unknown agent: {name}, skipping")
                continue
            cls, default_cfg = AGENT_REGISTRY[name]
            agent = cls(
                config_path=default_cfg,
                redis_url=redis_url,
                gateway_url=gateway_url,
                gateway_token=gateway_token,
            )
            self._agents.append(agent)

        logger.info(f"Hivemind initialized with {len(self._agents)} agents")

    async def start(self):
        """Start all agents, block until shutdown signal."""
        self._install_signal_handlers()
        for agent in self._agents:
            task = asyncio.create_task(agent.start())
            self._tasks.append(task)
        logger.info("All agents launched")
        await self._shutdown_event.wait()
        await self.stop()

    async def stop(self):
        """Stop all agents gracefully."""
        logger.info("Shutting down all agents...")
        if self._bridge:
            await self._bridge.stop()
        for agent in self._agents:
            try:
                await agent.stop()
            except Exception as e:
                logger.error(f"Error stopping {agent.name}: {e}")
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("All agents stopped")

    def get_agent(self, name: str):
        """Return agent by name, or None."""
        for agent in self._agents:
            if agent.name == name:
                return agent
        return None

    def _install_signal_handlers(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                pass  # Windows fallback below
        if sys.platform == "win32":
            signal.signal(signal.SIGINT, lambda s, f: self._shutdown_event.set())


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    if config_path is None:
        default = Path(__file__).parent / "hivemind_config.json"
        if default.exists():
            config_path = str(default)

    hive = Hivemind(config_path=config_path)
    try:
        asyncio.run(hive.start())
    except KeyboardInterrupt:
        print("\nHivemind shutting down...")


if __name__ == "__main__":
    main()
