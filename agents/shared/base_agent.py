"""Base class for all Hivemind agents."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from agents.shared.bus import RedisBus
from agents.shared.config import load_agent_config
from agents.shared.logger import get_agent_logger
from agents.shared.ai_client import AIClient


class BaseAgent(ABC):
    """Abstract base class. All Hivemind agents inherit from this.

    Provides:
    - Redis message bus connection
    - Heartbeat broadcasting
    - Structured logging
    - AI client access with token tracking
    - Config loading
    """

    def __init__(
        self,
        name: str,
        config_path: str | None = None,
        redis_url: str = "redis://localhost:6379",
        gateway_url: str = "ws://127.0.0.1:18789",
        gateway_token: str = "",
    ):
        self.name = name
        self.config = {}
        if config_path:
            self.config = load_agent_config(config_path)
        self.bus = RedisBus(redis_url=redis_url)
        self.logger = get_agent_logger(name)
        self.ai = AIClient(gateway_url=gateway_url, token=gateway_token)
        self._running = False
        self._heartbeat_interval = 30

    async def start(self):
        """Connect to bus, subscribe to dispatch channel, start heartbeat, then run."""
        await self.bus.connect()
        await self.bus.subscribe("prometheus/dispatch", self._handle_dispatch)
        self._running = True
        self.logger.info(f"Agent {self.name} started")

        await asyncio.gather(
            self._heartbeat_loop(),
            self.run(),
        )

    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        await self.bus.disconnect()
        self.logger.info(f"Agent {self.name} stopped")

    async def heartbeat(self):
        """Publish a single heartbeat to the bus."""
        await self.bus.publish(
            "agents/heartbeat",
            {"agent": self.name, "status": "alive"},
            sender=self.name,
        )

    async def _heartbeat_loop(self):
        while self._running:
            try:
                await self.heartbeat()
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
            await asyncio.sleep(self._heartbeat_interval)

    async def _handle_dispatch(self, channel: str, message: dict):
        """Route dispatch messages to the agent if targeted."""
        payload = message.get("payload", {})
        target = payload.get("target")
        if target is None or target == self.name:
            await self.on_dispatch(message)

    async def alert_telegram(self, message: str):
        """Send urgent message directly to Telegram via the bus."""
        await self.bus.publish(
            "prometheus/telegram",
            {"text": message, "urgent": True},
            sender=self.name,
        )

    @abstractmethod
    async def run(self):
        """Main agent loop. Override in subclass."""
        ...

    @abstractmethod
    async def on_dispatch(self, message: dict):
        """Handle task dispatched by Prometheus. Override in subclass."""
        ...
