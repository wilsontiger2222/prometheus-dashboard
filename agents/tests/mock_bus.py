"""In-memory mock bus for testing without Redis."""

import asyncio
import inspect
from collections import defaultdict
from datetime import datetime, timezone


class MockBus:
    """Drop-in replacement for RedisBus that delivers messages in-process."""

    def __init__(self):
        self._handlers: dict[str, list] = defaultdict(list)
        self.published: list[tuple[str, dict]] = []

    async def connect(self):
        pass

    async def disconnect(self):
        self._handlers.clear()
        self.published.clear()

    async def publish(self, channel: str, payload: dict, sender: str = "unknown"):
        envelope = {
            "from": sender,
            "channel": channel,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        self.published.append((channel, envelope))
        for handler in list(self._handlers.get(channel, [])):
            if inspect.iscoroutinefunction(handler):
                await handler(channel, envelope)
            else:
                handler(channel, envelope)

    async def subscribe(self, channel: str, handler):
        self._handlers[channel].append(handler)

    async def unsubscribe(self, channel: str):
        self._handlers.pop(channel, None)

    def get_published(self, channel: str | None = None) -> list[tuple[str, dict]]:
        if channel is None:
            return list(self.published)
        return [(ch, env) for ch, env in self.published if ch == channel]
