"""Redis pub/sub message bus for Hivemind inter-agent communication."""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

import redis.asyncio as aioredis


class RedisBus:
    """Thin async wrapper around Redis pub/sub with JSON envelope."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis_url = redis_url
        self._publisher = None
        self._subscriber = None
        self._pubsub = None
        self._handlers: dict[str, Callable] = {}
        self._listen_task = None

    async def connect(self):
        self._publisher = aioredis.from_url(self._redis_url)
        self._subscriber = aioredis.from_url(self._redis_url)
        self._pubsub = self._subscriber.pubsub()

    async def disconnect(self):
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
            self._pubsub = None
        if self._subscriber:
            await self._subscriber.close()
            self._subscriber = None
        if self._publisher:
            await self._publisher.close()
            self._publisher = None

    async def publish(self, channel: str, payload: dict[str, Any], sender: str = "unknown"):
        envelope = {
            "from": sender,
            "channel": channel,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        await self._publisher.publish(channel, json.dumps(envelope))

    async def subscribe(self, channel: str, handler: Callable):
        self._handlers[channel] = handler
        await self._pubsub.subscribe(channel)
        if self._listen_task is None or self._listen_task.done():
            self._listen_task = asyncio.create_task(self._listen())

    async def unsubscribe(self, channel: str):
        self._handlers.pop(channel, None)
        await self._pubsub.unsubscribe(channel)

    async def _listen(self):
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode()
                    data = json.loads(message["data"])
                    handler = self._handlers.get(channel)
                    if handler:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(channel, data)
                        else:
                            handler(channel, data)
        except asyncio.CancelledError:
            pass
