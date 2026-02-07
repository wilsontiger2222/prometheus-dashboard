"""Tests for Redis pub/sub message bus."""

import asyncio
import json
import pytest

try:
    import redis
    r = redis.Redis()
    r.ping()
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")

from agents.shared.bus import RedisBus


@pytest.fixture
async def bus():
    b = RedisBus(redis_url="redis://localhost:6379")
    await b.connect()
    yield b
    await b.disconnect()


@pytest.fixture
async def bus_pair():
    """Two bus instances to test pub/sub."""
    pub = RedisBus(redis_url="redis://localhost:6379")
    sub = RedisBus(redis_url="redis://localhost:6379")
    await pub.connect()
    await sub.connect()
    yield pub, sub
    await pub.disconnect()
    await sub.disconnect()


@pytest.mark.asyncio
async def test_publish_and_subscribe(bus_pair):
    pub, sub = bus_pair
    received = []

    async def handler(channel, message):
        received.append(message)

    await sub.subscribe("test/channel", handler)
    await asyncio.sleep(0.1)

    await pub.publish("test/channel", {"type": "test", "data": "hello"})
    await asyncio.sleep(0.3)

    assert len(received) == 1
    assert received[0]["payload"]["type"] == "test"
    assert received[0]["payload"]["data"] == "hello"

    await sub.unsubscribe("test/channel")


@pytest.mark.asyncio
async def test_publish_adds_envelope(bus_pair):
    pub, sub = bus_pair
    received = []

    async def handler(channel, message):
        received.append(message)

    await sub.subscribe("test/envelope", handler)
    await asyncio.sleep(0.1)

    await pub.publish("test/envelope", {"data": "payload"}, sender="scout")
    await asyncio.sleep(0.3)

    msg = received[0]
    assert msg["from"] == "scout"
    assert msg["channel"] == "test/envelope"
    assert "timestamp" in msg
    assert msg["payload"]["data"] == "payload"

    await sub.unsubscribe("test/envelope")


@pytest.mark.asyncio
async def test_multiple_subscribers(bus_pair):
    pub, sub = bus_pair
    sub2 = RedisBus(redis_url="redis://localhost:6379")
    await sub2.connect()

    received1 = []
    received2 = []

    await sub.subscribe("test/multi", lambda ch, msg: received1.append(msg))
    await sub2.subscribe("test/multi", lambda ch, msg: received2.append(msg))
    await asyncio.sleep(0.1)

    await pub.publish("test/multi", {"data": "broadcast"})
    await asyncio.sleep(0.3)

    assert len(received1) == 1
    assert len(received2) == 1

    await sub.unsubscribe("test/multi")
    await sub2.unsubscribe("test/multi")
    await sub2.disconnect()


@pytest.mark.asyncio
async def test_disconnect_cleans_up(bus):
    await bus.subscribe("test/cleanup", lambda ch, msg: None)
    await bus.disconnect()
    assert bus._subscriber is None
