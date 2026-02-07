"""Tests for the MockBus test utility."""

import pytest
from agents.tests.mock_bus import MockBus


class TestMockBus:
    @pytest.mark.asyncio
    async def test_publish_creates_envelope(self):
        bus = MockBus()
        await bus.connect()
        await bus.publish("test/channel", {"key": "value"}, sender="tester")
        assert len(bus.published) == 1
        channel, envelope = bus.published[0]
        assert channel == "test/channel"
        assert envelope["from"] == "tester"
        assert envelope["channel"] == "test/channel"
        assert envelope["payload"] == {"key": "value"}
        assert "timestamp" in envelope

    @pytest.mark.asyncio
    async def test_subscribe_and_deliver(self):
        bus = MockBus()
        await bus.connect()
        received = []

        async def handler(channel, message):
            received.append((channel, message))

        await bus.subscribe("test/channel", handler)
        await bus.publish("test/channel", {"data": 1}, sender="pub")
        assert len(received) == 1
        assert received[0][1]["payload"]["data"] == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self):
        bus = MockBus()
        await bus.connect()
        received = []

        async def handler(channel, message):
            received.append(message)

        await bus.subscribe("ch", handler)
        await bus.unsubscribe("ch")
        await bus.publish("ch", {"x": 1}, sender="s")
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        bus = MockBus()
        await bus.connect()
        r1, r2 = [], []

        async def h1(ch, msg):
            r1.append(msg)

        async def h2(ch, msg):
            r2.append(msg)

        await bus.subscribe("ch", h1)
        await bus.subscribe("ch", h2)
        await bus.publish("ch", {}, sender="s")
        assert len(r1) == 1 and len(r2) == 1

    @pytest.mark.asyncio
    async def test_get_published_filters_by_channel(self):
        bus = MockBus()
        await bus.connect()
        await bus.publish("a", {"x": 1}, sender="s")
        await bus.publish("b", {"x": 2}, sender="s")
        assert len(bus.get_published("a")) == 1
        assert len(bus.get_published("b")) == 1
        assert len(bus.get_published()) == 2

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self):
        bus = MockBus()
        await bus.connect()
        await bus.subscribe("ch", lambda c, m: None)
        await bus.publish("ch", {}, sender="s")
        await bus.disconnect()
        assert len(bus.published) == 0
