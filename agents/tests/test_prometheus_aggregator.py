"""Tests for the Prometheus aggregator module."""

import asyncio
import pytest
from agents.prometheus.aggregator import Aggregator


class TestAggregator:
    def test_add_response(self):
        agg = Aggregator(timeout=5)
        agg.start_request("req-1", expected=["sentinel"])
        agg.add_response("req-1", "sentinel", {"status": "ok"})
        assert agg.is_complete("req-1")

    def test_incomplete_until_all_respond(self):
        agg = Aggregator(timeout=5)
        agg.start_request("req-2", expected=["sentinel", "scout"])
        agg.add_response("req-2", "sentinel", {"status": "ok"})
        assert not agg.is_complete("req-2")
        agg.add_response("req-2", "scout", {"report": "data"})
        assert agg.is_complete("req-2")

    def test_get_responses(self):
        agg = Aggregator(timeout=5)
        agg.start_request("req-3", expected=["watchdog"])
        agg.add_response("req-3", "watchdog", {"cpu": 45})
        responses = agg.get_responses("req-3")
        assert responses["watchdog"] == {"cpu": 45}

    def test_get_responses_unknown_request(self):
        agg = Aggregator(timeout=5)
        assert agg.get_responses("unknown") == {}

    def test_cleanup_removes_request(self):
        agg = Aggregator(timeout=5)
        agg.start_request("req-4", expected=["sentinel"])
        agg.add_response("req-4", "sentinel", {"data": 1})
        agg.cleanup("req-4")
        assert agg.get_responses("req-4") == {}

    @pytest.mark.asyncio
    async def test_wait_for_complete(self):
        agg = Aggregator(timeout=2)
        agg.start_request("req-5", expected=["sentinel"])

        async def delayed_response():
            await asyncio.sleep(0.1)
            agg.add_response("req-5", "sentinel", {"done": True})

        asyncio.create_task(delayed_response())
        result = await agg.wait_for("req-5")
        assert result["sentinel"] == {"done": True}

    @pytest.mark.asyncio
    async def test_wait_for_timeout(self):
        agg = Aggregator(timeout=0.2)
        agg.start_request("req-6", expected=["sentinel", "scout"])
        agg.add_response("req-6", "sentinel", {"ok": True})
        # scout never responds — should timeout
        result = await agg.wait_for("req-6")
        assert "sentinel" in result
        assert "scout" not in result
