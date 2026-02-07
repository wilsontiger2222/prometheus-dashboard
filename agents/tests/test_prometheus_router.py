"""Tests for the Prometheus router module."""

import pytest
from unittest.mock import AsyncMock
from agents.prometheus.router import Router, RouteResult


class TestRouteResult:
    def test_route_result_creation(self):
        result = RouteResult(targets=["sentinel"], task="trade", args="BTC long")
        assert result.targets == ["sentinel"]
        assert result.task == "trade"
        assert result.args == "BTC long"

    def test_route_result_to_dict(self):
        result = RouteResult(targets=["scout"], task="research", args="NVDA earnings")
        d = result.to_dict()
        assert d["targets"] == ["scout"]
        assert d["task"] == "research"
        assert d["args"] == "NVDA earnings"


class TestRouterKeywords:
    def test_trade_shortcut(self):
        shortcuts = {"/trade": "sentinel", "/status": "watchdog", "/research": "scout"}
        router = Router(shortcuts=shortcuts, ai_client=None, agent_name="prometheus")
        result = router.route_keyword("/trade BTC long 2x")
        assert result is not None
        assert result.targets == ["sentinel"]
        assert result.task == "trade"
        assert result.args == "BTC long 2x"

    def test_status_shortcut(self):
        shortcuts = {"/trade": "sentinel", "/status": "watchdog", "/research": "scout"}
        router = Router(shortcuts=shortcuts, ai_client=None, agent_name="prometheus")
        result = router.route_keyword("/status")
        assert result.targets == ["watchdog"]
        assert result.task == "status"
        assert result.args == ""

    def test_research_shortcut(self):
        shortcuts = {"/trade": "sentinel", "/status": "watchdog", "/research": "scout"}
        router = Router(shortcuts=shortcuts, ai_client=None, agent_name="prometheus")
        result = router.route_keyword("/research gold price outlook")
        assert result.targets == ["scout"]
        assert result.task == "research"
        assert result.args == "gold price outlook"

    def test_all_shortcut_broadcasts(self):
        shortcuts = {"/trade": "sentinel", "/status": "watchdog", "/research": "scout"}
        agents = ["sentinel", "watchdog", "scout"]
        router = Router(shortcuts=shortcuts, ai_client=None, agent_name="prometheus", agents=agents)
        result = router.route_keyword("/all system check")
        assert result is not None
        assert set(result.targets) == {"sentinel", "watchdog", "scout"}
        assert result.task == "status"

    def test_unknown_command_returns_none(self):
        shortcuts = {"/trade": "sentinel"}
        router = Router(shortcuts=shortcuts, ai_client=None, agent_name="prometheus")
        result = router.route_keyword("hello world")
        assert result is None

    def test_unknown_slash_returns_none(self):
        shortcuts = {"/trade": "sentinel"}
        router = Router(shortcuts=shortcuts, ai_client=None, agent_name="prometheus")
        result = router.route_keyword("/unknown do something")
        assert result is None


class TestRouterAI:
    @pytest.mark.asyncio
    async def test_ai_route_parses_response(self):
        mock_ai = AsyncMock()
        mock_ai.call = AsyncMock(return_value='{"targets": ["sentinel"], "task": "trade", "args": "check BTC"}')

        router = Router(shortcuts={}, ai_client=mock_ai, agent_name="prometheus")
        result = await router.route_ai("What's happening with Bitcoin?")

        assert result.targets == ["sentinel"]
        assert result.task == "trade"

    @pytest.mark.asyncio
    async def test_ai_route_handles_bad_json(self):
        mock_ai = AsyncMock()
        mock_ai.call = AsyncMock(return_value="I don't understand")

        router = Router(shortcuts={}, ai_client=mock_ai, agent_name="prometheus")
        result = await router.route_ai("gibberish input")

        assert result is None

    @pytest.mark.asyncio
    async def test_route_tries_keyword_first(self):
        mock_ai = AsyncMock()
        shortcuts = {"/trade": "sentinel"}
        router = Router(shortcuts=shortcuts, ai_client=mock_ai, agent_name="prometheus")

        result = await router.route("/trade BTC")

        assert result.targets == ["sentinel"]
        mock_ai.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_falls_back_to_ai(self):
        mock_ai = AsyncMock()
        mock_ai.call = AsyncMock(return_value='{"targets": ["scout"], "task": "research", "args": "gold"}')
        shortcuts = {"/trade": "sentinel"}
        router = Router(shortcuts=shortcuts, ai_client=mock_ai, agent_name="prometheus")

        result = await router.route("research gold prices for me")

        assert result.targets == ["scout"]
        mock_ai.call.assert_called_once()
