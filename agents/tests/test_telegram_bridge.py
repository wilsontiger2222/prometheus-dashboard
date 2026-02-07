"""Tests for the Telegram bridge."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from agents.shared.telegram_bridge import TelegramBridge


@pytest.fixture
def mock_prometheus():
    prom = AsyncMock()
    prom.handle_message = AsyncMock(return_value={
        "sentinel": {"signal": "BUY", "symbol": "BTC"},
    })
    return prom


@pytest.fixture
def bridge(mock_prometheus):
    return TelegramBridge(
        prometheus_agent=mock_prometheus,
        gateway_url="ws://localhost:18789",
        reconnect_delay=0.01,
    )


class TestTelegramBridge:
    @pytest.mark.asyncio
    async def test_handle_text_message(self, bridge, mock_prometheus):
        raw = json.dumps({"type": "message", "text": "/trade BTC", "chat_id": 123})
        await bridge._handle_gateway_message(raw)
        mock_prometheus.handle_message.assert_called_once_with("/trade BTC")

    @pytest.mark.asyncio
    async def test_sends_reply(self, bridge):
        bridge._ws = AsyncMock()
        bridge._ws.closed = False
        raw = json.dumps({"type": "message", "text": "/status", "chat_id": 42})
        await bridge._handle_gateway_message(raw)
        bridge._ws.send_json.assert_called_once()
        call_args = bridge._ws.send_json.call_args[0][0]
        assert call_args["type"] == "reply"
        assert call_args["chat_id"] == 42

    @pytest.mark.asyncio
    async def test_handles_null_response(self, bridge, mock_prometheus):
        mock_prometheus.handle_message = AsyncMock(return_value=None)
        bridge._ws = AsyncMock()
        bridge._ws.closed = False
        raw = json.dumps({"type": "message", "text": "gibberish", "chat_id": 1})
        await bridge._handle_gateway_message(raw)
        call_args = bridge._ws.send_json.call_args[0][0]
        assert "couldn't" in call_args["text"].lower() or "try" in call_args["text"].lower()

    @pytest.mark.asyncio
    async def test_handles_empty_response(self, bridge, mock_prometheus):
        mock_prometheus.handle_message = AsyncMock(return_value={})
        bridge._ws = AsyncMock()
        bridge._ws.closed = False
        raw = json.dumps({"type": "message", "text": "test", "chat_id": 1})
        await bridge._handle_gateway_message(raw)
        call_args = bridge._ws.send_json.call_args[0][0]
        assert "timeout" in call_args["text"].lower() or "no agent" in call_args["text"].lower()

    @pytest.mark.asyncio
    async def test_ignores_non_message_types(self, bridge, mock_prometheus):
        raw = json.dumps({"type": "status", "data": "ok"})
        await bridge._handle_gateway_message(raw)
        mock_prometheus.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self, bridge, mock_prometheus):
        await bridge._handle_gateway_message("not json at all")
        mock_prometheus.handle_message.assert_not_called()

    def test_format_response_with_data(self, bridge):
        resp = {"sentinel": {"signal": "BUY"}, "watchdog": {"cpu": 45}}
        text = bridge._format_response(resp)
        assert "sentinel" in text.lower()
        assert "watchdog" in text.lower()

    def test_format_response_none(self, bridge):
        text = bridge._format_response(None)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, bridge):
        bridge._running = True
        bridge._session = AsyncMock()
        bridge._session.closed = False
        bridge._ws = None
        await bridge.stop()
        assert bridge._running is False
