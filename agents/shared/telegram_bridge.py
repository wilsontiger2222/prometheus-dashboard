"""Telegram bridge — connects OpenClaw gateway to Prometheus orchestrator."""

import asyncio
import json
import aiohttp
from agents.shared.logger import get_agent_logger


class TelegramBridge:
    """WebSocket bridge between OpenClaw Telegram gateway and Prometheus."""

    def __init__(
        self,
        prometheus_agent,
        gateway_url: str = "ws://127.0.0.1:18789",
        gateway_token: str = "",
        reconnect_delay: float = 5.0,
        max_reconnect_delay: float = 60.0,
    ):
        self._prometheus = prometheus_agent
        self._gateway_url = gateway_url
        self._gateway_token = gateway_token
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._running = False
        self._ws = None
        self._session = None
        self.logger = get_agent_logger("telegram_bridge")

    async def start(self):
        """Connect to gateway with auto-reconnect loop."""
        self._running = True
        self._session = aiohttp.ClientSession()
        delay = self._reconnect_delay
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                self.logger.error(f"Bridge connection error: {e}")
            if self._running:
                self.logger.info(f"Reconnecting in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

    async def stop(self):
        """Disconnect from gateway."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    async def _connect_and_listen(self):
        headers = {}
        if self._gateway_token:
            headers["Authorization"] = f"Bearer {self._gateway_token}"

        self._ws = await self._session.ws_connect(self._gateway_url, headers=headers)
        self.logger.info("Connected to OpenClaw gateway")

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_gateway_message(msg.data)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

    async def _handle_gateway_message(self, raw: str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON from gateway: {raw[:100]}")
            return

        if data.get("type") != "message":
            return

        text = data.get("text", "").strip()
        chat_id = data.get("chat_id")
        if not text:
            return

        self.logger.info(f"Received: {text[:50]}...")

        try:
            response = await self._prometheus.handle_message(text)
        except Exception as e:
            self.logger.error(f"Prometheus error: {e}")
            response = None

        reply = self._format_response(response)
        await self._send_reply(chat_id, reply)

    def _format_response(self, response: dict | None) -> str:
        if response is None:
            return "I couldn't understand that. Try /trade, /status, /research, /deploy, or /post."
        if not response:
            return "No agents responded (timeout)."
        parts = []
        for agent, data in response.items():
            parts.append(f"[{agent}] {json.dumps(data, default=str)}")
        return "\n\n".join(parts)

    async def _send_reply(self, chat_id, text: str):
        if self._ws and not self._ws.closed:
            await self._ws.send_json({
                "type": "reply",
                "chat_id": chat_id,
                "text": text,
            })
