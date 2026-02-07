"""AI client wrapper with token tracking for Hivemind agents.

Connects to the OpenClaw gateway WebSocket to make AI calls.
Tracks token usage per agent for budget monitoring.
"""

import json
from pathlib import Path
from typing import Any
from collections import defaultdict


class AIClient:
    def __init__(self, gateway_url: str, token: str, token_log: str | None = None):
        self._gateway_url = gateway_url
        self._token = token
        self._token_log = token_log
        self._total_tokens = 0
        self._agent_tokens: dict[str, int] = defaultdict(int)

    async def call(self, agent_name: str, prompt: str, context: str | None = None) -> str:
        """Send prompt to OpenClaw gateway and return response.

        This sends a WebSocket message to the gateway which forwards
        it to the configured AI model (GPT-5.2).
        """
        import aiohttp

        message = {"prompt": prompt}
        if context:
            message["context"] = context

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self._gateway_url,
                headers={"Authorization": f"Bearer {self._token}"}
            ) as ws:
                await ws.send_json(message)
                response = await ws.receive_json()

        # Track tokens if reported by gateway
        if "usage" in response:
            self._track_tokens(
                agent_name,
                response["usage"].get("prompt_tokens", 0),
                response["usage"].get("completion_tokens", 0),
            )

        return response.get("content", "")

    def _track_tokens(self, agent_name: str, prompt_tokens: int, completion_tokens: int):
        total = prompt_tokens + completion_tokens
        self._total_tokens += total
        self._agent_tokens[agent_name] += total
        if self._token_log:
            self._save_token_log()

    def _save_token_log(self):
        if not self._token_log:
            return
        data = {
            "total": self._total_tokens,
            "by_agent": dict(self._agent_tokens),
        }
        Path(self._token_log).write_text(json.dumps(data, indent=2))

    def get_usage_report(self) -> dict[str, Any]:
        return {
            "total": self._total_tokens,
            "by_agent": dict(self._agent_tokens),
        }
