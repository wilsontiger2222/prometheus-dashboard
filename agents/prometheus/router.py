"""Message router for the Prometheus orchestrator.

Routes inbound messages to the correct agent(s) via:
1. Keyword shortcuts (zero tokens) — e.g. /trade, /status, /research
2. AI-based intent detection (fallback) — parses natural language
"""

import json
from dataclasses import dataclass, field


@dataclass
class RouteResult:
    targets: list[str] = field(default_factory=list)
    task: str = ""
    args: str = ""

    def to_dict(self) -> dict:
        return {
            "targets": self.targets,
            "task": self.task,
            "args": self.args,
        }


class Router:
    """Route messages to agents via keywords or AI intent detection."""

    ROUTE_PROMPT = (
        "You are a message router for a multi-agent system. "
        "Given a user message, determine which agent(s) should handle it. "
        "Available agents: sentinel (trading/markets), watchdog (server/infra), "
        "scout (research/web), forge (code/deploy), herald (social media/content). "
        "Respond ONLY with a JSON object: "
        '{{"targets": ["agent_name"], "task": "task_type", "args": "relevant details"}}. '
        "If you cannot determine the intent, respond with null.\n\n"
        "User message: {message}"
    )

    def __init__(self, shortcuts: dict, ai_client, agent_name: str = "prometheus",
                 agents: list[str] | None = None):
        self._shortcuts = shortcuts
        self._ai = ai_client
        self._agent_name = agent_name
        self._agents = agents or []

    def route_keyword(self, message: str) -> RouteResult | None:
        """Try to route via keyword shortcut. Returns None if no match."""
        stripped = message.strip()
        if not stripped.startswith("/"):
            return None

        parts = stripped.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/all":
            return RouteResult(
                targets=list(self._agents),
                task="status",
                args=args,
            )

        target = self._shortcuts.get(command)
        if target is None:
            return None

        # Derive task from command name (strip the /)
        task = command.lstrip("/")
        return RouteResult(targets=[target], task=task, args=args)

    async def route_ai(self, message: str) -> RouteResult | None:
        """Route via AI intent detection. Returns None if AI can't parse."""
        prompt = self.ROUTE_PROMPT.format(message=message)
        response = await self._ai.call(self._agent_name, prompt)

        try:
            data = json.loads(response)
            if data is None:
                return None
            return RouteResult(
                targets=data.get("targets", []),
                task=data.get("task", ""),
                args=data.get("args", ""),
            )
        except (json.JSONDecodeError, TypeError):
            return None

    async def route(self, message: str) -> RouteResult | None:
        """Route a message: try keywords first, fall back to AI."""
        result = self.route_keyword(message)
        if result is not None:
            return result
        return await self.route_ai(message)
