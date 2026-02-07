"""Prometheus agent — orchestrator and dispatcher for the Hivemind."""

import asyncio
import uuid
from agents.shared.base_agent import BaseAgent
from agents.prometheus.router import Router
from agents.prometheus.aggregator import Aggregator
from agents.prometheus.heartbeat import HeartbeatTracker


class PrometheusAgent(BaseAgent):
    """Orchestrator that routes, dispatches, and aggregates."""

    def __init__(self, **kwargs):
        super().__init__(name="prometheus", **kwargs)

        shortcuts = self.config.get("keyword_shortcuts", {})
        self._agents = self.config.get("agents", [])
        timeout = self.config.get("response_timeout_seconds", 30)
        stale = self.config.get("heartbeat_stale_seconds", 90)

        self._router = Router(
            shortcuts=shortcuts,
            ai_client=self.ai,
            agent_name="prometheus",
            agents=self._agents,
        )
        self._aggregator = Aggregator(timeout=timeout)
        self._heartbeats = HeartbeatTracker(stale_seconds=stale)

    async def run(self):
        """Subscribe to agent channels and idle."""
        await self.bus.subscribe("agents/heartbeat", self._on_heartbeat)

        for channel in ["sentinel/alerts", "sentinel/trades", "watchdog/health",
                         "watchdog/critical", "scout/reports", "forge/deploys",
                         "herald/posts"]:
            await self.bus.subscribe(channel, self._on_agent_report)

        while self._running:
            await asyncio.sleep(1)

    async def on_dispatch(self, message: dict):
        payload = message.get("payload", {})
        task = payload.get("task", "")

        if task == "status":
            status = self._heartbeats.get_status()
            await self.bus.publish(
                "prometheus/status",
                {"agents": status, "alive": self._heartbeats.get_alive_agents()},
                sender="prometheus",
            )

    async def handle_message(self, text: str) -> dict | None:
        """Route an inbound user message and dispatch to agents."""
        route = await self._router.route(text)
        if route is None:
            return None

        request_id = str(uuid.uuid4())

        self._aggregator.start_request(request_id, expected=route.targets)

        for target in route.targets:
            await self.bus.publish(
                "prometheus/dispatch",
                {
                    "target": target,
                    "task": route.task,
                    "args": route.args,
                },
                sender="prometheus",
            )

        responses = await self._aggregator.wait_for(request_id)
        self._aggregator.cleanup(request_id)

        return responses

    async def _on_heartbeat(self, channel: str, message: dict):
        """Handle heartbeat from an agent."""
        payload = message.get("payload", {})
        agent = payload.get("agent")
        if agent:
            self._heartbeats.record(agent)

    async def _on_agent_report(self, channel: str, message: dict):
        """Handle report/alert from an agent — feed into aggregator."""
        request_id = message.get("request_id") or message.get("payload", {}).get("request_id")
        sender = message.get("from", "unknown")
        payload = message.get("payload", {})

        if request_id:
            self._aggregator.add_response(request_id, sender, payload)

        self.logger.info(f"Received on {channel} from {sender}")
