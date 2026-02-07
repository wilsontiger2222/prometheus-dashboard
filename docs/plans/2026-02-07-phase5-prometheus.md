# Phase 5: Prometheus (Orchestrator) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Prometheus orchestrator agent that receives user messages, routes them to the correct subagent(s) via keyword shortcuts or AI intent detection, tracks agent liveness via heartbeats, and aggregates multi-agent responses.

**Architecture:** Prometheus is the brain of the Hivemind. It subscribes to all agent report/alert channels, maintains a heartbeat registry for liveness tracking, routes inbound messages via keyword shortcuts (zero tokens) or AI-based intent detection, dispatches tasks to agents via `prometheus/dispatch`, collects responses with timeout, and aggregates them. It runs an always-on loop that listens to the bus.

**Tech Stack:** BaseAgent (existing), RedisBus (existing), AIClient (existing), uuid for request tracking

---

### Task 1: Scaffold Prometheus package and config

**Files:**
- Create: `agents/prometheus/__init__.py`
- Create: `agents/prometheus/config.json`

**Step 1: Create `agents/prometheus/__init__.py`**

```python
"""Prometheus — orchestrator and dispatcher for the Hivemind."""
```

**Step 2: Create `agents/prometheus/config.json`**

```json
{
  "name": "prometheus",
  "response_timeout_seconds": 30,
  "keyword_shortcuts": {
    "/trade": "sentinel",
    "/status": "watchdog",
    "/research": "scout",
    "/deploy": "forge",
    "/post": "herald"
  },
  "agents": ["sentinel", "watchdog", "scout", "forge", "herald"],
  "heartbeat_stale_seconds": 90
}
```

**Step 3: Commit**

```bash
git add agents/prometheus/__init__.py agents/prometheus/config.json
git commit -m "feat(prometheus): scaffold package and config"
```

---

### Task 2: Build router module

**Files:**
- Create: `agents/prometheus/router.py`
- Create: `agents/tests/test_prometheus_router.py`

**Step 1: Write the failing tests**

`agents/tests/test_prometheus_router.py`:

```python
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
        mock_ai.call.assert_not_called()  # AI should NOT be called

    @pytest.mark.asyncio
    async def test_route_falls_back_to_ai(self):
        mock_ai = AsyncMock()
        mock_ai.call = AsyncMock(return_value='{"targets": ["scout"], "task": "research", "args": "gold"}')
        shortcuts = {"/trade": "sentinel"}
        router = Router(shortcuts=shortcuts, ai_client=mock_ai, agent_name="prometheus")

        result = await router.route("research gold prices for me")

        assert result.targets == ["scout"]
        mock_ai.call.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_prometheus_router.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/prometheus/router.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_prometheus_router.py -v`
Expected: 11 PASSED

**Step 5: Commit**

```bash
git add agents/prometheus/router.py agents/tests/test_prometheus_router.py
git commit -m "feat(prometheus): add message router with keyword shortcuts and AI fallback"
```

---

### Task 3: Build aggregator module

**Files:**
- Create: `agents/prometheus/aggregator.py`
- Create: `agents/tests/test_prometheus_aggregator.py`

**Step 1: Write the failing tests**

`agents/tests/test_prometheus_aggregator.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_prometheus_aggregator.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/prometheus/aggregator.py`:

```python
"""Response aggregator for the Prometheus orchestrator.

Collects responses from multiple agents for a given request,
tracks completion, and supports timeout-based waiting.
"""

import asyncio


class Aggregator:
    """Collect and aggregate multi-agent responses."""

    def __init__(self, timeout: float = 30):
        self._timeout = timeout
        self._requests: dict[str, dict] = {}  # request_id -> {expected, responses}

    def start_request(self, request_id: str, expected: list[str]):
        """Register a new request with expected agent responses."""
        self._requests[request_id] = {
            "expected": set(expected),
            "responses": {},
        }

    def add_response(self, request_id: str, agent: str, data: dict):
        """Add a response from an agent."""
        if request_id in self._requests:
            self._requests[request_id]["responses"][agent] = data

    def is_complete(self, request_id: str) -> bool:
        """Check if all expected agents have responded."""
        req = self._requests.get(request_id)
        if not req:
            return False
        return req["expected"] <= set(req["responses"].keys())

    def get_responses(self, request_id: str) -> dict:
        """Get all collected responses for a request."""
        req = self._requests.get(request_id)
        if not req:
            return {}
        return dict(req["responses"])

    async def wait_for(self, request_id: str) -> dict:
        """Wait until all responses arrive or timeout. Returns collected responses."""
        try:
            async with asyncio.timeout(self._timeout):
                while not self.is_complete(request_id):
                    await asyncio.sleep(0.05)
        except TimeoutError:
            pass
        return self.get_responses(request_id)

    def cleanup(self, request_id: str):
        """Remove a completed request."""
        self._requests.pop(request_id, None)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_prometheus_aggregator.py -v`
Expected: 7 PASSED

**Step 5: Commit**

```bash
git add agents/prometheus/aggregator.py agents/tests/test_prometheus_aggregator.py
git commit -m "feat(prometheus): add response aggregator with timeout support"
```

---

### Task 4: Build heartbeat tracker

**Files:**
- Create: `agents/prometheus/heartbeat.py`
- Create: `agents/tests/test_prometheus_heartbeat.py`

**Step 1: Write the failing tests**

`agents/tests/test_prometheus_heartbeat.py`:

```python
"""Tests for the Prometheus heartbeat tracker."""

import time
import pytest
from agents.prometheus.heartbeat import HeartbeatTracker


class TestHeartbeatTracker:
    def test_record_heartbeat(self):
        tracker = HeartbeatTracker(stale_seconds=90)
        tracker.record("sentinel")
        status = tracker.get_status()
        assert "sentinel" in status
        assert status["sentinel"]["alive"] is True

    def test_stale_agent(self):
        tracker = HeartbeatTracker(stale_seconds=1)
        tracker.record("watchdog", timestamp=time.time() - 5)
        status = tracker.get_status()
        assert status["watchdog"]["alive"] is False

    def test_multiple_agents(self):
        tracker = HeartbeatTracker(stale_seconds=90)
        tracker.record("sentinel")
        tracker.record("watchdog")
        tracker.record("scout")
        status = tracker.get_status()
        assert len(status) == 3
        assert all(s["alive"] for s in status.values())

    def test_get_stale_agents(self):
        tracker = HeartbeatTracker(stale_seconds=1)
        tracker.record("sentinel")
        tracker.record("watchdog", timestamp=time.time() - 5)
        stale = tracker.get_stale_agents()
        assert "watchdog" in stale
        assert "sentinel" not in stale

    def test_get_alive_agents(self):
        tracker = HeartbeatTracker(stale_seconds=1)
        tracker.record("sentinel")
        tracker.record("watchdog", timestamp=time.time() - 5)
        alive = tracker.get_alive_agents()
        assert "sentinel" in alive
        assert "watchdog" not in alive
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_prometheus_heartbeat.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/prometheus/heartbeat.py`:

```python
"""Heartbeat tracker for agent liveness monitoring.

Prometheus uses this to track which agents are alive based on
their periodic heartbeat messages on the agents/heartbeat channel.
"""

import time


class HeartbeatTracker:
    """Track agent heartbeats and detect stale agents."""

    def __init__(self, stale_seconds: float = 90):
        self._stale_seconds = stale_seconds
        self._heartbeats: dict[str, float] = {}  # agent_name -> last_seen timestamp

    def record(self, agent: str, timestamp: float | None = None):
        """Record a heartbeat from an agent."""
        self._heartbeats[agent] = timestamp if timestamp is not None else time.time()

    def is_alive(self, agent: str) -> bool:
        """Check if an agent is alive (heartbeat within stale threshold)."""
        last_seen = self._heartbeats.get(agent)
        if last_seen is None:
            return False
        return (time.time() - last_seen) < self._stale_seconds

    def get_status(self) -> dict[str, dict]:
        """Get status of all known agents."""
        now = time.time()
        return {
            agent: {
                "alive": (now - ts) < self._stale_seconds,
                "last_seen": ts,
                "age_seconds": round(now - ts, 1),
            }
            for agent, ts in self._heartbeats.items()
        }

    def get_stale_agents(self) -> list[str]:
        """Return list of agents that have gone stale."""
        return [a for a in self._heartbeats if not self.is_alive(a)]

    def get_alive_agents(self) -> list[str]:
        """Return list of agents that are alive."""
        return [a for a in self._heartbeats if self.is_alive(a)]
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_prometheus_heartbeat.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add agents/prometheus/heartbeat.py agents/tests/test_prometheus_heartbeat.py
git commit -m "feat(prometheus): add heartbeat tracker for agent liveness"
```

---

### Task 5: Build PrometheusAgent class

**Files:**
- Create: `agents/prometheus/agent.py`
- Create: `agents/tests/test_prometheus_agent.py`

**Step 1: Write the failing tests**

`agents/tests/test_prometheus_agent.py`:

```python
"""Tests for the PrometheusAgent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.prometheus.agent import PrometheusAgent


@pytest.fixture
def prometheus():
    with patch("agents.shared.base_agent.load_agent_config") as mock_config:
        mock_config.return_value = {
            "name": "prometheus",
            "response_timeout_seconds": 2,
            "keyword_shortcuts": {
                "/trade": "sentinel",
                "/status": "watchdog",
                "/research": "scout",
            },
            "agents": ["sentinel", "watchdog", "scout"],
            "heartbeat_stale_seconds": 90,
        }
        agent = PrometheusAgent(config_path="dummy.json")
        agent.bus = AsyncMock()
        agent._running = True
        return agent


class TestPrometheusAgent:
    def test_prometheus_name(self, prometheus):
        assert prometheus.name == "prometheus"

    def test_loads_shortcuts(self, prometheus):
        assert "/trade" in prometheus._router._shortcuts

    def test_loads_agents_list(self, prometheus):
        assert "sentinel" in prometheus._agents

    @pytest.mark.asyncio
    async def test_handle_message_keyword(self, prometheus):
        """Keyword shortcut dispatches to correct agent."""
        await prometheus.handle_message("/trade BTC long")
        prometheus.bus.publish.assert_called()
        call_args = prometheus.bus.publish.call_args_list[0]
        assert call_args[0][0] == "prometheus/dispatch"

    @pytest.mark.asyncio
    async def test_handle_message_unknown(self, prometheus):
        """Unknown message with no AI returns error."""
        prometheus._router.route_ai = AsyncMock(return_value=None)
        result = await prometheus.handle_message("asdfghjkl")
        assert result is None or result == {}

    @pytest.mark.asyncio
    async def test_handle_heartbeat(self, prometheus):
        """Heartbeat messages update the tracker."""
        message = {"payload": {"agent": "sentinel", "status": "alive"}}
        await prometheus._on_heartbeat("agents/heartbeat", message)
        status = prometheus._heartbeats.get_status()
        assert "sentinel" in status
        assert status["sentinel"]["alive"] is True

    @pytest.mark.asyncio
    async def test_on_dispatch_status(self, prometheus):
        """Dispatch with task=status returns agent liveness."""
        prometheus._heartbeats.record("sentinel")
        prometheus._heartbeats.record("watchdog")
        message = {"payload": {"task": "status"}}
        await prometheus.on_dispatch(message)
        prometheus.bus.publish.assert_called()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_prometheus_agent.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/prometheus/agent.py`:

```python
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
        # Listen to heartbeats
        await self.bus.subscribe("agents/heartbeat", self._on_heartbeat)

        # Listen to agent report channels
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

        # Start aggregation tracking
        self._aggregator.start_request(request_id, expected=route.targets)

        # Dispatch to each target agent
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

        # Wait for responses
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_prometheus_agent.py -v`
Expected: 7 PASSED

**Step 5: Commit**

```bash
git add agents/prometheus/agent.py agents/tests/test_prometheus_agent.py
git commit -m "feat(prometheus): add PrometheusAgent orchestrator"
```

---

### Task 6: Add Prometheus `__main__.py` entry point

**Files:**
- Create: `agents/prometheus/__main__.py`

**Step 1: Write `__main__.py`**

```python
"""Run the Prometheus orchestrator: python -m agents.prometheus"""

import asyncio
import sys
from pathlib import Path

from agents.prometheus.agent import PrometheusAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = PrometheusAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nPrometheus shutting down...")


if __name__ == "__main__":
    main()
```

**Step 2: Verify import works**

Run: `python -c "from agents.prometheus.agent import PrometheusAgent; print('OK')"`

**Step 3: Commit**

```bash
git add agents/prometheus/__main__.py
git commit -m "feat(prometheus): add __main__.py entry point"
```

---

### Task 7: Final test run and push

**Step 1: Run full test suite**

Run: `python -m pytest agents/tests/ -v`
Expected: All tests pass (no regressions)

**Step 2: Commit plan doc**

```bash
git add docs/plans/2026-02-07-phase5-prometheus.md
git commit -m "docs: add Phase 5 Prometheus implementation plan"
```

**Step 3: Push to GitHub**

```bash
git push
```
