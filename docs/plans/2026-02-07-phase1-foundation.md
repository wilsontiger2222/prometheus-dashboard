# Phase 1: Foundation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the shared foundation that all Hivemind subagents inherit from — base agent class, Redis message bus wrapper, shared utilities, and directory structure.

**Architecture:** All agents inherit from `BaseAgent` which handles Redis pub/sub, heartbeats, logging, and AI calls. A thin `RedisBus` class wraps Redis pub/sub with JSON serialization. A shared `ai_client.py` manages API calls with token tracking. Everything lives in `agents/` at the repo root.

**Tech Stack:** Python 3.10+, Redis (pub/sub + streams), asyncio, `redis-py`, `aiohttp`

---

### Task 1: Create directory structure

**Files:**
- Create: `agents/shared/__init__.py`
- Create: `agents/shared/bus.py` (placeholder)
- Create: `agents/shared/base_agent.py` (placeholder)
- Create: `agents/shared/logger.py` (placeholder)
- Create: `agents/shared/ai_client.py` (placeholder)
- Create: `agents/shared/config.py` (placeholder)
- Create: `agents/prometheus/__init__.py` (placeholder)
- Create: `agents/sentinel/__init__.py` (placeholder)
- Create: `agents/watchdog/__init__.py` (placeholder)
- Create: `agents/scout/__init__.py` (placeholder)
- Create: `agents/forge/__init__.py` (placeholder)
- Create: `agents/herald/__init__.py` (placeholder)
- Create: `agents/requirements.txt`
- Create: `agents/README.md`

**Step 1: Create all directories and placeholder files**

```
agents/
├── shared/
│   ├── __init__.py
│   ├── bus.py
│   ├── base_agent.py
│   ├── logger.py
│   ├── ai_client.py
│   └── config.py
├── prometheus/
│   └── __init__.py
├── sentinel/
│   └── __init__.py
├── watchdog/
│   └── __init__.py
├── scout/
│   └── __init__.py
├── forge/
│   └── __init__.py
├── herald/
│   └── __init__.py
├── requirements.txt
└── README.md
```

`agents/requirements.txt`:
```
redis>=5.0.0
aiohttp>=3.9.0
```

`agents/README.md`:
```markdown
# OpenClaw Hivemind Agents

Subagent system for OpenClaw. See `docs/plans/2026-02-07-openclaw-hivemind-design.md` for full architecture.

## Setup

1. Install Redis: `sudo apt install redis-server`
2. Install Python deps: `pip install -r agents/requirements.txt`
3. Start Redis: `sudo systemctl start redis-server`

## Running an agent

```bash
python -m agents.watchdog
```
```

**Step 2: Commit**

```bash
git add agents/
git commit -m "feat: scaffold Hivemind agent directory structure"
```

---

### Task 2: Build the Redis message bus wrapper (`bus.py`)

**Files:**
- Create: `agents/tests/__init__.py`
- Create: `agents/tests/test_bus.py`
- Create: `agents/shared/bus.py`

**Step 1: Write the failing test**

`agents/tests/test_bus.py`:
```python
import asyncio
import json
import pytest
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
    assert received[0]["type"] == "test"
    assert received[0]["data"] == "hello"

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
```

**Step 2: Run test to verify it fails**

Run: `pytest agents/tests/test_bus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.shared.bus'` or `ImportError`

**Step 3: Write minimal implementation**

`agents/shared/bus.py`:
```python
"""Redis pub/sub message bus for Hivemind inter-agent communication."""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

import redis.asyncio as aioredis


class RedisBus:
    """Thin async wrapper around Redis pub/sub with JSON envelope."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis_url = redis_url
        self._publisher = None
        self._subscriber = None
        self._pubsub = None
        self._handlers: dict[str, Callable] = {}
        self._listen_task = None

    async def connect(self):
        self._publisher = aioredis.from_url(self._redis_url)
        self._subscriber = aioredis.from_url(self._redis_url)
        self._pubsub = self._subscriber.pubsub()

    async def disconnect(self):
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
            self._pubsub = None
        if self._subscriber:
            await self._subscriber.close()
            self._subscriber = None
        if self._publisher:
            await self._publisher.close()
            self._publisher = None

    async def publish(self, channel: str, payload: dict[str, Any], sender: str = "unknown"):
        envelope = {
            "from": sender,
            "channel": channel,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        await self._publisher.publish(channel, json.dumps(envelope))

    async def subscribe(self, channel: str, handler: Callable):
        self._handlers[channel] = handler
        await self._pubsub.subscribe(channel)
        if self._listen_task is None or self._listen_task.done():
            self._listen_task = asyncio.create_task(self._listen())

    async def unsubscribe(self, channel: str):
        self._handlers.pop(channel, None)
        await self._pubsub.unsubscribe(channel)

    async def _listen(self):
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode()
                    data = json.loads(message["data"])
                    handler = self._handlers.get(channel)
                    if handler:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(channel, data)
                        else:
                            handler(channel, data)
        except asyncio.CancelledError:
            pass
```

**Step 4: Run test to verify it passes**

Run: `pytest agents/tests/test_bus.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add agents/shared/bus.py agents/tests/
git commit -m "feat: add Redis pub/sub message bus with tests"
```

---

### Task 3: Build the shared logger (`logger.py`)

**Files:**
- Create: `agents/tests/test_logger.py`
- Create: `agents/shared/logger.py`

**Step 1: Write the failing test**

`agents/tests/test_logger.py`:
```python
import json
import logging
import pytest
from agents.shared.logger import get_agent_logger


def test_logger_returns_named_logger():
    logger = get_agent_logger("sentinel")
    assert logger.name == "hivemind.sentinel"


def test_logger_formats_json(capsys, tmp_path):
    log_file = tmp_path / "test.log"
    logger = get_agent_logger("watchdog", log_file=str(log_file))
    logger.info("health check passed", extra={"agent_data": {"cpu": 45}})

    content = log_file.read_text()
    record = json.loads(content.strip().split("\n")[-1])
    assert record["agent"] == "watchdog"
    assert record["message"] == "health check passed"
    assert record["data"]["cpu"] == 45


def test_logger_default_level():
    logger = get_agent_logger("scout")
    assert logger.level == logging.INFO
```

**Step 2: Run test to verify it fails**

Run: `pytest agents/tests/test_logger.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write minimal implementation**

`agents/shared/logger.py`:
```python
"""Structured JSON logging for Hivemind agents."""

import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": record.name.replace("hivemind.", ""),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "agent_data"):
            entry["data"] = record.agent_data
        return json.dumps(entry)


def get_agent_logger(agent_name: str, log_file: str | None = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(f"hivemind.{agent_name}")
    logger.setLevel(level)

    if not logger.handlers:
        if log_file:
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger
```

**Step 4: Run test to verify it passes**

Run: `pytest agents/tests/test_logger.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add agents/shared/logger.py agents/tests/test_logger.py
git commit -m "feat: add structured JSON logger for agents"
```

---

### Task 4: Build the config loader (`config.py`)

**Files:**
- Create: `agents/tests/test_config.py`
- Create: `agents/shared/config.py`

**Step 1: Write the failing test**

`agents/tests/test_config.py`:
```python
import json
import pytest
from agents.shared.config import load_agent_config


def test_load_config_from_file(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "name": "sentinel",
        "watchlist": ["BTC", "ETH"],
        "mode": "paper"
    }))
    config = load_agent_config(str(config_file))
    assert config["name"] == "sentinel"
    assert config["watchlist"] == ["BTC", "ETH"]


def test_load_config_with_defaults(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"name": "watchdog"}))
    defaults = {"check_interval": 60, "name": "default"}
    config = load_agent_config(str(config_file), defaults=defaults)
    assert config["name"] == "watchdog"
    assert config["check_interval"] == 60


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_agent_config("/nonexistent/config.json")
```

**Step 2: Run test to verify it fails**

Run: `pytest agents/tests/test_config.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write minimal implementation**

`agents/shared/config.py`:
```python
"""Configuration loader for Hivemind agents."""

import json
from pathlib import Path
from typing import Any


def load_agent_config(config_path: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = json.load(f)

    if defaults:
        merged = {**defaults, **config}
        return merged

    return config
```

**Step 4: Run test to verify it passes**

Run: `pytest agents/tests/test_config.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add agents/shared/config.py agents/tests/test_config.py
git commit -m "feat: add config loader with defaults merging"
```

---

### Task 5: Build the AI client wrapper (`ai_client.py`)

**Files:**
- Create: `agents/tests/test_ai_client.py`
- Create: `agents/shared/ai_client.py`

**Step 1: Write the failing test**

`agents/tests/test_ai_client.py`:
```python
import json
import pytest
from unittest.mock import AsyncMock, patch
from agents.shared.ai_client import AIClient


@pytest.fixture
def client(tmp_path):
    return AIClient(
        gateway_url="ws://127.0.0.1:18789",
        token="test-token",
        token_log=str(tmp_path / "tokens.json"),
    )


def test_client_init(client):
    assert client._gateway_url == "ws://127.0.0.1:18789"
    assert client._total_tokens == 0


def test_token_tracking(client):
    client._track_tokens("sentinel", prompt_tokens=100, completion_tokens=50)
    assert client._total_tokens == 150
    assert client._agent_tokens["sentinel"] == 150


def test_token_log_written(client, tmp_path):
    client._track_tokens("scout", prompt_tokens=200, completion_tokens=100)
    client._save_token_log()
    log_file = tmp_path / "tokens.json"
    data = json.loads(log_file.read_text())
    assert data["total"] == 300
    assert data["by_agent"]["scout"] == 300


def test_get_usage_report(client):
    client._track_tokens("sentinel", 100, 50)
    client._track_tokens("scout", 200, 100)
    report = client.get_usage_report()
    assert report["total"] == 450
    assert report["by_agent"]["sentinel"] == 150
    assert report["by_agent"]["scout"] == 300
```

**Step 2: Run test to verify it fails**

Run: `pytest agents/tests/test_ai_client.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write minimal implementation**

`agents/shared/ai_client.py`:
```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest agents/tests/test_ai_client.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add agents/shared/ai_client.py agents/tests/test_ai_client.py
git commit -m "feat: add AI client wrapper with token tracking"
```

---

### Task 6: Build the base agent class (`base_agent.py`)

**Files:**
- Create: `agents/tests/test_base_agent.py`
- Create: `agents/shared/base_agent.py`

**Step 1: Write the failing test**

`agents/tests/test_base_agent.py`:
```python
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.shared.base_agent import BaseAgent


class TestAgent(BaseAgent):
    """Concrete test agent."""

    def __init__(self, **kwargs):
        super().__init__(name="test_agent", **kwargs)
        self.dispatches = []
        self.started = False

    async def on_dispatch(self, message):
        self.dispatches.append(message)

    async def run(self):
        self.started = True


@pytest.fixture
def agent(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"name": "test_agent", "foo": "bar"}))
    return TestAgent(config_path=str(config_file), redis_url="redis://localhost:6379")


def test_agent_name(agent):
    assert agent.name == "test_agent"


def test_agent_loads_config(agent):
    assert agent.config["foo"] == "bar"


def test_agent_is_abstract():
    """BaseAgent requires run() and on_dispatch() to be implemented."""
    with pytest.raises(TypeError):
        BaseAgent(name="bad")


@pytest.mark.asyncio
async def test_agent_heartbeat(agent):
    agent.bus = AsyncMock()
    agent.bus.publish = AsyncMock()
    await agent.heartbeat()
    agent.bus.publish.assert_called_once()
    call_args = agent.bus.publish.call_args
    assert call_args[0][0] == "agents/heartbeat"
    assert call_args[0][1]["agent"] == "test_agent"
    assert call_args[0][1]["status"] == "alive"


@pytest.mark.asyncio
async def test_agent_dispatch_routing(agent):
    message = {
        "from": "prometheus",
        "channel": "prometheus/dispatch",
        "payload": {"task": "test", "target": "test_agent"},
    }
    await agent.on_dispatch(message)
    assert len(agent.dispatches) == 1
    assert agent.dispatches[0] == message
```

**Step 2: Run test to verify it fails**

Run: `pytest agents/tests/test_base_agent.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write minimal implementation**

`agents/shared/base_agent.py`:
```python
"""Base class for all Hivemind agents."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from agents.shared.bus import RedisBus
from agents.shared.config import load_agent_config
from agents.shared.logger import get_agent_logger
from agents.shared.ai_client import AIClient


class BaseAgent(ABC):
    """Abstract base class. All Hivemind agents inherit from this.

    Provides:
    - Redis message bus connection
    - Heartbeat broadcasting
    - Structured logging
    - AI client access with token tracking
    - Config loading
    """

    def __init__(
        self,
        name: str,
        config_path: str | None = None,
        redis_url: str = "redis://localhost:6379",
        gateway_url: str = "ws://127.0.0.1:18789",
        gateway_token: str = "",
    ):
        self.name = name
        self.config = {}
        if config_path:
            self.config = load_agent_config(config_path)
        self.bus = RedisBus(redis_url=redis_url)
        self.logger = get_agent_logger(name)
        self.ai = AIClient(gateway_url=gateway_url, token=gateway_token)
        self._running = False
        self._heartbeat_interval = 30

    async def start(self):
        """Connect to bus, subscribe to dispatch channel, start heartbeat, then run."""
        await self.bus.connect()
        await self.bus.subscribe("prometheus/dispatch", self._handle_dispatch)
        self._running = True
        self.logger.info(f"Agent {self.name} started")

        await asyncio.gather(
            self._heartbeat_loop(),
            self.run(),
        )

    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        await self.bus.disconnect()
        self.logger.info(f"Agent {self.name} stopped")

    async def heartbeat(self):
        """Publish a single heartbeat to the bus."""
        await self.bus.publish(
            "agents/heartbeat",
            {"agent": self.name, "status": "alive"},
            sender=self.name,
        )

    async def _heartbeat_loop(self):
        while self._running:
            try:
                await self.heartbeat()
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
            await asyncio.sleep(self._heartbeat_interval)

    async def _handle_dispatch(self, channel: str, message: dict):
        """Route dispatch messages to the agent if targeted."""
        payload = message.get("payload", {})
        target = payload.get("target")
        if target is None or target == self.name:
            await self.on_dispatch(message)

    async def alert_telegram(self, message: str):
        """Send urgent message directly to Telegram via the bus."""
        await self.bus.publish(
            "prometheus/telegram",
            {"text": message, "urgent": True},
            sender=self.name,
        )

    @abstractmethod
    async def run(self):
        """Main agent loop. Override in subclass."""
        ...

    @abstractmethod
    async def on_dispatch(self, message: dict):
        """Handle task dispatched by Prometheus. Override in subclass."""
        ...
```

**Step 4: Run test to verify it passes**

Run: `pytest agents/tests/test_base_agent.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add agents/shared/base_agent.py agents/tests/test_base_agent.py
git commit -m "feat: add BaseAgent abstract class with heartbeat and dispatch"
```

---

### Task 7: Add `__init__.py` exports and `__main__.py` runner

**Files:**
- Modify: `agents/shared/__init__.py`
- Create: `agents/__init__.py`
- Create: `agents/__main__.py`

**Step 1: Write the exports**

`agents/__init__.py`:
```python
"""OpenClaw Hivemind — Multi-agent system."""
```

`agents/shared/__init__.py`:
```python
"""Shared utilities for Hivemind agents."""

from agents.shared.base_agent import BaseAgent
from agents.shared.bus import RedisBus
from agents.shared.logger import get_agent_logger
from agents.shared.config import load_agent_config
from agents.shared.ai_client import AIClient

__all__ = ["BaseAgent", "RedisBus", "get_agent_logger", "load_agent_config", "AIClient"]
```

**Step 2: Run all tests**

Run: `pytest agents/tests/ -v`
Expected: All tests PASS (12 total)

**Step 3: Commit**

```bash
git add agents/__init__.py agents/shared/__init__.py
git commit -m "feat: add package exports for shared agent utilities"
```

---

### Task 8: Integration test — full agent lifecycle

**Files:**
- Create: `agents/tests/test_integration.py`

**Step 1: Write the integration test**

`agents/tests/test_integration.py`:
```python
"""Integration test: agent connects to Redis, heartbeats, receives dispatch."""

import asyncio
import pytest
from agents.shared.base_agent import BaseAgent
from agents.shared.bus import RedisBus


class EchoAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="echo", **kwargs)
        self.received = []

    async def on_dispatch(self, message):
        self.received.append(message)

    async def run(self):
        while self._running:
            await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_agent_receives_dispatch_via_bus(tmp_path):
    """Full lifecycle: agent starts, receives dispatch, heartbeats."""
    config = tmp_path / "config.json"
    config.write_text('{"name": "echo"}')

    agent = EchoAgent(config_path=str(config))

    # Start agent in background
    task = asyncio.create_task(agent.start())
    await asyncio.sleep(0.5)  # Let it connect and subscribe

    # Send a dispatch from a separate bus connection
    dispatcher = RedisBus()
    await dispatcher.connect()
    await dispatcher.publish(
        "prometheus/dispatch",
        {"task": "greet", "target": "echo", "data": "hello"},
        sender="prometheus",
    )
    await asyncio.sleep(0.5)  # Let message propagate

    # Verify agent received the dispatch
    assert len(agent.received) == 1
    assert agent.received[0]["payload"]["task"] == "greet"

    # Clean up
    await agent.stop()
    await dispatcher.disconnect()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
```

**Step 2: Run integration test**

Run: `pytest agents/tests/test_integration.py -v`
Expected: PASS — full lifecycle works end to end

Note: This test requires Redis running locally. On the server it will always be available. Locally, skip if Redis isn't running:

```python
# Add at top of test file:
import redis
pytestmark = pytest.mark.skipif(
    not redis.Redis().ping(), reason="Redis not available"
)
```

**Step 3: Commit**

```bash
git add agents/tests/test_integration.py
git commit -m "test: add full agent lifecycle integration test"
```

---

### Task 9: Final commit and push

**Step 1: Run all tests one final time**

Run: `pytest agents/tests/ -v`
Expected: All tests PASS

**Step 2: Push to GitHub**

```bash
git push origin main
```

---

## Summary

After Phase 1 is complete, you will have:

| Component | File | Purpose |
|-----------|------|---------|
| Message Bus | `agents/shared/bus.py` | Redis pub/sub with JSON envelopes |
| Logger | `agents/shared/logger.py` | Structured JSON logging |
| Config | `agents/shared/config.py` | Config loading with defaults |
| AI Client | `agents/shared/ai_client.py` | Gateway wrapper + token tracking |
| Base Agent | `agents/shared/base_agent.py` | Abstract class with heartbeat/dispatch |
| Tests | `agents/tests/` | 12+ unit tests + 1 integration test |

**Next:** Phase 2 — Build the Watchdog agent on top of this foundation.
