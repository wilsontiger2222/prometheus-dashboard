# Phase 7: Herald (Social Media & Content) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a hybrid agent that generates content via AI, manages a draft review queue (approval via Telegram), schedules posts, and publishes to platforms. Supports trigger-based content from other agents (Sentinel trade wins, Forge deploys, Scout trends).

**Architecture:** Herald is hybrid — it has a scheduled posting loop AND handles on-demand dispatch requests. Content drafts go through a review queue that requires user approval before publishing. Platform adapters are pluggable (starting with a base adapter). The AI client generates content from templates/prompts.

**Tech Stack:** BaseAgent (existing), AIClient (existing), JSON-based draft queue, pluggable platform adapters

---

### Task 1: Scaffold Herald package and config

**Files:**
- Create: `agents/herald/__init__.py`
- Create: `agents/herald/config.json`

**Step 1: Create `agents/herald/__init__.py`**

```python
"""Herald — social media & content creation agent."""
```

**Step 2: Create `agents/herald/config.json`**

```json
{
  "name": "herald",
  "platforms": ["twitter"],
  "require_approval": true,
  "scheduled_posts_per_day": 3,
  "content_templates_dir": "agents/herald/templates",
  "drafts_dir": "agents/herald/drafts",
  "schedule_interval_seconds": 3600
}
```

**Step 3: Commit**

```bash
git add agents/herald/__init__.py agents/herald/config.json
git commit -m "feat(herald): scaffold package and config"
```

---

### Task 2: Build content composer module

**Files:**
- Create: `agents/herald/composer.py`
- Create: `agents/tests/test_herald_composer.py`

**Step 1: Write the failing tests**

`agents/tests/test_herald_composer.py`:

```python
"""Tests for the Herald content composer."""

import pytest
from unittest.mock import AsyncMock
from agents.herald.composer import Composer, Draft


class TestDraft:
    def test_draft_creation(self):
        draft = Draft(
            content="BTC just hit 100k!",
            platform="twitter",
            trigger="sentinel",
            topic="BTC milestone",
        )
        assert draft.content == "BTC just hit 100k!"
        assert draft.platform == "twitter"
        assert draft.status == "pending"

    def test_draft_to_dict(self):
        draft = Draft(
            content="New deploy live",
            platform="twitter",
            trigger="forge",
            topic="deploy",
        )
        d = draft.to_dict()
        assert d["content"] == "New deploy live"
        assert d["platform"] == "twitter"
        assert d["status"] == "pending"
        assert "id" in d
        assert "created_at" in d

    def test_draft_approve(self):
        draft = Draft(content="test", platform="twitter", trigger="manual", topic="test")
        draft.approve()
        assert draft.status == "approved"

    def test_draft_reject(self):
        draft = Draft(content="test", platform="twitter", trigger="manual", topic="test")
        draft.reject()
        assert draft.status == "rejected"


class TestComposer:
    @pytest.mark.asyncio
    async def test_compose_from_prompt(self):
        mock_ai = AsyncMock()
        mock_ai.call = AsyncMock(return_value="Bitcoin breaks 100k — a new era for crypto!")

        composer = Composer(ai_client=mock_ai, agent_name="herald")
        draft = await composer.compose(
            topic="BTC milestone",
            platform="twitter",
            trigger="sentinel",
        )

        assert isinstance(draft, Draft)
        assert draft.content == "Bitcoin breaks 100k — a new era for crypto!"
        assert draft.platform == "twitter"
        mock_ai.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_compose_includes_platform_in_prompt(self):
        mock_ai = AsyncMock()
        mock_ai.call = AsyncMock(return_value="Post content")

        composer = Composer(ai_client=mock_ai, agent_name="herald")
        await composer.compose(topic="test", platform="twitter", trigger="manual")

        call_args = mock_ai.call.call_args
        assert "twitter" in call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_compose_with_context(self):
        mock_ai = AsyncMock()
        mock_ai.call = AsyncMock(return_value="Deployed v2.0!")

        composer = Composer(ai_client=mock_ai, agent_name="herald")
        draft = await composer.compose(
            topic="deploy",
            platform="twitter",
            trigger="forge",
            context="Successfully deployed prometheus-dashboard v2.0",
        )

        assert draft.content == "Deployed v2.0!"
        call_args = mock_ai.call.call_args
        assert "v2.0" in call_args[0][1]
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_herald_composer.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/herald/composer.py`:

```python
"""Content composer for the Herald agent.

Uses AI to generate platform-appropriate content from topics and context.
Produces Draft objects that go through the review queue before publishing.
"""

import uuid
import time
from dataclasses import dataclass, field


@dataclass
class Draft:
    content: str
    platform: str
    trigger: str
    topic: str
    status: str = "pending"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)

    def approve(self):
        self.status = "approved"

    def reject(self):
        self.status = "rejected"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "platform": self.platform,
            "trigger": self.trigger,
            "topic": self.topic,
            "status": self.status,
            "created_at": self.created_at,
        }


class Composer:
    """Generate content drafts using AI."""

    COMPOSE_PROMPT = (
        "You are a social media content creator. "
        "Write a {platform} post about: {topic}. "
        "{context_line}"
        "Keep it concise, engaging, and appropriate for {platform}. "
        "Do not include hashtags unless asked. "
        "Return ONLY the post text, nothing else."
    )

    def __init__(self, ai_client, agent_name: str = "herald"):
        self._ai = ai_client
        self._agent_name = agent_name

    async def compose(
        self,
        topic: str,
        platform: str,
        trigger: str,
        context: str = "",
    ) -> Draft:
        """Generate a content draft for a given topic and platform."""
        context_line = f"Context: {context}\n" if context else ""
        prompt = self.COMPOSE_PROMPT.format(
            platform=platform,
            topic=topic,
            context_line=context_line,
        )

        content = await self._ai.call(self._agent_name, prompt)

        return Draft(
            content=content,
            platform=platform,
            trigger=trigger,
            topic=topic,
        )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_herald_composer.py -v`
Expected: 7 PASSED

**Step 5: Commit**

```bash
git add agents/herald/composer.py agents/tests/test_herald_composer.py
git commit -m "feat(herald): add AI content composer with draft system"
```

---

### Task 3: Build review queue module

**Files:**
- Create: `agents/herald/queue.py`
- Create: `agents/tests/test_herald_queue.py`

**Step 1: Write the failing tests**

`agents/tests/test_herald_queue.py`:

```python
"""Tests for the Herald review queue."""

import pytest
from agents.herald.queue import ReviewQueue
from agents.herald.composer import Draft


def _make_draft(content="test", draft_id="d1"):
    return Draft(
        content=content, platform="twitter", trigger="manual",
        topic="test", id=draft_id,
    )


class TestReviewQueue:
    def test_add_draft(self):
        queue = ReviewQueue()
        draft = _make_draft()
        queue.add(draft)
        assert len(queue.get_pending()) == 1

    def test_get_pending_only_returns_pending(self):
        queue = ReviewQueue()
        d1 = _make_draft(content="pending", draft_id="d1")
        d2 = _make_draft(content="approved", draft_id="d2")
        d2.approve()
        queue.add(d1)
        queue.add(d2)
        pending = queue.get_pending()
        assert len(pending) == 1
        assert pending[0].id == "d1"

    def test_approve_draft(self):
        queue = ReviewQueue()
        draft = _make_draft(draft_id="d1")
        queue.add(draft)
        result = queue.approve("d1")
        assert result is True
        assert draft.status == "approved"

    def test_reject_draft(self):
        queue = ReviewQueue()
        draft = _make_draft(draft_id="d1")
        queue.add(draft)
        result = queue.reject("d1")
        assert result is True
        assert draft.status == "rejected"

    def test_approve_nonexistent_returns_false(self):
        queue = ReviewQueue()
        assert queue.approve("nope") is False

    def test_get_approved(self):
        queue = ReviewQueue()
        d1 = _make_draft(draft_id="d1")
        d2 = _make_draft(draft_id="d2")
        queue.add(d1)
        queue.add(d2)
        queue.approve("d1")
        approved = queue.get_approved()
        assert len(approved) == 1
        assert approved[0].id == "d1"

    def test_remove_draft(self):
        queue = ReviewQueue()
        draft = _make_draft(draft_id="d1")
        queue.add(draft)
        queue.remove("d1")
        assert len(queue.get_pending()) == 0

    def test_get_all(self):
        queue = ReviewQueue()
        queue.add(_make_draft(draft_id="d1"))
        queue.add(_make_draft(draft_id="d2"))
        queue.add(_make_draft(draft_id="d3"))
        assert len(queue.get_all()) == 3
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_herald_queue.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/herald/queue.py`:

```python
"""Review queue for Herald content drafts.

Drafts must be approved before publishing. The queue holds
drafts in memory and provides approve/reject/list operations.
"""

from agents.herald.composer import Draft


class ReviewQueue:
    """In-memory draft review queue."""

    def __init__(self):
        self._drafts: dict[str, Draft] = {}

    def add(self, draft: Draft):
        """Add a draft to the review queue."""
        self._drafts[draft.id] = draft

    def approve(self, draft_id: str) -> bool:
        """Approve a draft. Returns False if not found."""
        draft = self._drafts.get(draft_id)
        if not draft:
            return False
        draft.approve()
        return True

    def reject(self, draft_id: str) -> bool:
        """Reject a draft. Returns False if not found."""
        draft = self._drafts.get(draft_id)
        if not draft:
            return False
        draft.reject()
        return True

    def remove(self, draft_id: str):
        """Remove a draft from the queue."""
        self._drafts.pop(draft_id, None)

    def get_pending(self) -> list[Draft]:
        """Get all pending (unapproved) drafts."""
        return [d for d in self._drafts.values() if d.status == "pending"]

    def get_approved(self) -> list[Draft]:
        """Get all approved drafts ready to publish."""
        return [d for d in self._drafts.values() if d.status == "approved"]

    def get_all(self) -> list[Draft]:
        """Get all drafts regardless of status."""
        return list(self._drafts.values())
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_herald_queue.py -v`
Expected: 8 PASSED

**Step 5: Commit**

```bash
git add agents/herald/queue.py agents/tests/test_herald_queue.py
git commit -m "feat(herald): add review queue for content drafts"
```

---

### Task 4: Build HeraldAgent class

**Files:**
- Create: `agents/herald/agent.py`
- Create: `agents/tests/test_herald_agent.py`

**Step 1: Write the failing tests**

`agents/tests/test_herald_agent.py`:

```python
"""Tests for the HeraldAgent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.herald.agent import HeraldAgent
from agents.herald.composer import Draft


@pytest.fixture
def herald():
    with patch("agents.shared.base_agent.load_agent_config") as mock_config:
        mock_config.return_value = {
            "name": "herald",
            "platforms": ["twitter"],
            "require_approval": True,
            "scheduled_posts_per_day": 3,
            "content_templates_dir": "agents/herald/templates",
            "drafts_dir": "agents/herald/drafts",
            "schedule_interval_seconds": 3600,
        }
        agent = HeraldAgent(config_path="dummy.json")
        agent.bus = AsyncMock()
        agent._running = True
        return agent


class TestHeraldAgent:
    def test_herald_name(self, herald):
        assert herald.name == "herald"

    def test_loads_config(self, herald):
        assert herald._require_approval is True
        assert "twitter" in herald._platforms

    @pytest.mark.asyncio
    async def test_on_dispatch_compose(self, herald):
        """Dispatch with task=compose creates a draft."""
        herald._composer.compose = AsyncMock(return_value=Draft(
            content="Test post", platform="twitter",
            trigger="manual", topic="test",
        ))
        message = {
            "payload": {
                "task": "compose",
                "args": "test topic",
            }
        }
        await herald.on_dispatch(message)
        herald._composer.compose.assert_called_once()
        herald.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_on_dispatch_approve(self, herald):
        """Dispatch with task=approve approves a draft."""
        draft = Draft(content="test", platform="twitter",
                      trigger="manual", topic="test", id="d1")
        herald._queue.add(draft)
        message = {"payload": {"task": "approve", "args": "d1"}}
        await herald.on_dispatch(message)
        assert draft.status == "approved"

    @pytest.mark.asyncio
    async def test_on_dispatch_reject(self, herald):
        """Dispatch with task=reject rejects a draft."""
        draft = Draft(content="test", platform="twitter",
                      trigger="manual", topic="test", id="d1")
        herald._queue.add(draft)
        message = {"payload": {"task": "reject", "args": "d1"}}
        await herald.on_dispatch(message)
        assert draft.status == "rejected"

    @pytest.mark.asyncio
    async def test_on_dispatch_status(self, herald):
        """Dispatch with task=status returns queue info."""
        herald._queue.add(Draft(content="a", platform="twitter",
                                trigger="manual", topic="t", id="d1"))
        message = {"payload": {"task": "status"}}
        await herald.on_dispatch(message)
        herald.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_on_dispatch_list(self, herald):
        """Dispatch with task=list returns all drafts."""
        herald._queue.add(Draft(content="a", platform="twitter",
                                trigger="manual", topic="t", id="d1"))
        herald._queue.add(Draft(content="b", platform="twitter",
                                trigger="manual", topic="t", id="d2"))
        message = {"payload": {"task": "list"}}
        await herald.on_dispatch(message)
        herald.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_handle_trigger_creates_draft(self, herald):
        """Agent trigger (e.g. from Sentinel) creates a draft."""
        herald._composer.compose = AsyncMock(return_value=Draft(
            content="Trade win!", platform="twitter",
            trigger="sentinel", topic="trade",
        ))
        await herald._handle_trigger(
            trigger="sentinel",
            topic="BTC trade win",
            context="Closed BTC long +5%",
        )
        herald._composer.compose.assert_called_once()
        assert len(herald._queue.get_pending()) == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_herald_agent.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/herald/agent.py`:

```python
"""Herald agent — social media & content creation."""

import asyncio
from agents.shared.base_agent import BaseAgent
from agents.herald.composer import Composer
from agents.herald.queue import ReviewQueue


class HeraldAgent(BaseAgent):
    """Hybrid content creation and publishing agent."""

    def __init__(self, **kwargs):
        super().__init__(name="herald", **kwargs)

        self._platforms = self.config.get("platforms", ["twitter"])
        self._require_approval = self.config.get("require_approval", True)
        self._schedule_interval = self.config.get("schedule_interval_seconds", 3600)

        self._composer = Composer(ai_client=self.ai, agent_name="herald")
        self._queue = ReviewQueue()

    async def run(self):
        """Herald listens for triggers from other agents."""
        await self.bus.subscribe("sentinel/alerts", self._on_sentinel_alert)
        await self.bus.subscribe("forge/deploys", self._on_forge_deploy)

        while self._running:
            await asyncio.sleep(self._schedule_interval)

    async def on_dispatch(self, message: dict):
        payload = message.get("payload", {})
        task = payload.get("task", "")
        args = payload.get("args", "")

        if task == "compose":
            await self._handle_compose(args, payload)
        elif task == "approve":
            self._queue.approve(args)
            await self.bus.publish(
                "herald/status",
                {"action": "approved", "draft_id": args},
                sender="herald",
            )
        elif task == "reject":
            self._queue.reject(args)
            await self.bus.publish(
                "herald/status",
                {"action": "rejected", "draft_id": args},
                sender="herald",
            )
        elif task == "status":
            pending = self._queue.get_pending()
            approved = self._queue.get_approved()
            await self.bus.publish(
                "herald/status",
                {
                    "pending": len(pending),
                    "approved": len(approved),
                    "total": len(self._queue.get_all()),
                },
                sender="herald",
            )
        elif task == "list":
            drafts = [d.to_dict() for d in self._queue.get_all()]
            await self.bus.publish(
                "herald/status",
                {"drafts": drafts},
                sender="herald",
            )

    async def _handle_compose(self, topic: str, payload: dict):
        """Compose a new draft and add to review queue."""
        platform = payload.get("platform", self._platforms[0])
        context = payload.get("context", "")
        draft = await self._composer.compose(
            topic=topic,
            platform=platform,
            trigger="manual",
            context=context,
        )
        self._queue.add(draft)
        self.logger.info(f"Draft created: {draft.id} — '{topic}'")

        if self._require_approval:
            await self.alert_telegram(
                f"New draft [{draft.id}]: {draft.content[:100]}"
            )

        await self.bus.publish(
            "herald/posts",
            {"action": "draft_created", "draft": draft.to_dict()},
            sender="herald",
        )

    async def _handle_trigger(self, trigger: str, topic: str, context: str = ""):
        """Handle a trigger from another agent — compose a draft."""
        platform = self._platforms[0] if self._platforms else "twitter"
        draft = await self._composer.compose(
            topic=topic,
            platform=platform,
            trigger=trigger,
            context=context,
        )
        self._queue.add(draft)
        self.logger.info(f"Trigger draft [{trigger}]: {draft.id}")

        if self._require_approval:
            await self.alert_telegram(
                f"Auto-draft [{draft.id}] from {trigger}: {draft.content[:100]}"
            )

    async def _on_sentinel_alert(self, channel: str, message: dict):
        """Handle trading alerts — draft a post about wins."""
        payload = message.get("payload", {})
        if payload.get("type") == "take_profit":
            symbol = payload.get("symbol", "")
            pnl = payload.get("pnl", 0)
            await self._handle_trigger(
                trigger="sentinel",
                topic=f"{symbol} trade win",
                context=f"Closed {symbol} position with PnL: {pnl}",
            )

    async def _on_forge_deploy(self, channel: str, message: dict):
        """Handle deploy events — draft a post about releases."""
        payload = message.get("payload", {})
        if payload.get("success"):
            repo = payload.get("repo", "")
            await self._handle_trigger(
                trigger="forge",
                topic=f"{repo} deployment",
                context=f"Successfully deployed {repo}",
            )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_herald_agent.py -v`
Expected: 8 PASSED

**Step 5: Commit**

```bash
git add agents/herald/agent.py agents/tests/test_herald_agent.py
git commit -m "feat(herald): add HeraldAgent with compose, review queue, and triggers"
```

---

### Task 5: Add Herald `__main__.py` entry point

**Files:**
- Create: `agents/herald/__main__.py`

**Step 1: Write `__main__.py`**

```python
"""Run the Herald agent: python -m agents.herald"""

import asyncio
import sys
from pathlib import Path

from agents.herald.agent import HeraldAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = HeraldAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nHerald shutting down...")


if __name__ == "__main__":
    main()
```

**Step 2: Verify import works**

Run: `python -c "from agents.herald.agent import HeraldAgent; print('OK')"`

**Step 3: Commit**

```bash
git add agents/herald/__main__.py
git commit -m "feat(herald): add __main__.py entry point"
```

---

### Task 6: Final test run and push

**Step 1: Run full test suite**

Run: `python -m pytest agents/tests/ -v`
Expected: All tests pass (no regressions)

**Step 2: Commit plan doc**

```bash
git add docs/plans/2026-02-07-phase7-herald.md
git commit -m "docs: add Phase 7 Herald implementation plan"
```

**Step 3: Push to GitHub**

```bash
git push
```
