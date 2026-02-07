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
