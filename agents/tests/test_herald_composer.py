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
