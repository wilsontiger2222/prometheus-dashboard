"""Tests for the Scout summarizer module."""

import pytest
from unittest.mock import AsyncMock
from agents.scout.summarizer import Summarizer, ResearchReport


class TestResearchReport:
    def test_report_creation(self):
        report = ResearchReport(
            topic="BTC analysis",
            summary="Bitcoin is trending up",
            sources=["https://example.com"],
            key_facts=["Price above 100k"],
        )
        assert report.topic == "BTC analysis"
        assert report.summary == "Bitcoin is trending up"

    def test_report_to_dict(self):
        report = ResearchReport(
            topic="test",
            summary="summary",
            sources=["https://s.com"],
            key_facts=["fact1"],
        )
        d = report.to_dict()
        assert d["topic"] == "test"
        assert d["summary"] == "summary"
        assert d["sources"] == ["https://s.com"]
        assert d["key_facts"] == ["fact1"]


class TestSummarizer:
    @pytest.mark.asyncio
    async def test_summarize_text(self):
        mock_ai = AsyncMock()
        mock_ai.call = AsyncMock(return_value="Bitcoin surged past 100k on strong volume.")

        summarizer = Summarizer(ai_client=mock_ai, agent_name="scout")
        result = await summarizer.summarize_text("long article text here...", topic="BTC")

        assert result == "Bitcoin surged past 100k on strong volume."
        mock_ai.call.assert_called_once()
        call_args = mock_ai.call.call_args
        assert call_args[0][0] == "scout"
        assert "BTC" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_summarize_text_with_max_length(self):
        mock_ai = AsyncMock()
        mock_ai.call = AsyncMock(return_value="Short summary.")

        summarizer = Summarizer(ai_client=mock_ai, agent_name="scout", max_length=200)
        await summarizer.summarize_text("text", topic="test")

        call_args = mock_ai.call.call_args
        assert "200" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_build_report(self):
        mock_ai = AsyncMock()
        mock_ai.call = AsyncMock(return_value="Summary of research.\nKey: fact1, fact2")

        summarizer = Summarizer(ai_client=mock_ai, agent_name="scout")

        from agents.scout.scraper import ScrapedPage
        pages = [
            ScrapedPage(url="https://a.com", title="A", text="Content A", links=[]),
            ScrapedPage(url="https://b.com", title="B", text="Content B", links=[]),
        ]

        report = await summarizer.build_report("test topic", pages)

        assert isinstance(report, ResearchReport)
        assert report.topic == "test topic"
        assert "https://a.com" in report.sources
        assert "https://b.com" in report.sources

    @pytest.mark.asyncio
    async def test_build_report_empty_pages(self):
        mock_ai = AsyncMock()
        summarizer = Summarizer(ai_client=mock_ai, agent_name="scout")

        report = await summarizer.build_report("empty", [])

        assert report.summary == "No content available to summarize."
        mock_ai.call.assert_not_called()
