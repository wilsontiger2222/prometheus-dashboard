"""Tests for the ScoutAgent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.scout.agent import ScoutAgent


@pytest.fixture
def scout():
    with patch("agents.shared.base_agent.load_agent_config") as mock_config:
        mock_config.return_value = {
            "name": "scout",
            "cache_dir": "/tmp/scout_test_cache",
            "default_cache_ttl_hours": 24,
            "max_sources": 5,
            "scraper_timeout_seconds": 15,
            "max_summary_length": 500,
        }
        agent = ScoutAgent(config_path="dummy.json")
        agent.bus = AsyncMock()
        agent._running = True
        return agent


class TestScoutAgent:
    def test_scout_name(self, scout):
        assert scout.name == "scout"

    def test_scout_loads_config(self, scout):
        assert scout._max_sources == 5
        assert scout._cache is not None

    @pytest.mark.asyncio
    async def test_on_dispatch_research_url(self, scout):
        """Dispatch with task=research and a URL scrapes and summarizes."""
        from agents.scout.scraper import ScrapedPage
        mock_page = ScrapedPage(
            url="https://example.com",
            title="Test",
            text="Content about BTC",
            links=[],
        )
        scout._scraper.scrape_url = AsyncMock(return_value=mock_page)
        scout._summarizer.build_report = AsyncMock(return_value=MagicMock(
            to_dict=lambda: {"topic": "BTC", "summary": "BTC is up"},
        ))
        # Ensure cache miss
        scout._cache.get = MagicMock(return_value=None)

        message = {
            "payload": {
                "task": "research",
                "topic": "BTC",
                "urls": ["https://example.com"],
            },
            "request_id": "req-123",
        }
        await scout.on_dispatch(message)

        scout._scraper.scrape_url.assert_called_once_with("https://example.com")
        scout._summarizer.build_report.assert_called_once()
        scout.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_on_dispatch_uses_cache(self, scout):
        """If cached result exists, return it without scraping."""
        scout._cache.get = MagicMock(return_value={
            "topic": "BTC", "summary": "cached result",
        })
        scout._scraper.scrape_url = AsyncMock()

        message = {
            "payload": {
                "task": "research",
                "topic": "BTC",
                "urls": ["https://example.com"],
            },
            "request_id": "req-456",
        }
        await scout.on_dispatch(message)

        # Should NOT scrape since cache hit
        scout._scraper.scrape_url.assert_not_called()
        # Should publish cached result
        scout.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_on_dispatch_status(self, scout):
        """Dispatch with task=status returns cache stats."""
        scout._cache.list_keys = MagicMock(return_value=["btc", "eth"])
        message = {"payload": {"task": "status"}}
        await scout.on_dispatch(message)
        scout.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_run_is_idle(self, scout):
        """Scout run() should just idle (on-demand agent)."""
        scout._running = False  # stop immediately
        await scout.run()  # should return without error
