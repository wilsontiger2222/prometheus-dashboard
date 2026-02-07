"""Tests for the Scout scraper module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.scout.scraper import Scraper, ScrapedPage


class TestScrapedPage:
    def test_scraped_page_creation(self):
        page = ScrapedPage(
            url="https://example.com",
            title="Example",
            text="Hello world",
            links=["https://example.com/about"],
        )
        assert page.url == "https://example.com"
        assert page.title == "Example"
        assert page.text == "Hello world"
        assert len(page.links) == 1

    def test_scraped_page_to_dict(self):
        page = ScrapedPage(
            url="https://example.com",
            title="Test",
            text="Content here",
            links=[],
        )
        d = page.to_dict()
        assert d["url"] == "https://example.com"
        assert d["title"] == "Test"
        assert d["text"] == "Content here"
        assert d["links"] == []


class TestScraper:
    @pytest.mark.asyncio
    async def test_scrape_url_returns_scraped_page(self):
        html = """<html><head><title>Test Page</title></head>
        <body><p>Hello world</p><a href="https://example.com/about">About</a></body></html>"""

        mock_response = AsyncMock()
        mock_response.text = html
        mock_response.status_code = 200

        scraper = Scraper(timeout=10)
        with patch.object(scraper, '_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await scraper.scrape_url("https://example.com")

        assert isinstance(result, ScrapedPage)
        assert result.title == "Test Page"
        assert "Hello world" in result.text

    @pytest.mark.asyncio
    async def test_scrape_url_returns_none_on_error(self):
        scraper = Scraper(timeout=10)
        with patch.object(scraper, '_client') as mock_client:
            mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))
            result = await scraper.scrape_url("https://bad-url.example")

        assert result is None

    @pytest.mark.asyncio
    async def test_scrape_url_extracts_links(self):
        html = """<html><head><title>Links</title></head><body>
        <a href="https://a.com">A</a>
        <a href="https://b.com">B</a>
        </body></html>"""

        mock_response = AsyncMock()
        mock_response.text = html
        mock_response.status_code = 200

        scraper = Scraper(timeout=10)
        with patch.object(scraper, '_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await scraper.scrape_url("https://example.com")

        assert "https://a.com" in result.links
        assert "https://b.com" in result.links

    @pytest.mark.asyncio
    async def test_scrape_url_strips_script_and_style(self):
        html = """<html><head><title>Clean</title>
        <style>body{color:red}</style></head>
        <body><script>alert('x')</script><p>Real content</p></body></html>"""

        mock_response = AsyncMock()
        mock_response.text = html
        mock_response.status_code = 200

        scraper = Scraper(timeout=10)
        with patch.object(scraper, '_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await scraper.scrape_url("https://example.com")

        assert "alert" not in result.text
        assert "color:red" not in result.text
        assert "Real content" in result.text

    @pytest.mark.asyncio
    async def test_scrape_url_handles_non_200(self):
        mock_response = AsyncMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        scraper = Scraper(timeout=10)
        with patch.object(scraper, '_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await scraper.scrape_url("https://example.com/missing")

        assert result is None
