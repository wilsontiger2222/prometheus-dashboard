# Phase 4: Scout (Research & Intelligence) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an on-demand research agent that scrapes web content, summarizes it via AI, caches results, and publishes reports to the message bus.

**Architecture:** Scout listens for dispatch requests (from Prometheus or other agents), scrapes the requested URL/topic using httpx + BeautifulSoup, summarizes content via AIClient, caches results with configurable TTL, and publishes reports to `scout/reports`. Scout is on-demand — no polling loop, just an idle `run()` that waits while dispatch handles requests.

**Tech Stack:** httpx (already in requirements), beautifulsoup4 (new dep), AIClient (existing shared module), JSON file cache

---

### Task 1: Scaffold Scout package and config

**Files:**
- Create: `agents/scout/__init__.py`
- Create: `agents/scout/config.json`
- Modify: `agents/requirements.txt` (add beautifulsoup4)

**Step 1: Create `agents/scout/__init__.py`**

```python
"""Scout — research & intelligence gathering agent."""
```

**Step 2: Create `agents/scout/config.json`**

```json
{
  "name": "scout",
  "cache_dir": "agents/scout/cache",
  "default_cache_ttl_hours": 24,
  "max_sources": 5,
  "scraper_timeout_seconds": 15,
  "max_summary_length": 500
}
```

**Step 3: Add beautifulsoup4 to requirements**

Add `beautifulsoup4>=4.12.0` to `agents/requirements.txt`.

**Step 4: Install new dependency**

Run: `python -m pip install beautifulsoup4>=4.12.0`

**Step 5: Commit**

```bash
git add agents/scout/__init__.py agents/scout/config.json agents/requirements.txt
git commit -m "feat(scout): scaffold package and config"
```

---

### Task 2: Build scraper module

**Files:**
- Create: `agents/scout/scraper.py`
- Create: `agents/tests/test_scout_scraper.py`

**Step 1: Write the failing tests**

`agents/tests/test_scout_scraper.py`:

```python
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
        <body><p>Hello world</p><a href="/about">About</a></body></html>"""

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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_scout_scraper.py -v`
Expected: FAIL (ImportError — module doesn't exist yet)

**Step 3: Write implementation**

`agents/scout/scraper.py`:

```python
"""Web scraper for the Scout agent.

Uses httpx for async HTTP and BeautifulSoup for HTML parsing.
Extracts page title, text content (no scripts/styles), and links.
"""

from dataclasses import dataclass, field
import httpx
from bs4 import BeautifulSoup


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "links": self.links,
        }


class Scraper:
    """Async web scraper using httpx + BeautifulSoup."""

    def __init__(self, timeout: int = 15):
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "OpenClaw-Scout/1.0"},
        )

    async def scrape_url(self, url: str) -> ScrapedPage | None:
        """Fetch and parse a URL. Returns None on failure."""
        try:
            response = await self._client.get(url)
            if response.status_code != 200:
                return None
        except Exception:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator=" ", strip=True)
        links = [a["href"] for a in soup.find_all("a", href=True)
                 if a["href"].startswith("http")]

        return ScrapedPage(url=url, title=title, text=text, links=links)

    async def close(self):
        await self._client.aclose()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_scout_scraper.py -v`
Expected: 7 PASSED

**Step 5: Commit**

```bash
git add agents/scout/scraper.py agents/tests/test_scout_scraper.py
git commit -m "feat(scout): add web scraper with httpx + BeautifulSoup"
```

---

### Task 3: Build cache module

**Files:**
- Create: `agents/scout/cache.py`
- Create: `agents/tests/test_scout_cache.py`

**Step 1: Write the failing tests**

`agents/tests/test_scout_cache.py`:

```python
"""Tests for the Scout cache module."""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch
from agents.scout.cache import ResearchCache


class TestResearchCache:
    def test_get_miss_returns_none(self, tmp_path):
        cache = ResearchCache(cache_dir=str(tmp_path), default_ttl_hours=24)
        assert cache.get("nonexistent-key") is None

    def test_put_and_get(self, tmp_path):
        cache = ResearchCache(cache_dir=str(tmp_path), default_ttl_hours=24)
        data = {"summary": "BTC is up", "sources": ["https://example.com"]}
        cache.put("btc-analysis", data)
        result = cache.get("btc-analysis")
        assert result is not None
        assert result["summary"] == "BTC is up"

    def test_expired_entry_returns_none(self, tmp_path):
        cache = ResearchCache(cache_dir=str(tmp_path), default_ttl_hours=24)
        data = {"summary": "old data"}
        cache.put("old-key", data)

        # Manually expire by rewriting the timestamp
        cache_file = tmp_path / "old-key.json"
        entry = json.loads(cache_file.read_text())
        entry["expires_at"] = time.time() - 1  # expired 1s ago
        cache_file.write_text(json.dumps(entry))

        assert cache.get("old-key") is None

    def test_put_with_custom_ttl(self, tmp_path):
        cache = ResearchCache(cache_dir=str(tmp_path), default_ttl_hours=24)
        data = {"summary": "custom ttl"}
        cache.put("custom", data, ttl_hours=1)

        cache_file = tmp_path / "custom.json"
        entry = json.loads(cache_file.read_text())
        # TTL of 1 hour means expires_at should be ~3600s from created_at
        assert entry["expires_at"] - entry["created_at"] == pytest.approx(3600, abs=1)

    def test_list_keys(self, tmp_path):
        cache = ResearchCache(cache_dir=str(tmp_path), default_ttl_hours=24)
        cache.put("key-a", {"data": "a"})
        cache.put("key-b", {"data": "b"})
        keys = cache.list_keys()
        assert "key-a" in keys
        assert "key-b" in keys

    def test_delete_key(self, tmp_path):
        cache = ResearchCache(cache_dir=str(tmp_path), default_ttl_hours=24)
        cache.put("to-delete", {"data": "bye"})
        assert cache.get("to-delete") is not None
        cache.delete("to-delete")
        assert cache.get("to-delete") is None

    def test_clear_expired(self, tmp_path):
        cache = ResearchCache(cache_dir=str(tmp_path), default_ttl_hours=24)
        cache.put("valid", {"data": "keep"})
        cache.put("expired", {"data": "remove"})

        # Manually expire one entry
        cache_file = tmp_path / "expired.json"
        entry = json.loads(cache_file.read_text())
        entry["expires_at"] = time.time() - 1
        cache_file.write_text(json.dumps(entry))

        removed = cache.clear_expired()
        assert removed == 1
        assert cache.get("valid") is not None
        assert cache.get("expired") is None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_scout_cache.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/scout/cache.py`:

```python
"""File-based research cache with TTL expiration.

Stores research results as JSON files keyed by topic/query.
Each entry includes created_at and expires_at timestamps.
"""

import json
import time
from pathlib import Path


class ResearchCache:
    """JSON file cache with per-entry TTL."""

    def __init__(self, cache_dir: str, default_ttl_hours: float = 24):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._default_ttl = default_ttl_hours * 3600  # convert to seconds

    def get(self, key: str) -> dict | None:
        """Get cached data by key. Returns None if missing or expired."""
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None

        entry = json.loads(path.read_text())
        if time.time() > entry.get("expires_at", 0):
            return None

        return entry.get("data")

    def put(self, key: str, data: dict, ttl_hours: float | None = None):
        """Store data with TTL. Uses default TTL if not specified."""
        ttl = (ttl_hours * 3600) if ttl_hours is not None else self._default_ttl
        now = time.time()
        entry = {
            "key": key,
            "data": data,
            "created_at": now,
            "expires_at": now + ttl,
        }
        path = self._dir / f"{key}.json"
        path.write_text(json.dumps(entry, indent=2))

    def delete(self, key: str):
        """Remove a cache entry."""
        path = self._dir / f"{key}.json"
        if path.exists():
            path.unlink()

    def list_keys(self) -> list[str]:
        """List all cache keys (including expired)."""
        return [p.stem for p in self._dir.glob("*.json")]

    def clear_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        removed = 0
        now = time.time()
        for path in self._dir.glob("*.json"):
            entry = json.loads(path.read_text())
            if now > entry.get("expires_at", 0):
                path.unlink()
                removed += 1
        return removed
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_scout_cache.py -v`
Expected: 7 PASSED

**Step 5: Commit**

```bash
git add agents/scout/cache.py agents/tests/test_scout_cache.py
git commit -m "feat(scout): add file-based research cache with TTL"
```

---

### Task 4: Build summarizer module

**Files:**
- Create: `agents/scout/summarizer.py`
- Create: `agents/tests/test_scout_summarizer.py`

**Step 1: Write the failing tests**

`agents/tests/test_scout_summarizer.py`:

```python
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
        assert call_args[0][0] == "scout"  # agent_name
        assert "BTC" in call_args[0][1]    # topic in prompt

    @pytest.mark.asyncio
    async def test_summarize_text_with_max_length(self):
        mock_ai = AsyncMock()
        mock_ai.call = AsyncMock(return_value="Short summary.")

        summarizer = Summarizer(ai_client=mock_ai, agent_name="scout", max_length=200)
        await summarizer.summarize_text("text", topic="test")

        call_args = mock_ai.call.call_args
        assert "200" in call_args[0][1]  # max_length in prompt

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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_scout_summarizer.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/scout/summarizer.py`:

```python
"""AI-powered summarizer for the Scout agent.

Takes scraped page content and produces concise summaries
via the shared AIClient. Builds structured ResearchReports
from multiple sources.
"""

from dataclasses import dataclass, field
from agents.scout.scraper import ScrapedPage


@dataclass
class ResearchReport:
    topic: str
    summary: str
    sources: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "sources": self.sources,
            "key_facts": self.key_facts,
        }


class Summarizer:
    """Summarize text content using the AI client."""

    def __init__(self, ai_client, agent_name: str = "scout", max_length: int = 500):
        self._ai = ai_client
        self._agent_name = agent_name
        self._max_length = max_length

    async def summarize_text(self, text: str, topic: str) -> str:
        """Summarize a block of text about a given topic."""
        prompt = (
            f"Summarize the following content about '{topic}' "
            f"in {self._max_length} words or less. "
            f"Focus on key facts, numbers, and actionable insights.\n\n"
            f"{text[:5000]}"  # truncate to avoid token burn
        )
        return await self._ai.call(self._agent_name, prompt)

    async def build_report(self, topic: str, pages: list[ScrapedPage]) -> ResearchReport:
        """Build a ResearchReport from multiple scraped pages."""
        if not pages:
            return ResearchReport(
                topic=topic,
                summary="No content available to summarize.",
                sources=[],
                key_facts=[],
            )

        combined_text = "\n\n---\n\n".join(
            f"Source: {p.title}\n{p.text[:2000]}" for p in pages
        )
        sources = [p.url for p in pages]

        summary = await self.summarize_text(combined_text, topic)

        # Extract key facts (lines starting with dash or number)
        key_facts = [
            line.strip().lstrip("- ").lstrip("* ")
            for line in summary.split("\n")
            if line.strip().startswith(("-", "*", "•"))
        ]

        return ResearchReport(
            topic=topic,
            summary=summary,
            sources=sources,
            key_facts=key_facts,
        )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_scout_summarizer.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add agents/scout/summarizer.py agents/tests/test_scout_summarizer.py
git commit -m "feat(scout): add AI-powered summarizer with report builder"
```

---

### Task 5: Build ScoutAgent class

**Files:**
- Create: `agents/scout/agent.py`
- Create: `agents/tests/test_scout_agent.py`

**Step 1: Write the failing tests**

`agents/tests/test_scout_agent.py`:

```python
"""Tests for the ScoutAgent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.scout.agent import ScoutAgent


@pytest.fixture
def scout():
    with patch("agents.scout.agent.load_agent_config") as mock_config:
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
        scout._scraper.scrape_url = AsyncMock()
        scout._scraper.scrape_url.assert_not_called()
        # Should publish cached result
        scout.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_on_dispatch_status(self, scout):
        """Dispatch with task=status returns cache stats."""
        message = {"payload": {"task": "status"}}
        await scout.on_dispatch(message)
        scout.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_run_is_idle(self, scout):
        """Scout run() should just idle (on-demand agent)."""
        scout._running = False  # stop immediately
        await scout.run()  # should return without error
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_scout_agent.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

`agents/scout/agent.py`:

```python
"""Scout agent — on-demand research and intelligence gathering."""

import asyncio
from agents.shared.base_agent import BaseAgent
from agents.shared.config import load_agent_config
from agents.scout.scraper import Scraper
from agents.scout.cache import ResearchCache
from agents.scout.summarizer import Summarizer


class ScoutAgent(BaseAgent):
    """On-demand research agent. Scrapes, summarizes, and caches."""

    def __init__(self, **kwargs):
        super().__init__(name="scout", **kwargs)

        cache_dir = self.config.get("cache_dir", "agents/scout/cache")
        ttl = self.config.get("default_cache_ttl_hours", 24)
        timeout = self.config.get("scraper_timeout_seconds", 15)
        max_len = self.config.get("max_summary_length", 500)

        self._max_sources = self.config.get("max_sources", 5)
        self._scraper = Scraper(timeout=timeout)
        self._cache = ResearchCache(cache_dir=cache_dir, default_ttl_hours=ttl)
        self._summarizer = Summarizer(
            ai_client=self.ai,
            agent_name="scout",
            max_length=max_len,
        )

    async def run(self):
        """Scout is on-demand — idle until dispatch."""
        while self._running:
            await asyncio.sleep(5)

    async def on_dispatch(self, message: dict):
        payload = message.get("payload", {})
        task = payload.get("task", "")
        request_id = message.get("request_id", "")

        if task == "research":
            await self._handle_research(payload, request_id)
        elif task == "status":
            keys = self._cache.list_keys()
            await self.bus.publish(
                "scout/status",
                {"cached_topics": keys, "count": len(keys)},
                sender="scout",
            )

    async def _handle_research(self, payload: dict, request_id: str):
        topic = payload.get("topic", "unknown")
        urls = payload.get("urls", [])[:self._max_sources]

        # Check cache first
        cache_key = topic.lower().replace(" ", "-")
        cached = self._cache.get(cache_key)
        if cached:
            self.logger.info(f"Cache hit for '{topic}'")
            await self.bus.publish(
                "scout/reports",
                {"report": cached, "cached": True, "request_id": request_id},
                sender="scout",
            )
            return

        # Scrape all URLs
        pages = []
        for url in urls:
            page = await self._scraper.scrape_url(url)
            if page:
                pages.append(page)

        # Build report
        report = await self._summarizer.build_report(topic, pages)
        report_dict = report.to_dict()

        # Cache the result
        self._cache.put(cache_key, report_dict)

        # Publish to bus
        self.logger.info(f"Research complete: '{topic}' ({len(pages)} sources)")
        await self.bus.publish(
            "scout/reports",
            {"report": report_dict, "cached": False, "request_id": request_id},
            sender="scout",
        )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_scout_agent.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add agents/scout/agent.py agents/tests/test_scout_agent.py
git commit -m "feat(scout): add ScoutAgent with research, cache, and summarization"
```

---

### Task 6: Add Scout `__main__.py` entry point

**Files:**
- Create: `agents/scout/__main__.py`

**Step 1: Write `__main__.py`**

```python
"""Run the Scout agent: python -m agents.scout"""

import asyncio
import sys
from pathlib import Path

from agents.scout.agent import ScoutAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = ScoutAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nScout shutting down...")


if __name__ == "__main__":
    main()
```

**Step 2: Verify import works**

Run: `python -c "from agents.scout.agent import ScoutAgent; print('OK')"`

**Step 3: Commit**

```bash
git add agents/scout/__main__.py
git commit -m "feat(scout): add __main__.py entry point"
```

---

### Task 7: Final test run and push

**Step 1: Run full test suite**

Run: `python -m pytest agents/tests/ -v`
Expected: All tests pass (no regressions)

**Step 2: Commit plan doc**

```bash
git add docs/plans/2026-02-07-phase4-scout.md
git commit -m "docs: add Phase 4 Scout implementation plan"
```

**Step 3: Push to GitHub**

```bash
git push
```
