"""Scout agent — on-demand research and intelligence gathering."""

import asyncio
from agents.shared.base_agent import BaseAgent
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
