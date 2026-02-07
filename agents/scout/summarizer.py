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
            f"{text[:5000]}"
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
