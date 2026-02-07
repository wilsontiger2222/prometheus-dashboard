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
