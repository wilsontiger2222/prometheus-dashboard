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
