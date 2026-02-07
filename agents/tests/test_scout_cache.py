"""Tests for the Scout cache module."""

import json
import time
import pytest
from pathlib import Path
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
