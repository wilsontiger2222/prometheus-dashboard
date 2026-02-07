"""Tests for the Prometheus heartbeat tracker."""

import time
import pytest
from agents.prometheus.heartbeat import HeartbeatTracker


class TestHeartbeatTracker:
    def test_record_heartbeat(self):
        tracker = HeartbeatTracker(stale_seconds=90)
        tracker.record("sentinel")
        status = tracker.get_status()
        assert "sentinel" in status
        assert status["sentinel"]["alive"] is True

    def test_stale_agent(self):
        tracker = HeartbeatTracker(stale_seconds=1)
        tracker.record("watchdog", timestamp=time.time() - 5)
        status = tracker.get_status()
        assert status["watchdog"]["alive"] is False

    def test_multiple_agents(self):
        tracker = HeartbeatTracker(stale_seconds=90)
        tracker.record("sentinel")
        tracker.record("watchdog")
        tracker.record("scout")
        status = tracker.get_status()
        assert len(status) == 3
        assert all(s["alive"] for s in status.values())

    def test_get_stale_agents(self):
        tracker = HeartbeatTracker(stale_seconds=1)
        tracker.record("sentinel")
        tracker.record("watchdog", timestamp=time.time() - 5)
        stale = tracker.get_stale_agents()
        assert "watchdog" in stale
        assert "sentinel" not in stale

    def test_get_alive_agents(self):
        tracker = HeartbeatTracker(stale_seconds=1)
        tracker.record("sentinel")
        tracker.record("watchdog", timestamp=time.time() - 5)
        alive = tracker.get_alive_agents()
        assert "sentinel" in alive
        assert "watchdog" not in alive
