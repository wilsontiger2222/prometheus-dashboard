"""Heartbeat tracker for agent liveness monitoring.

Prometheus uses this to track which agents are alive based on
their periodic heartbeat messages on the agents/heartbeat channel.
"""

import time


class HeartbeatTracker:
    """Track agent heartbeats and detect stale agents."""

    def __init__(self, stale_seconds: float = 90):
        self._stale_seconds = stale_seconds
        self._heartbeats: dict[str, float] = {}

    def record(self, agent: str, timestamp: float | None = None):
        """Record a heartbeat from an agent."""
        self._heartbeats[agent] = timestamp if timestamp is not None else time.time()

    def is_alive(self, agent: str) -> bool:
        """Check if an agent is alive (heartbeat within stale threshold)."""
        last_seen = self._heartbeats.get(agent)
        if last_seen is None:
            return False
        return (time.time() - last_seen) < self._stale_seconds

    def get_status(self) -> dict[str, dict]:
        """Get status of all known agents."""
        now = time.time()
        return {
            agent: {
                "alive": (now - ts) < self._stale_seconds,
                "last_seen": ts,
                "age_seconds": round(now - ts, 1),
            }
            for agent, ts in self._heartbeats.items()
        }

    def get_stale_agents(self) -> list[str]:
        """Return list of agents that have gone stale."""
        return [a for a in self._heartbeats if not self.is_alive(a)]

    def get_alive_agents(self) -> list[str]:
        """Return list of agents that are alive."""
        return [a for a in self._heartbeats if self.is_alive(a)]
