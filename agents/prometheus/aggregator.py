"""Response aggregator for the Prometheus orchestrator.

Collects responses from multiple agents for a given request,
tracks completion, and supports timeout-based waiting.
"""

import asyncio


class Aggregator:
    """Collect and aggregate multi-agent responses."""

    def __init__(self, timeout: float = 30):
        self._timeout = timeout
        self._requests: dict[str, dict] = {}

    def start_request(self, request_id: str, expected: list[str]):
        """Register a new request with expected agent responses."""
        self._requests[request_id] = {
            "expected": set(expected),
            "responses": {},
        }

    def add_response(self, request_id: str, agent: str, data: dict):
        """Add a response from an agent."""
        if request_id in self._requests:
            self._requests[request_id]["responses"][agent] = data

    def is_complete(self, request_id: str) -> bool:
        """Check if all expected agents have responded."""
        req = self._requests.get(request_id)
        if not req:
            return False
        return req["expected"] <= set(req["responses"].keys())

    def get_responses(self, request_id: str) -> dict:
        """Get all collected responses for a request."""
        req = self._requests.get(request_id)
        if not req:
            return {}
        return dict(req["responses"])

    async def wait_for(self, request_id: str) -> dict:
        """Wait until all responses arrive or timeout. Returns collected responses."""
        try:
            async with asyncio.timeout(self._timeout):
                while not self.is_complete(request_id):
                    await asyncio.sleep(0.05)
        except TimeoutError:
            pass
        return self.get_responses(request_id)

    def cleanup(self, request_id: str):
        """Remove a completed request."""
        self._requests.pop(request_id, None)
