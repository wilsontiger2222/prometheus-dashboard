"""Price feed for Sentinel agent.

Fetches prices from Hyperliquid's info API and maintains local price history.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class PricePoint:
    symbol: str
    price: float
    volume: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "price": self.price, "volume": self.volume, "timestamp": self.timestamp}


class PriceFeed:
    """Fetches prices from Hyperliquid and maintains history."""

    def __init__(self, api_url: str = "https://api.hyperliquid.xyz", max_history: int = 100):
        self._api_url = api_url
        self._max_history = max_history
        self._history: dict[str, deque[PricePoint]] = defaultdict(lambda: deque(maxlen=max_history))

    async def fetch_price(self, symbol: str) -> PricePoint | None:
        prices = await self._fetch_meta_and_prices()
        for item in prices:
            if item.get("coin") == symbol:
                point = PricePoint(
                    symbol=symbol,
                    price=float(item["markPx"]),
                    volume=float(item.get("dayNtlVlm", 0)),
                    timestamp=time.time(),
                )
                self.record_price(point)
                return point
        return None

    async def fetch_all(self, symbols: list[str]) -> dict[str, PricePoint]:
        prices = await self._fetch_meta_and_prices()
        results = {}
        for item in prices:
            coin = item.get("coin")
            if coin in symbols:
                point = PricePoint(
                    symbol=coin,
                    price=float(item["markPx"]),
                    volume=float(item.get("dayNtlVlm", 0)),
                    timestamp=time.time(),
                )
                self.record_price(point)
                results[coin] = point
        return results

    async def _fetch_meta_and_prices(self) -> list[dict]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._api_url}/info",
                json={"type": "metaAndAssetCtxs"},
            )
            data = response.json()
            # Hyperliquid returns either a flat list of asset contexts
            # or a [meta, assetCtxs] pair. Handle both formats.
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and "coin" in data[0]:
                    return data
                if len(data) > 1 and isinstance(data[1], list):
                    return data[1]
            return data if isinstance(data, list) else []

    def record_price(self, point: PricePoint):
        self._history[point.symbol].append(point)

    def get_history(self, symbol: str) -> list[PricePoint]:
        return list(self._history[symbol])
