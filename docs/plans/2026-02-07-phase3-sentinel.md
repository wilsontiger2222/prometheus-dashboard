# Phase 3: Sentinel Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Sentinel trading agent — a proactive market monitor that connects to Hyperliquid, tracks BTC/gold/silver/copper, executes breakout strategies, manages positions, and alerts via the message bus.

**Architecture:** Sentinel inherits from `BaseAgent` and runs a continuous price monitoring loop. A `PriceFeed` abstraction fetches prices from Hyperliquid's API. A `StrategyBase` class defines the strategy interface, with `BreakoutStrategy` as the first implementation. A `PositionManager` tracks open positions, P&L, and risk limits. All exchange interactions go through a thin `HyperliquidClient` wrapper that's mockable for testing.

**Tech Stack:** Python 3.10+, `hyperliquid-python-sdk`, `httpx` (async HTTP), asyncio, BaseAgent (from Phase 1)

---

### Task 1: Add dependencies and Sentinel config

**Files:**
- Modify: `agents/requirements.txt`
- Create: `agents/sentinel/config.json`

**Step 1: Update requirements**

Add to `agents/requirements.txt`:
```
redis>=5.0.0
aiohttp>=3.9.0
psutil>=5.9.0
httpx>=0.27.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

**Step 2: Create default config**

`agents/sentinel/config.json`:
```json
{
  "name": "sentinel",
  "mode": "paper",
  "poll_interval_seconds": 30,
  "watchlist": [
    {"symbol": "BTC", "asset_type": "crypto"},
    {"symbol": "GLD", "asset_type": "commodity", "note": "gold"},
    {"symbol": "SLV", "asset_type": "commodity", "note": "silver"},
    {"symbol": "HG", "asset_type": "commodity", "note": "copper"}
  ],
  "strategy": "breakout",
  "risk": {
    "max_position_size_pct": 2.0,
    "max_positions": 5,
    "stop_loss_pct": 3.0,
    "take_profit_pct": 6.0
  },
  "breakout": {
    "lookback_periods": 20,
    "breakout_threshold_pct": 1.5,
    "volume_confirmation": true
  },
  "hyperliquid": {
    "api_url": "https://api.hyperliquid.xyz",
    "testnet": true
  }
}
```

**Step 3: Install deps**

Run: `python -m pip install httpx`

**Step 4: Commit**

```bash
git add agents/requirements.txt agents/sentinel/config.json
git commit -m "feat(sentinel): add config and httpx dependency"
```

---

### Task 2: Build the price feed abstraction

**Files:**
- Create: `agents/sentinel/price_feed.py`
- Create: `agents/tests/test_sentinel_price_feed.py`

**Step 1: Write the failing tests**

`agents/tests/test_sentinel_price_feed.py`:
```python
"""Tests for Sentinel price feed."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.sentinel.price_feed import PriceFeed, PricePoint


def test_price_point_creation():
    p = PricePoint(symbol="BTC", price=105000.0, volume=1500.0, timestamp=1707300000.0)
    assert p.symbol == "BTC"
    assert p.price == 105000.0
    assert p.to_dict()["volume"] == 1500.0


def test_price_point_to_dict():
    p = PricePoint(symbol="GLD", price=2050.0, volume=300.0, timestamp=1707300000.0)
    d = p.to_dict()
    assert d == {"symbol": "GLD", "price": 2050.0, "volume": 300.0, "timestamp": 1707300000.0}


@pytest.mark.asyncio
async def test_fetch_price_returns_price_point():
    feed = PriceFeed(api_url="https://api.hyperliquid.xyz")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"coin": "BTC", "markPx": "105000.50", "dayNtlVlm": "1500000000"}
    ]

    with patch("agents.sentinel.price_feed.httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = mock_instance

        result = await feed.fetch_price("BTC")
        assert result is not None
        assert result.symbol == "BTC"
        assert result.price == 105000.50


@pytest.mark.asyncio
async def test_fetch_price_unknown_symbol():
    feed = PriceFeed(api_url="https://api.hyperliquid.xyz")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"coin": "BTC", "markPx": "105000.50", "dayNtlVlm": "1500000000"}
    ]

    with patch("agents.sentinel.price_feed.httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = mock_instance

        result = await feed.fetch_price("DOESNTEXIST")
        assert result is None


@pytest.mark.asyncio
async def test_fetch_all_prices():
    feed = PriceFeed(api_url="https://api.hyperliquid.xyz")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"coin": "BTC", "markPx": "105000.50", "dayNtlVlm": "1500000000"},
        {"coin": "ETH", "markPx": "3200.00", "dayNtlVlm": "500000000"},
    ]

    with patch("agents.sentinel.price_feed.httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = mock_instance

        results = await feed.fetch_all(["BTC", "ETH"])
        assert len(results) == 2
        assert results["BTC"].price == 105000.50
        assert results["ETH"].price == 3200.00


def test_price_history_tracking():
    feed = PriceFeed(api_url="https://api.hyperliquid.xyz")
    p1 = PricePoint("BTC", 100000.0, 1000.0, 1.0)
    p2 = PricePoint("BTC", 101000.0, 1100.0, 2.0)
    p3 = PricePoint("BTC", 102000.0, 1200.0, 3.0)

    feed.record_price(p1)
    feed.record_price(p2)
    feed.record_price(p3)

    history = feed.get_history("BTC")
    assert len(history) == 3
    assert history[-1].price == 102000.0


def test_price_history_max_length():
    feed = PriceFeed(api_url="https://api.hyperliquid.xyz", max_history=2)
    feed.record_price(PricePoint("BTC", 100.0, 10.0, 1.0))
    feed.record_price(PricePoint("BTC", 200.0, 20.0, 2.0))
    feed.record_price(PricePoint("BTC", 300.0, 30.0, 3.0))

    history = feed.get_history("BTC")
    assert len(history) == 2
    assert history[0].price == 200.0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_sentinel_price_feed.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

`agents/sentinel/price_feed.py`:
```python
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
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "timestamp": self.timestamp,
        }


class PriceFeed:
    """Fetches prices from Hyperliquid and maintains history."""

    def __init__(self, api_url: str = "https://api.hyperliquid.xyz", max_history: int = 100):
        self._api_url = api_url
        self._max_history = max_history
        self._history: dict[str, deque[PricePoint]] = defaultdict(lambda: deque(maxlen=max_history))

    async def fetch_price(self, symbol: str) -> PricePoint | None:
        """Fetch current price for a single symbol from Hyperliquid."""
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
        """Fetch prices for multiple symbols in one API call."""
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
        """Call Hyperliquid info API for all asset prices."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._api_url}/info",
                json={"type": "metaAndAssetCtxs"},
            )
            data = response.json()
            # Hyperliquid returns [meta, [assetCtx, ...]]
            # For simplicity, handle both list-of-dicts and nested format
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and "coin" in data[0]:
                    return data
                # Nested format: data[1] is the list of asset contexts
                if len(data) > 1 and isinstance(data[1], list):
                    return data[1]
            return data if isinstance(data, list) else []

    def record_price(self, point: PricePoint):
        """Add a price point to history."""
        self._history[point.symbol].append(point)

    def get_history(self, symbol: str) -> list[PricePoint]:
        """Get price history for a symbol."""
        return list(self._history[symbol])
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_sentinel_price_feed.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add agents/sentinel/price_feed.py agents/tests/test_sentinel_price_feed.py
git commit -m "feat(sentinel): add price feed with Hyperliquid API and history tracking"
```

---

### Task 3: Build the strategy base class and breakout strategy

**Files:**
- Create: `agents/sentinel/strategies/strategy_base.py`
- Create: `agents/sentinel/strategies/breakout.py`
- Create: `agents/sentinel/strategies/__init__.py`
- Create: `agents/tests/test_sentinel_strategies.py`

**Step 1: Write the failing tests**

`agents/tests/test_sentinel_strategies.py`:
```python
"""Tests for Sentinel trading strategies."""

import pytest
from agents.sentinel.price_feed import PricePoint
from agents.sentinel.strategies.strategy_base import Signal, SignalType
from agents.sentinel.strategies.breakout import BreakoutStrategy


# --- Signal ---

def test_signal_creation():
    s = Signal(signal_type=SignalType.BUY, symbol="BTC", price=105000.0, reason="Breakout above resistance")
    assert s.signal_type == SignalType.BUY
    assert s.symbol == "BTC"


def test_signal_to_dict():
    s = Signal(signal_type=SignalType.SELL, symbol="GLD", price=2050.0, reason="Below support")
    d = s.to_dict()
    assert d["signal_type"] == "SELL"
    assert d["symbol"] == "GLD"


# --- BreakoutStrategy ---

def make_history(prices: list[float], symbol: str = "BTC") -> list[PricePoint]:
    return [PricePoint(symbol=symbol, price=p, volume=1000.0, timestamp=float(i)) for i, p in enumerate(prices)]


def test_breakout_needs_enough_history():
    strategy = BreakoutStrategy(lookback=5, threshold_pct=1.5)
    history = make_history([100, 101, 102])  # Only 3, need 5
    signal = strategy.evaluate("BTC", history)
    assert signal is None


def test_breakout_no_signal_in_range():
    strategy = BreakoutStrategy(lookback=5, threshold_pct=2.0)
    # Prices stay flat — no breakout
    history = make_history([100, 100, 100, 100, 100, 100.5])
    signal = strategy.evaluate("BTC", history)
    assert signal is None


def test_breakout_buy_signal():
    strategy = BreakoutStrategy(lookback=5, threshold_pct=1.5)
    # Last price breaks above lookback high by > 1.5%
    history = make_history([100, 101, 99, 100, 101, 103])
    signal = strategy.evaluate("BTC", history)
    assert signal is not None
    assert signal.signal_type == SignalType.BUY


def test_breakout_sell_signal():
    strategy = BreakoutStrategy(lookback=5, threshold_pct=1.5)
    # Last price breaks below lookback low by > 1.5%
    history = make_history([100, 101, 99, 100, 101, 96])
    signal = strategy.evaluate("BTC", history)
    assert signal is not None
    assert signal.signal_type == SignalType.SELL


def test_breakout_uses_configurable_params():
    strategy = BreakoutStrategy(lookback=3, threshold_pct=5.0)
    # With 5% threshold, a small move doesn't trigger
    history = make_history([100, 101, 100, 103])
    signal = strategy.evaluate("BTC", history)
    assert signal is None

    # But a big move does
    history2 = make_history([100, 101, 100, 106])
    signal2 = strategy.evaluate("BTC", history2)
    assert signal2 is not None
    assert signal2.signal_type == SignalType.BUY
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_sentinel_strategies.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

`agents/sentinel/strategies/__init__.py`:
```python
"""Trading strategies for Sentinel agent."""
```

`agents/sentinel/strategies/strategy_base.py`:
```python
"""Base class and types for trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agents.sentinel.price_feed import PricePoint


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    signal_type: SignalType
    symbol: str
    price: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "price": self.price,
            "reason": self.reason,
        }


class StrategyBase(ABC):
    """Abstract base for all trading strategies."""

    @abstractmethod
    def evaluate(self, symbol: str, history: list[PricePoint]) -> Signal | None:
        """Evaluate price history and return a signal or None."""
        ...
```

`agents/sentinel/strategies/breakout.py`:
```python
"""Breakout trading strategy.

Generates BUY when price breaks above the lookback high by threshold_pct.
Generates SELL when price breaks below the lookback low by threshold_pct.
"""

from agents.sentinel.price_feed import PricePoint
from agents.sentinel.strategies.strategy_base import StrategyBase, Signal, SignalType


class BreakoutStrategy(StrategyBase):
    def __init__(self, lookback: int = 20, threshold_pct: float = 1.5):
        self._lookback = lookback
        self._threshold_pct = threshold_pct

    def evaluate(self, symbol: str, history: list[PricePoint]) -> Signal | None:
        if len(history) < self._lookback + 1:
            return None

        lookback_window = history[-(self._lookback + 1):-1]
        current = history[-1]

        high = max(p.price for p in lookback_window)
        low = min(p.price for p in lookback_window)

        breakout_above = ((current.price - high) / high) * 100
        breakout_below = ((low - current.price) / low) * 100

        if breakout_above >= self._threshold_pct:
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current.price,
                reason=f"Breakout above {high:.2f} by {breakout_above:.1f}%",
            )
        elif breakout_below >= self._threshold_pct:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current.price,
                reason=f"Breakdown below {low:.2f} by {breakout_below:.1f}%",
            )

        return None
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_sentinel_strategies.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add agents/sentinel/strategies/ agents/tests/test_sentinel_strategies.py
git commit -m "feat(sentinel): add strategy base and breakout strategy"
```

---

### Task 4: Build the position manager

**Files:**
- Create: `agents/sentinel/positions.py`
- Create: `agents/tests/test_sentinel_positions.py`

**Step 1: Write the failing tests**

`agents/tests/test_sentinel_positions.py`:
```python
"""Tests for Sentinel position manager."""

import pytest
from agents.sentinel.positions import PositionManager, Position


def test_open_position():
    pm = PositionManager(max_positions=5)
    pos = pm.open_position("BTC", "buy", 105000.0, 0.1)
    assert pos.symbol == "BTC"
    assert pos.side == "buy"
    assert pos.entry_price == 105000.0
    assert pos.size == 0.1
    assert pos.is_open is True


def test_close_position():
    pm = PositionManager(max_positions=5)
    pos = pm.open_position("BTC", "buy", 105000.0, 0.1)
    result = pm.close_position(pos.id, exit_price=107000.0)
    assert result.is_open is False
    assert result.exit_price == 107000.0
    assert result.pnl == pytest.approx(200.0)  # (107000 - 105000) * 0.1


def test_pnl_short():
    pm = PositionManager(max_positions=5)
    pos = pm.open_position("BTC", "sell", 105000.0, 0.1)
    result = pm.close_position(pos.id, exit_price=103000.0)
    assert result.pnl == pytest.approx(200.0)  # (105000 - 103000) * 0.1


def test_max_positions_enforced():
    pm = PositionManager(max_positions=2)
    pm.open_position("BTC", "buy", 100.0, 1.0)
    pm.open_position("ETH", "buy", 200.0, 1.0)
    pos3 = pm.open_position("GLD", "buy", 300.0, 1.0)
    assert pos3 is None


def test_get_open_positions():
    pm = PositionManager(max_positions=5)
    pm.open_position("BTC", "buy", 100.0, 1.0)
    pm.open_position("ETH", "buy", 200.0, 1.0)
    open_pos = pm.get_open_positions()
    assert len(open_pos) == 2


def test_check_stop_loss():
    pm = PositionManager(max_positions=5)
    pos = pm.open_position("BTC", "buy", 100.0, 1.0)
    # Price dropped 5% — should trigger 3% stop loss
    triggered = pm.check_stop_loss(pos.id, current_price=96.0, stop_pct=3.0)
    assert triggered is True


def test_check_stop_loss_not_triggered():
    pm = PositionManager(max_positions=5)
    pos = pm.open_position("BTC", "buy", 100.0, 1.0)
    # Price only dropped 1% — should NOT trigger 3% stop loss
    triggered = pm.check_stop_loss(pos.id, current_price=99.0, stop_pct=3.0)
    assert triggered is False


def test_check_take_profit():
    pm = PositionManager(max_positions=5)
    pos = pm.open_position("BTC", "buy", 100.0, 1.0)
    # Price up 7% — should trigger 6% take profit
    triggered = pm.check_take_profit(pos.id, current_price=107.0, profit_pct=6.0)
    assert triggered is True


def test_portfolio_summary():
    pm = PositionManager(max_positions=5)
    pm.open_position("BTC", "buy", 100.0, 1.0)
    pm.open_position("ETH", "sell", 200.0, 0.5)
    summary = pm.get_summary({"BTC": 105.0, "ETH": 190.0})
    assert summary["open_positions"] == 2
    assert summary["unrealized_pnl"] == pytest.approx(10.0)  # BTC: +5, ETH: +5
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_sentinel_positions.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

`agents/sentinel/positions.py`:
```python
"""Position management for Sentinel agent."""

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Position:
    id: str
    symbol: str
    side: str  # "buy" or "sell"
    entry_price: float
    size: float
    is_open: bool = True
    exit_price: float | None = None
    pnl: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "size": self.size,
            "is_open": self.is_open,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
        }


class PositionManager:
    """Tracks open and closed positions with P&L."""

    def __init__(self, max_positions: int = 5):
        self._max_positions = max_positions
        self._positions: dict[str, Position] = {}

    def open_position(self, symbol: str, side: str, entry_price: float, size: float) -> Position | None:
        open_count = sum(1 for p in self._positions.values() if p.is_open)
        if open_count >= self._max_positions:
            return None

        pos = Position(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
        )
        self._positions[pos.id] = pos
        return pos

    def close_position(self, position_id: str, exit_price: float) -> Position:
        pos = self._positions[position_id]
        pos.is_open = False
        pos.exit_price = exit_price
        if pos.side == "buy":
            pos.pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pos.pnl = (pos.entry_price - exit_price) * pos.size
        return pos

    def get_open_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.is_open]

    def check_stop_loss(self, position_id: str, current_price: float, stop_pct: float) -> bool:
        pos = self._positions[position_id]
        if pos.side == "buy":
            loss_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
        else:
            loss_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
        return loss_pct >= stop_pct

    def check_take_profit(self, position_id: str, current_price: float, profit_pct: float) -> bool:
        pos = self._positions[position_id]
        if pos.side == "buy":
            gain_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
        else:
            gain_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
        return gain_pct >= profit_pct

    def get_summary(self, current_prices: dict[str, float]) -> dict[str, Any]:
        open_positions = self.get_open_positions()
        unrealized = 0.0
        for pos in open_positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            if pos.side == "buy":
                unrealized += (price - pos.entry_price) * pos.size
            else:
                unrealized += (pos.entry_price - price) * pos.size

        return {
            "open_positions": len(open_positions),
            "unrealized_pnl": unrealized,
            "positions": [p.to_dict() for p in open_positions],
        }
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_sentinel_positions.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add agents/sentinel/positions.py agents/tests/test_sentinel_positions.py
git commit -m "feat(sentinel): add position manager with P&L and risk checks"
```

---

### Task 5: Build the Sentinel agent class

**Files:**
- Create: `agents/sentinel/agent.py`
- Create: `agents/tests/test_sentinel_agent.py`

**Step 1: Write the failing tests**

`agents/tests/test_sentinel_agent.py`:
```python
"""Tests for the Sentinel agent."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from agents.sentinel.agent import SentinelAgent
from agents.sentinel.price_feed import PricePoint
from agents.sentinel.strategies.strategy_base import Signal, SignalType


@pytest.fixture
def agent(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "name": "sentinel",
        "mode": "paper",
        "poll_interval_seconds": 1,
        "watchlist": [
            {"symbol": "BTC", "asset_type": "crypto"},
            {"symbol": "GLD", "asset_type": "commodity"},
        ],
        "strategy": "breakout",
        "risk": {
            "max_position_size_pct": 2.0,
            "max_positions": 5,
            "stop_loss_pct": 3.0,
            "take_profit_pct": 6.0,
        },
        "breakout": {
            "lookback_periods": 20,
            "breakout_threshold_pct": 1.5,
        },
        "hyperliquid": {
            "api_url": "https://api.hyperliquid.xyz",
            "testnet": True,
        },
    }))
    return SentinelAgent(config_path=str(config))


def test_sentinel_name(agent):
    assert agent.name == "sentinel"


def test_sentinel_loads_watchlist(agent):
    assert len(agent._watchlist) == 2
    assert agent._watchlist[0]["symbol"] == "BTC"


def test_sentinel_creates_strategy(agent):
    assert agent._strategy is not None


@pytest.mark.asyncio
async def test_poll_cycle_fetches_prices(agent):
    agent.bus = AsyncMock()
    agent.bus.publish = AsyncMock()

    mock_prices = {
        "BTC": PricePoint("BTC", 105000.0, 1500.0, 1.0),
        "GLD": PricePoint("GLD", 2050.0, 300.0, 2.0),
    }

    with patch.object(agent._feed, "fetch_all", new_callable=AsyncMock, return_value=mock_prices):
        with patch.object(agent._strategy, "evaluate", return_value=None):
            await agent.poll_cycle()

    # Should publish prices to sentinel/prices
    channels = [call[0][0] for call in agent.bus.publish.call_args_list]
    assert "sentinel/prices" in channels


@pytest.mark.asyncio
async def test_poll_cycle_publishes_signal(agent):
    agent.bus = AsyncMock()
    agent.bus.publish = AsyncMock()

    mock_prices = {
        "BTC": PricePoint("BTC", 105000.0, 1500.0, 1.0),
        "GLD": PricePoint("GLD", 2050.0, 300.0, 2.0),
    }
    buy_signal = Signal(SignalType.BUY, "BTC", 105000.0, "Breakout above 103000")

    with patch.object(agent._feed, "fetch_all", new_callable=AsyncMock, return_value=mock_prices):
        with patch.object(agent._strategy, "evaluate", side_effect=[buy_signal, None]):
            await agent.poll_cycle()

    channels = [call[0][0] for call in agent.bus.publish.call_args_list]
    assert "sentinel/alerts" in channels


@pytest.mark.asyncio
async def test_on_dispatch_status(agent):
    agent.bus = AsyncMock()
    agent.bus.publish = AsyncMock()

    message = {"from": "prometheus", "payload": {"task": "status"}}
    await agent.on_dispatch(message)
    agent.bus.publish.assert_called()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest agents/tests/test_sentinel_agent.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

`agents/sentinel/agent.py`:
```python
"""Sentinel agent — proactive market monitoring and trading."""

import asyncio
from agents.shared.base_agent import BaseAgent
from agents.sentinel.price_feed import PriceFeed
from agents.sentinel.positions import PositionManager
from agents.sentinel.strategies.strategy_base import Signal, SignalType
from agents.sentinel.strategies.breakout import BreakoutStrategy


class SentinelAgent(BaseAgent):
    """Monitors markets and executes trading strategies.

    Runs a continuous poll loop that fetches prices, evaluates strategies,
    manages positions, and publishes signals to the bus.
    """

    def __init__(self, **kwargs):
        super().__init__(name="sentinel", **kwargs)
        self._poll_interval = self.config.get("poll_interval_seconds", 30)
        self._watchlist = self.config.get("watchlist", [])
        self._mode = self.config.get("mode", "paper")
        self._risk = self.config.get("risk", {})

        # Price feed
        hl_config = self.config.get("hyperliquid", {})
        self._feed = PriceFeed(api_url=hl_config.get("api_url", "https://api.hyperliquid.xyz"))

        # Strategy
        breakout_config = self.config.get("breakout", {})
        self._strategy = BreakoutStrategy(
            lookback=breakout_config.get("lookback_periods", 20),
            threshold_pct=breakout_config.get("breakout_threshold_pct", 1.5),
        )

        # Positions
        self._positions = PositionManager(
            max_positions=self._risk.get("max_positions", 5),
        )

    async def run(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self.poll_cycle()
            except Exception as e:
                self.logger.error(f"Poll cycle failed: {e}")
            await asyncio.sleep(self._poll_interval)

    async def on_dispatch(self, message: dict):
        """Handle dispatch commands from Prometheus."""
        payload = message.get("payload", {})
        task = payload.get("task", "")

        if task == "status":
            prices = {}
            for pos in self._positions.get_open_positions():
                history = self._feed.get_history(pos.symbol)
                if history:
                    prices[pos.symbol] = history[-1].price
            summary = self._positions.get_summary(prices)
            await self.bus.publish("sentinel/status", summary, sender="sentinel")
        elif task == "poll":
            await self.poll_cycle()

    async def poll_cycle(self):
        """Single poll iteration: fetch prices, evaluate, manage positions."""
        symbols = [w["symbol"] for w in self._watchlist]
        prices = await self._feed.fetch_all(symbols)

        if not prices:
            return

        # Publish current prices
        price_data = {s: p.to_dict() for s, p in prices.items()}
        await self.bus.publish("sentinel/prices", price_data, sender="sentinel")

        # Evaluate strategy for each symbol
        for symbol in symbols:
            history = self._feed.get_history(symbol)
            if not history:
                continue

            signal = self._strategy.evaluate(symbol, history)
            if signal:
                await self._handle_signal(signal)

        # Check stop-loss / take-profit on open positions
        await self._check_positions(prices)

    async def _handle_signal(self, signal: Signal):
        """Process a trading signal."""
        self.logger.info(f"Signal: {signal.signal_type.value} {signal.symbol} @ {signal.price} — {signal.reason}")

        await self.bus.publish(
            "sentinel/alerts",
            signal.to_dict(),
            sender="sentinel",
        )

        if self._mode == "paper":
            if signal.signal_type == SignalType.BUY:
                size = self._risk.get("max_position_size_pct", 2.0) / 100
                pos = self._positions.open_position(signal.symbol, "buy", signal.price, size)
                if pos:
                    self.logger.info(f"Paper BUY: {signal.symbol} @ {signal.price}, size={size}")
                    await self.bus.publish("sentinel/trades", pos.to_dict(), sender="sentinel")
            elif signal.signal_type == SignalType.SELL:
                # Close any open buy positions for this symbol
                for pos in self._positions.get_open_positions():
                    if pos.symbol == signal.symbol and pos.side == "buy":
                        closed = self._positions.close_position(pos.id, signal.price)
                        self.logger.info(f"Paper SELL: {signal.symbol} @ {signal.price}, PnL={closed.pnl:.2f}")
                        await self.bus.publish("sentinel/trades", closed.to_dict(), sender="sentinel")

    async def _check_positions(self, prices: dict):
        """Check stop-loss and take-profit on open positions."""
        stop_pct = self._risk.get("stop_loss_pct", 3.0)
        profit_pct = self._risk.get("take_profit_pct", 6.0)

        for pos in self._positions.get_open_positions():
            if pos.symbol not in prices:
                continue
            current_price = prices[pos.symbol].price

            if self._positions.check_stop_loss(pos.id, current_price, stop_pct):
                closed = self._positions.close_position(pos.id, current_price)
                self.logger.warning(f"STOP LOSS: {pos.symbol} @ {current_price}, PnL={closed.pnl:.2f}")
                await self.bus.publish("sentinel/trades", closed.to_dict(), sender="sentinel")
                await self.bus.publish("sentinel/alerts", {"type": "stop_loss", **closed.to_dict()}, sender="sentinel")

            elif self._positions.check_take_profit(pos.id, current_price, profit_pct):
                closed = self._positions.close_position(pos.id, current_price)
                self.logger.info(f"TAKE PROFIT: {pos.symbol} @ {current_price}, PnL={closed.pnl:.2f}")
                await self.bus.publish("sentinel/trades", closed.to_dict(), sender="sentinel")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest agents/tests/test_sentinel_agent.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add agents/sentinel/agent.py agents/tests/test_sentinel_agent.py
git commit -m "feat(sentinel): add SentinelAgent with poll loop, signals, and position management"
```

---

### Task 6: Add `__main__.py` entry point

**Files:**
- Create: `agents/sentinel/__main__.py`

**Step 1: Write the entry point**

`agents/sentinel/__main__.py`:
```python
"""Run the Sentinel agent: python -m agents.sentinel"""

import asyncio
import sys
from pathlib import Path

from agents.sentinel.agent import SentinelAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = SentinelAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nSentinel shutting down...")


if __name__ == "__main__":
    main()
```

**Step 2: Verify it imports**

Run: `python -c "from agents.sentinel.agent import SentinelAgent; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add agents/sentinel/__main__.py
git commit -m "feat(sentinel): add __main__.py entry point"
```

---

### Task 7: Run all tests and push

**Step 1: Run full test suite**

Run: `python -m pytest agents/tests/ -v`
Expected: All Phase 1 + Phase 2 + Phase 3 tests pass

**Step 2: Push to GitHub**

```bash
git push origin main
```

---

## Summary

After Phase 3 is complete, you will have:

| Component | File | Purpose |
|-----------|------|---------|
| **Price Feed** | `agents/sentinel/price_feed.py` | Hyperliquid API, price history |
| **Strategies** | `agents/sentinel/strategies/` | StrategyBase + BreakoutStrategy |
| **Positions** | `agents/sentinel/positions.py` | Position tracking, P&L, risk |
| **Agent** | `agents/sentinel/agent.py` | Poll loop, signal handling, bus |
| **Config** | `agents/sentinel/config.json` | Watchlist, risk params, strategy |
| **Entry Point** | `agents/sentinel/__main__.py` | `python -m agents.sentinel` |
| **Tests** | `agents/tests/test_sentinel_*.py` | ~30 new tests |

**To run on server:**
```bash
cd ~/prometheus-dashboard
pip install -r agents/requirements.txt
python -m agents.sentinel
```

**Next:** Phase 4 — Scout (Research Agent)
