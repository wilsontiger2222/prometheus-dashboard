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
