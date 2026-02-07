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
