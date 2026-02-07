"""Tests for Sentinel trading strategies."""

import pytest
from agents.sentinel.price_feed import PricePoint
from agents.sentinel.strategies.strategy_base import Signal, SignalType
from agents.sentinel.strategies.breakout import BreakoutStrategy


def test_signal_creation():
    s = Signal(signal_type=SignalType.BUY, symbol="BTC", price=105000.0, reason="Breakout above resistance")
    assert s.signal_type == SignalType.BUY
    assert s.symbol == "BTC"


def test_signal_to_dict():
    s = Signal(signal_type=SignalType.SELL, symbol="GLD", price=2050.0, reason="Below support")
    d = s.to_dict()
    assert d["signal_type"] == "SELL"
    assert d["symbol"] == "GLD"


def make_history(prices: list[float], symbol: str = "BTC") -> list[PricePoint]:
    return [PricePoint(symbol=symbol, price=p, volume=1000.0, timestamp=float(i)) for i, p in enumerate(prices)]


def test_breakout_needs_enough_history():
    strategy = BreakoutStrategy(lookback=5, threshold_pct=1.5)
    history = make_history([100, 101, 102])
    signal = strategy.evaluate("BTC", history)
    assert signal is None


def test_breakout_no_signal_in_range():
    strategy = BreakoutStrategy(lookback=5, threshold_pct=2.0)
    history = make_history([100, 100, 100, 100, 100, 100.5])
    signal = strategy.evaluate("BTC", history)
    assert signal is None


def test_breakout_buy_signal():
    strategy = BreakoutStrategy(lookback=5, threshold_pct=1.5)
    history = make_history([100, 101, 99, 100, 101, 103])
    signal = strategy.evaluate("BTC", history)
    assert signal is not None
    assert signal.signal_type == SignalType.BUY


def test_breakout_sell_signal():
    strategy = BreakoutStrategy(lookback=5, threshold_pct=1.5)
    history = make_history([100, 101, 99, 100, 101, 96])
    signal = strategy.evaluate("BTC", history)
    assert signal is not None
    assert signal.signal_type == SignalType.SELL


def test_breakout_uses_configurable_params():
    strategy = BreakoutStrategy(lookback=3, threshold_pct=5.0)
    history = make_history([100, 101, 100, 103])
    signal = strategy.evaluate("BTC", history)
    assert signal is None

    history2 = make_history([100, 101, 100, 106])
    signal2 = strategy.evaluate("BTC", history2)
    assert signal2 is not None
    assert signal2.signal_type == SignalType.BUY
