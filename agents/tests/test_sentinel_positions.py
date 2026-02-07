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
    assert result.pnl == pytest.approx(200.0)


def test_pnl_short():
    pm = PositionManager(max_positions=5)
    pos = pm.open_position("BTC", "sell", 105000.0, 0.1)
    result = pm.close_position(pos.id, exit_price=103000.0)
    assert result.pnl == pytest.approx(200.0)


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
    triggered = pm.check_stop_loss(pos.id, current_price=96.0, stop_pct=3.0)
    assert triggered is True


def test_check_stop_loss_not_triggered():
    pm = PositionManager(max_positions=5)
    pos = pm.open_position("BTC", "buy", 100.0, 1.0)
    triggered = pm.check_stop_loss(pos.id, current_price=99.0, stop_pct=3.0)
    assert triggered is False


def test_check_take_profit():
    pm = PositionManager(max_positions=5)
    pos = pm.open_position("BTC", "buy", 100.0, 1.0)
    triggered = pm.check_take_profit(pos.id, current_price=107.0, profit_pct=6.0)
    assert triggered is True


def test_portfolio_summary():
    pm = PositionManager(max_positions=5)
    pm.open_position("BTC", "buy", 100.0, 1.0)
    pm.open_position("ETH", "sell", 200.0, 0.5)
    summary = pm.get_summary({"BTC": 105.0, "ETH": 190.0})
    assert summary["open_positions"] == 2
    assert summary["unrealized_pnl"] == pytest.approx(10.0)
