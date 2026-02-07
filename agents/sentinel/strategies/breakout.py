"""Breakout trading strategy."""

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

        pct_above_high = ((current.price - high) / high) * 100
        pct_below_low = ((low - current.price) / low) * 100
        pct_from_low = ((current.price - low) / low) * 100
        pct_from_high = ((high - current.price) / high) * 100

        if pct_above_high > 0 and pct_from_low >= self._threshold_pct:
            return Signal(signal_type=SignalType.BUY, symbol=symbol, price=current.price, reason=f"Breakout above {high:.2f} by {pct_from_low:.1f}%")
        elif pct_below_low > 0 and pct_from_high >= self._threshold_pct:
            return Signal(signal_type=SignalType.SELL, symbol=symbol, price=current.price, reason=f"Breakdown below {low:.2f} by {pct_from_high:.1f}%")

        return None
