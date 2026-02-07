"""Position management for Sentinel agent."""

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class Position:
    id: str
    symbol: str
    side: str
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
    def __init__(self, max_positions: int = 5):
        self._max_positions = max_positions
        self._positions: dict[str, Position] = {}

    def open_position(
        self, symbol: str, side: str, entry_price: float, size: float
    ) -> Position | None:
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

    def check_stop_loss(
        self, position_id: str, current_price: float, stop_pct: float
    ) -> bool:
        pos = self._positions[position_id]
        if pos.side == "buy":
            loss_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
        else:
            loss_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
        return loss_pct >= stop_pct

    def check_take_profit(
        self, position_id: str, current_price: float, profit_pct: float
    ) -> bool:
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
