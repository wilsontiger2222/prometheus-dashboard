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
        return {"signal_type": self.signal_type.value, "symbol": self.symbol, "price": self.price, "reason": self.reason}


class StrategyBase(ABC):
    @abstractmethod
    def evaluate(self, symbol: str, history: list[PricePoint]) -> Signal | None:
        ...
