"""Sentinel agent — proactive market monitoring and trading."""

import asyncio
from agents.shared.base_agent import BaseAgent
from agents.sentinel.price_feed import PriceFeed
from agents.sentinel.positions import PositionManager
from agents.sentinel.strategies.strategy_base import Signal, SignalType
from agents.sentinel.strategies.breakout import BreakoutStrategy


class SentinelAgent(BaseAgent):
    """Monitors markets and executes trading strategies."""

    def __init__(self, **kwargs):
        super().__init__(name="sentinel", **kwargs)
        self._poll_interval = self.config.get("poll_interval_seconds", 30)
        self._watchlist = self.config.get("watchlist", [])
        self._mode = self.config.get("mode", "paper")
        self._risk = self.config.get("risk", {})

        hl_config = self.config.get("hyperliquid", {})
        self._feed = PriceFeed(api_url=hl_config.get("api_url", "https://api.hyperliquid.xyz"))

        breakout_config = self.config.get("breakout", {})
        self._strategy = BreakoutStrategy(
            lookback=breakout_config.get("lookback_periods", 20),
            threshold_pct=breakout_config.get("breakout_threshold_pct", 1.5),
        )

        self._positions = PositionManager(
            max_positions=self._risk.get("max_positions", 5),
        )

    async def run(self):
        while self._running:
            try:
                await self.poll_cycle()
            except Exception as e:
                self.logger.error(f"Poll cycle failed: {e}")
            await asyncio.sleep(self._poll_interval)

    async def on_dispatch(self, message: dict):
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
        symbols = [w["symbol"] for w in self._watchlist]
        prices = await self._feed.fetch_all(symbols)

        if not prices:
            return

        # Record fetched prices into the feed history so strategies
        # can access them even when fetch_all is mocked in tests.
        for symbol, point in prices.items():
            self._feed.record_price(point)

        price_data = {s: p.to_dict() for s, p in prices.items()}
        await self.bus.publish("sentinel/prices", price_data, sender="sentinel")

        for symbol in symbols:
            history = self._feed.get_history(symbol)
            if not history:
                continue
            signal = self._strategy.evaluate(symbol, history)
            if signal:
                await self._handle_signal(signal)

        await self._check_positions(prices)

    async def _handle_signal(self, signal: Signal):
        self.logger.info(f"Signal: {signal.signal_type.value} {signal.symbol} @ {signal.price} — {signal.reason}")
        await self.bus.publish("sentinel/alerts", signal.to_dict(), sender="sentinel")

        if self._mode == "paper":
            if signal.signal_type == SignalType.BUY:
                size = self._risk.get("max_position_size_pct", 2.0) / 100
                pos = self._positions.open_position(signal.symbol, "buy", signal.price, size)
                if pos:
                    self.logger.info(f"Paper BUY: {signal.symbol} @ {signal.price}, size={size}")
                    await self.bus.publish("sentinel/trades", pos.to_dict(), sender="sentinel")
            elif signal.signal_type == SignalType.SELL:
                for pos in self._positions.get_open_positions():
                    if pos.symbol == signal.symbol and pos.side == "buy":
                        closed = self._positions.close_position(pos.id, signal.price)
                        self.logger.info(f"Paper SELL: {signal.symbol} @ {signal.price}, PnL={closed.pnl:.2f}")
                        await self.bus.publish("sentinel/trades", closed.to_dict(), sender="sentinel")

    async def _check_positions(self, prices: dict):
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
