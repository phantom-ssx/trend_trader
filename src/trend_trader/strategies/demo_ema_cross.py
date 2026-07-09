from __future__ import annotations

from decimal import Decimal
from typing import Literal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

SignalSide = Literal["BUY", "SELL"]


class DemoEmaCrossSignal:
    """Shared EMA cross signal logic used by backtests and Nautilus strategy wrappers."""

    def __init__(self, fast_period: int = 10, slow_period: int = 30) -> None:
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("EMA periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast: float | None = None
        self.slow: float | None = None
        self.last_signal = 0

    def on_price(self, close: float) -> SignalSide | None:
        self.fast = self._ema(self.fast, close, self.fast_period)
        self.slow = self._ema(self.slow, close, self.slow_period)

        signal = 1 if self.fast > self.slow else -1
        if signal == self.last_signal:
            return None

        self.last_signal = signal
        return "BUY" if signal > 0 else "SELL"

    @staticmethod
    def _ema(previous: float | None, value: float, period: int) -> float:
        if previous is None:
            return value
        alpha = 2.0 / (period + 1.0)
        return alpha * value + (1.0 - alpha) * previous


class DemoEmaCrossConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal = Decimal("0.001")
    fast_period: int = 10
    slow_period: int = 30
    size_precision: int = 3


class DemoEmaCrossStrategy(Strategy):
    """Minimal EMA cross strategy for NautilusTrader wiring tests."""

    def __init__(self, config: DemoEmaCrossConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.trade_size = config.trade_size
        self.fast_period = config.fast_period
        self.slow_period = config.slow_period
        self.size_precision = config.size_precision
        self.signal = DemoEmaCrossSignal(
            fast_period=config.fast_period,
            slow_period=config.slow_period,
        )

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar) -> None:
        side_text = self.signal.on_price(float(bar.close))
        if side_text is None:
            return

        side = OrderSide.BUY if side_text == "BUY" else OrderSide.SELL
        quantity = self._target_order_quantity(side)
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=quantity,
        )
        self.submit_order(order)
        self.log.info(f"Submitted {side_text} order for {quantity} {self.instrument_id}")

    def on_event(self, event: Event) -> None:
        self.log.debug(str(event))

    def _target_order_quantity(self, side: OrderSide) -> Quantity:
        signed_position = self.portfolio.net_position(self.instrument_id)
        target_position = self.trade_size if side == OrderSide.BUY else -self.trade_size
        delta = target_position - Decimal(str(signed_position))
        return Quantity.from_str(f"{abs(delta):.{self.size_precision}f}")
