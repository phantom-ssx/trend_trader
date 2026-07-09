from __future__ import annotations

from collections import deque
from decimal import ROUND_DOWN, Decimal
from typing import Literal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Currency, Quantity
from nautilus_trader.trading.strategy import Strategy

SignalSide = Literal["BUY", "SELL"]


class MaSpreadAtrSignal:
    """MA spread threshold cross filtered by ATR percentage."""

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 20,
        spread_threshold: float = 0.0035,
        atr_period: int = 14,
        atr_pct_min: float = 0.005,
    ) -> None:
        if fast_period <= 0 or slow_period <= 0 or atr_period <= 0:
            raise ValueError("periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        if spread_threshold < 0 or atr_pct_min < 0:
            raise ValueError("thresholds must not be negative")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.spread_threshold = spread_threshold
        self.atr_period = atr_period
        self.atr_pct_min = atr_pct_min

        self.fast_window: deque[float] = deque()
        self.slow_window: deque[float] = deque()
        self.fast_sum = 0.0
        self.slow_sum = 0.0
        self.previous_close: float | None = None
        self.atr: float | None = None
        self.atr_count = 0
        self.previous_spread_pct: float | None = None

    def on_bar(self, high: float, low: float, close: float) -> SignalSide | None:
        self.fast_sum = self._push(self.fast_window, self.fast_sum, close, self.fast_period)
        self.slow_sum = self._push(self.slow_window, self.slow_sum, close, self.slow_period)
        self._update_atr(high=high, low=low, close=close)

        if len(self.slow_window) < self.slow_period:
            return None

        fast = self.fast_sum / self.fast_period
        slow = self.slow_sum / self.slow_period
        if slow == 0 or close == 0:
            return None

        spread_pct = (fast - slow) / slow
        previous_spread_pct = self.previous_spread_pct
        self.previous_spread_pct = spread_pct

        if previous_spread_pct is None or self.atr is None or self.atr_count < self.atr_period:
            return None
        if self.atr / close < self.atr_pct_min:
            return None
        if previous_spread_pct <= self.spread_threshold < spread_pct:
            return "BUY"
        if previous_spread_pct >= -self.spread_threshold > spread_pct:
            return "SELL"
        return None

    def _update_atr(self, *, high: float, low: float, close: float) -> None:
        if self.previous_close is None:
            true_range = high - low
        else:
            true_range = max(
                high - low,
                abs(high - self.previous_close),
                abs(low - self.previous_close),
            )

        if self.atr is None:
            self.atr = true_range
        else:
            alpha = 1.0 / self.atr_period
            self.atr = alpha * true_range + (1.0 - alpha) * self.atr
        self.atr_count += 1
        self.previous_close = close

    @staticmethod
    def _push(window: deque[float], running_sum: float, value: float, period: int) -> float:
        window.append(value)
        running_sum += value
        if len(window) > period:
            running_sum -= window.popleft()
        return running_sum


class MaSpreadAtrConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    settlement_currency: Currency
    trade_size: Decimal = Decimal("0.001")
    sizing: str = "all-in"
    leverage: Decimal = Decimal("1")
    fast_period: int = 5
    slow_period: int = 20
    spread_threshold: float = 0.0035
    atr_period: int = 14
    atr_pct_min: float = 0.005
    size_precision: int = 6


class MaSpreadAtrStrategy(Strategy):
    """NautilusTrader wrapper for the best-performing MA spread + ATR filter."""

    def __init__(self, config: MaSpreadAtrConfig) -> None:
        super().__init__(config)
        if config.sizing not in {"fixed", "all-in"}:
            raise ValueError("sizing must be either 'fixed' or 'all-in'")

        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.settlement_currency = config.settlement_currency
        self.trade_size = config.trade_size
        self.sizing = config.sizing
        self.leverage = config.leverage
        self.size_precision = config.size_precision
        self.signal = MaSpreadAtrSignal(
            fast_period=config.fast_period,
            slow_period=config.slow_period,
            spread_threshold=config.spread_threshold,
            atr_period=config.atr_period,
            atr_pct_min=config.atr_pct_min,
        )

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)

    def on_stop(self) -> None:
        self.unsubscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar) -> None:
        side_text = self.signal.on_bar(
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
        )
        if side_text is None:
            return

        side = OrderSide.BUY if side_text == "BUY" else OrderSide.SELL
        quantity = self._target_order_quantity(side=side, price=Decimal(str(bar.close)))
        if quantity is None:
            return

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=quantity,
        )
        self.submit_order(order)
        self.log.info(f"Submitted {side_text} order for {quantity} {self.instrument_id}")

    def on_event(self, event: Event) -> None:
        self.log.debug(str(event))

    def _target_order_quantity(self, *, side: OrderSide, price: Decimal) -> Quantity | None:
        signed_position = Decimal(str(self.portfolio.net_position(self.instrument_id)))
        target_position = self._target_position(side=side, price=price)
        delta = target_position - signed_position
        quantity = self._round_quantity(abs(delta))
        if quantity <= 0:
            return None
        return Quantity.from_str(f"{quantity:.{self.size_precision}f}")

    def _target_position(self, *, side: OrderSide, price: Decimal) -> Decimal:
        direction = Decimal("1") if side == OrderSide.BUY else Decimal("-1")
        if self.sizing == "fixed":
            return direction * self.trade_size

        equity = self._current_equity()
        if price <= 0 or equity <= 0:
            return Decimal("0")
        return direction * equity * self.leverage / price

    def _current_equity(self) -> Decimal:
        equities = self.portfolio.equity(self.instrument_id.venue)
        money = equities.get(self.settlement_currency)
        if money is None and equities:
            money = next(iter(equities.values()))
        if money is None:
            return Decimal("0")
        return money.as_decimal()

    def _round_quantity(self, quantity: Decimal) -> Decimal:
        quantum = Decimal(1).scaleb(-self.size_precision)
        return quantity.quantize(quantum, rounding=ROUND_DOWN)
