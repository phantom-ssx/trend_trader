from __future__ import annotations

from collections import deque
from decimal import ROUND_DOWN, Decimal
from typing import Literal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Currency, Quantity
from nautilus_trader.trading.strategy import Strategy

from trend_trader.notifications import BarkNotifier

SignalSide = Literal["BUY", "SELL"]


def is_below_min_order_notional(
    *,
    quantity: Decimal,
    price: Decimal,
    min_order_notional: Decimal,
) -> bool:
    return min_order_notional > 0 and quantity * price < min_order_notional


def order_side_for_position_delta(delta: Decimal) -> OrderSide | None:
    if delta > 0:
        return OrderSide.BUY
    if delta < 0:
        return OrderSide.SELL
    return None


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
    min_order_notional: Decimal = Decimal("50")
    size_precision: int = 6
    warmup_bars: int = 100
    load_history_on_start: bool = False
    bark_url: str | None = None
    trading_mode: str | None = None


class MaSpreadAtrStrategy(Strategy):
    """NautilusTrader wrapper for the best-performing MA spread + ATR filter."""

    def __init__(self, config: MaSpreadAtrConfig) -> None:
        super().__init__(config)
        if config.sizing not in {"fixed", "all-in"}:
            raise ValueError("sizing must be either 'fixed' or 'all-in'")
        if config.min_order_notional < 0:
            raise ValueError("min_order_notional must not be negative")
        minimum_warmup = max(config.slow_period, config.atr_period) + 1
        if config.load_history_on_start and config.warmup_bars < minimum_warmup:
            raise ValueError(f"warmup_bars must be at least {minimum_warmup}")

        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.settlement_currency = config.settlement_currency
        self.trade_size = config.trade_size
        self.sizing = config.sizing
        self.leverage = config.leverage
        self.min_order_notional = config.min_order_notional
        self.size_precision = config.size_precision
        self.warmup_bars = config.warmup_bars
        self.load_history_on_start = config.load_history_on_start
        self.indicators_initialized = not config.load_history_on_start
        self.historical_bars_loaded = 0
        self.market_data_started = False
        self.notifier = (
            BarkNotifier(config.bark_url, config.trading_mode, on_error=self.log.warning)
            if config.bark_url and config.trading_mode in {"模拟盘", "实盘"}
            else None
        )
        self.signal = MaSpreadAtrSignal(
            fast_period=config.fast_period,
            slow_period=config.slow_period,
            spread_threshold=config.spread_threshold,
            atr_period=config.atr_period,
            atr_pct_min=config.atr_pct_min,
        )

    def on_start(self) -> None:
        if not self.load_history_on_start:
            self._start_market_data()
            return
        # Live clients load their instrument providers asynchronously. Request the
        # instrument first so the bar request cannot race the cache population.
        self.request_instrument(
            self.instrument_id,
            callback=self._on_instrument_ready,
        )

    def _on_instrument_ready(self, _request_id: object) -> None:
        if self.cache.instrument(self.instrument_id) is None:
            self.log.error(
                f"Instrument {self.instrument_id} was not loaded into cache; "
                "market data and order submission remain disabled"
            )
            return
        self._start_market_data()

    def _start_market_data(self) -> None:
        if self.market_data_started:
            return
        self.market_data_started = True
        self.subscribe_bars(self.bar_type)
        if not self.load_history_on_start:
            return
        interval = self.bar_type.spec.timedelta
        self.request_bars(
            self.bar_type,
            start=self.clock.utc_now() - interval * self.warmup_bars,
            limit=self.warmup_bars,
            callback=self._on_warmup_complete,
        )

    def on_historical_data(self, data: object) -> None:
        if not isinstance(data, Bar):
            return
        self._warm_up_indicator(data)
        self.historical_bars_loaded += 1

    def _warm_up_indicator(self, bar: Bar) -> None:
        self.signal.on_bar(
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
        )

    def _on_warmup_complete(self, _request_id: object) -> None:
        minimum = max(self.signal.slow_period, self.signal.atr_period) + 1
        self.indicators_initialized = self.historical_bars_loaded >= minimum
        if self.indicators_initialized:
            self.log.info(
                f"Indicators initialized from {self.historical_bars_loaded} historical bars"
            )
        else:
            self.log.error(
                f"Indicator warmup incomplete: loaded {self.historical_bars_loaded}, "
                f"need at least {minimum}; order submission remains disabled"
            )

    def on_stop(self) -> None:
        if self.market_data_started:
            self.unsubscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self._notify_bar(bar)
        if not self.indicators_initialized:
            self.log.debug("Skipping live bar while indicators are warming up")
            return
        side_text = self.signal.on_bar(
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
        )
        if side_text is None:
            return

        target_side = OrderSide.BUY if side_text == "BUY" else OrderSide.SELL
        target_order = self._target_order(
            side=target_side,
            price=Decimal(str(bar.close)),
        )
        if target_order is None:
            return
        order_side, quantity = target_order

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=quantity,
        )
        self.submit_order(order)
        self.log.info(f"Submitted {order_side.name} order for {quantity} {self.instrument_id}")

    def on_event(self, event: Event) -> None:
        if isinstance(event, OrderFilled):
            self._notify_trade(event)
        self.log.debug(str(event))

    def _notify_bar(self, bar: Bar) -> None:
        if self.notifier is None:
            return
        body = (
            f"标的: {self.instrument_id}\n"
            f"周期: {self.bar_type.spec}\n"
            f"时间(ns): {bar.ts_event}\n"
            f"开: {bar.open}  高: {bar.high}\n"
            f"低: {bar.low}  收: {bar.close}\n"
            f"成交量: {bar.volume}"
        )
        self.notifier.send(f"[{self.notifier.mode}] K线", body)

    def _notify_trade(self, fill: OrderFilled) -> None:
        if self.notifier is None:
            return
        body = (
            f"标的: {fill.instrument_id}\n"
            f"方向: {fill.order_side.name}\n"
            f"数量: {fill.last_qty}\n"
            f"成交价: {fill.last_px}\n"
            f"手续费: {fill.commission}\n"
            f"成交ID: {fill.trade_id}\n"
            f"时间(ns): {fill.ts_event}"
        )
        self.notifier.send(f"[{self.notifier.mode}] 交易成交", body)

    def _target_order(
        self,
        *,
        side: OrderSide,
        price: Decimal,
    ) -> tuple[OrderSide, Quantity] | None:
        signed_position = Decimal(str(self.portfolio.net_position(self.instrument_id)))
        target_position = self._target_position(side=side, price=price)
        return self._order_for_target_position(
            target_position=target_position,
            signed_position=signed_position,
            price=price,
        )

    def _order_for_target_position(
        self,
        *,
        target_position: Decimal,
        signed_position: Decimal,
        price: Decimal,
    ) -> tuple[OrderSide, Quantity] | None:
        delta = target_position - signed_position
        order_side = order_side_for_position_delta(delta)
        if order_side is None:
            return None
        quantity = self._round_quantity(abs(delta))
        if quantity <= 0:
            return None
        if is_below_min_order_notional(
            quantity=quantity,
            price=price,
            min_order_notional=self.min_order_notional,
        ):
            return None
        return order_side, Quantity.from_str(f"{quantity:.{self.size_precision}f}")

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
