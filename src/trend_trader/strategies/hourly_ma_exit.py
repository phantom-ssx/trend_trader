from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Literal

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide

from trend_trader.strategies.ma_spread_atr import MaSpreadAtrConfig, MaSpreadAtrStrategy

HourlyAction = Literal["ENTER_LONG", "ENTER_SHORT", "EXIT"]


class HourlyMaExitStateMachine:
    """Position-aware entry, exit, and cooldown rules for the hourly strategy."""

    def __init__(
        self,
        *,
        entry_threshold: float = 0.0025,
        exit_threshold: float = 0.0,
        atr_pct_min: float = 0.005,
        cooldown_bars: int = 10,
    ) -> None:
        if entry_threshold < 0 or atr_pct_min < 0 or cooldown_bars < 0:
            raise ValueError("thresholds and cooldown_bars must not be negative")
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.atr_pct_min = atr_pct_min
        self.cooldown_bars = cooldown_bars
        self.position_state = 0
        self.cooldown_remaining = 0
        self.previous_spread_pct: float | None = None

    def on_value(self, *, spread_pct: float, atr_pct: float) -> HourlyAction | None:
        previous = self.previous_spread_pct
        self.previous_spread_pct = spread_pct

        if self.position_state == 0:
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                return None
            if previous is None or atr_pct < self.atr_pct_min:
                return None
            if previous <= self.entry_threshold < spread_pct:
                self.position_state = 1
                return "ENTER_LONG"
            if previous >= -self.entry_threshold > spread_pct:
                self.position_state = -1
                return "ENTER_SHORT"
            return None

        if self.position_state == 1 and spread_pct <= self.exit_threshold:
            self.position_state = 0
            self.cooldown_remaining = self.cooldown_bars
            return "EXIT"
        if self.position_state == -1 and spread_pct >= -self.exit_threshold:
            self.position_state = 0
            self.cooldown_remaining = self.cooldown_bars
            return "EXIT"
        return None


class HourlyMaExitSignal:
    def __init__(
        self,
        *,
        fast_period: int = 5,
        slow_period: int = 20,
        atr_period: int = 14,
        entry_threshold: float = 0.0025,
        exit_threshold: float = 0.0,
        atr_pct_min: float = 0.005,
        cooldown_bars: int = 10,
    ) -> None:
        if fast_period <= 0 or slow_period <= 0 or atr_period <= 0:
            raise ValueError("periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.fast_window: deque[float] = deque()
        self.slow_window: deque[float] = deque()
        self.fast_sum = 0.0
        self.slow_sum = 0.0
        self.previous_close: float | None = None
        self.atr: float | None = None
        self.atr_count = 0
        self.state = HourlyMaExitStateMachine(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            atr_pct_min=atr_pct_min,
            cooldown_bars=cooldown_bars,
        )

    def on_bar(self, *, high: float, low: float, close: float) -> HourlyAction | None:
        self.fast_sum = self._push(self.fast_window, self.fast_sum, close, self.fast_period)
        self.slow_sum = self._push(self.slow_window, self.slow_sum, close, self.slow_period)
        self._update_atr(high=high, low=low, close=close)
        if len(self.slow_window) < self.slow_period or self.atr is None or close == 0:
            return None
        fast = self.fast_sum / self.fast_period
        slow = self.slow_sum / self.slow_period
        if slow == 0 or self.atr_count < self.atr_period:
            return None
        return self.state.on_value(spread_pct=(fast - slow) / slow, atr_pct=self.atr / close)

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
    def _push(window: deque[float], total: float, value: float, period: int) -> float:
        window.append(value)
        total += value
        if len(window) > period:
            total -= window.popleft()
        return total


class HourlyMaExitConfig(MaSpreadAtrConfig, frozen=True):
    fast_period: int = 5
    slow_period: int = 20
    spread_threshold: float = 0.0025
    exit_threshold: float = 0.0
    atr_pct_min: float = 0.005
    cooldown_bars: int = 10


class HourlyMaExitStrategy(MaSpreadAtrStrategy):
    def __init__(self, config: HourlyMaExitConfig) -> None:
        super().__init__(config)
        self.hourly_signal = HourlyMaExitSignal(
            fast_period=config.fast_period,
            slow_period=config.slow_period,
            atr_period=config.atr_period,
            entry_threshold=config.spread_threshold,
            exit_threshold=config.exit_threshold,
            atr_pct_min=config.atr_pct_min,
            cooldown_bars=config.cooldown_bars,
        )

    def _warm_up_indicator(self, bar: Bar) -> None:
        self.hourly_signal.on_bar(
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
        )

    def _on_warmup_complete(self, _request_id: object) -> None:
        minimum = max(self.hourly_signal.slow_period, self.hourly_signal.atr_period) + 1
        self.indicators_initialized = self.historical_bars_loaded >= minimum
        if self.indicators_initialized:
            self.log.info(
                f"Hourly indicators initialized from {self.historical_bars_loaded} historical bars"
            )
        else:
            self.log.error(
                f"Hourly indicator warmup incomplete: loaded {self.historical_bars_loaded}, "
                f"need at least {minimum}; order submission remains disabled"
            )

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized:
            self.log.debug("Skipping live bar while hourly indicators are warming up")
            return
        action = self.hourly_signal.on_bar(
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
        )
        if action is None:
            return

        price = Decimal(str(bar.close))
        signed_position = Decimal(str(self.portfolio.net_position(self.instrument_id)))
        if action == "EXIT":
            target_position = Decimal("0")
        else:
            target_side = OrderSide.BUY if action == "ENTER_LONG" else OrderSide.SELL
            target_position = self._target_position(side=target_side, price=price)
        target_order = self._order_for_target_position(
            target_position=target_position,
            signed_position=signed_position,
            price=price,
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
        self.log.info(f"Submitted {action} as {order_side.name} {quantity} {self.instrument_id}")
