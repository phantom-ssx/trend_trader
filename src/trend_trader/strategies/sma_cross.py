from __future__ import annotations

from collections import deque
from typing import Literal

SignalSide = Literal["BUY", "SELL"]


class SmaCrossSignal:
    """Simple moving-average crossover signal on close prices."""

    def __init__(self, fast_period: int = 5, slow_period: int = 20) -> None:
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("SMA periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_window: deque[float] = deque()
        self.slow_window: deque[float] = deque()
        self.fast_sum = 0.0
        self.slow_sum = 0.0
        self.previous_spread: float | None = None

    def on_price(self, close: float) -> SignalSide | None:
        self.fast_sum = self._push(self.fast_window, self.fast_sum, close, self.fast_period)
        self.slow_sum = self._push(self.slow_window, self.slow_sum, close, self.slow_period)
        if len(self.slow_window) < self.slow_period:
            return None

        fast = self.fast_sum / self.fast_period
        slow = self.slow_sum / self.slow_period
        spread = fast - slow
        previous_spread = self.previous_spread
        self.previous_spread = spread

        if previous_spread is None:
            return None
        if previous_spread <= 0 < spread:
            return "BUY"
        if previous_spread >= 0 > spread:
            return "SELL"
        return None

    @staticmethod
    def _push(window: deque[float], running_sum: float, value: float, period: int) -> float:
        window.append(value)
        running_sum += value
        if len(window) > period:
            running_sum -= window.popleft()
        return running_sum
