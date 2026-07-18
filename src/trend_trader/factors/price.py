"""Price, trend, breakout, and mean-reversion factors."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from trend_trader.data.models import DataType
from trend_trader.factors.base import Factor, duration_bars, positive_int
from trend_trader.factors.models import FactorSpec


class MomentumFactor(Factor):
    name = "momentum"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return duration_bars(spec.params.get("lookback"), bar_type, default="24h")

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        lookback = self.required_history_bars(spec, bar_type)
        return inputs[DataType.CANDLES].select(
            "timestamp",
            (pl.col("close").log() - pl.col("close").shift(lookback).log()).alias("raw_value"),
        )


class MaSpreadFactor(Factor):
    name = "ma_spread"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "slow_period", 20)

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        fast = positive_int(spec.params, "fast_period", 5)
        slow = positive_int(spec.params, "slow_period", 20)
        if fast >= slow:
            raise ValueError("fast_period must be less than slow_period")
        close = pl.col("close")
        fast_ma = close.rolling_mean(fast, min_samples=fast)
        slow_ma = close.rolling_mean(slow, min_samples=slow)
        return inputs[DataType.CANDLES].select(
            "timestamp",
            pl.when(slow_ma != 0).then((fast_ma - slow_ma) / slow_ma).alias("raw_value"),
        )


class TrendSlopeFactor(Factor):
    name = "trend_slope"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "period", 20)

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        period = positive_int(spec.params, "period", 20)
        weight_r2 = bool(spec.params.get("weight_r2", False))

        def slope(values: pl.Series) -> float | None:
            if len(values) < period or values.null_count():
                return None
            observed = [float(value) for value in values]
            center = (period - 1) / 2
            y_mean = sum(observed) / period
            denominator = sum((index - center) ** 2 for index in range(period))
            beta = (
                sum((index - center) * (value - y_mean) for index, value in enumerate(observed))
                / denominator
            )
            normalized = beta * (period - 1)
            if not weight_r2:
                return normalized
            fitted = [y_mean + beta * (index - center) for index in range(period)]
            total = sum((value - y_mean) ** 2 for value in observed)
            residual = sum(
                (value - estimate) ** 2 for value, estimate in zip(observed, fitted, strict=True)
            )
            r_squared = 1 - residual / total if total > 0 else 0.0
            return normalized * max(0.0, r_squared)

        return inputs[DataType.CANDLES].select(
            "timestamp",
            pl.col("close")
            .log()
            .rolling_map(slope, window_size=period, min_samples=period)
            .alias("raw_value"),
        )


class BreakoutFactor(Factor):
    name = "breakout"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "period", 20) + 1

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        period = positive_int(spec.params, "period", 20)
        upper = pl.col("high").rolling_max(period, min_samples=period).shift(1)
        lower = pl.col("low").rolling_min(period, min_samples=period).shift(1)
        width = upper - lower
        position = 2 * (pl.col("close") - lower) / width - 1
        return inputs[DataType.CANDLES].select(
            "timestamp",
            pl.when(width > 0).then(position).alias("raw_value"),
        )


class MeanReversionFactor(Factor):
    name = "mean_reversion"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "period", 20)

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        period = positive_int(spec.params, "period", 20)
        price = pl.col("close").log()
        mean = price.rolling_mean(period, min_samples=period)
        std = price.rolling_std(period, min_samples=period)
        return inputs[DataType.CANDLES].select(
            "timestamp",
            pl.when(std > 0).then(-(price - mean) / std).alias("raw_value"),
        )


PRICE_FACTORS = (
    MomentumFactor(),
    MaSpreadFactor(),
    TrendSlopeFactor(),
    BreakoutFactor(),
    MeanReversionFactor(),
)
