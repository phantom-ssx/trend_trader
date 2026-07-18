"""Volatility factors."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from trend_trader.data.models import DataType
from trend_trader.factors.base import Factor, annualization_factor, positive_int
from trend_trader.factors.models import FactorSpec


def _log_return() -> pl.Expr:
    return pl.col("close").log().diff()


class HistoricalVolatilityFactor(Factor):
    name = "historical_volatility"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "period", 24) + 1

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        period = positive_int(spec.params, "period", 24)
        annualize = bool(spec.params.get("annualize", True))
        scale = annualization_factor(bar_type) if annualize else 1.0
        return inputs[DataType.CANDLES].select(
            "timestamp",
            (_log_return().rolling_std(period, min_samples=period) * scale).alias("raw_value"),
        )


class AtrFactor(Factor):
    name = "atr"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "period", 14) + 1

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        period = positive_int(spec.params, "period", 14)
        normalized = bool(spec.params.get("normalized", True))
        previous_close = pl.col("close").shift(1)
        true_range = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - previous_close).abs(),
            (pl.col("low") - previous_close).abs(),
        )
        atr = true_range.ewm_mean(alpha=1 / period, adjust=False, min_samples=period)
        value = pl.when(pl.col("close") != 0).then(atr / pl.col("close")) if normalized else atr
        return inputs[DataType.CANDLES].select("timestamp", value.alias("raw_value"))


class VolatilityChangeFactor(Factor):
    name = "volatility_change"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "long_period", 72) + 1

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        short = positive_int(spec.params, "short_period", 24)
        long = positive_int(spec.params, "long_period", 72)
        if short >= long:
            raise ValueError("short_period must be less than long_period")
        returns = _log_return()
        short_vol = returns.rolling_std(short, min_samples=short)
        long_vol = returns.rolling_std(long, min_samples=long)
        return inputs[DataType.CANDLES].select(
            "timestamp",
            pl.when(long_vol > 0).then(short_vol / long_vol - 1).alias("raw_value"),
        )


class UpDownVolatilityAsymmetryFactor(Factor):
    name = "up_down_volatility_asymmetry"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "period", 24) + 1

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        period = positive_int(spec.params, "period", 24)
        returns = _log_return()
        upside = (
            pl.when(returns > 0)
            .then(returns.pow(2))
            .otherwise(0.0)
            .rolling_mean(period, min_samples=period)
            .sqrt()
        )
        downside = (
            pl.when(returns < 0)
            .then(returns.pow(2))
            .otherwise(0.0)
            .rolling_mean(period, min_samples=period)
            .sqrt()
        )
        denominator = downside + upside
        return inputs[DataType.CANDLES].select(
            "timestamp",
            pl.when(denominator > 0).then((downside - upside) / denominator).alias("raw_value"),
        )


VOLATILITY_FACTORS = (
    HistoricalVolatilityFactor(),
    AtrFactor(),
    VolatilityChangeFactor(),
    UpDownVolatilityAsymmetryFactor(),
)
