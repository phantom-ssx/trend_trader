"""Volume and liquidity factors."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from trend_trader.data.models import DataType
from trend_trader.factors.base import Factor, positive_int
from trend_trader.factors.models import FactorSpec


class VolumeChangeFactor(Factor):
    name = "volume_change"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "period", 24) * 2

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        period = positive_int(spec.params, "period", 24)
        volume = pl.col(str(spec.params.get("column", "volume_quote")))
        current = volume.rolling_sum(period, min_samples=period)
        previous = current.shift(period)
        return inputs[DataType.CANDLES].select(
            "timestamp",
            pl.when((current > 0) & (previous > 0))
            .then((current / previous).log())
            .alias("raw_value"),
        )


class TurnoverFactor(Factor):
    name = "turnover"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "period", 24)

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        period = positive_int(spec.params, "period", 24)
        turnover = pl.col("volume_quote").rolling_sum(period, min_samples=period)
        return inputs[DataType.CANDLES].select("timestamp", turnover.log1p().alias("raw_value"))


class AmihudFactor(Factor):
    name = "amihud"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "period", 24) + 1

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        period = positive_int(spec.params, "period", 24)
        turnover = pl.col("volume_quote")
        ratio = pl.when(turnover > 0).then(pl.col("close").log().diff().abs() / turnover)
        value = ratio.rolling_mean(period, min_samples=period)
        if bool(spec.params.get("log_transform", False)):
            scale = float(spec.params.get("scale", 1e12))
            value = (value * scale).log1p()
        return inputs[DataType.CANDLES].select("timestamp", value.alias("raw_value"))


class VolumePriceDivergenceFactor(Factor):
    name = "volume_price_divergence"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "period", 24) * 3

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        period = positive_int(spec.params, "period", 24)
        price_momentum = pl.col("close").log() - pl.col("close").shift(period).log()
        volume_growth = (
            pl.col("volume_quote").rolling_sum(period, min_samples=period).log().diff(period)
        )
        price_mean = price_momentum.rolling_mean(period, min_samples=period)
        price_std = price_momentum.rolling_std(period, min_samples=period)
        volume_mean = volume_growth.rolling_mean(period, min_samples=period)
        volume_std = volume_growth.rolling_std(period, min_samples=period)
        divergence = pl.when((price_std > 0) & (volume_std > 0)).then(
            (price_momentum - price_mean) / price_std - (volume_growth - volume_mean) / volume_std
        )
        return inputs[DataType.CANDLES].select("timestamp", divergence.alias("raw_value"))


LIQUIDITY_FACTORS = (
    VolumeChangeFactor(),
    TurnoverFactor(),
    AmihudFactor(),
    VolumePriceDivergenceFactor(),
)
