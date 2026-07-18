"""Derivative-market, positioning, liquidation, and size factors."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from trend_trader.data.models import DataType
from trend_trader.factors.base import Factor, anchor, bar_timedelta, duration_bars, positive_int
from trend_trader.factors.models import DataDependency, FactorSpec


def _asof_value(
    inputs: Mapping[DataType, pl.DataFrame],
    data_type: DataType,
    column: str,
    *,
    tolerance: str | None = None,
) -> pl.DataFrame:
    timeline = anchor(inputs)
    source = inputs[data_type].select("timestamp", column).sort("timestamp")
    if source.is_empty():
        return timeline.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    return timeline.join_asof(source, on="timestamp", strategy="backward", tolerance=tolerance)


class FundingRateFactor(Factor):
    name = "funding_rate"
    dependencies = (
        DataDependency(DataType.CANDLES),
        DataDependency(DataType.FUNDING_RATES),
    )

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return duration_bars(spec.params.get("max_staleness"), bar_type, default="3d")

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        column = str(spec.params.get("column", "funding_rate"))
        if column not in {"funding_rate", "realized_rate"}:
            raise ValueError("funding column must be funding_rate or realized_rate")
        tolerance = str(spec.params.get("max_staleness", "3d"))
        timeline = anchor(inputs).with_columns(
            pl.col("timestamp").alias("_source_timestamp"),
            (pl.col("timestamp") + bar_timedelta(bar_type)).alias("_query_timestamp"),
        )
        source = (
            inputs[DataType.FUNDING_RATES]
            .select(pl.col("timestamp").alias("_funding_timestamp"), column)
            .sort("_funding_timestamp")
        )
        result = timeline.join_asof(
            source,
            left_on="_query_timestamp",
            right_on="_funding_timestamp",
            strategy="backward",
            tolerance=tolerance,
        ).select(pl.col("_source_timestamp").alias("timestamp"), column)
        smoothing = positive_int(spec.params, "smoothing", 1)
        value = pl.col(column)
        if smoothing > 1:
            value = value.rolling_mean(smoothing, min_samples=smoothing)
        if bool(spec.params.get("annualize", False)):
            periods_per_year = float(spec.params.get("periods_per_year", 3 * 365))
            value = value * periods_per_year
        return result.select("timestamp", value.alias("raw_value"))


class BasisFactor(Factor):
    name = "basis"
    dependencies = (
        DataDependency(DataType.CANDLES),
        DataDependency(DataType.CONTRACT_BASIS),
    )

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        return _asof_value(inputs, DataType.CONTRACT_BASIS, "basis_rate").select(
            "timestamp", pl.col("basis_rate").alias("raw_value")
        )


class OpenInterestFactor(Factor):
    name = "open_interest"
    dependencies = (
        DataDependency(DataType.CANDLES),
        DataDependency(DataType.OPEN_INTEREST),
    )

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        if str(spec.params.get("mode", "change")) == "level":
            return 0
        return duration_bars(spec.params.get("lookback"), bar_type, default="24h")

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        result = _asof_value(inputs, DataType.OPEN_INTEREST, "open_interest_usd")
        mode = str(spec.params.get("mode", "change"))
        if mode == "level":
            value = pl.when(pl.col("open_interest_usd") > 0).then(pl.col("open_interest_usd").log())
        elif mode == "change":
            lookback = duration_bars(spec.params.get("lookback"), bar_type, default="24h")
            current = pl.col("open_interest_usd")
            previous = current.shift(lookback)
            value = pl.when((current > 0) & (previous > 0)).then((current / previous).log())
        else:
            raise ValueError("open_interest mode must be 'change' or 'level'")
        return result.select("timestamp", value.alias("raw_value"))


class MarketCapFactor(Factor):
    name = "market_cap"
    dependencies = (
        DataDependency(DataType.CANDLES),
        DataDependency(DataType.MARKET_CAP),
    )

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return duration_bars(spec.params.get("max_staleness"), bar_type, default="3d")

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        tolerance = str(spec.params.get("max_staleness", "3d"))
        result = _asof_value(
            inputs,
            DataType.MARKET_CAP,
            "market_cap_usd",
            tolerance=tolerance,
        )
        return result.select(
            "timestamp",
            pl.when(pl.col("market_cap_usd") > 0)
            .then(pl.col("market_cap_usd").log())
            .alias("raw_value"),
        )


class LongShortRatioFactor(Factor):
    name = "long_short_ratio"
    dependencies = (
        DataDependency(DataType.CANDLES),
        DataDependency(DataType.LONG_SHORT_RATIO),
    )

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        result = _asof_value(inputs, DataType.LONG_SHORT_RATIO, "long_short_ratio")
        return result.select(
            "timestamp",
            pl.when(pl.col("long_short_ratio") > 0)
            .then(pl.col("long_short_ratio").log())
            .alias("raw_value"),
        )


class LiquidationImbalanceFactor(Factor):
    name = "liquidation_imbalance"
    dependencies = (
        DataDependency(DataType.CANDLES),
        DataDependency(DataType.LIQUIDATIONS),
    )

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        events = inputs[DataType.LIQUIDATIONS]
        timeline = anchor(inputs)
        if events.is_empty():
            return timeline.with_columns(pl.lit(0.0).alias("raw_value"))
        measure = str(spec.params.get("measure", "bankruptcy_loss"))
        if measure == "notional":
            amount = pl.col("bankruptcy_price") * pl.col("size")
        elif measure in {"bankruptcy_loss", "size"}:
            amount = pl.col(measure)
        else:
            raise ValueError("liquidation measure must be bankruptcy_loss, notional, or size")
        grouped = (
            events.with_columns(pl.col("timestamp").dt.truncate(bar_type).alias("timestamp"))
            .group_by("timestamp")
            .agg(
                amount.filter(pl.col("position_side") == "short").sum().alias("short_liq"),
                amount.filter(pl.col("position_side") == "long").sum().alias("long_liq"),
            )
        )
        joined = timeline.join(grouped, on="timestamp", how="left").with_columns(
            pl.col("short_liq").fill_null(0.0),
            pl.col("long_liq").fill_null(0.0),
        )
        total = pl.col("short_liq") + pl.col("long_liq")
        return joined.select(
            "timestamp",
            pl.when(total > 0)
            .then((pl.col("short_liq") - pl.col("long_liq")) / total)
            .otherwise(0.0)
            .alias("raw_value"),
        )


class TakerImbalanceFactor(Factor):
    name = "taker_imbalance"
    dependencies = (
        DataDependency(DataType.CANDLES),
        DataDependency(DataType.TAKER_VOLUME),
    )

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        timeline = anchor(inputs)
        source = inputs[DataType.TAKER_VOLUME].select("timestamp", "buy_volume", "sell_volume")
        joined = timeline.join(source, on="timestamp", how="left")
        total = pl.col("buy_volume") + pl.col("sell_volume")
        return joined.select(
            "timestamp",
            pl.when(total > 0)
            .then((pl.col("buy_volume") - pl.col("sell_volume")) / total)
            .alias("raw_value"),
        )


DERIVATIVE_FACTORS = (
    FundingRateFactor(),
    BasisFactor(),
    OpenInterestFactor(),
    MarketCapFactor(),
    LongShortRatioFactor(),
    LiquidationImbalanceFactor(),
    TakerImbalanceFactor(),
)
