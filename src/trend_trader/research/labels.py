"""Execution-aware forward-return labels for offline research only."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

import polars as pl

from trend_trader.data.models import bar_minutes
from trend_trader.research.models import ExecutionReturnSpec, LabelResult

_KEYS = ["venue", "instrument_id", "bar_type"]


class ExecutionReturnLabeler:
    """Build next-open entry and open-to-open exit labels."""

    def compute(
        self,
        candles: pl.DataFrame,
        specs: Sequence[ExecutionReturnSpec],
    ) -> LabelResult:
        specs = tuple(specs)
        if not specs:
            raise ValueError("at least one execution-return label is required")
        missing = sorted({*_KEYS, "timestamp", "open"}.difference(candles.columns))
        if missing:
            raise ValueError(f"candles are missing label columns: {missing}")
        if candles.is_empty():
            return LabelResult(_empty_labels(candles.schema["timestamp"]))
        bar_types = candles.get_column("bar_type").unique().to_list()
        if len(bar_types) != 1:
            raise ValueError("label calculation requires exactly one bar_type")
        bar_type = str(bar_types[0])
        step = timedelta(minutes=bar_minutes(bar_type))
        ordered = candles.sort(*_KEYS, "timestamp")
        frames = [self._compute_one(ordered, spec, step) for spec in specs]
        return LabelResult(
            pl.concat(frames, how="vertical_relaxed").sort("label_name", *_KEYS, "timestamp")
        )

    @staticmethod
    def _compute_one(
        candles: pl.DataFrame,
        spec: ExecutionReturnSpec,
        step: timedelta,
    ) -> pl.DataFrame:
        horizon = spec.horizon_bars
        future_price = pl.col("open").shift(-horizon).over(_KEYS)
        observed_exit = pl.col("timestamp").shift(-horizon).over(_KEYS)
        expected_exit = pl.col("timestamp") + step * horizon
        frame = candles.select(
            *_KEYS,
            "timestamp",
            pl.col("timestamp").alias("entry_time"),
            observed_exit.alias("_observed_exit_time"),
            expected_exit.alias("exit_time"),
            pl.col("open").cast(pl.Float64).alias("entry_price"),
            future_price.cast(pl.Float64).alias("exit_price"),
        )
        valid_prices = (
            pl.col("entry_price").is_not_null()
            & pl.col("exit_price").is_not_null()
            & pl.col("entry_price").is_finite()
            & pl.col("exit_price").is_finite()
            & (pl.col("entry_price") > 0)
            & (pl.col("exit_price") > 0)
        )
        continuous = pl.col("_observed_exit_time") == pl.col("exit_time")
        gross = pl.col("exit_price") / pl.col("entry_price") - 1
        net = gross - spec.round_trip_cost_bps / 10_000
        valid = valid_prices & continuous
        return (
            frame.with_columns(
                pl.lit(spec.label_name).alias("label_name"),
                pl.lit(horizon).cast(pl.Int32).alias("horizon_bars"),
                pl.lit(spec.round_trip_cost_bps).alias("round_trip_cost_bps"),
                pl.when(valid).then(gross).alias("gross_return"),
                pl.when(valid).then(net).alias("net_return"),
                pl.when(valid).then(net).alias("label_value"),
                valid.alias("label_is_valid"),
                pl.when(pl.col("exit_price").is_null())
                .then(pl.lit("MISSING_FUTURE_PRICE"))
                .when(~continuous)
                .then(pl.lit("NON_CONTIGUOUS_HORIZON"))
                .when(~valid_prices)
                .then(pl.lit("INVALID_PRICE"))
                .otherwise(pl.lit(""))
                .alias("label_quality_flags"),
            )
            .drop("_observed_exit_time")
            .select(
                *_KEYS,
                "timestamp",
                "label_name",
                "horizon_bars",
                "round_trip_cost_bps",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "gross_return",
                "net_return",
                "label_value",
                "label_is_valid",
                "label_quality_flags",
            )
        )


def _empty_labels(timestamp_dtype: pl.DataType) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "venue": pl.Utf8,
            "instrument_id": pl.Utf8,
            "bar_type": pl.Utf8,
            "timestamp": timestamp_dtype,
            "label_name": pl.Utf8,
            "horizon_bars": pl.Int32,
            "round_trip_cost_bps": pl.Float64,
            "entry_time": timestamp_dtype,
            "exit_time": timestamp_dtype,
            "entry_price": pl.Float64,
            "exit_price": pl.Float64,
            "gross_return": pl.Float64,
            "net_return": pl.Float64,
            "label_value": pl.Float64,
            "label_is_valid": pl.Boolean,
            "label_quality_flags": pl.Utf8,
        }
    )
