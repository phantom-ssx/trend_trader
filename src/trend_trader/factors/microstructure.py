"""OHLCV-based market-microstructure proxy factors."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from trend_trader.data.models import DataType
from trend_trader.factors.base import Factor, positive_int
from trend_trader.factors.models import FactorSpec


class QuarterHourVolumePressureFactor(Factor):
    """Signed relative-volume pressure observed at quarter-hour boundaries.

    Historical candle data does not identify buyer- and seller-initiated trades, so
    close location within the bar is used as an explicitly approximate direction
    measure. Multiplying it by relative quote volume emphasizes the periodic bursts
    documented at cryptocurrency quarter-hour openings.
    """

    name = "quarter_hour_volume_pressure"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return positive_int(spec.params, "volume_period", 60)

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        volume_period = positive_int(spec.params, "volume_period", 60)
        boundary_minutes = positive_int(spec.params, "boundary_minutes", 15)
        if 60 % boundary_minutes:
            raise ValueError("boundary_minutes must divide one hour")

        candle_range = pl.col("high") - pl.col("low")
        close_location = (2 * pl.col("close") - pl.col("high") - pl.col("low")) / candle_range
        volume = pl.col("volume_quote")
        trailing_volume = volume.rolling_mean(volume_period, min_samples=volume_period)
        at_boundary = pl.col("timestamp").dt.minute().mod(boundary_minutes) == 0
        pressure = close_location * volume / trailing_volume
        value = (
            pl.when(trailing_volume.is_not_null() & (trailing_volume > 0))
            .then(
                pl.when(at_boundary & (candle_range > 0) & volume.is_not_null())
                .then(pressure)
                .otherwise(0.0)
            )
            .otherwise(None)
        )
        return inputs[DataType.CANDLES].select("timestamp", value.alias("raw_value"))


MICROSTRUCTURE_FACTORS = (QuarterHourVolumePressureFactor(),)
