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


class VolumeConfirmedReversalEventFactor(Factor):
    """One-bar event for an unusually sharp, high-volume short-term selloff.

    The signal standardizes short-horizon momentum against its own trailing
    history, smooths that surprise, and emits a positive score only on the first
    threshold crossing when relative quote volume also confirms the shock. All
    rolling inputs include only the current and earlier candles.
    """

    name = "volume_confirmed_reversal_event"

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return (
            positive_int(spec.params, "momentum_lookback", 5)
            + positive_int(spec.params, "normalization_window", 43_200)
            + positive_int(spec.params, "smoothing_period", 45)
        )

    def compute(
        self, inputs: Mapping[DataType, pl.DataFrame], spec: FactorSpec, bar_type: str
    ) -> pl.DataFrame:
        momentum_lookback = positive_int(spec.params, "momentum_lookback", 5)
        normalization_window = positive_int(
            spec.params, "normalization_window", 43_200
        )
        normalization_min_periods = positive_int(
            spec.params, "normalization_min_periods", 14_400
        )
        smoothing_period = positive_int(spec.params, "smoothing_period", 45)
        volume_period = positive_int(spec.params, "volume_period", 60)
        signal_threshold = float(spec.params.get("signal_threshold", 1.5))
        minimum_relative_volume = float(spec.params.get("minimum_relative_volume", 1.5))
        if normalization_min_periods > normalization_window:
            raise ValueError("normalization_min_periods must not exceed normalization_window")
        if signal_threshold <= 0 or minimum_relative_volume <= 0:
            raise ValueError("event thresholds must be positive")

        close = pl.col("close").log()
        momentum = close - close.shift(momentum_lookback)
        momentum_mean = momentum.rolling_mean(
            normalization_window, min_samples=normalization_min_periods
        )
        momentum_std = momentum.rolling_std(
            normalization_window, min_samples=normalization_min_periods
        )
        normalized_momentum = pl.when(momentum_std > 0).then(
            (momentum - momentum_mean) / momentum_std
        )
        reversal_score = -normalized_momentum.rolling_mean(
            smoothing_period, min_samples=smoothing_period
        )

        volume = pl.col(str(spec.params.get("volume_column", "volume_quote")))
        trailing_volume = volume.rolling_mean(volume_period, min_samples=volume_period)
        relative_volume = pl.when(trailing_volume > 0).then(volume / trailing_volume)
        inputs_ready = reversal_score.is_not_null() & relative_volume.is_not_null()
        crossed_threshold = (reversal_score > signal_threshold) & (
            reversal_score.shift(1) <= signal_threshold
        )
        value = (
            pl.when(inputs_ready)
            .then(
                pl.when(crossed_threshold & (relative_volume > minimum_relative_volume))
                .then(reversal_score)
                .otherwise(0.0)
            )
            .otherwise(None)
        )
        return inputs[DataType.CANDLES].select("timestamp", value.alias("raw_value"))


MICROSTRUCTURE_FACTORS = (
    QuarterHourVolumePressureFactor(),
    VolumeConfirmedReversalEventFactor(),
)
