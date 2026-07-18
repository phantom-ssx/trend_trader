"""Outlier handling, standardization, and optional neutralization."""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl

from trend_trader.factors.models import ProcessingConfig

_CROSS_KEYS = ["factor_name", "timestamp"]
_TIME_KEYS = ["factor_name", "venue", "instrument_id"]


class FactorProcessor:
    def apply(
        self,
        frame: pl.DataFrame,
        config: ProcessingConfig,
        *,
        exposures: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        result = frame.sort("factor_name", "instrument_id", "timestamp").with_columns(
            pl.col("raw_value").alias("value")
        )
        result = self._mark_non_finite(result)
        result = self._handle_outliers(result, config)
        result = self._standardize(result, config)
        if config.neutralize.exposures:
            result = self._neutralize(result, config, exposures)
        return result.with_columns(
            (
                pl.col("is_valid") & pl.col("value").is_not_null() & pl.col("value").is_finite()
            ).alias("is_valid")
        )

    @staticmethod
    def _mark_non_finite(frame: pl.DataFrame) -> pl.DataFrame:
        valid = pl.col("value").is_not_null() & pl.col("value").is_finite()
        return frame.with_columns(
            valid.alias("is_valid"),
            pl.when(valid)
            .then(pl.lit(""))
            .otherwise(pl.lit("NON_FINITE_RAW_VALUE"))
            .alias("quality_flags"),
            pl.when(valid).then(pl.col("value")).otherwise(None).alias("value"),
        )

    def _handle_outliers(self, frame: pl.DataFrame, config: ProcessingConfig) -> pl.DataFrame:
        settings = config.outlier
        if settings.method == "none":
            return frame
        keys = _CROSS_KEYS if settings.scope == "cross_sectional" else _TIME_KEYS
        value = pl.col("value")
        if settings.scope == "time_series":
            rolling = {"window_size": settings.window, "min_samples": settings.min_periods}
            if settings.method == "mad":
                frame = frame.with_columns(
                    value.rolling_median(**rolling).over(keys).alias("_center")
                )
                frame = frame.with_columns(
                    value.rolling_map(_mad, **rolling).over(keys).alias("_mad")
                )
                scale = pl.col("_mad") * 1.4826
                lower = pl.col("_center") - settings.threshold * scale
                upper = pl.col("_center") + settings.threshold * scale
            elif settings.method == "std":
                frame = frame.with_columns(
                    value.rolling_mean(**rolling).over(keys).alias("_center"),
                    value.rolling_std(**rolling).over(keys).alias("_scale"),
                )
                lower = pl.col("_center") - settings.threshold * pl.col("_scale")
                upper = pl.col("_center") + settings.threshold * pl.col("_scale")
            else:
                frame = frame.with_columns(
                    value.rolling_quantile(settings.lower_quantile, **rolling)
                    .over(keys)
                    .alias("_lo"),
                    value.rolling_quantile(settings.upper_quantile, **rolling)
                    .over(keys)
                    .alias("_hi"),
                )
                lower, upper = pl.col("_lo"), pl.col("_hi")
        elif settings.method == "mad":
            frame = frame.with_columns(value.median().over(keys).alias("_center"))
            frame = frame.with_columns(
                (value - pl.col("_center")).abs().median().over(keys).alias("_mad")
            )
            scale = pl.col("_mad") * 1.4826
            lower = pl.col("_center") - settings.threshold * scale
            upper = pl.col("_center") + settings.threshold * scale
        elif settings.method == "std":
            frame = frame.with_columns(
                value.mean().over(keys).alias("_center"),
                value.std().over(keys).alias("_scale"),
            )
            lower = pl.col("_center") - settings.threshold * pl.col("_scale")
            upper = pl.col("_center") + settings.threshold * pl.col("_scale")
        else:
            lower = value.quantile(settings.lower_quantile).over(keys)
            upper = value.quantile(settings.upper_quantile).over(keys)

        frame = frame.with_columns(value.clip(lower, upper).alias("value"))
        return frame.drop(column for column in frame.columns if column.startswith("_"))

    def _standardize(self, frame: pl.DataFrame, config: ProcessingConfig) -> pl.DataFrame:
        settings = config.standardize
        if settings.method == "none":
            return frame
        value = pl.col("value")
        if settings.scope == "cross_sectional":
            keys = _CROSS_KEYS
            frame = frame.with_columns(value.count().over(keys).alias("_count"))
            enough = pl.col("_count") >= settings.min_cross_section
            if settings.method == "rank":
                rank = value.rank(method="average").over(keys)
                standardized = pl.when(pl.col("_count") > 1).then(
                    2 * (rank - 1) / (pl.col("_count") - 1) - 1
                )
            elif settings.method == "robust_zscore":
                frame = frame.with_columns(value.median().over(keys).alias("_center"))
                frame = frame.with_columns(
                    (value - pl.col("_center")).abs().median().over(keys).alias("_mad")
                )
                denominator = pl.col("_mad") * 1.4826
                standardized = pl.when(denominator > 0).then(
                    (value - pl.col("_center")) / denominator
                )
            else:
                mean = value.mean().over(keys)
                std = value.std().over(keys)
                standardized = pl.when(std > 0).then((value - mean) / std)
            frame = frame.with_columns(
                pl.when(enough).then(standardized).otherwise(None).alias("value"),
                pl.when(enough)
                .then(pl.col("quality_flags"))
                .otherwise(_append_flag("quality_flags", "INSUFFICIENT_CROSS_SECTION"))
                .alias("quality_flags"),
            )
            frame = frame.with_columns(
                pl.when(pl.col("value").is_null() & (pl.col("quality_flags") == ""))
                .then(pl.lit("STANDARDIZATION_FAILED"))
                .otherwise(pl.col("quality_flags"))
                .alias("quality_flags")
            )
            return frame.drop(column for column in frame.columns if column.startswith("_"))

        keys = _TIME_KEYS
        rolling = {"window_size": settings.window, "min_samples": settings.min_periods}
        frame = frame.with_columns(
            value.is_not_null().cast(pl.Int64).rolling_sum(**rolling).over(keys).alias("_count")
        )
        enough = pl.col("_count") >= settings.min_periods
        if settings.method == "rank":
            standardized = value.rolling_map(
                lambda values: (
                    float(values.rank(method="average")[-1] - 1) * 2 / (len(values) - 1) - 1
                    if len(values) > 1
                    else None
                ),
                **rolling,
            ).over(keys)
        elif settings.method == "robust_zscore":
            center = value.rolling_median(**rolling).over(keys)
            frame = frame.with_columns(center.alias("_center"))
            mad = value.rolling_map(_mad, **rolling).over(keys)
            standardized = pl.when(mad > 0).then((value - pl.col("_center")) / (1.4826 * mad))
        else:
            mean = value.rolling_mean(**rolling).over(keys)
            std = value.rolling_std(**rolling).over(keys)
            standardized = pl.when(std > 0).then((value - mean) / std)
        frame = frame.with_columns(
            pl.when(enough).then(standardized).otherwise(None).alias("value"),
            pl.when(enough)
            .then(pl.col("quality_flags"))
            .otherwise(_append_flag("quality_flags", "WARMUP_INCOMPLETE"))
            .alias("quality_flags"),
        )
        frame = frame.with_columns(
            pl.when(pl.col("value").is_null() & (pl.col("quality_flags") == ""))
            .then(pl.lit("STANDARDIZATION_FAILED"))
            .otherwise(pl.col("quality_flags"))
            .alias("quality_flags")
        )
        return frame.drop(column for column in frame.columns if column.startswith("_"))

    def _neutralize(
        self,
        frame: pl.DataFrame,
        config: ProcessingConfig,
        exposures: pl.DataFrame | None,
    ) -> pl.DataFrame:
        if exposures is None or exposures.is_empty():
            raise ValueError("neutralization exposures were requested but not calculated")
        exposure_names = config.neutralize.exposures
        wide = exposures.pivot(
            on="factor_key",
            index=["venue", "instrument_id", "timestamp"],
            values="raw_value",
        )
        missing = sorted(set(exposure_names).difference(wide.columns))
        if missing:
            raise ValueError(f"missing neutralization exposures: {missing}")
        joined = frame.join(wide, on=["venue", "instrument_id", "timestamp"], how="left")
        output: list[pl.DataFrame] = []
        for group in joined.partition_by(["factor_name", "timestamp"], maintain_order=True):
            output.append(self._neutralize_group(group, exposure_names, config))
        return pl.concat(output, how="vertical_relaxed").select(frame.columns)

    @staticmethod
    def _neutralize_group(
        group: pl.DataFrame,
        exposures: Sequence[str],
        config: ProcessingConfig,
    ) -> pl.DataFrame:
        valid_rows: list[int] = []
        matrix: list[list[float]] = []
        target: list[float] = []
        for index, row in enumerate(group.iter_rows(named=True)):
            values = [row.get(name) for name in exposures]
            if (
                row["value"] is not None
                and math.isfinite(float(row["value"]))
                and all(value is not None and math.isfinite(float(value)) for value in values)
            ):
                valid_rows.append(index)
                matrix.append([1.0, *(float(value) for value in values)])
                target.append(float(row["value"]))
        minimum = max(config.neutralize.min_observations, len(exposures) + 2)
        if len(valid_rows) < minimum:
            return group.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("value"),
                _append_flag("quality_flags", "NEUTRALIZATION_FAILED").alias("quality_flags"),
            )
        coefficients = _ridge_solve(matrix, target, config.neutralize.ridge)
        if coefficients is None:
            return group.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("value"),
                _append_flag("quality_flags", "NEUTRALIZATION_FAILED").alias("quality_flags"),
            )
        residuals: list[float | None] = [None] * group.height
        for index, row, observed in zip(valid_rows, matrix, target, strict=True):
            residuals[index] = observed - sum(
                x * beta for x, beta in zip(row, coefficients, strict=True)
            )
        group = group.with_columns(pl.Series("value", residuals, dtype=pl.Float64))
        return group.with_columns(
            pl.when(pl.col("value").is_null() & (pl.col("quality_flags") == ""))
            .then(pl.lit("NEUTRALIZATION_FAILED"))
            .otherwise(pl.col("quality_flags"))
            .alias("quality_flags")
        )


def _append_flag(column: str, flag: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) == "")
        .then(pl.lit(flag))
        .otherwise(pl.concat_str(pl.col(column), pl.lit(flag), separator="|"))
    )


def _mad(values: pl.Series) -> float | None:
    if values.is_empty() or values.null_count() == len(values):
        return None
    center = values.median()
    if center is None:
        return None
    return float((values - center).abs().median())


def _ridge_solve(
    matrix: list[list[float]], target: list[float], ridge: float
) -> list[float] | None:
    width = len(matrix[0])
    normal = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for row, observed in zip(matrix, target, strict=True):
        for left in range(width):
            rhs[left] += row[left] * observed
            for right in range(width):
                normal[left][right] += row[left] * row[right]
    for index in range(1, width):
        normal[index][index] += ridge
    augmented = [normal[index] + [rhs[index]] for index in range(width)]
    for pivot_index in range(width):
        pivot_row = max(range(pivot_index, width), key=lambda row: abs(augmented[row][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            return None
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]
        for row_index in range(width):
            if row_index == pivot_index:
                continue
            multiplier = augmented[row_index][pivot_index]
            augmented[row_index] = [
                value - multiplier * pivot_value
                for value, pivot_value in zip(
                    augmented[row_index], augmented[pivot_index], strict=True
                )
            ]
    return [augmented[index][-1] for index in range(width)]
