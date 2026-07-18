"""Multivariate factor redundancy and unique-contribution analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import polars as pl

from trend_trader.research.models import RedundancyAnalysisReport, ResearchDataset

CorrelationMethod = Literal["pearson", "spearman"]

_OBSERVATION_KEYS = ["venue", "instrument_id", "timestamp"]


class FactorRedundancyAnalyzer:
    def __init__(self, dataset: ResearchDataset | pl.DataFrame) -> None:
        self.frame = dataset.frame if isinstance(dataset, ResearchDataset) else dataset

    def vif(
        self,
        *,
        factors: Sequence[str] | None = None,
        min_observations: int = 20,
        ridge: float = 1e-8,
        moderate_threshold: float = 5.0,
        high_threshold: float = 10.0,
    ) -> pl.DataFrame:
        """Variance inflation factor for each factor against all other factors."""

        _validate_regression_options(min_observations, ridge)
        if not 1 < moderate_threshold < high_threshold:
            raise ValueError("VIF thresholds must satisfy 1 < moderate < high")
        wide = self._feature_wide()
        available = sorted(set(wide.columns).difference(_OBSERVATION_KEYS))
        selected = _select_factors(available, factors)
        rows: list[dict[str, object]] = []
        for target in selected:
            controls = [name for name in selected if name != target]
            complete = wide.select(target, *controls).drop_nulls()
            minimum = max(min_observations, len(controls) + 3)
            r_squared: float | None = None
            vif_value: float | None = None
            if complete.height >= minimum and controls:
                target_values = _standardize(_float_column(complete, target))
                control_values = [_standardize(_float_column(complete, name)) for name in controls]
                matrix = _columns_to_rows(control_values)
                residuals, r_squared = _regression_residuals(
                    target_values,
                    matrix,
                    ridge=ridge,
                )
                if residuals is not None and r_squared is not None:
                    vif_value = math.inf if r_squared >= 1 - 1e-12 else 1 / (1 - r_squared)
            if not controls and complete.height >= min_observations:
                r_squared = 0.0
                vif_value = 1.0
            if vif_value is None:
                status = "UNAVAILABLE"
            elif vif_value >= high_threshold:
                status = "HIGH"
            elif vif_value >= moderate_threshold:
                status = "MODERATE"
            else:
                status = "LOW"
            rows.append(
                {
                    "factor_name": target,
                    "controls": ",".join(controls),
                    "observations": complete.height,
                    "r_squared": r_squared,
                    "vif": vif_value,
                    "status": status,
                }
            )
        return pl.DataFrame(rows, infer_schema_length=None).sort("factor_name")

    def unique_contribution(
        self,
        *,
        target_factors: Sequence[str] | None = None,
        control_factors: Sequence[str] | None = None,
        method: CorrelationMethod = "spearman",
        min_observations: int = 5,
        ridge: float = 1e-8,
    ) -> pl.DataFrame:
        """Conditional IC and incremental R² after controlling for other factors."""

        _validate_regression_options(min_observations, ridge)
        if method not in {"pearson", "spearman"}:
            raise ValueError("method must be pearson or spearman")
        wide = self._labeled_wide()
        metadata = {
            *_OBSERVATION_KEYS,
            "label_name",
            "horizon_bars",
            "label_value",
        }
        available = sorted(set(wide.columns).difference(metadata))
        targets = _select_factors(available, target_factors)
        requested_controls = (
            _select_factors(available, control_factors) if control_factors is not None else None
        )
        rows: list[dict[str, object]] = []
        group_keys = ["venue", "label_name", "horizon_bars", "timestamp"]
        for group in wide.partition_by(group_keys, maintain_order=True):
            group_info = group.row(0, named=True)
            for target in targets:
                controls = (
                    [name for name in requested_controls if name != target]
                    if requested_controls is not None
                    else [name for name in available if name != target]
                )
                columns = [target, *controls, "label_value"]
                complete = group.select(*columns).drop_nulls()
                minimum = max(min_observations, len(controls) + 3)
                if complete.height < minimum:
                    continue
                target_values = _float_column(complete, target)
                label_values = _float_column(complete, "label_value")
                control_values = [_float_column(complete, name) for name in controls]
                if method == "spearman":
                    target_values = _average_ranks(target_values)
                    label_values = _average_ranks(label_values)
                    control_values = [_average_ranks(values) for values in control_values]
                target_values = _standardize(target_values)
                label_values = _standardize(label_values)
                control_values = [_standardize(values) for values in control_values]
                raw_ic = _pearson(target_values, label_values)
                control_matrix = _columns_to_rows(control_values)
                residuals, _ = _regression_residuals(
                    target_values,
                    control_matrix,
                    ridge=ridge,
                )
                conditional_ic = (
                    _pearson(residuals, label_values) if residuals is not None else None
                )
                baseline_r_squared = 0.0
                if controls:
                    _, baseline_r_squared = _regression_residuals(
                        label_values,
                        control_matrix,
                        ridge=ridge,
                    )
                full_matrix = _columns_to_rows([*control_values, target_values])
                _, full_r_squared = _regression_residuals(
                    label_values,
                    full_matrix,
                    ridge=ridge,
                )
                incremental_r_squared = (
                    max(0.0, full_r_squared - baseline_r_squared)
                    if full_r_squared is not None and baseline_r_squared is not None
                    else None
                )
                rows.append(
                    {
                        "venue": group_info["venue"],
                        "factor_name": target,
                        "controls": ",".join(controls),
                        "label_name": group_info["label_name"],
                        "horizon_bars": group_info["horizon_bars"],
                        "timestamp": group_info["timestamp"],
                        "observations": complete.height,
                        "raw_ic": raw_ic,
                        "conditional_ic": conditional_ic,
                        "baseline_r_squared": baseline_r_squared,
                        "full_r_squared": full_r_squared,
                        "incremental_r_squared": incremental_r_squared,
                        "method": method,
                    }
                )
        return _unique_contribution_frame(rows, self.frame.schema["timestamp"])

    @staticmethod
    def unique_contribution_summary(series: pl.DataFrame) -> pl.DataFrame:
        """Aggregate conditional IC and incremental R² through time."""

        keys = ["factor_name", "controls", "label_name", "horizon_bars", "method"]
        result = series.group_by(keys).agg(
            pl.len().alias("periods"),
            pl.col("raw_ic").mean().alias("mean_raw_ic"),
            pl.col("conditional_ic").mean().alias("mean_conditional_ic"),
            pl.col("conditional_ic").std().alias("std_conditional_ic"),
            (pl.col("conditional_ic") > 0).mean().alias("conditional_positive_rate"),
            pl.col("incremental_r_squared").mean().alias("mean_incremental_r_squared"),
            pl.col("incremental_r_squared").median().alias("median_incremental_r_squared"),
        )
        return result.with_columns(
            (pl.col("mean_conditional_ic") - pl.col("mean_raw_ic")).alias("mean_ic_change"),
            pl.when((pl.col("std_conditional_ic") > 0) & (pl.col("periods") > 1))
            .then(
                pl.col("mean_conditional_ic")
                / (pl.col("std_conditional_ic") / pl.col("periods").sqrt())
            )
            .alias("conditional_ic_t_stat"),
        ).sort(*keys)

    @staticmethod
    def clusters(
        pairwise_correlation: pl.DataFrame,
        overall_ic: pl.DataFrame,
        *,
        threshold: float = 0.8,
    ) -> pl.DataFrame:
        """Connected components of highly correlated factors."""

        if not 0 < threshold <= 1:
            raise ValueError("correlation threshold must be in (0, 1]")
        factors = sorted(
            set(pairwise_correlation.get_column("factor_left").to_list())
            | set(pairwise_correlation.get_column("factor_right").to_list())
        )
        adjacency = {factor: set() for factor in factors}
        correlation_lookup: dict[frozenset[str], float] = {}
        for row in pairwise_correlation.iter_rows(named=True):
            left = str(row["factor_left"])
            right = str(row["factor_right"])
            correlation = row["correlation"]
            if correlation is None or left == right:
                continue
            absolute = abs(float(correlation))
            correlation_lookup[frozenset((left, right))] = absolute
            if absolute >= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)
        scores = {
            str(row["factor_name"]): abs(float(row["mean_abs_ic"]))
            for row in overall_ic.group_by("factor_name")
            .agg(pl.col("ic").abs().mean().alias("mean_abs_ic"))
            .iter_rows(named=True)
            if row["mean_abs_ic"] is not None
        }
        components: list[list[str]] = []
        unseen = set(factors)
        while unseen:
            root = min(unseen)
            stack = [root]
            component: list[str] = []
            unseen.remove(root)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in sorted(adjacency[current]):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
            components.append(sorted(component))
        rows: list[dict[str, object]] = []
        for index, component in enumerate(components, start=1):
            representative = max(component, key=lambda name: (scores.get(name, -1.0), name))
            pair_correlations = [
                correlation_lookup[frozenset((left, right))]
                for left_index, left in enumerate(component)
                for right in component[left_index + 1 :]
                if frozenset((left, right)) in correlation_lookup
            ]
            for factor in component:
                rows.append(
                    {
                        "cluster_id": f"R{index}",
                        "factor_name": factor,
                        "cluster_size": len(component),
                        "representative": representative,
                        "is_representative": factor == representative,
                        "mean_abs_ic": scores.get(factor),
                        "max_abs_correlation": max(pair_correlations, default=None),
                        "threshold": threshold,
                    }
                )
        return pl.DataFrame(rows, infer_schema_length=None).sort("cluster_id", "factor_name")

    def run(
        self,
        pairwise_correlation: pl.DataFrame,
        overall_ic: pl.DataFrame,
        *,
        method: CorrelationMethod = "spearman",
        min_observations: int = 5,
        ridge: float = 1e-8,
        cluster_threshold: float = 0.8,
        target_factors: Sequence[str] | None = None,
        control_factors: Sequence[str] | None = None,
    ) -> RedundancyAnalysisReport:
        contributions = self.unique_contribution(
            target_factors=target_factors,
            control_factors=control_factors,
            method=method,
            min_observations=min_observations,
            ridge=ridge,
        )
        return RedundancyAnalysisReport(
            pairwise_correlation=pairwise_correlation,
            vif=self.vif(
                factors=None,
                min_observations=min_observations,
                ridge=ridge,
            ),
            unique_contribution=contributions,
            unique_contribution_summary=self.unique_contribution_summary(contributions),
            clusters=self.clusters(
                pairwise_correlation,
                overall_ic,
                threshold=cluster_threshold,
            ),
        )

    def _feature_wide(self) -> pl.DataFrame:
        unique = (
            self.frame.filter(pl.col("factor_is_valid"))
            .select(
                *_OBSERVATION_KEYS,
                "factor_name",
                "value",
            )
            .unique(subset=[*_OBSERVATION_KEYS, "factor_name"])
        )
        return unique.pivot(on="factor_name", index=_OBSERVATION_KEYS, values="value")

    def _labeled_wide(self) -> pl.DataFrame:
        keys = [*_OBSERVATION_KEYS, "label_name", "horizon_bars", "label_value"]
        unique = (
            self.frame.filter(pl.col("is_valid"))
            .select(
                *keys,
                "factor_name",
                "value",
            )
            .unique(subset=[*_OBSERVATION_KEYS, "label_name", "factor_name"])
        )
        return unique.pivot(on="factor_name", index=keys, values="value").sort(
            "label_name", "timestamp", "instrument_id"
        )


def _select_factors(available: list[str], requested: Sequence[str] | None) -> list[str]:
    if requested is None:
        return available
    selected = list(dict.fromkeys(str(name) for name in requested))
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(f"unknown factors in redundancy analysis: {missing}")
    if not selected:
        raise ValueError("at least one factor must be selected")
    return selected


def _validate_regression_options(min_observations: int, ridge: float) -> None:
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")
    if ridge < 0:
        raise ValueError("ridge must not be negative")


def _float_column(frame: pl.DataFrame, name: str) -> list[float]:
    return [float(value) for value in frame.get_column(name).to_list()]


def _columns_to_rows(columns: list[list[float]]) -> list[list[float]]:
    if not columns:
        return []
    return [list(row) for row in zip(*columns, strict=True)]


def _standardize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= 1e-24:
        return [0.0] * len(values)
    scale = variance**0.5
    return [(value - mean) / scale for value in values]


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[ordered[position][0]] = average
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss <= 1e-16 or right_ss <= 1e-16:
        return None
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    return covariance / (left_ss * right_ss) ** 0.5


def _regression_residuals(
    target: list[float],
    features: list[list[float]],
    *,
    ridge: float,
) -> tuple[list[float] | None, float | None]:
    if not target:
        return None, None
    if not features:
        mean = sum(target) / len(target)
        residuals = [value - mean for value in target]
        return residuals, 0.0
    matrix = [[1.0, *row] for row in features]
    coefficients = _ridge_solve(matrix, target, ridge)
    if coefficients is None:
        return None, None
    fitted = [
        sum(value * coefficient for value, coefficient in zip(row, coefficients, strict=True))
        for row in matrix
    ]
    residuals = [observed - estimate for observed, estimate in zip(target, fitted, strict=True)]
    mean = sum(target) / len(target)
    total = sum((value - mean) ** 2 for value in target)
    residual = sum(value**2 for value in residuals)
    r_squared = 1 - residual / total if total > 1e-24 else None
    if r_squared is not None:
        r_squared = min(1.0, max(0.0, r_squared))
    return residuals, r_squared


def _ridge_solve(
    matrix: list[list[float]],
    target: list[float],
    ridge: float,
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
        pivot_row = max(
            range(pivot_index, width),
            key=lambda row: abs(augmented[row][pivot_index]),
        )
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            return None
        augmented[pivot_index], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot_index],
        )
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


def _unique_contribution_frame(
    rows: list[dict[str, object]],
    timestamp_dtype: pl.DataType,
) -> pl.DataFrame:
    if rows:
        return pl.DataFrame(rows, infer_schema_length=None).sort(
            "factor_name", "label_name", "timestamp"
        )
    return pl.DataFrame(
        schema={
            "venue": pl.Utf8,
            "factor_name": pl.Utf8,
            "controls": pl.Utf8,
            "label_name": pl.Utf8,
            "horizon_bars": pl.Int32,
            "timestamp": timestamp_dtype,
            "observations": pl.Int64,
            "raw_ic": pl.Float64,
            "conditional_ic": pl.Float64,
            "baseline_r_squared": pl.Float64,
            "full_r_squared": pl.Float64,
            "incremental_r_squared": pl.Float64,
            "method": pl.Utf8,
        }
    )
