"""Leakage-safe rolling IC-weighted factor combination."""

from __future__ import annotations

import math
from collections import deque
from datetime import timedelta
from typing import Any

import polars as pl

from trend_trader.combinations.base import (
    KEYS,
    FactorCombinationRequest,
    FactorCombinationResult,
    FactorCombiner,
    assemble_combined_dataset,
    prepare_features,
)
from trend_trader.data.models import bar_minutes
from trend_trader.research import ResearchDataset


class IcWeightedCombiner(FactorCombiner):
    method = "ic_weighted"

    def combine(
        self, dataset: ResearchDataset, request: FactorCombinationRequest
    ) -> FactorCombinationResult:
        params = request.params
        window = int(params.get("window", 60))
        min_periods = int(params.get("min_periods", 20))
        min_cross_section = int(params.get("min_cross_section", 5))
        min_factors = int(params.get("min_factors", len(request.factor_names)))
        method = str(params.get("ic_method", "spearman")).lower()
        normalization = str(params.get("normalization", "sum_abs")).lower()
        missing = str(params.get("missing", "renormalize")).lower()
        half_life = params.get("half_life")
        clip = float(params.get("clip", 1.0))
        label_lag_bars = int(params.get("label_lag_bars", 1))
        if window <= 0 or min_periods <= 0 or min_periods > window:
            raise ValueError("invalid IC rolling window")
        if min_cross_section < 2 or not 1 <= min_factors <= len(request.factor_names):
            raise ValueError("invalid IC minimum cross-section or factor count")
        if method not in {"pearson", "spearman"}:
            raise ValueError("IC method must be pearson or spearman")
        if missing not in {"drop", "zero", "renormalize"}:
            raise ValueError("IC missing policy must be drop, zero, or renormalize")
        if clip <= 0 or label_lag_bars < 1:
            raise ValueError("IC clip must be positive and label_lag_bars must be at least 1")

        history = self._ic_history(dataset, request, method, min_cross_section)
        features = prepare_features(dataset, request.factor_names)
        bar_types = features["bar_type"].unique().to_list()
        if len(bar_types) != 1:
            raise ValueError("IC combination requires exactly one bar_type")
        availability_lag = timedelta(minutes=bar_minutes(str(bar_types[0])) * label_lag_bars)
        history_by_factor: dict[str, list[tuple[object, object, float]]] = {
            name: [] for name in request.factor_names
        }
        for row in history.iter_rows(named=True):
            history_by_factor[str(row["factor_name"])].append(
                (
                    row["exit_time"] + availability_lag,
                    row["exit_time"],
                    float(row["ic"]),
                )
            )
        state: dict[str, dict[str, Any]] = {
            name: {
                "cursor": 0,
                "values": deque(maxlen=window),
                "latest_exit": None,
            }
            for name in request.factor_names
        }
        score_rows: list[dict[str, object]] = []
        weight_rows: list[dict[str, object]] = []
        for cross_section in features.partition_by("timestamp", maintain_order=True):
            timestamp = cross_section["timestamp"][0]
            estimates: dict[str, float] = {}
            for factor_name in request.factor_names:
                observations = history_by_factor[factor_name]
                factor_state = state[factor_name]
                cursor = int(factor_state["cursor"])
                values = factor_state["values"]
                while cursor < len(observations) and observations[cursor][0] <= timestamp:
                    _, exit_time, ic = observations[cursor]
                    values.append(ic)
                    factor_state["latest_exit"] = exit_time
                    cursor += 1
                factor_state["cursor"] = cursor
                if len(values) < min_periods:
                    continue
                estimate = _weighted_average(list(values), half_life)
                estimates[factor_name] = max(-clip, min(clip, estimate))
            weights = _normalize(estimates, normalization, params)
            if len(weights) < min_factors:
                score_rows.extend(_invalid_rows(cross_section, "IC_WARMUP_INCOMPLETE"))
                continue
            for factor_name, weight in weights.items():
                factor_state = state[factor_name]
                weight_rows.append(
                    {
                        "timestamp": timestamp,
                        "factor_name": factor_name,
                        "weight": weight,
                        "rolling_ic": estimates[factor_name],
                        "ic_observations": len(factor_state["values"]),
                        "latest_label_exit_time": factor_state["latest_exit"],
                    }
                )
            score_rows.extend(
                _weighted_scores(cross_section, request.factor_names, weights, missing)
            )
        scores = pl.DataFrame(score_rows, infer_schema_length=None)
        weights_frame = (
            pl.DataFrame(weight_rows, infer_schema_length=None)
            if weight_rows
            else pl.DataFrame(
                schema={
                    "timestamp": features.schema["timestamp"],
                    "factor_name": pl.Utf8,
                    "weight": pl.Float64,
                    "rolling_ic": pl.Float64,
                    "ic_observations": pl.Int64,
                    "latest_label_exit_time": features.schema["timestamp"],
                }
            )
        )
        return FactorCombinationResult(
            assemble_combined_dataset(dataset, scores, request=request, version=self.version),
            weights=weights_frame,
            diagnostics={
                "method": self.method,
                "training_horizon": request.training_horizon,
                "window": window,
                "min_periods": min_periods,
                "min_cross_section": min_cross_section,
                "normalization": normalization,
                "label_lag_bars": label_lag_bars,
                "leakage_guard": (
                    "only IC observations with exit_time + label lag <= prediction timestamp"
                ),
            },
        )

    @staticmethod
    def _ic_history(
        dataset: ResearchDataset,
        request: FactorCombinationRequest,
        method: str,
        min_cross_section: int,
    ) -> pl.DataFrame:
        keys = ["factor_name", "timestamp"]
        return (
            dataset.frame.filter(
                pl.col("factor_name").is_in(request.factor_names)
                & (pl.col("horizon_bars") == request.training_horizon)
                & pl.col("is_valid")
                & pl.col("value").is_not_null()
                & pl.col("label_value").is_not_null()
            )
            .group_by(keys)
            .agg(
                pl.len().alias("observations"),
                pl.col("exit_time").max().alias("exit_time"),
                pl.corr("value", "label_value", method=method).alias("ic"),
            )
            .filter(
                (pl.col("observations") >= min_cross_section)
                & pl.col("ic").is_not_null()
                & pl.col("ic").is_finite()
            )
            .sort("factor_name", "exit_time")
        )


def _weighted_average(values: list[float], half_life: object) -> float:
    if half_life is None:
        return sum(values) / len(values)
    half_life_value = float(half_life)
    if half_life_value <= 0:
        raise ValueError("IC half_life must be positive")
    weights = [0.5 ** ((len(values) - 1 - index) / half_life_value) for index in range(len(values))]
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / sum(weights)


def _normalize(
    estimates: dict[str, float],
    method: str,
    params: dict[str, Any],
) -> dict[str, float]:
    if not estimates:
        return {}
    if method == "sum_abs":
        denominator = sum(abs(value) for value in estimates.values())
        return (
            {name: value / denominator for name, value in estimates.items()}
            if denominator > 0
            else {}
        )
    if method == "equal_sign":
        active = {name: math.copysign(1.0, value) for name, value in estimates.items() if value}
        return {name: value / len(active) for name, value in active.items()} if active else {}
    if method == "long_only":
        active = {name: max(0.0, value) for name, value in estimates.items()}
        denominator = sum(active.values())
        return {name: value / denominator for name, value in active.items()} if denominator else {}
    if method == "softmax":
        temperature = float(params.get("temperature", 0.1))
        if temperature <= 0:
            raise ValueError("IC softmax temperature must be positive")
        maximum = max(estimates.values())
        active = {
            name: math.exp((value - maximum) / temperature) for name, value in estimates.items()
        }
        denominator = sum(active.values())
        return {name: value / denominator for name, value in active.items()}
    raise ValueError("IC normalization must be sum_abs, equal_sign, long_only, or softmax")


def _weighted_scores(
    frame: pl.DataFrame,
    factor_names: tuple[str, ...],
    weights: dict[str, float],
    missing: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in frame.iter_rows(named=True):
        available = {
            name: weights[name]
            for name in factor_names
            if name in weights and item[name] is not None and math.isfinite(float(item[name]))
        }
        if missing == "drop" and len(available) != len(weights):
            score = None
        elif missing == "zero":
            score = sum(
                (float(item[name]) if name in available else 0.0) * weight
                for name, weight in weights.items()
            )
        else:
            denominator = sum(abs(value) for value in available.values())
            score = (
                sum(float(item[name]) * weight for name, weight in available.items()) / denominator
                if denominator
                else None
            )
        rows.append(
            {
                **{key: item[key] for key in KEYS},
                "raw_value": score,
                "factor_is_valid": score is not None,
                "factor_quality_flags": "" if score is not None else "MISSING_COMPONENT_FACTOR",
            }
        )
    return rows


def _invalid_rows(frame: pl.DataFrame, flag: str) -> list[dict[str, object]]:
    return [
        {
            **{key: item[key] for key in KEYS},
            "raw_value": None,
            "factor_is_valid": False,
            "factor_quality_flags": flag,
        }
        for item in frame.iter_rows(named=True)
    ]
