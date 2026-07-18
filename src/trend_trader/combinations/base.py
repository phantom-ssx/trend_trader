"""Unified contracts and dataset helpers for multi-factor combinations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from trend_trader.research import ResearchDataset

KEYS = ["venue", "instrument_id", "bar_type", "timestamp"]
LABEL_COLUMNS = [
    *KEYS,
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
]


@dataclass(frozen=True, slots=True)
class FactorCombinationRequest:
    method: str
    factor_names: tuple[str, ...]
    name: str = "combined_factor"
    training_horizon: int = 1
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        method = self.method.strip().lower()
        name = self.name.strip()
        factors = tuple(dict.fromkeys(item.strip() for item in self.factor_names))
        params = dict(self.params)
        if not method or not name:
            raise ValueError("combination method and name must not be empty")
        if not factors or any(not item for item in factors):
            raise ValueError("combination requires at least one factor")
        if self.training_horizon <= 0:
            raise ValueError("training_horizon must be positive")
        try:
            json.dumps(params, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("combination params must be JSON serializable") from exc
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "factor_names", factors)
        object.__setattr__(self, "params", params)


@dataclass(slots=True)
class FactorCombinationResult:
    dataset: ResearchDataset
    weights: pl.DataFrame = field(default_factory=pl.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    model_bytes: bytes | None = None


class FactorCombiner(ABC):
    method: str
    version = "1"

    @abstractmethod
    def combine(
        self,
        dataset: ResearchDataset,
        request: FactorCombinationRequest,
    ) -> FactorCombinationResult:
        """Fit if necessary and return a point-in-time combined factor dataset."""


class FactorCombinationRegistry:
    def __init__(self) -> None:
        self._items: dict[str, FactorCombiner] = {}

    def register(self, combiner: FactorCombiner, *, replace: bool = False) -> None:
        method = combiner.method.strip().lower()
        if not method:
            raise ValueError("combination method must not be empty")
        if method in self._items and not replace:
            raise ValueError(f"combination method already registered: {method}")
        self._items[method] = combiner

    def get(self, method: str) -> FactorCombiner:
        normalized = method.strip().lower()
        try:
            return self._items[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self._items))
            raise KeyError(
                f"unknown combination method {normalized!r}; available: {available}"
            ) from exc

    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


def prepare_features(
    dataset: ResearchDataset,
    factor_names: tuple[str, ...],
) -> pl.DataFrame:
    frame = dataset.frame.filter(pl.col("factor_name").is_in(factor_names))
    available = set(frame["factor_name"].unique().to_list())
    missing = sorted(set(factor_names).difference(available))
    if missing:
        raise ValueError(f"research dataset does not contain factors: {missing}")
    unique = (
        frame.select(
            *KEYS,
            "factor_name",
            pl.when(pl.col("factor_is_valid")).then(pl.col("value")).alias("value"),
        )
        .unique(subset=[*KEYS, "factor_name"])
        .sort(*KEYS, "factor_name")
    )
    result = unique.pivot(on="factor_name", index=KEYS, values="value")
    return result.select(*KEYS, *factor_names).sort(*KEYS)


def prepare_target(dataset: ResearchDataset, horizon: int) -> pl.DataFrame:
    selected = dataset.frame.filter(pl.col("horizon_bars") == horizon)
    if selected.is_empty():
        raise ValueError(f"dataset does not contain training horizon {horizon}")
    return (
        selected.select(
            *KEYS,
            "exit_time",
            "label_value",
            "label_is_valid",
        )
        .unique(subset=KEYS)
        .sort("timestamp", "instrument_id")
    )


def assemble_combined_dataset(
    source: ResearchDataset,
    scores: pl.DataFrame,
    *,
    request: FactorCombinationRequest,
    version: str,
) -> ResearchDataset:
    required = {*KEYS, "raw_value", "factor_is_valid", "factor_quality_flags"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"combined scores are missing columns: {missing}")
    labels = source.frame.select(*LABEL_COLUMNS).unique(subset=[*KEYS, "label_name"])
    combined = labels.join(scores, on=KEYS, how="left").with_columns(
        pl.col("raw_value").alias("value"),
        pl.lit(request.name).alias("factor_name"),
        pl.lit(request.method).alias("factor_key"),
        pl.lit(version).alias("factor_version"),
    )
    combined = combined.with_columns(
        (
            pl.col("factor_is_valid")
            & pl.col("label_is_valid")
            & pl.col("value").is_not_null()
            & pl.col("value").is_finite()
        ).alias("is_valid")
    )
    return ResearchDataset(
        combined.select(
            *KEYS,
            "factor_name",
            "factor_key",
            "factor_version",
            "raw_value",
            "value",
            "factor_is_valid",
            "factor_quality_flags",
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
            "is_valid",
        ).sort("label_name", "instrument_id", "timestamp")
    )


def impute_frame(
    frame: pl.DataFrame,
    factor_names: tuple[str, ...],
    policy: str,
) -> pl.DataFrame:
    if policy == "drop":
        return frame
    if policy == "zero":
        return frame.with_columns(pl.col(name).fill_null(0.0) for name in factor_names)
    if policy == "cross_sectional_median":
        groups = ["venue", "bar_type", "timestamp"]
        return frame.with_columns(
            pl.col(name).fill_null(pl.col(name).median().over(groups)) for name in factor_names
        )
    raise ValueError("missing policy must be drop, zero, or cross_sectional_median")
