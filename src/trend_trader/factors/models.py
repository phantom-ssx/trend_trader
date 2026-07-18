"""Public models for factor calculation and post-processing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import polars as pl

from trend_trader.data.models import DataType, as_utc, bar_minutes, normalize_bar_type

ProcessingScope = Literal["cross_sectional", "time_series"]


@dataclass(frozen=True, slots=True)
class FactorSpec:
    """A registered factor name plus serializable parameters."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("factor name must not be empty")
        params = dict(self.params)
        if any(not isinstance(key, str) for key in params):
            raise TypeError("factor parameter names must be strings")
        try:
            json.dumps(params, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("factor parameters must be JSON serializable") from exc
        object.__setattr__(self, "name", self.name.strip().lower())
        object.__setattr__(self, "params", params)


@dataclass(frozen=True, slots=True)
class DataDependency:
    """A market dataset required by a factor."""

    data_type: DataType


@dataclass(frozen=True, slots=True)
class OutlierConfig:
    method: Literal["none", "mad", "winsorize", "std"] = "none"
    scope: ProcessingScope = "cross_sectional"
    threshold: float = 5.0
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    window: int = 60
    min_periods: int = 20

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("outlier threshold must be positive")
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("outlier quantiles must satisfy 0 <= lower < upper <= 1")
        if self.window <= 0 or self.min_periods <= 0 or self.min_periods > self.window:
            raise ValueError("invalid outlier rolling window")


@dataclass(frozen=True, slots=True)
class StandardizeConfig:
    method: Literal["none", "zscore", "robust_zscore", "rank"] = "none"
    scope: ProcessingScope = "cross_sectional"
    window: int = 60
    min_periods: int = 20
    min_cross_section: int = 5

    def __post_init__(self) -> None:
        if self.window <= 0 or self.min_periods <= 0 or self.min_periods > self.window:
            raise ValueError("invalid standardization rolling window")
        if self.min_cross_section < 2:
            raise ValueError("min_cross_section must be at least 2")


@dataclass(frozen=True, slots=True)
class NeutralizeConfig:
    exposures: tuple[str, ...] = ()
    min_observations: int = 5
    ridge: float = 1e-8

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(item.strip().lower() for item in self.exposures))
        if any(not item for item in normalized):
            raise ValueError("neutralization exposure names must not be empty")
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2")
        if self.ridge < 0:
            raise ValueError("ridge must not be negative")
        object.__setattr__(self, "exposures", normalized)


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    standardize: StandardizeConfig = field(default_factory=StandardizeConfig)
    neutralize: NeutralizeConfig = field(default_factory=NeutralizeConfig)


@dataclass(frozen=True, slots=True)
class FactorRequest:
    factors: tuple[FactorSpec, ...]
    instrument_ids: tuple[str, ...]
    start: datetime | str
    end: datetime | str
    bar_type: str
    venue: str = "OKX"
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    def __post_init__(self) -> None:
        factors = tuple(
            factor if isinstance(factor, FactorSpec) else FactorSpec(str(factor))
            for factor in self.factors
        )
        instruments = tuple(dict.fromkeys(item.strip() for item in self.instrument_ids))
        start = as_utc(self.start)
        end = as_utc(self.end)
        normalized_bar_type = normalize_bar_type(self.bar_type)
        if not factors:
            raise ValueError("at least one factor is required")
        if not instruments or any(not item for item in instruments):
            raise ValueError("at least one non-empty instrument_id is required")
        if end <= start:
            raise ValueError("end must be after start")
        duration_seconds = bar_minutes(normalized_bar_type) * 60
        if int(start.timestamp()) % duration_seconds or int(end.timestamp()) % duration_seconds:
            raise ValueError("start and end must align with bar_type boundaries in UTC")
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "instrument_ids", instruments)
        object.__setattr__(self, "start", start.astimezone(UTC))
        object.__setattr__(self, "end", end.astimezone(UTC))
        object.__setattr__(self, "bar_type", normalized_bar_type)
        object.__setattr__(self, "venue", self.venue.upper())


@dataclass(slots=True)
class FactorResult:
    """Long-form factor output with a convenience wide view."""

    frame: pl.DataFrame

    def to_wide(self, *, value_column: str = "value") -> pl.DataFrame:
        if value_column not in {"value", "raw_value"}:
            raise ValueError("value_column must be 'value' or 'raw_value'")
        keys = ["venue", "instrument_id", "bar_type", "timestamp"]
        if self.frame.is_empty():
            return self.frame.select(*keys).unique()
        return (
            self.frame.select(*keys, "factor_name", value_column)
            .pivot(on="factor_name", index=keys, values=value_column)
            .sort("instrument_id", "timestamp")
        )

    def __len__(self) -> int:
        return self.frame.height


def factor_request(
    factors: Sequence[FactorSpec | str],
    instrument_ids: Sequence[str],
    start: datetime | str,
    end: datetime | str,
    bar_type: str,
    *,
    venue: str = "OKX",
    processing: ProcessingConfig | None = None,
) -> FactorRequest:
    """Convenience constructor accepting ordinary sequences."""

    return FactorRequest(
        factors=tuple(
            item if isinstance(item, FactorSpec) else FactorSpec(item) for item in factors
        ),
        instrument_ids=tuple(instrument_ids),
        start=start,
        end=end,
        bar_type=bar_type,
        venue=venue,
        processing=processing or ProcessingConfig(),
    )
