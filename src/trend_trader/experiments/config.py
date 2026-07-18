"""Strict YAML configuration models for reproducible factor experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trend_trader.data.models import as_utc, bar_minutes


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentMetaConfig(StrictModel):
    name: str
    description: str | None = None
    allow_dirty_git: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("experiment name must not be empty")
        return value


class UniverseExperimentConfig(StrictModel):
    mode: Literal["start_snapshot", "explicit"] = "start_snapshot"
    name: str = "experiment_universe"
    venue: str = "OKX"
    instruments: tuple[str, ...] = ()
    instrument_type: str = "SWAP"
    settle_currency: str = "USDT"
    contract_type: str = "linear"
    min_quote_volume_24h: float = 5_000_000
    min_open_interest_usd: float = 0
    min_listing_days: int = 90
    max_spread_bps: float | None = 50
    top_n_by_volume: int = 50

    @model_validator(mode="after")
    def validate_universe(self) -> UniverseExperimentConfig:
        if self.mode == "explicit" and not self.instruments:
            raise ValueError("explicit universe requires instruments")
        if self.min_quote_volume_24h < 0 or self.min_open_interest_usd < 0:
            raise ValueError("universe liquidity thresholds must not be negative")
        if self.min_listing_days < 0 or self.top_n_by_volume <= 0:
            raise ValueError("invalid universe size or listing-days threshold")
        normalized = tuple(dict.fromkeys(item.strip() for item in self.instruments))
        if any(not item for item in normalized):
            raise ValueError("universe instruments must not be empty")
        self.instruments = normalized
        self.venue = self.venue.upper()
        return self


class ExperimentDataConfig(StrictModel):
    start: datetime
    end: datetime
    timeframe: str = "1h"
    universe: UniverseExperimentConfig = Field(default_factory=UniverseExperimentConfig)

    @field_validator("start", "end", mode="before")
    @classmethod
    def parse_datetime(cls, value: object) -> datetime:
        if isinstance(value, str) and len(value.strip()) == 10:
            value = f"{value.strip()}T00:00:00Z"
        if not isinstance(value, (str, datetime)):
            raise TypeError("experiment dates must be ISO strings or datetimes")
        return as_utc(value)

    @field_validator("timeframe")
    @classmethod
    def normalize_timeframe(cls, value: str) -> str:
        minutes = bar_minutes(value)
        normalized = value.strip().lower()
        if minutes <= 0:
            raise ValueError("timeframe must be positive")
        return f"{int(normalized[:-1])}{normalized[-1]}"

    @model_validator(mode="after")
    def validate_interval(self) -> ExperimentDataConfig:
        if self.end <= self.start:
            raise ValueError("data.end must be after data.start")
        duration = bar_minutes(self.timeframe) * 60
        if int(self.start.timestamp()) % duration or int(self.end.timestamp()) % duration:
            raise ValueError("data start and end must align with timeframe boundaries")
        return self


class ExperimentFactorConfig(StrictModel):
    name: str
    alias: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("factor name must not be empty")
        return value

    @field_validator("alias")
    @classmethod
    def normalize_alias(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("factor alias must not be empty")
        return value

    @property
    def reference(self) -> str:
        return self.alias or self.name


class ExperimentCombinationConfig(StrictModel):
    method: Literal[
        "rule",
        "linear",
        "rank",
        "ic_weighted",
        "machine_learning",
        "deep_learning",
    ]
    name: str = "combined_signal"
    training_horizon: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("combination name must not be empty")
        return value

    @field_validator("training_horizon")
    @classmethod
    def validate_training_horizon(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("combination training_horizon must be positive")
        return value


class ExperimentLabelConfig(StrictModel):
    price: Literal["next_open"] = "next_open"
    horizons: tuple[int, ...] = (1, 4, 8, 24)

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        value = tuple(dict.fromkeys(value))
        if not value or any(item <= 0 for item in value):
            raise ValueError("label horizons must contain positive integers")
        return value


class WinsorizeExperimentConfig(StrictModel):
    lower: float = 0.01
    upper: float = 0.99
    scope: Literal["cross_sectional", "time_series"] = "cross_sectional"

    @model_validator(mode="after")
    def validate_limits(self) -> WinsorizeExperimentConfig:
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("winsorize limits must satisfy 0 <= lower < upper <= 1")
        return self


class ExperimentPreprocessConfig(StrictModel):
    winsorize: WinsorizeExperimentConfig | None = None
    normalize: Literal["none", "zscore", "robust_zscore", "rank"] = "none"
    normalize_scope: Literal["cross_sectional", "time_series"] = "cross_sectional"
    rolling_window: int = 60
    rolling_min_periods: int = 20
    neutralize: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_windows(self) -> ExperimentPreprocessConfig:
        if (
            self.rolling_window <= 0
            or self.rolling_min_periods <= 0
            or self.rolling_min_periods > self.rolling_window
        ):
            raise ValueError("invalid preprocessing rolling window")
        return self


class ExperimentEvaluationConfig(StrictModel):
    quantiles: int = 5
    ic_method: Literal["pearson", "spearman"] = "spearman"
    min_cross_section: int = 5
    stability_period: str = "1mo"
    stability_min_observations: int = 20
    primary_horizon: int | None = None

    @model_validator(mode="after")
    def validate_evaluation(self) -> ExperimentEvaluationConfig:
        if self.quantiles < 2 or self.min_cross_section < 2:
            raise ValueError("quantiles and min_cross_section must be at least 2")
        if self.stability_min_observations < 2:
            raise ValueError("stability_min_observations must be at least 2")
        return self


class ExperimentCostConfig(StrictModel):
    fee_bps: float = 5
    slippage_bps: float = 3

    @model_validator(mode="after")
    def validate_costs(self) -> ExperimentCostConfig:
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs must not be negative")
        return self

    @property
    def round_trip_bps(self) -> float:
        return 2 * (self.fee_bps + self.slippage_bps)


class ExperimentConfig:
    """Compatibility facade dispatching payloads to a typed experiment config."""

    @classmethod
    def model_validate(cls, payload: object) -> BaseModel:
        if not isinstance(payload, dict):
            raise TypeError("experiment config must be a mapping")
        if "combination" in payload or "factors" in payload:
            from trend_trader.experiments.strategy.config import StrategyExperimentConfig

            return StrategyExperimentConfig.model_validate(payload)
        from trend_trader.experiments.factor.config import FactorExperimentConfig

        return FactorExperimentConfig.model_validate(payload)


def load_experiment_config(path: Path | str) -> BaseModel:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("experiment YAML must contain a mapping at the top level")
    return ExperimentConfig.model_validate(payload)


def dump_experiment_config(config: BaseModel) -> str:
    return yaml.safe_dump(
        config.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
    )
