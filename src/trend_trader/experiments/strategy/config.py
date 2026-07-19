"""Configuration contract for a multi-factor strategy experiment."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from trend_trader.experiments.config import (
    ExperimentCombinationConfig,
    ExperimentCostConfig,
    ExperimentDataConfig,
    ExperimentEvaluationConfig,
    ExperimentFactorConfig,
    ExperimentLabelConfig,
    ExperimentMetaConfig,
    ExperimentPreprocessConfig,
    StrictModel,
)


class StrategyPortfolioConfig(StrictModel):
    """Construction rules applied after the factors have been combined."""

    mode: Literal["long_short", "long_only", "short_only", "time_series_threshold"] = "long_short"
    long_threshold_bps: float = 0.0
    short_threshold_bps: float | None = None
    signal_multiplier: Literal[-1.0, 1.0] = 1.0
    signal_smoothing_periods: int = 1
    signal_standardization_periods: int | None = None
    signal_standardization_min_periods: int | None = None
    long_threshold_zscore: float | None = None
    short_threshold_zscore: float | None = None
    long_trend_filter_bars: int | None = None
    long_trend_min_return_bps: float = 0.0
    position_size: float = 1.0

    @model_validator(mode="after")
    def validate_thresholds(self) -> StrategyPortfolioConfig:
        if self.long_threshold_bps < 0 or (
            self.short_threshold_bps is not None and self.short_threshold_bps < 0
        ):
            raise ValueError("portfolio signal thresholds must not be negative")
        if self.signal_smoothing_periods < 1:
            raise ValueError("signal_smoothing_periods must be at least 1")
        if self.signal_standardization_periods is not None:
            minimum = self.signal_standardization_min_periods
            if self.signal_standardization_periods < 2:
                raise ValueError("signal_standardization_periods must be at least 2")
            if minimum is not None and not 2 <= minimum <= self.signal_standardization_periods:
                raise ValueError("invalid signal_standardization_min_periods")
            if self.long_threshold_zscore is None:
                raise ValueError("standardized signals require long_threshold_zscore")
        elif self.long_threshold_zscore is not None or self.short_threshold_zscore is not None:
            raise ValueError("z-score thresholds require signal_standardization_periods")
        if (self.long_threshold_zscore is not None and self.long_threshold_zscore < 0) or (
            self.short_threshold_zscore is not None and self.short_threshold_zscore < 0
        ):
            raise ValueError("portfolio z-score thresholds must not be negative")
        if self.long_trend_filter_bars is not None and self.long_trend_filter_bars < 1:
            raise ValueError("long_trend_filter_bars must be at least 1")
        if not 0 < self.position_size <= 1:
            raise ValueError("position_size must be in (0, 1]")
        return self


class StrategyExperimentConfig(StrictModel):
    experiment: ExperimentMetaConfig
    data: ExperimentDataConfig
    factors: tuple[ExperimentFactorConfig, ...]
    combination: ExperimentCombinationConfig
    label: ExperimentLabelConfig = Field(default_factory=ExperimentLabelConfig)
    preprocess: ExperimentPreprocessConfig = Field(default_factory=ExperimentPreprocessConfig)
    evaluation: ExperimentEvaluationConfig = Field(default_factory=ExperimentEvaluationConfig)
    portfolio: StrategyPortfolioConfig = Field(default_factory=StrategyPortfolioConfig)
    cost: ExperimentCostConfig = Field(default_factory=ExperimentCostConfig)

    @model_validator(mode="after")
    def validate_strategy(self) -> StrategyExperimentConfig:
        if len(self.factors) < 2:
            raise ValueError("strategy experiment requires at least two factors")
        references = [item.reference for item in self.factors]
        if len(references) != len(set(references)):
            raise ValueError("factor aliases must be unique")
        primary = self.evaluation.primary_horizon
        if primary is not None and primary not in self.label.horizons:
            raise ValueError("evaluation.primary_horizon must be one of label.horizons")
        training = self.combination.training_horizon
        if training is not None and training not in self.label.horizons:
            raise ValueError("combination.training_horizon must be one of label.horizons")
        if (
            self.portfolio.mode == "time_series_threshold"
            and self.evaluation.scope != "time_series"
        ):
            raise ValueError("time_series_threshold portfolio requires time_series evaluation")
        return self

    @property
    def primary_horizon(self) -> int:
        return self.evaluation.primary_horizon or self.label.horizons[0]

    @property
    def factor_configs(self) -> tuple[ExperimentFactorConfig, ...]:
        return self.factors

    @property
    def combination_training_horizon(self) -> int:
        return self.combination.training_horizon or self.primary_horizon
