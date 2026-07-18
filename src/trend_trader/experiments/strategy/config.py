"""Configuration contract for a multi-factor strategy experiment."""

from __future__ import annotations

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


class StrategyExperimentConfig(StrictModel):
    experiment: ExperimentMetaConfig
    data: ExperimentDataConfig
    factors: tuple[ExperimentFactorConfig, ...]
    combination: ExperimentCombinationConfig
    label: ExperimentLabelConfig = Field(default_factory=ExperimentLabelConfig)
    preprocess: ExperimentPreprocessConfig = Field(default_factory=ExperimentPreprocessConfig)
    evaluation: ExperimentEvaluationConfig = Field(default_factory=ExperimentEvaluationConfig)
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
