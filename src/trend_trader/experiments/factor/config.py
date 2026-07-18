"""Configuration contract for a single-factor quality experiment."""

from __future__ import annotations

from pydantic import Field, model_validator

from trend_trader.experiments.config import (
    ExperimentDataConfig,
    ExperimentEvaluationConfig,
    ExperimentFactorConfig,
    ExperimentLabelConfig,
    ExperimentMetaConfig,
    ExperimentPreprocessConfig,
    StrictModel,
)


class FactorExperimentConfig(StrictModel):
    experiment: ExperimentMetaConfig
    data: ExperimentDataConfig
    factor: ExperimentFactorConfig
    label: ExperimentLabelConfig = Field(default_factory=ExperimentLabelConfig)
    preprocess: ExperimentPreprocessConfig = Field(default_factory=ExperimentPreprocessConfig)
    evaluation: ExperimentEvaluationConfig = Field(default_factory=ExperimentEvaluationConfig)

    @model_validator(mode="after")
    def validate_horizons(self) -> FactorExperimentConfig:
        primary = self.evaluation.primary_horizon
        if primary is not None and primary not in self.label.horizons:
            raise ValueError("evaluation.primary_horizon must be one of label.horizons")
        return self

    @property
    def primary_horizon(self) -> int:
        return self.evaluation.primary_horizon or self.label.horizons[0]

    @property
    def factor_configs(self) -> tuple[ExperimentFactorConfig, ...]:
        return (self.factor,)
