"""Configuration-driven factor-quality and strategy-performance experiments."""

from trend_trader.experiments.config import (
    ExperimentConfig,
    dump_experiment_config,
    load_experiment_config,
)
from trend_trader.experiments.factor import FactorExperimentConfig, FactorExperimentRunner
from trend_trader.experiments.runner import ExperimentResult, ExperimentRunner
from trend_trader.experiments.strategy import (
    StrategyExperimentConfig,
    StrategyExperimentRunner,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "FactorExperimentConfig",
    "FactorExperimentRunner",
    "StrategyExperimentConfig",
    "StrategyExperimentRunner",
    "dump_experiment_config",
    "load_experiment_config",
]
