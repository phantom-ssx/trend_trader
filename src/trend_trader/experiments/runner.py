"""Compatibility dispatcher for typed factor and strategy experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trend_trader.data import MarketDataClient
from trend_trader.experiments.common import ExperimentResult
from trend_trader.experiments.config import load_experiment_config
from trend_trader.experiments.factor.config import FactorExperimentConfig
from trend_trader.experiments.factor.runner import FactorExperimentRunner
from trend_trader.experiments.strategy.config import StrategyExperimentConfig
from trend_trader.experiments.strategy.runner import StrategyExperimentRunner


class ExperimentRunner:
    """Dispatch legacy callers to the explicit experiment pipeline."""

    def __init__(
        self,
        data: MarketDataClient | None = None,
        *,
        output_root: Path | str = "experiments",
        workdir: Path | str = ".",
    ) -> None:
        self.data = data or MarketDataClient()
        self.output_root = Path(output_root)
        self.workdir = Path(workdir)

    def run(
        self,
        config: FactorExperimentConfig | StrategyExperimentConfig,
    ) -> ExperimentResult:
        if isinstance(config, FactorExperimentConfig):
            return FactorExperimentRunner(
                self.data,
                output_root=self.output_root,
                workdir=self.workdir,
            ).run(config)
        if isinstance(config, StrategyExperimentConfig):
            return StrategyExperimentRunner(
                self.data,
                output_root=self.output_root,
                workdir=self.workdir,
            ).run(config)
        raise TypeError(f"unsupported experiment config: {type(config).__name__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dispatch a factor or strategy experiment from its YAML shape"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--workdir", type=Path, default=Path("."))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = ExperimentRunner(
        MarketDataClient(data_root=args.data_root),
        output_root=args.output_dir,
        workdir=args.workdir,
    ).run(load_experiment_config(args.config))
    print(
        json.dumps(
            {
                "experiment_id": result.experiment_id,
                "experiment_type": result.summary["experiment_type"],
                "artifact_path": str(result.artifact_path),
                "primary_metrics": result.summary["primary_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
