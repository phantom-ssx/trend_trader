"""Execution pipeline for single-factor predictive-quality experiments."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from trend_trader.data import MarketDataClient
from trend_trader.experiments.common import (
    ExperimentResult,
    build_component_research,
    fingerprint_experiment_data,
    first_float,
    prepare_experiment,
)
from trend_trader.experiments.factor.config import FactorExperimentConfig
from trend_trader.experiments.factor.report import render_factor_report
from trend_trader.experiments.storage import ExperimentRepository, base_record, write_config
from trend_trader.research import FactorAnalyzer


class FactorExperimentRunner:
    """Evaluate whether one factor predicts future returns reliably."""

    def __init__(
        self,
        data: MarketDataClient | None = None,
        *,
        output_root: Path | str = "experiments",
        workdir: Path | str = ".",
    ) -> None:
        self.data = data or MarketDataClient()
        self.repository = ExperimentRepository(output_root)
        self.workdir = Path(workdir).resolve()

    def run(self, config: FactorExperimentConfig) -> ExperimentResult:
        created_at = datetime.now(tz=UTC)
        experiment_id = self.repository.new_experiment_id(
            f"factor_{config.experiment.name}", created_at
        )
        artifacts = self.repository.artifacts(experiment_id)
        try:
            prepared = prepare_experiment(self.data, config, self.workdir)
            dataset = build_component_research(self.data, config, prepared)
            analysis = FactorAnalyzer(dataset).run(
                method=config.evaluation.ic_method,
                min_cross_section=config.evaluation.min_cross_section,
                quantiles=config.evaluation.quantiles,
                stability_period=config.evaluation.stability_period,
                stability_min_observations=config.evaluation.stability_min_observations,
            )
            data_version, data_manifest = fingerprint_experiment_data(self.data, config, prepared)
            primary = _factor_primary_metrics(
                analysis,
                horizon=config.primary_horizon,
            )
            version = prepared.factor_versions[0]
            summary: dict[str, Any] = {
                "experiment_id": experiment_id,
                "experiment_type": "factor",
                "research_objective": "predictive_quality",
                "name": config.experiment.name,
                "created_at": created_at.isoformat(),
                "status": "completed",
                "git_commit": prepared.git_commit,
                "git_dirty": prepared.git_dirty,
                "factor": {**version, "params": config.factor.params},
                "data_version": data_version,
                "data_range": {
                    "start": config.data.start.isoformat(),
                    "end": config.data.end.isoformat(),
                    "timeframe": config.data.timeframe,
                },
                "universe": {
                    "rules": config.data.universe.model_dump(mode="json"),
                    "instrument_count": len(prepared.instruments),
                    "instrument_ids": list(prepared.instruments),
                    "selection_time": config.data.start.isoformat(),
                },
                "label": {
                    **config.label.model_dump(mode="json"),
                    "definition": (
                        "gross open(T+horizon)/open(T)-1; T is the first open after signal close"
                    ),
                    "transaction_costs_applied": False,
                },
                "primary_horizon": config.primary_horizon,
                "primary_metrics": primary,
                "interpretation": {
                    "focus": "IC level, stability, coverage, monotonicity, and horizon decay",
                    "not_strategy_performance": (
                        "quantile spread is a factor diagnostic, not a tradable strategy PnL"
                    ),
                },
            }

            write_config(artifacts, config)
            artifacts.write_json("summary.json", summary)
            artifacts.write_json("data_manifest.json", data_manifest)
            artifacts.write_csv("universe.csv", prepared.universe)
            artifacts.write_csv("dataset_summary.csv", analysis.summary)
            artifacts.write_csv("ic.csv", analysis.ic_series)
            artifacts.write_csv("ic_summary.csv", analysis.ic_summary)
            artifacts.write_csv("overall_ic.csv", analysis.overall_ic)
            artifacts.write_csv("factor_returns.csv", analysis.factor_returns)
            artifacts.write_csv("factor_return_summary.csv", analysis.factor_return_summary)
            artifacts.write_csv("quantile_returns.csv", analysis.quantile_returns)
            artifacts.write_csv("quantile_spread.csv", analysis.quantile_spread)
            artifacts.write_csv("periodic_ic.csv", analysis.periodic_ic)
            artifacts.write_csv("decay.csv", analysis.decay)
            artifacts.write_csv("autocorrelation.csv", analysis.autocorrelation)
            artifacts.write_text(
                "report.html",
                render_factor_report(
                    summary,
                    ic_summary=analysis.ic_summary,
                    overall_ic=analysis.overall_ic,
                    quantile_returns=analysis.quantile_returns,
                    periodic_ic=analysis.periodic_ic,
                    decay=analysis.decay,
                ),
            )
            final_path = artifacts.publish()
            record = base_record(
                experiment_id=experiment_id,
                experiment_type="factor",
                config=config,
                created_at=created_at,
                git_commit=prepared.git_commit,
                git_dirty=prepared.git_dirty,
                factor_version=version["version"],
                factor_params=[config.factor.model_dump(mode="json")],
                data_version=data_version,
                artifact_path=final_path,
            )
            record.update(
                {
                    "mean_ic": primary["mean_ic"],
                    "ic_ir": primary["ic_ir"],
                }
            )
            self.repository.save(record)
            return ExperimentResult(experiment_id, final_path, summary)
        except Exception:
            artifacts.discard()
            raise


def _factor_primary_metrics(analysis: Any, *, horizon: int) -> dict[str, float | None]:
    ic = analysis.ic_summary.filter(pl.col("horizon_bars") == horizon)
    coverage = analysis.summary.filter(pl.col("horizon_bars") == horizon)
    spread = analysis.quantile_spread.filter(pl.col("horizon_bars") == horizon)
    return {
        "mean_ic": first_float(ic, "mean_ic"),
        "ic_ir": first_float(ic, "icir"),
        "ic_t_stat": first_float(ic, "t_stat"),
        "positive_ic_rate": first_float(ic, "positive_rate"),
        "coverage": first_float(coverage, "coverage"),
        "quantile_spread": first_float(spread, "long_short_return"),
        "monotonicity": first_float(spread, "monotonicity"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single-factor quality experiment")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--workdir", type=Path, default=Path("."))
    return parser


def main() -> None:
    from trend_trader.experiments.config import load_experiment_config

    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    if not isinstance(config, FactorExperimentConfig):
        raise TypeError("factor experiment CLI requires a single-factor config")
    result = FactorExperimentRunner(
        MarketDataClient(data_root=args.data_root),
        output_root=args.output_dir,
        workdir=args.workdir,
    ).run(config)
    print(
        json.dumps(
            {
                "experiment_id": result.experiment_id,
                "artifact_path": str(result.artifact_path),
                "factor_metrics": result.summary["primary_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
