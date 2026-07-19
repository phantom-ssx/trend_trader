"""Execution pipeline for multi-factor strategy-performance experiments."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from trend_trader.combinations import FactorCombinationClient, FactorCombinationRequest
from trend_trader.data import MarketDataClient
from trend_trader.experiments.common import (
    ExperimentResult,
    build_component_research,
    fingerprint_experiment_data,
    first_float,
    prepare_experiment,
    translate_combination_params,
)
from trend_trader.experiments.storage import ExperimentRepository, base_record, write_config
from trend_trader.experiments.strategy.config import StrategyExperimentConfig
from trend_trader.experiments.strategy.portfolio import (
    build_portfolio_returns,
    portfolio_metrics,
    portfolio_yearly_metrics,
)
from trend_trader.experiments.strategy.report import render_strategy_report
from trend_trader.experiments.versioning import combination_code_version
from trend_trader.research import FactorAnalyzer


class StrategyExperimentRunner:
    """Evaluate tradable performance of a configured multi-factor signal."""

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

    def run(self, config: StrategyExperimentConfig) -> ExperimentResult:
        created_at = datetime.now(tz=UTC)
        experiment_id = self.repository.new_experiment_id(
            f"strategy_{config.experiment.name}", created_at
        )
        artifacts = self.repository.artifacts(experiment_id)
        try:
            prepared = prepare_experiment(self.data, config, self.workdir)
            component_dataset = build_component_research(self.data, config, prepared)
            references = {
                item.reference: version["resolved_name"]
                for item, version in zip(
                    config.factor_configs,
                    prepared.factor_versions,
                    strict=True,
                )
            }
            combination_version = combination_code_version(config.combination.method)
            combination = FactorCombinationClient().combine(
                component_dataset,
                FactorCombinationRequest(
                    method=config.combination.method,
                    factor_names=tuple(references.values()),
                    name=config.combination.name,
                    training_horizon=config.combination_training_horizon,
                    params=translate_combination_params(
                        config.combination.params,
                        references,
                    ),
                ),
            )
            analysis = FactorAnalyzer(combination.dataset).run(
                method=config.evaluation.ic_method,
                min_cross_section=config.evaluation.min_cross_section,
                quantiles=config.evaluation.quantiles,
                quantile_scope=config.evaluation.scope,
                stability_period=config.evaluation.stability_period,
                stability_min_observations=config.evaluation.stability_min_observations,
            )
            returns = build_portfolio_returns(
                combination.dataset,
                factor_name=config.combination.name,
                timeframe=config.data.timeframe,
                start=config.data.start,
                quantiles=config.evaluation.quantiles,
                round_trip_cost_bps=config.cost.round_trip_bps,
                mode=config.portfolio.mode,
                long_threshold_bps=config.portfolio.long_threshold_bps,
                short_threshold_bps=config.portfolio.short_threshold_bps,
                signal_multiplier=config.portfolio.signal_multiplier,
                signal_smoothing_periods=config.portfolio.signal_smoothing_periods,
                signal_standardization_periods=(
                    config.portfolio.signal_standardization_periods
                ),
                signal_standardization_min_periods=(
                    config.portfolio.signal_standardization_min_periods
                ),
                long_threshold_zscore=config.portfolio.long_threshold_zscore,
                short_threshold_zscore=config.portfolio.short_threshold_zscore,
                long_trend_filter_bars=config.portfolio.long_trend_filter_bars,
                long_trend_min_return_bps=config.portfolio.long_trend_min_return_bps,
                position_size=config.portfolio.position_size,
            )
            metrics = portfolio_metrics(returns, timeframe=config.data.timeframe)
            yearly_metrics = portfolio_yearly_metrics(
                returns,
                timeframe=config.data.timeframe,
            )
            data_version, data_manifest = fingerprint_experiment_data(self.data, config, prepared)
            primary = _strategy_primary_metrics(
                analysis.ic_summary,
                metrics,
                signal_horizon=config.combination_training_horizon,
                portfolio_horizon=config.primary_horizon,
                signal_multiplier=config.portfolio.signal_multiplier,
            )
            component_versions = ",".join(item["version"] for item in prepared.factor_versions)
            strategy_version = f"{combination_version['version']}|{component_versions}"
            cost_model = {
                **config.cost.model_dump(mode="json"),
                "round_trip_bps": config.cost.round_trip_bps,
                "interpretation": "fee and slippage are per-side costs",
                "portfolio_application": "round-trip bps multiplied by weight turnover",
            }
            summary: dict[str, Any] = {
                "experiment_id": experiment_id,
                "experiment_type": "strategy",
                "research_objective": "tradable_strategy_performance",
                "name": config.experiment.name,
                "created_at": created_at.isoformat(),
                "status": "completed",
                "git_commit": prepared.git_commit,
                "git_dirty": prepared.git_dirty,
                "factors": [
                    {
                        **version,
                        "alias": item.reference,
                        "params": item.params,
                    }
                    for item, version in zip(
                        config.factor_configs,
                        prepared.factor_versions,
                        strict=True,
                    )
                ],
                "combination": {
                    **combination_version,
                    "name": config.combination.name,
                    "training_horizon": config.combination_training_horizon,
                    "params": config.combination.params,
                    "diagnostics": combination.diagnostics,
                },
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
                    "transaction_costs_applied_to_training_target": False,
                },
                "cost": cost_model,
                "portfolio": config.portfolio.model_dump(mode="json"),
                "primary_horizon": config.primary_horizon,
                "prediction_horizon": config.combination_training_horizon,
                "execution_horizon": config.primary_horizon,
                "primary_metrics": primary,
                "metrics_by_horizon": metrics.to_dicts(),
                "yearly_metrics": yearly_metrics.to_dicts(),
                "interpretation": {
                    "focus": "net portfolio return, drawdown, risk-adjusted return, and turnover",
                    "signal_ic_role": (
                        "diagnostic only; portfolio performance is the decision metric"
                    ),
                },
            }

            write_config(artifacts, config)
            artifacts.write_json("summary.json", summary)
            artifacts.write_json("data_manifest.json", data_manifest)
            artifacts.write_json("combination_diagnostics.json", combination.diagnostics)
            artifacts.write_csv("universe.csv", prepared.universe)
            artifacts.write_csv(
                "component_factor_summary.csv", FactorAnalyzer(component_dataset).summary()
            )
            artifacts.write_csv("signal_ic.csv", analysis.ic_series)
            artifacts.write_csv("signal_ic_summary.csv", analysis.ic_summary)
            artifacts.write_csv("quantile_returns.csv", analysis.quantile_returns)
            artifacts.write_csv("combination_weights.csv", combination.weights)
            artifacts.write_csv("portfolio_returns.csv", returns)
            artifacts.write_csv("portfolio_metrics.csv", metrics)
            artifacts.write_csv("yearly_portfolio_metrics.csv", yearly_metrics)
            if combination.model_bytes is not None:
                artifacts.write_bytes("model.pkl", combination.model_bytes)
            artifacts.write_text(
                "report.html",
                render_strategy_report(
                    summary,
                    ic_summary=analysis.ic_summary,
                    quantile_returns=analysis.quantile_returns,
                    portfolio_returns=returns,
                    portfolio_metrics=metrics,
                    yearly_metrics=yearly_metrics,
                    combination_weights=combination.weights,
                ),
            )
            final_path = artifacts.publish()
            record = base_record(
                experiment_id=experiment_id,
                experiment_type="strategy",
                config=config,
                created_at=created_at,
                git_commit=prepared.git_commit,
                git_dirty=prepared.git_dirty,
                factor_version=strategy_version,
                factor_params=[item.model_dump(mode="json") for item in config.factor_configs],
                data_version=data_version,
                artifact_path=final_path,
                cost_model=cost_model,
            )
            record.update(
                {
                    "mean_ic": primary["effective_signal_mean_ic"],
                    "ic_ir": primary["effective_signal_ic_ir"],
                    "long_short_return": primary["long_short_return"],
                    "annual_return": primary["annual_return"],
                    "sharpe": primary["sharpe"],
                    "max_drawdown": primary["max_drawdown"],
                    "turnover": primary["turnover"],
                }
            )
            self.repository.save(record)
            return ExperimentResult(experiment_id, final_path, summary)
        except Exception:
            artifacts.discard()
            raise


def _strategy_primary_metrics(
    ic_summary: pl.DataFrame,
    metrics: pl.DataFrame,
    *,
    signal_horizon: int,
    portfolio_horizon: int,
    signal_multiplier: float = 1.0,
) -> dict[str, float | None]:
    ic = ic_summary.filter(pl.col("horizon_bars") == signal_horizon)
    performance = metrics.filter(pl.col("horizon_bars") == portfolio_horizon)
    raw_mean_ic = first_float(ic, "mean_ic")
    raw_ic_ir = first_float(ic, "icir")
    return {
        "signal_mean_ic": raw_mean_ic,
        "signal_ic_ir": raw_ic_ir,
        "effective_signal_mean_ic": (
            raw_mean_ic * signal_multiplier if raw_mean_ic is not None else None
        ),
        "effective_signal_ic_ir": (
            raw_ic_ir * signal_multiplier if raw_ic_ir is not None else None
        ),
        "portfolio_return": first_float(performance, "total_return"),
        "long_short_return": first_float(performance, "total_return"),
        "annual_return": first_float(performance, "annual_return"),
        "sharpe": first_float(performance, "sharpe"),
        "max_drawdown": first_float(performance, "max_drawdown"),
        "active_return": first_float(performance, "active_total_return"),
        "active_annual_return": first_float(performance, "active_annual_return"),
        "active_sharpe": first_float(performance, "active_sharpe"),
        "active_max_drawdown": first_float(performance, "active_max_drawdown"),
        "benchmark_return": first_float(performance, "benchmark_total_return"),
        "benchmark_annual_return": first_float(performance, "benchmark_annual_return"),
        "benchmark_sharpe": first_float(performance, "benchmark_sharpe"),
        "benchmark_max_drawdown": first_float(performance, "benchmark_max_drawdown"),
        "relative_total_return": first_float(performance, "relative_total_return"),
        "turnover": first_float(performance, "turnover"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multi-factor strategy experiment")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--workdir", type=Path, default=Path("."))
    return parser


def main() -> None:
    from trend_trader.experiments.config import load_experiment_config

    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    if not isinstance(config, StrategyExperimentConfig):
        raise TypeError("strategy experiment CLI requires a factors + combination config")
    result = StrategyExperimentRunner(
        MarketDataClient(data_root=args.data_root),
        output_root=args.output_dir,
        workdir=args.workdir,
    ).run(config)
    print(
        json.dumps(
            {
                "experiment_id": result.experiment_id,
                "artifact_path": str(result.artifact_path),
                "strategy_metrics": result.summary["primary_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
