"""Run a reproducible overfitting audit for the frozen v8 portfolio."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from trend_trader.combinations import FactorCombinationClient, FactorCombinationRequest
from trend_trader.data import MarketDataClient
from trend_trader.data.models import bar_minutes
from trend_trader.experiments.common import (
    build_component_research,
    prepare_experiment,
    translate_combination_params,
)
from trend_trader.experiments.config import load_experiment_config
from trend_trader.experiments.strategy.config import StrategyExperimentConfig
from trend_trader.experiments.strategy.portfolio import build_portfolio_returns

V8_COMPONENTS = (
    ("minute_return", 0.20),
    ("btc_72h_return", 0.10),
    ("eth_168h_return", 0.20),
    ("eth_24h_long_return", 0.50),
)
DEBIASED_COMPONENTS = (
    ("minute_return", 0.40),
    ("btc_72h_return", 0.00),
    ("eth_168h_return", 0.15),
    ("eth_24h_long_return", 0.45),
)
TARGET_LEVERAGE = 2.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-artifact", type=Path, required=True)
    parser.add_argument("--portfolio-screen", type=Path, required=True)
    parser.add_argument("--btc-config", type=Path, required=True)
    parser.add_argument("--eth-168h-config", type=Path, required=True)
    parser.add_argument("--eth-funding", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--development-end", default="2025-01-01")
    parser.add_argument("--phase-step-hours", type=int, default=24)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_719)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def performance(values: list[float], *, periods_per_year: float) -> dict[str, float]:
    if not values:
        raise ValueError("performance values are empty")
    wealth = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1.0)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    annual_return = wealth ** (periods_per_year / len(values)) - 1.0 if wealth > 0 else -1.0
    sharpe = mean / math.sqrt(variance) * math.sqrt(periods_per_year) if variance > 0 else 0.0
    return {
        "periods": len(values),
        "total_return": wealth - 1.0,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
    }


def bootstrap_annual_returns(
    monthly_returns: list[float], *, samples: int, seed: int
) -> dict[str, float]:
    if not monthly_returns:
        raise ValueError("monthly returns are empty")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    generator = random.Random(seed)
    simulated: list[float] = []
    for _ in range(samples):
        sample = [generator.choice(monthly_returns) for _ in monthly_returns]
        wealth = math.prod(1.0 + value for value in sample)
        simulated.append(wealth ** (12.0 / len(sample)) - 1.0 if wealth > 0 else -1.0)
    simulated.sort()

    def percentile(probability: float) -> float:
        index = min(len(simulated) - 1, max(0, int(probability * len(simulated))))
        return simulated[index]

    return {
        "samples": samples,
        "months": len(monthly_returns),
        "p05": percentile(0.05),
        "median": percentile(0.50),
        "p95": percentile(0.95),
        "probability_positive": sum(value > 0 for value in simulated) / samples,
        "probability_at_least_100pct": sum(value >= 1.0 for value in simulated) / samples,
    }


def build_combined_signal(
    config_path: Path, *, data_root: Path, workdir: Path
) -> tuple[StrategyExperimentConfig, Any]:
    config = load_experiment_config(config_path)
    if not isinstance(config, StrategyExperimentConfig):
        raise TypeError(f"phase audit requires a strategy config: {config_path}")
    data = MarketDataClient(data_root=data_root)
    prepared = prepare_experiment(data, config, workdir)
    components = build_component_research(data, config, prepared)
    references = {
        item.reference: version["resolved_name"]
        for item, version in zip(
            config.factor_configs,
            prepared.factor_versions,
            strict=True,
        )
    }
    combination = FactorCombinationClient().combine(
        components,
        FactorCombinationRequest(
            method=config.combination.method,
            factor_names=tuple(references.values()),
            name=config.combination.name,
            training_horizon=config.combination_training_horizon,
            params=translate_combination_params(config.combination.params, references),
        ),
    )
    return config, combination.dataset


def returns_for_anchor(
    config: StrategyExperimentConfig, dataset: Any, *, offset: int
) -> pl.DataFrame:
    step = timedelta(minutes=bar_minutes(config.data.timeframe))
    return build_portfolio_returns(
        dataset,
        factor_name=config.combination.name,
        timeframe=config.data.timeframe,
        start=config.data.start + step * offset,
        quantiles=config.evaluation.quantiles,
        round_trip_cost_bps=config.cost.round_trip_bps,
        mode=config.portfolio.mode,
        long_threshold_bps=config.portfolio.long_threshold_bps,
        short_threshold_bps=config.portfolio.short_threshold_bps,
        long_threshold_value=config.portfolio.long_threshold_value,
        short_threshold_value=config.portfolio.short_threshold_value,
        signal_multiplier=config.portfolio.signal_multiplier,
        signal_smoothing_periods=config.portfolio.signal_smoothing_periods,
        signal_standardization_periods=config.portfolio.signal_standardization_periods,
        signal_standardization_min_periods=(
            config.portfolio.signal_standardization_min_periods
        ),
        long_threshold_zscore=config.portfolio.long_threshold_zscore,
        short_threshold_zscore=config.portfolio.short_threshold_zscore,
        long_trend_filter_bars=config.portfolio.long_trend_filter_bars,
        long_trend_min_return_bps=config.portfolio.long_trend_min_return_bps,
        position_size=config.portfolio.position_size,
        fixed_holding_periods=config.portfolio.fixed_holding_periods,
        monthly_loss_limit=config.portfolio.monthly_loss_limit,
    ).filter(pl.col("horizon_bars") == config.primary_horizon)


def phase_audit(
    config_path: Path,
    *,
    data_root: Path,
    workdir: Path,
    development_end: datetime,
    phase_step_hours: int,
) -> pl.DataFrame:
    config, dataset = build_combined_signal(config_path, data_root=data_root, workdir=workdir)
    timeframe_minutes = bar_minutes(config.data.timeframe)
    phase_step_bars = phase_step_hours * 60 // timeframe_minutes
    if phase_step_bars < 1 or phase_step_bars * timeframe_minutes != phase_step_hours * 60:
        raise ValueError("phase-step-hours must align with the strategy timeframe")
    horizon = config.primary_horizon
    offsets = list(range(0, horizon, phase_step_bars))
    rows: list[dict[str, object]] = []
    periods_per_year = 365 * 24 * 60 / timeframe_minutes / horizon
    for offset in offsets:
        frame = returns_for_anchor(config, dataset, offset=offset)
        development = frame.filter(pl.col("timestamp") < development_end)
        out_of_sample = frame.filter(pl.col("timestamp") >= development_end)
        rows.append(
            {
                "strategy": config.experiment.name,
                "horizon_bars": horizon,
                "offset_bars": offset,
                "offset_hours": offset * timeframe_minutes / 60,
                **{
                    f"development_{name}": value
                    for name, value in performance(
                        development["portfolio_return"].to_list(),
                        periods_per_year=periods_per_year,
                    ).items()
                },
                **{
                    f"out_of_sample_{name}": value
                    for name, value in performance(
                        out_of_sample["portfolio_return"].to_list(),
                        periods_per_year=periods_per_year,
                    ).items()
                },
            }
        )
    return pl.DataFrame(rows)


def phase_summary(frame: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in frame.partition_by("strategy", maintain_order=True):
        rows.append(
            {
                "strategy": group["strategy"][0],
                "phases": group.height,
                "oos_annual_min": float(group["out_of_sample_annual_return"].min()),
                "oos_annual_median": float(group["out_of_sample_annual_return"].median()),
                "oos_annual_max": float(group["out_of_sample_annual_return"].max()),
                "oos_positive_phase_rate": float(
                    (group["out_of_sample_annual_return"] > 0).mean()
                ),
                "oos_worst_drawdown": float(group["out_of_sample_max_drawdown"].min()),
            }
        )
    return rows


def weight_neighborhood(screen_path: Path) -> dict[str, object]:
    frame = pl.read_csv(screen_path)
    required_oos = {
        "oos_annual_return",
        "oos_max_drawdown",
    }
    if not required_oos.issubset(frame.columns):
        return {"available": False}
    distance = (
        (pl.col("minute_weight") - 0.20).abs()
        + (pl.col("btc_72h_weight") - 0.10).abs()
        + (pl.col("eth_168h_weight") - 0.20).abs()
        + (pl.col("eth_24h_long_weight") - 0.50).abs()
    )
    neighbors = frame.filter(pl.col("leverage") == 2.25).with_columns(
        distance.alias("distance")
    ).filter(pl.col("distance") <= 0.20 + 1e-12)
    return {
        "available": True,
        "neighbors": neighbors.height,
        "oos_annual_min": float(neighbors["oos_annual_return"].min()),
        "oos_annual_median": float(neighbors["oos_annual_return"].median()),
        "oos_annual_max": float(neighbors["oos_annual_return"].max()),
        "share_oos_at_least_100pct": float(
            (neighbors["oos_annual_return"] >= 1.0).mean()
        ),
        "worst_oos_drawdown": float(neighbors["oos_max_drawdown"].min()),
    }


def portfolio_diagnostics(
    artifact: Path,
    *,
    development_end: datetime,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    daily = pl.read_csv(artifact / "selected_daily_returns.csv", try_parse_dates=True)
    monthly = pl.read_csv(artifact / "selected_monthly_returns.csv", try_parse_dates=True)
    out_of_sample = daily.filter(pl.col("date") >= development_end)
    contributions = {
        column: TARGET_LEVERAGE * weight * float(out_of_sample[column].sum())
        for column, weight in V8_COMPONENTS
    }
    total_contribution = sum(contributions.values())
    contribution_shares = {
        name: value / total_contribution if total_contribution else 0.0
        for name, value in contributions.items()
    }
    full_months = monthly.filter(
        (pl.col("month") >= development_end) & (pl.col("days") >= 28)
    )
    bootstrap = bootstrap_annual_returns(
        full_months["return"].to_list(), samples=bootstrap_samples, seed=seed
    )
    ordered_months = monthly.filter(pl.col("month") >= development_end).sort(
        "return", descending=True
    )
    values = ordered_months["return"].to_list()
    return {
        "bootstrap": bootstrap,
        "arithmetic_contribution": contributions,
        "arithmetic_contribution_share": contribution_shares,
        "oos_months": ordered_months.height,
        "positive_month_rate": float((ordered_months["return"] > 0).mean()),
        "best_month": float(ordered_months["return"].max()),
        "worst_month": float(ordered_months["return"].min()),
        "return_without_best_month": math.prod(1.0 + value for value in values[1:]) - 1.0,
        "return_without_best_three_months": (
            math.prod(1.0 + value for value in values[3:]) - 1.0
        ),
    }


def fixed_candidate_diagnostics(
    daily: pl.DataFrame,
    *,
    development_end: datetime,
) -> dict[str, object]:
    candidate = daily.with_columns(
        (
            TARGET_LEVERAGE
            * sum(pl.col(column) * weight for column, weight in DEBIASED_COMPONENTS)
        ).alias("candidate_return")
    )
    development = candidate.filter(pl.col("date") < development_end)
    out_of_sample = candidate.filter(pl.col("date") >= development_end)
    yearly = (
        candidate.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            (pl.col("candidate_return") + 1.0).product().sub(1.0).alias("return"),
            pl.len().alias("days"),
        )
        .sort("year")
    )
    return {
        "status": "forward_shadow_only_historical_metrics_are_contaminated",
        "weights": dict(DEBIASED_COMPONENTS),
        "target_leverage": TARGET_LEVERAGE,
        "selection_rule": (
            "remove BTC after zero-of-three positive daily phases; cap phase-sensitive "
            "ETH 168h at 15%; cap concentrated ETH 24h at 45%; assign the remainder "
            "to the low-correlation minute sleeve"
        ),
        "development": performance(
            development["candidate_return"].to_list(), periods_per_year=365
        ),
        "post_2025_diagnostic_only": performance(
            out_of_sample["candidate_return"].to_list(), periods_per_year=365
        ),
        "yearly": yearly.to_dicts(),
    }


def funding_coverage(path: Path | None, *, start: datetime, end: datetime) -> dict[str, object]:
    if path is None or not path.exists():
        return {"available": False, "complete": False}
    frame = pl.read_parquet(path)
    timestamp_column = "ts" if "ts" in frame.columns else "timestamp"
    funding_start = frame[timestamp_column].min()
    funding_end = frame[timestamp_column].max()
    covered_seconds = max(0.0, (min(funding_end, end) - max(funding_start, start)).total_seconds())
    required_seconds = (end - start).total_seconds()
    return {
        "available": True,
        "complete": bool(funding_start <= start and funding_end >= end),
        "records": frame.height,
        "start": funding_start.isoformat(),
        "end": funding_end.isoformat(),
        "required_start": start.isoformat(),
        "required_end": end.isoformat(),
        "duration_coverage_ratio": covered_seconds / required_seconds,
    }


def main() -> None:
    args = parse_args()
    if args.phase_step_hours < 1:
        raise ValueError("phase-step-hours must be positive")
    development_end = datetime.fromisoformat(args.development_end).replace(tzinfo=UTC)
    workdir = Path.cwd()
    phase_results = pl.concat(
        [
            phase_audit(
                args.btc_config,
                data_root=args.data_root,
                workdir=workdir,
                development_end=development_end,
                phase_step_hours=args.phase_step_hours,
            ),
            phase_audit(
                args.eth_168h_config,
                data_root=args.data_root,
                workdir=workdir,
                development_end=development_end,
                phase_step_hours=args.phase_step_hours,
            ),
        ],
        how="vertical_relaxed",
    )
    daily = pl.read_csv(args.v8_artifact / "selected_daily_returns.csv", try_parse_dates=True)
    portfolio_start = daily["date"].min()
    portfolio_end = daily["date"].max()
    summary = {
        "audit": "v8_overfit",
        "development_end": development_end.isoformat(),
        "phase_step_hours": args.phase_step_hours,
        "phase_summary": phase_summary(phase_results),
        "weight_neighborhood": weight_neighborhood(args.portfolio_screen),
        "portfolio": portfolio_diagnostics(
            args.v8_artifact,
            development_end=development_end,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "debiased_candidate": fixed_candidate_diagnostics(
            daily,
            development_end=development_end,
        ),
        "eth_funding_coverage": funding_coverage(
            args.eth_funding,
            start=portfolio_start,
            end=portfolio_end,
        ),
        "activation_blockers": [
            "oos_was_observed_during_iterative_research",
            "funding_history_is_incomplete",
            "requires_52_frozen_forward_168h_periods",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    phase_results.write_csv(args.output / "phase_results.csv")
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
