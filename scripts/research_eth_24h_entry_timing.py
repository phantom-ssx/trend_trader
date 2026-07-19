"""Screen causal rebound-confirmed entries for the ETH 24h contrarian sleeve."""

from __future__ import annotations

import argparse
import itertools
import json
import math
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--development-end", default="2025-01-01")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def performance(values: list[float], *, periods_per_year: float) -> dict[str, float]:
    if not values:
        raise ValueError("performance values are empty")
    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        max_drawdown = min(max_drawdown, wealth / peak - 1.0)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    return {
        "periods": len(values),
        "total_return": wealth - 1.0,
        "annual_return": wealth ** (periods_per_year / len(values)) - 1.0,
        "sharpe": mean / math.sqrt(variance) * math.sqrt(periods_per_year)
        if variance > 0
        else 0.0,
        "max_drawdown": max_drawdown,
    }


def build_signal_dataset(
    config_path: Path,
    *,
    data: MarketDataClient,
    workdir: Path,
) -> tuple[StrategyExperimentConfig, Any]:
    config = load_experiment_config(config_path)
    if not isinstance(config, StrategyExperimentConfig):
        raise TypeError(f"expected strategy config: {config_path}")
    prepared = prepare_experiment(data, config, workdir)
    components = build_component_research(data, config, prepared)
    references = {
        item.reference: version["resolved_name"]
        for item, version in zip(config.factor_configs, prepared.factor_versions, strict=True)
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


def candidate_portfolio(
    config: StrategyExperimentConfig,
    dataset: Any,
    *,
    offset_bars: int,
    threshold_bps: float,
    smoothing_periods: int,
    trend_bars: int | None,
    trend_min_bps: float,
) -> pl.DataFrame:
    step = timedelta(minutes=bar_minutes(config.data.timeframe))
    return build_portfolio_returns(
        dataset,
        factor_name=config.combination.name,
        timeframe=config.data.timeframe,
        start=config.data.start + step * offset_bars,
        quantiles=config.evaluation.quantiles,
        round_trip_cost_bps=config.cost.round_trip_bps,
        mode=config.portfolio.mode,
        long_threshold_bps=threshold_bps,
        short_threshold_bps=None,
        signal_multiplier=config.portfolio.signal_multiplier,
        signal_smoothing_periods=smoothing_periods,
        long_trend_filter_bars=trend_bars,
        long_trend_min_return_bps=trend_min_bps,
        position_size=config.portfolio.position_size,
    ).filter(pl.col("horizon_bars") == config.primary_horizon)


def half_year_returns(frame: pl.DataFrame) -> list[float]:
    values = (
        frame.with_columns(
            (
                pl.col("timestamp").dt.year().cast(pl.Utf8)
                + pl.lit("-H")
                + pl.when(pl.col("timestamp").dt.month() <= 6)
                .then(pl.lit("1"))
                .otherwise(pl.lit("2"))
            ).alias("half")
        )
        .group_by("half")
        .agg(
            (pl.col("portfolio_return") + 1.0).product().sub(1.0).alias("return"),
            pl.len().alias("periods"),
        )
        .filter(pl.col("periods") >= 150)
    )
    return values["return"].to_list()


def main() -> None:
    args = parse_args()
    development_end = datetime.fromisoformat(args.development_end).replace(tzinfo=UTC)
    data = MarketDataClient(data_root=args.data_root)
    config, dataset = build_signal_dataset(args.config, data=data, workdir=Path.cwd())
    if config.primary_horizon != 24 or config.data.timeframe != "1h":
        raise ValueError("entry timing screen expects a 24h horizon on 1h data")

    parameter_grid = list(
        itertools.product(
            (2.0, 10.0, 20.0),
            (1, 3, 6),
            (None, 6, 12, 24, 72),
            (0.0, 10.0),
        )
    )
    parameter_grid = [
        values for values in parameter_grid if values[2] is not None or values[3] == 0.0
    ]
    rows: list[dict[str, object]] = []
    portfolios: dict[tuple[float, int, int | None, float, int], pl.DataFrame] = {}
    for threshold, smoothing, trend_bars, trend_min_bps in parameter_grid:
        for offset in (0, 6, 12, 18):
            frame = candidate_portfolio(
                config,
                dataset,
                offset_bars=offset,
                threshold_bps=threshold,
                smoothing_periods=smoothing,
                trend_bars=trend_bars,
                trend_min_bps=trend_min_bps,
            )
            key = (threshold, smoothing, trend_bars, trend_min_bps, offset)
            portfolios[key] = frame
            development = frame.filter(pl.col("timestamp") < development_end)
            out_of_sample = frame.filter(pl.col("timestamp") >= development_end)
            dev_halves = half_year_returns(development)
            rows.append(
                {
                    "threshold_bps": threshold,
                    "smoothing_periods": smoothing,
                    "trend_bars": trend_bars,
                    "trend_min_bps": trend_min_bps,
                    "offset_bars": offset,
                    **{
                        f"dev_{name}": value
                        for name, value in performance(
                            development["portfolio_return"].to_list(),
                            periods_per_year=365.25,
                        ).items()
                    },
                    "dev_worst_half_return": min(dev_halves),
                    "dev_positive_half_rate": sum(value > 0 for value in dev_halves)
                    / len(dev_halves),
                    **{
                        f"oos_{name}": value
                        for name, value in performance(
                            out_of_sample["portfolio_return"].to_list(),
                            periods_per_year=365.25,
                        ).items()
                    },
                    "full_position_changes": int((frame["turnover"] > 0).sum()),
                    "full_long_rate": float((frame["position"] > 0).mean()),
                }
            )
    phase_results = pl.DataFrame(rows)
    keys = ["threshold_bps", "smoothing_periods", "trend_bars", "trend_min_bps"]
    screen = (
        phase_results.group_by(keys)
        .agg(
            (pl.col("dev_annual_return") > 0).mean().alias("dev_positive_phase_rate"),
            pl.col("dev_annual_return").min().alias("dev_annual_min"),
            pl.col("dev_annual_return").median().alias("dev_annual_median"),
            pl.col("dev_sharpe").median().alias("dev_sharpe_median"),
            pl.col("dev_max_drawdown").min().alias("dev_worst_drawdown"),
            pl.col("dev_worst_half_return").min().alias("dev_worst_half_return"),
            pl.col("dev_positive_half_rate").min().alias("dev_positive_half_rate"),
            pl.col("oos_annual_return").median().alias("oos_annual_median_diagnostic"),
            pl.col("oos_max_drawdown").min().alias("oos_worst_drawdown_diagnostic"),
            pl.col("full_position_changes").median().alias("position_changes_median"),
            pl.col("full_long_rate").median().alias("long_rate_median"),
        )
        .sort(
            [
                "dev_positive_phase_rate",
                "dev_positive_half_rate",
                "dev_worst_half_return",
                "dev_sharpe_median",
            ],
            descending=True,
        )
    )
    eligible = screen.filter(
        (pl.col("dev_positive_phase_rate") == 1.0)
        & (pl.col("dev_positive_half_rate") == 1.0)
        & (pl.col("dev_worst_drawdown") >= -0.35)
        & (pl.col("dev_annual_median") >= 0.15)
    )
    gates_passed = not eligible.is_empty()
    selected = (eligible if gates_passed else screen).row(0, named=True)
    selected_key = (
        float(selected["threshold_bps"]),
        int(selected["smoothing_periods"]),
        selected["trend_bars"],
        float(selected["trend_min_bps"]),
        0,
    )
    selected_portfolio = portfolios[selected_key]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_results.write_csv(args.output_dir / "phase_results.csv")
    screen.write_csv(args.output_dir / "development_screen.csv")
    selected_portfolio.write_csv(args.output_dir / "selected_portfolio_returns.csv")
    summary = {
        "research": "eth_24h_entry_timing",
        "selection_uses_dates_before": args.development_end,
        "selection_does_not_use_oos_columns": True,
        "cost": {"fee_bps_per_side": 5.0, "slippage_bps_per_side": 3.0},
        "phase_offsets_hours": [0, 6, 12, 18],
        "selection_rule": (
            "require every phase and complete development half-year positive, "
            "development median annual return >= 15%, and worst drawdown <= 35%; "
            "then maximize worst development half-year return and median Sharpe"
        ),
        "development_gates_passed": gates_passed,
        "selected": selected,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    print("top development-selected candidates (OOS shown only after ordering)")
    print(screen.head(20))


if __name__ == "__main__":
    main()
