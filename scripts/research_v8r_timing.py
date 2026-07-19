"""Research staggered long-horizon execution without targeting specific OOS months."""

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
from trend_trader.experiments.strategy.timing import (
    aligned_daily_returns,
    equal_weight_daily_wealth,
    interval_daily_wealth,
    streaming_daily_wealth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-artifact", type=Path, required=True)
    parser.add_argument("--eth-168h-artifact", type=Path, required=True)
    parser.add_argument("--eth-24h-artifact", type=Path, required=True)
    parser.add_argument("--trend-artifact", type=Path, required=True)
    parser.add_argument("--eth-168h-config", type=Path, required=True)
    parser.add_argument("--eth-24h-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--development-end", default="2025-01-01")
    parser.add_argument("--leverage", type=float, default=2.25)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def performance(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("performance values are empty")
    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        max_drawdown = min(max_drawdown, wealth / peak - 1.0)
    years = len(values) / 365.25
    annual_return = wealth ** (1.0 / years) - 1.0 if wealth > 0 else -1.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    sharpe = mean / math.sqrt(variance) * math.sqrt(365.25) if variance > 0 else 0.0
    return {
        "days": len(values),
        "total_return": wealth - 1.0,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def compound(values: list[float]) -> float:
    return math.prod(1.0 + value for value in values) - 1.0


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


def phase_portfolio(
    config: StrategyExperimentConfig,
    dataset: Any,
    *,
    offset_bars: int,
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


def staggered_wealth(
    config_path: Path,
    *,
    phase_step_hours: int,
    data: MarketDataClient,
    workdir: Path,
) -> tuple[dict[datetime, float], list[dict[str, object]]]:
    config, dataset = build_signal_dataset(config_path, data=data, workdir=workdir)
    timeframe_minutes = bar_minutes(config.data.timeframe)
    step_bars = phase_step_hours * 60 // timeframe_minutes
    if step_bars < 1 or step_bars * timeframe_minutes != phase_step_hours * 60:
        raise ValueError("phase step must align with strategy timeframe")
    offsets = list(range(0, config.primary_horizon, step_bars))
    portfolios = {
        f"offset_{offset}": phase_portfolio(config, dataset, offset_bars=offset)
        for offset in offsets
    }
    if any(frame.is_empty() for frame in portfolios.values()):
        raise ValueError(f"one or more empty phases for {config.experiment.name}")
    start = min(frame["timestamp"].min() for frame in portfolios.values())
    end = max(frame["exit_time"].max() for frame in portfolios.values())
    instrument_id = config.data.universe.instruments[0]
    candles = data.candles(instrument_id, config.data.timeframe, start, end + timedelta(hours=1))
    phases = {
        name: interval_daily_wealth(frame, candles)
        for name, frame in portfolios.items()
    }
    diagnostics = []
    periods_per_year = 365.25 * 24 / config.primary_horizon
    for name, frame in portfolios.items():
        values = frame["portfolio_return"].to_list()
        wealth = math.prod(1.0 + value for value in values)
        annual = wealth ** (periods_per_year / len(values)) - 1.0 if wealth > 0 else -1.0
        diagnostics.append(
            {
                "strategy": config.experiment.name,
                "phase": name,
                "offset_bars": int(name.removeprefix("offset_")),
                "periods": len(values),
                "annual_return": annual,
                "total_return": wealth - 1.0,
                "position_changes": int((frame["turnover"] > 0).sum()),
            }
        )
    return equal_weight_daily_wealth(phases), diagnostics


def artifact_interval_wealth(
    artifact: Path,
    *,
    data: MarketDataClient,
) -> dict[datetime, float]:
    frame = pl.read_csv(artifact / "portfolio_returns.csv", try_parse_dates=True)
    start = frame["timestamp"].min()
    end = frame["exit_time"].max()
    candles = data.candles("ETH-USDT-SWAP", "1h", start, end + timedelta(hours=1))
    return interval_daily_wealth(frame, candles)


def artifact_streaming_wealth(artifact: Path) -> dict[datetime, float]:
    frame = pl.read_csv(artifact / "portfolio_returns.csv", try_parse_dates=True)
    return streaming_daily_wealth(frame, horizon_bars=1)


def weighted_return_expression(weights: dict[str, float], *, leverage: float) -> pl.Expr:
    return (
        leverage * sum(pl.col(name) * weight for name, weight in weights.items())
    )


def half_year_key(date: datetime) -> str:
    return f"{date.year}-H{1 if date.month <= 6 else 2}"


def screen_weights(
    frame: pl.DataFrame,
    *,
    development_end: datetime,
    leverage: float,
    step: float,
) -> pl.DataFrame:
    units = round(1.0 / step)
    if units < 1 or not math.isclose(units * step, 1.0):
        raise ValueError("weight step must divide one")
    development = frame.filter(pl.col("date") < development_end)
    rows: list[dict[str, object]] = []
    for cuts in itertools.combinations_with_replacement(range(units + 1), 3):
        a, b, c = cuts
        weights = {
            "minute": a / units,
            "eth_168h_staggered": (b - a) / units,
            "eth_24h_staggered": (c - b) / units,
            "trend_24h": (units - c) / units,
        }
        if not (
            0.30 <= weights["minute"] <= 0.50
            and 0.10 <= weights["eth_168h_staggered"] <= 0.20
            and 0.20 <= weights["eth_24h_staggered"] <= 0.50
            and weights["trend_24h"] <= 0.20
        ):
            continue
        values = development.select(
            "date", weighted_return_expression(weights, leverage=leverage).alias("return")
        )
        block_returns = []
        for block in sorted({half_year_key(value) for value in values["date"].to_list()}):
            block_values = values.filter(
                pl.col("date").map_elements(
                    lambda value, expected=block: half_year_key(value) == expected,
                    return_dtype=pl.Boolean,
                )
            )["return"].to_list()
            if len(block_values) >= 150:
                block_returns.append(compound(block_values))
        stats = performance(values["return"].to_list())
        rows.append(
            {
                **{f"{name}_weight": weight for name, weight in weights.items()},
                "dev_worst_half_return": min(block_returns),
                "dev_positive_half_rate": sum(value > 0 for value in block_returns)
                / len(block_returns),
                **{f"dev_{name}": value for name, value in stats.items()},
            }
        )
    return pl.DataFrame(rows).sort(
        ["dev_positive_half_rate", "dev_worst_half_return", "dev_sharpe"],
        descending=[True, True, True],
    )


def add_variant(
    base: pl.DataFrame,
    *,
    name: str,
    weights: dict[str, float],
    leverage: float,
) -> pl.DataFrame:
    return base.select(
        "date",
        pl.lit(name).alias("variant"),
        weighted_return_expression(weights, leverage=leverage).alias("return"),
    )


def period_summary(frame: pl.DataFrame, *, development_end: datetime) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    periods = {
        "full": pl.lit(True),
        "development": pl.col("date") < development_end,
        "post_2025_diagnostic": pl.col("date") >= development_end,
        "2026_ytd_diagnostic": pl.col("date").dt.year() == 2026,
    }
    for variant in frame["variant"].unique(maintain_order=True).to_list():
        selected = frame.filter(pl.col("variant") == variant)
        for period, predicate in periods.items():
            values = selected.filter(predicate)["return"].to_list()
            rows.append(
                {
                    "variant": variant,
                    "period": period,
                    **performance(values),
                }
            )
    return rows


def monthly_returns(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("variant", "month")
        .agg((pl.col("return") + 1.0).product().sub(1.0).alias("return"), pl.len().alias("days"))
        .sort("variant", "month")
    )


def yearly_returns(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("variant", "year")
        .agg((pl.col("return") + 1.0).product().sub(1.0).alias("return"), pl.len().alias("days"))
        .sort("variant", "year")
    )


def main() -> None:
    args = parse_args()
    if args.leverage <= 0:
        raise ValueError("leverage must be positive")
    development_end = datetime.fromisoformat(args.development_end).replace(tzinfo=UTC)
    data = MarketDataClient(data_root=args.data_root)
    workdir = Path.cwd()

    eth_24h_staggered, phase_24 = staggered_wealth(
        args.eth_24h_config,
        phase_step_hours=1,
        data=data,
        workdir=workdir,
    )
    eth_168h_staggered, phase_168 = staggered_wealth(
        args.eth_168h_config,
        phase_step_hours=24,
        data=data,
        workdir=workdir,
    )
    sleeves = {
        "minute": artifact_streaming_wealth(args.minute_artifact),
        "eth_168h_anchor": artifact_interval_wealth(args.eth_168h_artifact, data=data),
        "eth_24h_anchor": artifact_interval_wealth(args.eth_24h_artifact, data=data),
        "eth_168h_staggered": eth_168h_staggered,
        "eth_24h_staggered": eth_24h_staggered,
        "trend_24h": artifact_streaming_wealth(args.trend_artifact),
    }
    aligned = aligned_daily_returns(sleeves)
    screen = screen_weights(
        aligned,
        development_end=development_end,
        leverage=args.leverage,
        step=args.weight_step,
    )
    selected = screen.row(0, named=True)
    selected_weights = {
        name: float(selected[f"{name}_weight"])
        for name in ("minute", "eth_168h_staggered", "eth_24h_staggered", "trend_24h")
    }
    baseline_weights = {
        "minute": 0.40,
        "eth_168h_anchor": 0.15,
        "eth_24h_anchor": 0.45,
    }
    same_weights_staggered = {
        "minute": 0.40,
        "eth_168h_staggered": 0.15,
        "eth_24h_staggered": 0.45,
    }
    variants = pl.concat(
        [
            add_variant(
                aligned,
                name="v8r_anchor",
                weights=baseline_weights,
                leverage=args.leverage,
            ),
            add_variant(
                aligned,
                name="same_weights_staggered",
                weights=same_weights_staggered,
                leverage=args.leverage,
            ),
            add_variant(
                aligned,
                name="development_selected_staggered",
                weights=selected_weights,
                leverage=args.leverage,
            ),
        ],
        how="vertical",
    )
    monthly = monthly_returns(variants)
    yearly = yearly_returns(variants)
    summaries = period_summary(variants, development_end=development_end)
    phase_frame = pl.DataFrame(phase_24 + phase_168)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned.write_csv(args.output_dir / "sleeve_daily_returns.csv")
    variants.write_csv(args.output_dir / "variant_daily_returns.csv")
    monthly.write_csv(args.output_dir / "variant_monthly_returns.csv")
    yearly.write_csv(args.output_dir / "variant_yearly_returns.csv")
    phase_frame.write_csv(args.output_dir / "phase_diagnostics.csv")
    screen.write_csv(args.output_dir / "development_weight_screen.csv")
    summary = {
        "research": "v8r_timing",
        "selection_uses_dates_before": args.development_end,
        "selection_does_not_use_2025_or_2026": True,
        "leverage": args.leverage,
        "cost": {"fee_bps_per_side": 5.0, "slippage_bps_per_side": 3.0},
        "execution": {
            "eth_24h": "24 equal-capital hourly phases, each holding 24h",
            "eth_168h": "7 equal-capital daily phases, each holding 168h",
        },
        "selection_rule": (
            "constrained 5% grid; maximize positive development half-year rate, "
            "then worst development half-year return, then development Sharpe"
        ),
        "selected_weights": selected_weights,
        "selected_development_row": selected,
        "phase_summary": {
            "eth_24h_annual_min": float(
                phase_frame.filter(pl.col("strategy").str.contains("24h"))["annual_return"].min()
            ),
            "eth_24h_annual_median": float(
                phase_frame.filter(pl.col("strategy").str.contains("24h"))["annual_return"].median()
            ),
            "eth_24h_positive_phase_rate": float(
                (
                    phase_frame.filter(pl.col("strategy").str.contains("24h"))[
                        "annual_return"
                    ]
                    > 0
                ).mean()
            ),
            "eth_168h_annual_min": float(
                phase_frame.filter(pl.col("strategy").str.contains("168h"))["annual_return"].min()
            ),
            "eth_168h_annual_median": float(
                phase_frame.filter(pl.col("strategy").str.contains("168h"))["annual_return"].median()
            ),
            "eth_168h_positive_phase_rate": float(
                (
                    phase_frame.filter(pl.col("strategy").str.contains("168h"))[
                        "annual_return"
                    ]
                    > 0
                ).mean()
            ),
        },
        "performance": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print("2026 monthly returns")
    print(monthly.filter(pl.col("month").dt.year() == 2026))


if __name__ == "__main__":
    main()
