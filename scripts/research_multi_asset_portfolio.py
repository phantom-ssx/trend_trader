"""Evaluate the ETH/BTC core after adding capped SOL and DOGE sleeves."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from research_high_return_portfolio import (
    daily_returns,
    metrics,
    minute_strategy_wealth,
    reconstruct_interval_wealth,
    scaled_returns,
)

SLEEVE_NAMES = ("eth_1m", "btc_72h", "eth_168h", "eth_24h", "sol_1m", "doge_1m")
WEIGHTS = (0.20, 0.10, 0.20, 0.40, 0.08, 0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth-minute-artifact", type=Path, required=True)
    parser.add_argument("--btc-72h-artifact", type=Path, required=True)
    parser.add_argument("--eth-168h-artifact", type=Path, required=True)
    parser.add_argument("--eth-24h-artifact", type=Path, required=True)
    parser.add_argument("--sol-minute-artifact", type=Path, required=True)
    parser.add_argument("--doge-minute-artifact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--development-end", default="2025-01-01")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_sleeves(
    args: argparse.Namespace, *, cost_multiplier: float = 1.0
) -> dict[str, dict[datetime, float]]:
    return {
        "eth_1m": minute_strategy_wealth(
            args.eth_minute_artifact, cost_multiplier=cost_multiplier
        ),
        "btc_72h": reconstruct_interval_wealth(
            args.btc_72h_artifact,
            "BTC-USDT-SWAP",
            data_root=args.data_root,
            cost_multiplier=cost_multiplier,
        ),
        "eth_168h": reconstruct_interval_wealth(
            args.eth_168h_artifact,
            "ETH-USDT-SWAP",
            data_root=args.data_root,
            cost_multiplier=cost_multiplier,
        ),
        "eth_24h": reconstruct_interval_wealth(
            args.eth_24h_artifact,
            "ETH-USDT-SWAP",
            data_root=args.data_root,
            cost_multiplier=cost_multiplier,
        ),
        "sol_1m": minute_strategy_wealth(
            args.sol_minute_artifact, cost_multiplier=cost_multiplier
        ),
        "doge_1m": minute_strategy_wealth(
            args.doge_minute_artifact, cost_multiplier=cost_multiplier
        ),
    }


def combine(
    sleeves: dict[str, dict[datetime, float]],
) -> tuple[list[datetime], list[float], dict[str, list[float]]]:
    common_dates = sorted(set.intersection(*(set(values) for values in sleeves.values())))
    dates = common_dates[1:]
    components = {
        name: daily_returns(values, common_dates) for name, values in sleeves.items()
    }
    returns = [
        sum(
            WEIGHTS[sleeve_index] * components[name][index]
            for sleeve_index, name in enumerate(SLEEVE_NAMES)
        )
        for index in range(len(dates))
    ]
    return dates, returns, components


def compound(values: list[float]) -> float:
    return math.prod(1.0 + value for value in values) - 1.0


def average_absolute_position(
    artifact: Path, *, start: datetime, end: datetime
) -> float:
    frame = pl.read_csv(
        artifact / "portfolio_returns.csv",
        columns=["horizon_bars", "timestamp", "position"],
        try_parse_dates=True,
    ).filter(
        (pl.col("horizon_bars") == pl.col("horizon_bars").min())
        & (pl.col("timestamp") >= start)
        & (pl.col("timestamp") < end)
    )
    return float(frame.select(pl.col("position").abs().mean()).item())


def main() -> None:
    args = parse_args()
    sleeves = load_sleeves(args)
    dates, unscaled, components = combine(sleeves)
    development_end = datetime.fromisoformat(args.development_end).replace(tzinfo=UTC)
    development = [
        value
        for date, value in zip(dates, unscaled, strict=True)
        if date < development_end
    ]
    out_of_sample = [
        value
        for date, value in zip(dates, unscaled, strict=True)
        if date >= development_end
    ]
    screen_rows: list[dict[str, float]] = []
    for leverage_units in range(4, 13):
        leverage = leverage_units / 4
        dev = metrics(scaled_returns(development, leverage))
        oos = metrics(scaled_returns(out_of_sample, leverage))
        screen_rows.append(
            {
                "leverage": leverage,
                "dev_annual_return": dev.annual_return,
                "dev_sharpe": dev.sharpe,
                "dev_max_drawdown": dev.max_drawdown,
                "oos_annual_return": oos.annual_return,
                "oos_sharpe": oos.sharpe,
                "oos_max_drawdown": oos.max_drawdown,
            }
        )
    screen = pl.DataFrame(screen_rows)
    candidates = screen.filter(pl.col("dev_annual_return") >= 1.0).sort("leverage")
    if candidates.is_empty():
        raise ValueError("portfolio does not clear the development 100% annual-return gate")
    selected = candidates.row(0, named=True)
    leverage = float(selected["leverage"])
    returns = scaled_returns(unscaled, leverage)

    wealth = 1.0
    daily_rows: list[dict[str, object]] = []
    for index, (date, value) in enumerate(zip(dates, returns, strict=True)):
        wealth *= 1.0 + value
        daily_rows.append(
            {
                "date": date,
                "portfolio_return": value,
                "wealth": wealth,
                **{
                    f"{name}_return": components[name][index]
                    for name in SLEEVE_NAMES
                },
            }
        )
    daily = pl.DataFrame(daily_rows)
    monthly = (
        daily.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg((pl.col("portfolio_return") + 1).product().sub(1).alias("return"))
        .sort("month")
    )
    yearly = (
        daily.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            (pl.col("portfolio_return") + 1).product().sub(1).alias("return"),
            pl.len().alias("days"),
        )
        .sort("year")
    )

    oos_end = dates[-1]
    artifact_paths = {
        "eth_1m": args.eth_minute_artifact,
        "btc_72h": args.btc_72h_artifact,
        "eth_168h": args.eth_168h_artifact,
        "eth_24h": args.eth_24h_artifact,
        "sol_1m": args.sol_minute_artifact,
        "doge_1m": args.doge_minute_artifact,
    }
    sleeve_exposure = {
        name: average_absolute_position(path, start=development_end, end=oos_end)
        for name, path in artifact_paths.items()
    }
    weighted_exposure = {
        name: leverage * weight * sleeve_exposure[name]
        for name, weight in zip(SLEEVE_NAMES, WEIGHTS, strict=True)
    }

    stress_rows: list[dict[str, object]] = []
    for label, multiplier in (("5bp_fee", 1.0), ("8bp_fee", 1.375), ("10bp_fee", 1.625)):
        stressed_dates, stressed_unscaled, _ = combine(
            load_sleeves(args, cost_multiplier=multiplier)
        )
        stressed = scaled_returns(stressed_unscaled, leverage)
        oos_values = [
            value
            for date, value in zip(stressed_dates, stressed, strict=True)
            if date >= development_end
        ]
        values_2025 = [
            value
            for date, value in zip(stressed_dates, stressed, strict=True)
            if date.year == 2025
        ]
        oos = metrics(oos_values)
        stress_rows.append(
            {
                "scenario": label,
                "oos_annual_return": oos.annual_return,
                "oos_max_drawdown": oos.max_drawdown,
                "calendar_2025_return": compound(values_2025),
            }
        )
    stress = pl.DataFrame(stress_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    screen.write_csv(args.output)
    daily.write_csv(args.output.parent / "selected_daily_returns.csv")
    monthly.write_csv(args.output.parent / "selected_monthly_returns.csv")
    yearly.write_csv(args.output.parent / "selected_yearly_returns.csv")
    stress.write_csv(args.output.parent / "selected_cost_stress.csv")
    summary = {
        "selection_rule": (
            "fixed 20/10/20/40/8/2 weights; minimum development-only leverage "
            "clearing 100% annual return"
        ),
        "weights": dict(zip(SLEEVE_NAMES, WEIGHTS, strict=True)),
        "selected": selected,
        "oos_average_gross_leverage": sum(weighted_exposure.values()),
        "oos_weighted_exposure": weighted_exposure,
        "oos_sleeve_active_rate": sleeve_exposure,
        "period": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
    }
    (args.output.parent / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(yearly)
    print(stress)


if __name__ == "__main__":
    main()
