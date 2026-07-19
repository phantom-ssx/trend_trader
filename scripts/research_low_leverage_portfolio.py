"""Search a four-sleeve portfolio with a hard three-times leverage ceiling."""

from __future__ import annotations

import argparse
import itertools
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

SLEEVE_NAMES = ("minute", "btc_72h", "eth_168h", "eth_24h_long")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-artifact", type=Path, required=True)
    parser.add_argument("--btc-artifact", type=Path, required=True)
    parser.add_argument("--eth-168h-artifact", type=Path, required=True)
    parser.add_argument("--eth-24h-artifact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--development-end", default="2025-01-01")
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--max-development-drawdown", type=float, default=0.30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_sleeves(
    args: argparse.Namespace, *, cost_multiplier: float = 1.0
) -> dict[str, dict[datetime, float]]:
    return {
        "minute": minute_strategy_wealth(
            args.minute_artifact, cost_multiplier=cost_multiplier
        ),
        "btc_72h": reconstruct_interval_wealth(
            args.btc_artifact,
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
        "eth_24h_long": reconstruct_interval_wealth(
            args.eth_24h_artifact,
            "ETH-USDT-SWAP",
            data_root=args.data_root,
            cost_multiplier=cost_multiplier,
        ),
    }


def weight_grid(units: int = 20):
    for cuts in itertools.combinations_with_replacement(range(units + 1), 3):
        first, second, third = cuts
        yield (
            first / units,
            (second - first) / units,
            (third - second) / units,
            (units - third) / units,
        )


def compound(values: list[float]) -> float:
    return math.prod(1.0 + value for value in values) - 1.0


def combine_returns(
    sleeves: dict[str, dict[datetime, float]],
    weights: tuple[float, float, float, float],
) -> tuple[list[datetime], list[float], dict[str, list[float]]]:
    common_dates = sorted(set.intersection(*(set(values) for values in sleeves.values())))
    dates = common_dates[1:]
    sleeve_returns = {
        name: daily_returns(values, common_dates) for name, values in sleeves.items()
    }
    combined = [
        sum(
            weights[sleeve_index] * sleeve_returns[name][index]
            for sleeve_index, name in enumerate(SLEEVE_NAMES)
        )
        for index in range(len(dates))
    ]
    return dates, combined, sleeve_returns


def main() -> None:
    args = parse_args()
    sleeves = load_sleeves(args)
    common_dates = sorted(set.intersection(*(set(values) for values in sleeves.values())))
    dates, _, sleeve_returns = combine_returns(sleeves, (1.0, 0.0, 0.0, 0.0))
    development_end = datetime.fromisoformat(args.development_end).replace(tzinfo=UTC)
    development = [index for index, date in enumerate(dates) if date < development_end]
    out_of_sample = [index for index, date in enumerate(dates) if date >= development_end]
    leverage_values = [
        units / 4
        for units in range(4, int(round(args.max_leverage * 4)) + 1)
    ]
    results: list[dict[str, float]] = []
    for weights in weight_grid():
        combined = [
            sum(
                weights[sleeve_index] * sleeve_returns[name][index]
                for sleeve_index, name in enumerate(SLEEVE_NAMES)
            )
            for index in range(len(dates))
        ]
        dev_unscaled = [combined[index] for index in development]
        oos_unscaled = [combined[index] for index in out_of_sample]
        for leverage in leverage_values:
            leveraged = scaled_returns(combined, leverage)
            dev = metrics(scaled_returns(dev_unscaled, leverage))
            oos = metrics(scaled_returns(oos_unscaled, leverage))
            development_segments = [
                compound(
                    [
                        value
                        for date, value in zip(dates, leveraged, strict=True)
                        if date.year == year and date < development_end
                    ]
                )
                for year in sorted(
                    {date.year for date in dates if date < development_end}
                )
            ]
            results.append(
                {
                    **{
                        f"{name}_weight": weight
                        for name, weight in zip(SLEEVE_NAMES, weights, strict=True)
                    },
                    "leverage": leverage,
                    "dev_annual_return": dev.annual_return,
                    "dev_sharpe": dev.sharpe,
                    "dev_max_drawdown": dev.max_drawdown,
                    "dev_worst_year_return": min(development_segments),
                    "oos_annual_return": oos.annual_return,
                    "oos_sharpe": oos.sharpe,
                    "oos_max_drawdown": oos.max_drawdown,
                    "oos_total_return": oos.total_return,
                }
            )
    frame = pl.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the grid-search artifact blind to OOS so it cannot be sorted after the
    # fact to choose weights. OOS metrics remain available only for the three
    # predeclared selection rules recorded in summary.json.
    frame.drop(
        "oos_annual_return",
        "oos_sharpe",
        "oos_max_drawdown",
        "oos_total_return",
    ).write_csv(args.output)

    weight_columns = [f"{name}_weight" for name in SLEEVE_NAMES]
    best_sharpe_weight_row = (
        frame.filter(pl.col("leverage") == 1.0)
        .sort(["dev_sharpe", "dev_annual_return"], descending=True)
        .row(0, named=True)
    )
    selected = (
        frame.filter(
            pl.all_horizontal(
                pl.col(column) == float(best_sharpe_weight_row[column])
                for column in weight_columns
            )
            & (pl.col("dev_max_drawdown") >= -args.max_development_drawdown)
        )
        .sort("leverage", descending=True)
        .row(0, named=True)
    )
    stable_weight_row = (
        frame.filter(pl.col("leverage") == 1.0)
        .sort(
            ["dev_worst_year_return", "dev_sharpe"],
            descending=[True, True],
        )
        .row(0, named=True)
    )
    stable_candidates = frame.filter(
        pl.all_horizontal(
            pl.col(column) == float(stable_weight_row[column])
            for column in weight_columns
        )
        & (pl.col("dev_annual_return") >= 1.0)
        & (pl.col("dev_max_drawdown") >= -args.max_development_drawdown)
    ).sort("leverage")
    stable_selected = (
        stable_candidates.row(0, named=True) if not stable_candidates.is_empty() else None
    )
    # This allocation is fixed before inspecting OOS results. Half the risk budget
    # goes to the only sleeve with positive returns in every complete 2020-2024
    # calendar year; the shorter-history directional sleeves are capped at 20%.
    robust_weights = (0.20, 0.10, 0.20, 0.50)
    robust_candidates = frame.filter(
        pl.all_horizontal(
            pl.col(column) == weight
            for column, weight in zip(weight_columns, robust_weights, strict=True)
        )
        & (pl.col("dev_annual_return") >= 1.0)
        & (pl.col("dev_max_drawdown") >= -args.max_development_drawdown)
    ).sort("leverage")
    robust_selected = (
        robust_candidates.row(0, named=True) if not robust_candidates.is_empty() else None
    )
    summary = {
        "selection_rule": (
            "maximize development Sharpe on a 5% four-sleeve grid, then use the "
            "largest allowed leverage within the development drawdown budget"
        ),
        "selected": selected,
        "stable_selection_rule": (
            "maximize the worst development calendar-segment return at 1x, then "
            "choose the minimum leverage that clears 100% development annual return"
        ),
        "stable_selected": stable_selected,
        "robust_selection_rule": (
            "allocate 50% to the positive-in-every-complete-2020-2024-year ETH 24h "
            "sleeve, cap shorter-history sleeves at 20%, then choose the minimum "
            "development-only leverage clearing 100% annual return"
        ),
        "robust_selected": robust_selected,
        "period": {"start": str(common_dates[0]), "end": str(common_dates[-1])},
    }
    (args.output.parent / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    if robust_selected is None:
        raise ValueError("robust allocation does not clear the development return gate")
    robust_leverage = float(robust_selected["leverage"])
    _, robust_combined, component_returns = combine_returns(sleeves, robust_weights)
    robust_returns = scaled_returns(robust_combined, robust_leverage)
    wealth = 1.0
    daily_rows: list[dict[str, object]] = []
    for index, (date, value) in enumerate(zip(dates, robust_returns, strict=True)):
        wealth *= 1.0 + value
        daily_rows.append(
            {
                "date": date,
                "portfolio_return": value,
                "wealth": wealth,
                **{
                    f"{name}_return": component_returns[name][index]
                    for name in SLEEVE_NAMES
                },
            }
        )
    daily_frame = pl.DataFrame(daily_rows)
    monthly = (
        daily_frame.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(
            (pl.col("portfolio_return") + 1.0).product().sub(1.0).alias("return"),
            pl.len().alias("days"),
        )
        .sort("month")
    )
    yearly = (
        daily_frame.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            (pl.col("portfolio_return") + 1.0).product().sub(1.0).alias("return"),
            pl.len().alias("days"),
        )
        .sort("year")
    )
    daily_frame.write_csv(args.output.parent / "selected_daily_returns.csv")
    monthly.write_csv(args.output.parent / "selected_monthly_returns.csv")
    yearly.write_csv(args.output.parent / "selected_yearly_returns.csv")

    stress_rows: list[dict[str, object]] = []
    for label, multiplier in (("5bp_fee", 1.0), ("8bp_fee", 1.375), ("10bp_fee", 1.625)):
        stressed_sleeves = load_sleeves(args, cost_multiplier=multiplier)
        stressed_dates, stressed_combined, _ = combine_returns(
            stressed_sleeves, robust_weights
        )
        stressed_returns = scaled_returns(stressed_combined, robust_leverage)
        oos_values = [
            value
            for date, value in zip(stressed_dates, stressed_returns, strict=True)
            if date >= development_end
        ]
        year_2025_values = [
            value
            for date, value in zip(stressed_dates, stressed_returns, strict=True)
            if date.year == 2025
        ]
        oos = metrics(oos_values)
        stress_rows.append(
            {
                "scenario": label,
                "cost_multiplier": multiplier,
                "oos_annual_return": oos.annual_return,
                "oos_max_drawdown": oos.max_drawdown,
                "calendar_2025_return": compound(year_2025_values),
            }
        )
    stress = pl.DataFrame(stress_rows)
    stress.write_csv(args.output.parent / "selected_cost_stress.csv")
    print("robust monthly returns")
    print(monthly)
    print("robust cost stress")
    print(stress)


if __name__ == "__main__":
    main()
