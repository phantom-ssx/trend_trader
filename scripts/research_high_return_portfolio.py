"""Research whether diversified causal strategy sleeves can clear a 100% return gate.

The long-horizon experiment artifacts report one return per holding interval.  This
script reconstructs their mark-to-market equity at daily boundaries from hourly
open prices before combining them with a minute-strategy equity curve.  Portfolio
weights and leverage are selected on a development period and then frozen for the
out-of-sample period.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from trend_trader.data import MarketDataClient


@dataclass(frozen=True)
class Metrics:
    annual_return: float
    sharpe: float
    max_drawdown: float
    total_return: float


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(UTC)


def _daily_boundaries(start: datetime, end: datetime) -> list[datetime]:
    boundary = start.replace(hour=0, minute=0, second=0, microsecond=0)
    if boundary < start:
        boundary += timedelta(days=1)
    values: list[datetime] = []
    while boundary <= end:
        values.append(boundary)
        boundary += timedelta(days=1)
    return values


def reconstruct_interval_wealth(
    artifact: Path,
    instrument_id: str,
    *,
    data_root: Path,
    cost_multiplier: float = 1.0,
) -> dict[datetime, float]:
    rows = list(csv.DictReader((artifact / "portfolio_returns.csv").open()))
    if not rows:
        raise ValueError(f"empty portfolio returns: {artifact}")
    start = _parse_timestamp(rows[0]["timestamp"])
    end = _parse_timestamp(rows[-1]["exit_time"])
    client = MarketDataClient(data_root=data_root, sources=[])
    candles = client.candles(instrument_id, "1h", start, end + timedelta(hours=1))
    opens = {
        timestamp: float(value)
        for timestamp, value in candles.select("timestamp", "open").iter_rows()
    }
    boundaries = iter(_daily_boundaries(start, end))
    boundary = next(boundaries, None)
    daily_wealth: dict[datetime, float] = {}
    base_wealth = 1.0
    for row in rows:
        entry = _parse_timestamp(row["timestamp"])
        exit_time = _parse_timestamp(row["exit_time"])
        position = float(row["position"])
        cost = float(row["transaction_cost"]) * cost_multiplier
        entry_price = opens[entry]
        while boundary is not None and boundary <= exit_time:
            if boundary >= entry:
                marked_return = position * (opens[boundary] / entry_price - 1.0) - cost
                daily_wealth[boundary] = base_wealth * (1.0 + marked_return)
            boundary = next(boundaries, None)
        base_wealth *= 1.0 + float(row["portfolio_return"])
        reported = float(row["wealth"])
        if cost_multiplier == 1.0 and not math.isclose(
            base_wealth, reported, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError(
                f"wealth reconstruction mismatch at {entry}: {base_wealth} != {reported}"
            )
    return daily_wealth


def minute_strategy_wealth(
    artifact: Path, *, cost_multiplier: float = 1.0
) -> dict[datetime, float]:
    frame = pl.read_csv(artifact / "portfolio_returns.csv", try_parse_dates=True)
    frame = frame.with_columns(
        (
            pl.col("portfolio_return")
            - (cost_multiplier - 1.0) * pl.col("transaction_cost")
        ).alias("stressed_return")
    ).with_columns((pl.col("stressed_return") + 1.0).cum_prod().alias("stressed_wealth"))
    daily = (
        frame.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("stressed_wealth").last())
        .sort("date")
    )
    return {
        datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=UTC): float(
            wealth
        )
        for day, wealth in daily.iter_rows()
    }


def daily_returns(wealth: dict[datetime, float], dates: list[datetime]) -> list[float]:
    values = [wealth[date] for date in dates]
    return [right / left - 1.0 for left, right in zip(values, values[1:], strict=False)]


def metrics(returns: list[float]) -> Metrics:
    if not returns:
        raise ValueError("returns are empty")
    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        max_drawdown = min(max_drawdown, wealth / peak - 1.0)
    years = len(returns) / 365.0
    annual_return = wealth ** (1.0 / years) - 1.0 if wealth > 0 else -1.0
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / max(1, len(returns) - 1)
    sharpe = mean / math.sqrt(variance) * math.sqrt(365.0) if variance > 0 else 0.0
    return Metrics(annual_return, sharpe, max_drawdown, wealth - 1.0)


def scaled_returns(returns: list[float], leverage: float) -> list[float]:
    return [leverage * value for value in returns]


def load_sleeves(args: argparse.Namespace, cost_multiplier: float = 1.0):
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
            args.eth_artifact,
            "ETH-USDT-SWAP",
            data_root=args.data_root,
            cost_multiplier=cost_multiplier,
        ),
    }


def combine_sleeves(
    sleeves: dict[str, dict[datetime, float]],
    weights: tuple[float, float, float],
) -> tuple[list[datetime], list[float], dict[str, list[float]]]:
    common_dates = sorted(set.intersection(*(set(values) for values in sleeves.values())))
    sleeve_returns = {
        name: daily_returns(values, common_dates) for name, values in sleeves.items()
    }
    combined = [
        weights[0] * sleeve_returns["minute"][index]
        + weights[1] * sleeve_returns["btc_72h"][index]
        + weights[2] * sleeve_returns["eth_168h"][index]
        for index in range(len(common_dates) - 1)
    ]
    return common_dates[1:], combined, sleeve_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-artifact", type=Path, required=True)
    parser.add_argument("--btc-artifact", type=Path, required=True)
    parser.add_argument("--eth-artifact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--development-end", default="2025-01-01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-development-drawdown", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sleeves = load_sleeves(args)
    common_dates = sorted(set.intersection(*(set(values) for values in sleeves.values())))
    if len(common_dates) < 3:
        raise ValueError("insufficient common daily observations")
    return_dates, _, sleeve_returns = combine_sleeves(sleeves, (1.0, 0.0, 0.0))
    development_end = datetime.fromisoformat(args.development_end).replace(tzinfo=UTC)
    development = [index for index, date in enumerate(return_dates) if date < development_end]
    out_of_sample = [index for index, date in enumerate(return_dates) if date >= development_end]
    if not development or not out_of_sample:
        raise ValueError("development or out-of-sample period is empty")

    results: list[dict[str, float]] = []
    for minute_units in range(0, 21):
        for btc_units in range(0, 21 - minute_units):
            eth_units = 20 - minute_units - btc_units
            weights = [minute_units / 20, btc_units / 20, eth_units / 20]
            combined = [
                weights[0] * sleeve_returns["minute"][index]
                + weights[1] * sleeve_returns["btc_72h"][index]
                + weights[2] * sleeve_returns["eth_168h"][index]
                for index in range(len(return_dates))
            ]
            dev_unscaled = [combined[index] for index in development]
            oos_unscaled = [combined[index] for index in out_of_sample]
            for leverage_units in range(4, 25):
                leverage = leverage_units / 4
                dev_metrics = metrics(scaled_returns(dev_unscaled, leverage))
                oos_metrics = metrics(scaled_returns(oos_unscaled, leverage))
                results.append(
                    {
                        "minute_weight": weights[0],
                        "btc_72h_weight": weights[1],
                        "eth_168h_weight": weights[2],
                        "leverage": leverage,
                        "dev_annual_return": dev_metrics.annual_return,
                        "dev_sharpe": dev_metrics.sharpe,
                        "dev_max_drawdown": dev_metrics.max_drawdown,
                        "dev_total_return": dev_metrics.total_return,
                        "oos_annual_return": oos_metrics.annual_return,
                        "oos_sharpe": oos_metrics.sharpe,
                        "oos_max_drawdown": oos_metrics.max_drawdown,
                        "oos_total_return": oos_metrics.total_return,
                    }
                )
    output = pl.DataFrame(results).sort(
        ["dev_sharpe", "dev_annual_return"], descending=[True, True]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.write_csv(args.output)
    feasible = output.filter(
        (pl.col("dev_annual_return") >= 1.0)
        & (pl.col("dev_max_drawdown") >= -0.5)
    )
    print(f"common period: {common_dates[0]} to {common_dates[-1]}")
    print(f"development days: {len(development)}, out-of-sample days: {len(out_of_sample)}")
    print("best development candidates with annual return >= 100% and drawdown >= -50%")
    print(feasible.head(20))
    print("candidates that also clear 100% out of sample")
    print(feasible.filter(pl.col("oos_annual_return") >= 1.0).head(20))

    # Select weights using development Sharpe only, then consume the development
    # drawdown budget to select leverage.  Neither decision observes OOS metrics.
    weight_selection = (
        output.filter(pl.col("leverage") == 1.0)
        .sort(["dev_sharpe", "dev_annual_return"], descending=[True, True])
        .row(0, named=True)
    )
    selected_weights = (
        float(weight_selection["minute_weight"]),
        float(weight_selection["btc_72h_weight"]),
        float(weight_selection["eth_168h_weight"]),
    )
    selected = (
        output.filter(
            (pl.col("minute_weight") == selected_weights[0])
            & (pl.col("btc_72h_weight") == selected_weights[1])
            & (pl.col("eth_168h_weight") == selected_weights[2])
            & (pl.col("dev_max_drawdown") >= -args.max_development_drawdown)
        )
        .sort("leverage", descending=True)
        .row(0, named=True)
    )
    leverage = float(selected["leverage"])
    _, combined, component_returns = combine_sleeves(sleeves, selected_weights)
    leveraged = scaled_returns(combined, leverage)
    wealth = 1.0
    daily_rows: list[dict[str, object]] = []
    for index, (date, value) in enumerate(zip(return_dates, leveraged, strict=True)):
        wealth *= 1.0 + value
        daily_rows.append(
            {
                "date": date,
                "portfolio_return": value,
                "wealth": wealth,
                "minute_return": component_returns["minute"][index],
                "btc_72h_return": component_returns["btc_72h"][index],
                "eth_168h_return": component_returns["eth_168h"][index],
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
    output_dir = args.output.parent
    daily_frame.write_csv(output_dir / "selected_daily_returns.csv")
    monthly.write_csv(output_dir / "selected_monthly_returns.csv")
    yearly.write_csv(output_dir / "selected_yearly_returns.csv")

    stress_rows = []
    for label, multiplier in (("5bp_fee", 1.0), ("8bp_fee", 1.375), ("10bp_fee", 1.625)):
        stressed_sleeves = load_sleeves(args, cost_multiplier=multiplier)
        stressed_dates, stressed_combined, _ = combine_sleeves(
            stressed_sleeves, selected_weights
        )
        stressed_returns = scaled_returns(stressed_combined, leverage)
        dev_values = [
            value
            for date, value in zip(stressed_dates, stressed_returns, strict=True)
            if date < development_end
        ]
        oos_values = [
            value
            for date, value in zip(stressed_dates, stressed_returns, strict=True)
            if date >= development_end
        ]
        dev_stress = metrics(dev_values)
        oos_stress = metrics(oos_values)
        stress_rows.append(
            {
                "scenario": label,
                "cost_multiplier": multiplier,
                "dev_annual_return": dev_stress.annual_return,
                "dev_max_drawdown": dev_stress.max_drawdown,
                "oos_annual_return": oos_stress.annual_return,
                "oos_max_drawdown": oos_stress.max_drawdown,
            }
        )
    pl.DataFrame(stress_rows).write_csv(output_dir / "selected_cost_stress.csv")

    summary = {
        "selection_rule": (
            "maximize development Sharpe across 5% sleeve-weight grid, then choose "
            f"the largest 0.25x leverage with development drawdown <= "
            f"{args.max_development_drawdown:.0%}"
        ),
        "development_end": args.development_end,
        "weights": {
            "minute": selected_weights[0],
            "btc_72h": selected_weights[1],
            "eth_168h": selected_weights[2],
        },
        "leverage": leverage,
        "development": {
            key.removeprefix("dev_"): value
            for key, value in selected.items()
            if key.startswith("dev_")
        },
        "out_of_sample": {
            key.removeprefix("oos_"): value
            for key, value in selected.items()
            if key.startswith("oos_")
        },
        "cost_assumption": {
            "fee_bps_per_side": 5,
            "slippage_bps_per_side": 3,
        },
    }
    (output_dir / "selected_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("development-only selected candidate")
    print(json.dumps(summary, indent=2))
    print("monthly returns")
    print(monthly)
    print("cost stress")
    print(pl.DataFrame(stress_rows))


if __name__ == "__main__":
    main()
