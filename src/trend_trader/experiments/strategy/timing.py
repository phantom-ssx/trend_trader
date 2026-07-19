"""Helpers for evaluating staggered executions of long-horizon strategies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timedelta

import polars as pl


def interval_daily_wealth(
    portfolio: pl.DataFrame,
    candles: pl.DataFrame,
    *,
    cost_multiplier: float = 1.0,
) -> dict[datetime, float]:
    """Mark a non-overlapping interval portfolio to market at UTC day boundaries."""

    if portfolio.is_empty():
        raise ValueError("portfolio must not be empty")
    if cost_multiplier <= 0:
        raise ValueError("cost_multiplier must be positive")
    required_portfolio = {
        "timestamp",
        "exit_time",
        "position",
        "transaction_cost",
        "portfolio_return",
    }
    missing = required_portfolio.difference(portfolio.columns)
    if missing:
        raise ValueError(f"portfolio is missing columns: {sorted(missing)}")
    if not {"timestamp", "open"}.issubset(candles.columns):
        raise ValueError("candles must contain timestamp and open")

    ordered = portfolio.sort("timestamp")
    start = ordered["timestamp"][0]
    end = ordered["exit_time"][-1]
    boundary = start.replace(hour=0, minute=0, second=0, microsecond=0)
    if boundary < start:
        boundary += timedelta(days=1)
    opens = {
        timestamp: float(value)
        for timestamp, value in candles.select("timestamp", "open").iter_rows()
    }
    daily_wealth: dict[datetime, float] = {}
    base_wealth = 1.0
    verify_reported = cost_multiplier == 1.0 and "wealth" in ordered.columns
    for row in ordered.iter_rows(named=True):
        entry = row["timestamp"]
        exit_time = row["exit_time"]
        position = float(row["position"])
        cost = float(row["transaction_cost"]) * cost_multiplier
        entry_price = opens[entry]
        while boundary <= exit_time and boundary <= end:
            if boundary >= entry:
                marked_return = position * (opens[boundary] / entry_price - 1.0) - cost
                daily_wealth[boundary] = base_wealth * (1.0 + marked_return)
            boundary += timedelta(days=1)
        stressed_return = float(row["portfolio_return"]) - (
            cost_multiplier - 1.0
        ) * float(row["transaction_cost"])
        base_wealth *= 1.0 + stressed_return
        if verify_reported and not math.isclose(
            base_wealth,
            float(row["wealth"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"reported wealth mismatch at {entry}")
    if not daily_wealth:
        raise ValueError("portfolio contains no complete daily boundary")
    return daily_wealth


def streaming_daily_wealth(
    portfolio: pl.DataFrame,
    *,
    horizon_bars: int,
    cost_multiplier: float = 1.0,
) -> dict[datetime, float]:
    """Compound a streaming strategy to UTC day-end wealth observations."""

    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive")
    if cost_multiplier <= 0:
        raise ValueError("cost_multiplier must be positive")
    frame = portfolio.filter(pl.col("horizon_bars") == horizon_bars).sort("timestamp")
    if frame.is_empty():
        raise ValueError(f"portfolio has no horizon {horizon_bars}")
    daily = (
        frame.with_columns(
            (
                pl.col("portfolio_return")
                - (cost_multiplier - 1.0) * pl.col("transaction_cost")
            ).alias("stressed_return")
        )
        .with_columns((pl.col("stressed_return") + 1.0).cum_prod().alias("wealth"))
        .with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("wealth").last())
        .sort("date")
    )
    timezone = frame["timestamp"][0].tzinfo
    return {
        datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone): float(
            wealth
        )
        for day, wealth in daily.iter_rows()
    }


def equal_weight_daily_wealth(
    sleeves: Mapping[str, Mapping[datetime, float]],
) -> dict[datetime, float]:
    """Daily-rebalance equally weighted phase wealth curves on their common dates."""

    if not sleeves:
        raise ValueError("sleeves must not be empty")
    common_dates = sorted(set.intersection(*(set(values) for values in sleeves.values())))
    if len(common_dates) < 2:
        raise ValueError("sleeves have fewer than two common dates")
    phase_returns = {
        name: [
            values[right] / values[left] - 1.0
            for left, right in zip(common_dates, common_dates[1:], strict=False)
        ]
        for name, values in sleeves.items()
    }
    wealth = 1.0
    result = {common_dates[0]: wealth}
    for index, date in enumerate(common_dates[1:]):
        daily_return = sum(values[index] for values in phase_returns.values()) / len(
            phase_returns
        )
        wealth *= 1.0 + daily_return
        result[date] = wealth
    return result


def aligned_daily_returns(
    sleeves: Mapping[str, Mapping[datetime, float]],
) -> pl.DataFrame:
    """Align named wealth curves and return one daily return column per sleeve."""

    if not sleeves:
        raise ValueError("sleeves must not be empty")
    common_dates = sorted(set.intersection(*(set(values) for values in sleeves.values())))
    if len(common_dates) < 2:
        raise ValueError("sleeves have fewer than two common dates")
    rows: list[dict[str, object]] = []
    for left, right in zip(common_dates, common_dates[1:], strict=False):
        rows.append(
            {
                "date": right,
                **{
                    name: float(values[right] / values[left] - 1.0)
                    for name, values in sleeves.items()
                },
            }
        )
    return pl.DataFrame(rows)
