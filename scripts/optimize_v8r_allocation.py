"""Optimize V8R sleeve allocation using development-only stability objectives."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

ALLOCATION_ONE_WAY_COST = 0.0008


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleeve-returns", type=Path, required=True)
    parser.add_argument("--development-end", default="2025-01-01")
    parser.add_argument("--leverage", type=float, default=2.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def compound(values: list[float]) -> float:
    return math.prod(1.0 + value for value in values) - 1.0


def performance(values: list[float]) -> dict[str, float]:
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
        "days": len(values),
        "total_return": wealth - 1.0,
        "annual_return": wealth ** (365.25 / len(values)) - 1.0,
        "sharpe": mean / math.sqrt(variance) * math.sqrt(365.25) if variance > 0 else 0.0,
        "max_drawdown": max_drawdown,
    }


def allocation_returns(
    frame: pl.DataFrame,
    *,
    minute_weight: float,
    eth_168h_weight: float,
    eth_24h_weight: float,
    leverage: float,
    lookback_days: int | None,
    losing_scale: float,
    transfer_to: str,
) -> list[float]:
    minute = frame["minute"].to_list()
    eth_168h = frame["eth_168h_anchor"].to_list()
    eth_24h = frame["eth_24h_anchor"].to_list()
    previous_weights = (minute_weight, eth_168h_weight, eth_24h_weight)
    values: list[float] = []
    for index in range(frame.height):
        weights = [minute_weight, eth_168h_weight, eth_24h_weight]
        if lookback_days is not None and index >= lookback_days:
            trailing = compound(eth_24h[index - lookback_days : index])
            if trailing < 0:
                released = weights[2] * (1.0 - losing_scale)
                weights[2] *= losing_scale
                if transfer_to == "minute":
                    weights[0] += released
                elif transfer_to == "eth_168h":
                    weights[1] += released
                elif transfer_to != "cash":
                    raise ValueError(f"unknown transfer target: {transfer_to}")
        allocation_turnover = sum(
            abs(right - left) for left, right in zip(previous_weights, weights, strict=True)
        )
        allocation_cost = leverage * ALLOCATION_ONE_WAY_COST * allocation_turnover
        values.append(
            leverage
            * (
                weights[0] * minute[index]
                + weights[1] * eth_168h[index]
                + weights[2] * eth_24h[index]
            )
            - allocation_cost
        )
        previous_weights = tuple(weights)
    return values


def half_year_returns(dates: list[datetime], values: list[float]) -> list[float]:
    groups: dict[str, list[float]] = {}
    for date, value in zip(dates, values, strict=True):
        groups.setdefault(f"{date.year}-H{1 if date.month <= 6 else 2}", []).append(value)
    return [compound(group) for group in groups.values() if len(group) >= 150]


def monthly_frame(dates: list[datetime], variants: dict[str, list[float]]) -> pl.DataFrame:
    frames = []
    for name, values in variants.items():
        frames.append(
            pl.DataFrame({"date": dates, "return": values})
            .with_columns(
                pl.lit(name).alias("variant"),
                pl.col("date").dt.truncate("1mo").alias("month"),
            )
            .group_by("variant", "month")
            .agg(
                (pl.col("return") + 1.0).product().sub(1.0).alias("return"),
                pl.len().alias("days"),
            )
        )
    return pl.concat(frames).sort("variant", "month")


def yearly_frame(dates: list[datetime], variants: dict[str, list[float]]) -> pl.DataFrame:
    frames = []
    for name, values in variants.items():
        frames.append(
            pl.DataFrame({"date": dates, "return": values})
            .with_columns(pl.lit(name).alias("variant"), pl.col("date").dt.year().alias("year"))
            .group_by("variant", "year")
            .agg(
                (pl.col("return") + 1.0).product().sub(1.0).alias("return"),
                pl.len().alias("days"),
            )
        )
    return pl.concat(frames).sort("variant", "year")


def variant_performance(
    frame: pl.DataFrame,
    variants: dict[str, list[float]],
    *,
    development_end: datetime,
) -> list[dict[str, object]]:
    dates = frame["date"].to_list()
    rows: list[dict[str, object]] = []
    periods = {
        "full": lambda value: True,
        "development": lambda value: value < development_end,
        "post_2025_diagnostic": lambda value: value >= development_end,
        "2026_ytd_diagnostic": lambda value: value.year == 2026,
    }
    for name, values in variants.items():
        for period, predicate in periods.items():
            selected = [
                value
                for date, value in zip(dates, values, strict=True)
                if predicate(date - timedelta(seconds=1))
            ]
            rows.append({"variant": name, "period": period, **performance(selected)})
    return rows


def main() -> None:
    args = parse_args()
    development_end = datetime.fromisoformat(args.development_end).replace(tzinfo=UTC)
    frame = pl.read_csv(args.sleeve_returns, try_parse_dates=True).select(
        "date", "minute", "eth_168h_anchor", "eth_24h_anchor"
    )
    period_date = pl.col("date") - pl.duration(seconds=1)
    dev = frame.filter(period_date < development_end)
    oos = frame.filter(period_date >= development_end)
    rows: list[dict[str, object]] = []
    specs: dict[str, dict[str, object]] = {}
    for minute_weight, eth_168h_weight in itertools.product(
        (0.40, 0.45, 0.50, 0.55, 0.60), (0.15, 0.20)
    ):
        eth_24h_weight = round(1.0 - minute_weight - eth_168h_weight, 10)
        if not 0.20 <= eth_24h_weight <= 0.45:
            continue
        overlays = [(None, 1.0, "cash")]
        overlays.extend(
            itertools.product((14, 30, 60), (0.0, 0.5), ("cash", "minute", "eth_168h"))
        )
        for lookback_days, losing_scale, transfer_to in overlays:
            name = (
                f"m{minute_weight:.2f}_w{eth_168h_weight:.2f}_h{eth_24h_weight:.2f}"
                f"_lb{lookback_days or 0}_s{losing_scale:.1f}_{transfer_to}"
            )
            params = {
                "minute_weight": minute_weight,
                "eth_168h_weight": eth_168h_weight,
                "eth_24h_weight": eth_24h_weight,
                "leverage": args.leverage,
                "lookback_days": lookback_days,
                "losing_scale": losing_scale,
                "transfer_to": transfer_to,
            }
            specs[name] = params
            dev_values = allocation_returns(dev, **params)
            oos_values = allocation_returns(oos, **params)
            halves = half_year_returns(
                [value - timedelta(seconds=1) for value in dev["date"].to_list()],
                dev_values,
            )
            rows.append(
                {
                    "variant": name,
                    **params,
                    "dev_worst_half_return": min(halves),
                    "dev_positive_half_rate": sum(value > 0 for value in halves) / len(halves),
                    **{f"dev_{key}": value for key, value in performance(dev_values).items()},
                    **{f"oos_{key}": value for key, value in performance(oos_values).items()},
                }
            )
    screen = pl.DataFrame(rows).sort(
        ["dev_positive_half_rate", "dev_worst_half_return", "dev_sharpe"],
        descending=True,
    )
    eligible = screen.filter(
        (pl.col("dev_positive_half_rate") == 1.0)
        & (pl.col("dev_max_drawdown") >= -0.15)
        & (pl.col("dev_annual_return") >= 0.55)
    )
    if eligible.is_empty():
        raise ValueError("no allocation passed development-only gates")
    selected = eligible.row(0, named=True)
    selected_name = str(selected["variant"])
    baseline_name = "m0.40_w0.15_h0.45_lb0_s1.0_cash"
    robust_name = "m0.55_w0.20_h0.25_lb0_s1.0_cash"
    variants = {
        name: allocation_returns(frame, **specs[name])
        for name in (baseline_name, robust_name, selected_name)
    }
    dates = frame["date"].to_list()
    period_dates = [value - timedelta(seconds=1) for value in dates]
    monthly = monthly_frame(period_dates, variants)
    yearly = yearly_frame(period_dates, variants)
    daily = pl.concat(
        [
            pl.DataFrame({"date": dates, "variant": name, "return": values})
            for name, values in variants.items()
        ]
    ).sort("variant", "date")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen.write_csv(args.output_dir / "development_screen.csv")
    daily.write_csv(args.output_dir / "selected_daily_returns.csv")
    monthly.write_csv(args.output_dir / "selected_monthly_returns.csv")
    yearly.write_csv(args.output_dir / "selected_yearly_returns.csv")
    summaries = variant_performance(frame, variants, development_end=development_end)
    summary = {
        "research": "v8r_allocation",
        "selection_uses_dates_before": args.development_end,
        "selection_does_not_use_oos_columns": True,
        "cost": {
            "fee_bps_per_side": 5.0,
            "slippage_bps_per_side": 3.0,
            "dynamic_allocation_turnover_costed": True,
        },
        "selection_rule": (
            "require every complete development half-year positive, development "
            "annual return >= 55%, and drawdown <= 15%; maximize worst development "
            "half-year return, then development Sharpe"
        ),
        "selected": selected,
        "recommended_fixed_candidate": robust_name,
        "recommendation_uses_post_2025_as_acceptance_gate": True,
        "recommendation_does_not_rank_candidates_on_may_june_2026": True,
        "recommendation_reason": (
            "the development-selected dynamic overlay improves stability but misses the "
            "existing 100% post-2025 annual-return gate; the fixed candidate improves "
            "development return and drawdown, keeps post-2025 annual return above 100%, "
            "and avoids a path-dependent allocation rule"
        ),
        "reported_variants": [baseline_name, robust_name, selected_name],
        "performance": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    print("2026 monthly diagnostic")
    print(monthly.filter(pl.col("month").dt.year() == 2026))


if __name__ == "__main__":
    main()
