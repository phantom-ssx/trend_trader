"""Develop a causal hourly ETH trend sleeve on pre-2025 data."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from trend_trader.data import MarketDataClient

ONE_WAY_COST = 0.0008  # 5bp fee + 3bp slippage.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--start", default="2020-01-11")
    parser.add_argument("--end", default="2026-07-11")
    parser.add_argument("--development-end", default="2025-01-01")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def indicators(
    candles: pl.DataFrame,
    *,
    fast: int,
    slow: int,
    atr_period: int = 24,
    kdj_lookback: int = 9,
) -> pl.DataFrame:
    close = pl.col("close")
    previous_close = close.shift(1)
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - previous_close).abs(),
        (pl.col("low") - previous_close).abs(),
    )
    lowest = pl.col("low").rolling_min(kdj_lookback, min_samples=kdj_lookback)
    highest = pl.col("high").rolling_max(kdj_lookback, min_samples=kdj_lookback)
    rsv = (close - lowest) / (highest - lowest)
    k_value = rsv.ewm_mean(alpha=1 / 3, adjust=False, min_samples=3)
    d_value = k_value.ewm_mean(alpha=1 / 3, adjust=False, min_samples=3)
    return candles.with_columns(
        (
            close.rolling_mean(fast, min_samples=fast)
            / close.rolling_mean(slow, min_samples=slow)
            - 1.0
        ).alias("ma_spread"),
        (true_range.ewm_mean(alpha=1 / atr_period, adjust=False) / close).alias("atr_pct"),
        (3 * (k_value - d_value)).alias("kdj_j_minus_d"),
        (pl.col("open").shift(-2) / pl.col("open").shift(-1) - 1.0).alias(
            "next_open_return"
        ),
        pl.col("timestamp").shift(-1).alias("entry_time"),
    ).drop_nulls(["ma_spread", "atr_pct", "kdj_j_minus_d", "next_open_return"])


def simulate(
    frame: pl.DataFrame,
    *,
    threshold_bps: float,
    atr_min_bps: float,
    use_kdj: bool,
) -> pl.DataFrame:
    threshold = threshold_bps / 10_000
    atr_min = atr_min_bps / 10_000
    position = 0.0
    rows: list[dict[str, object]] = []
    for row in frame.select(
        "entry_time", "ma_spread", "atr_pct", "kdj_j_minus_d", "next_open_return"
    ).iter_rows(named=True):
        spread = float(row["ma_spread"])
        atr_pct = float(row["atr_pct"])
        kdj = float(row["kdj_j_minus_d"])
        next_position = position
        if atr_pct >= atr_min:
            if spread > threshold and (not use_kdj or kdj > 0):
                next_position = 1.0
            elif spread < -threshold and (not use_kdj or kdj < 0):
                next_position = -1.0
        turnover = 0.5 * abs(next_position - position)
        transaction_cost = ONE_WAY_COST * abs(next_position - position)
        gross_return = next_position * float(row["next_open_return"])
        rows.append(
            {
                "timestamp": row["entry_time"],
                "exit_time": row["entry_time"] + timedelta(hours=1),
                "horizon_bars": 1,
                "position": next_position,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "gross_portfolio_return": gross_return,
                "portfolio_return": gross_return - transaction_cost,
            }
        )
        position = next_position
    return pl.DataFrame(rows)


def performance(frame: pl.DataFrame) -> dict[str, float]:
    values = frame["portfolio_return"].to_list()
    if not values:
        raise ValueError("performance frame is empty")
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
        "annual_return": wealth ** (365.25 * 24 / len(values)) - 1.0,
        "sharpe": mean / math.sqrt(variance) * math.sqrt(365.25 * 24)
        if variance > 0
        else 0.0,
        "max_drawdown": max_drawdown,
    }


def calendar_returns(frame: pl.DataFrame, *, period: str) -> list[float]:
    if period == "year":
        key = pl.col("timestamp").dt.year().alias("key")
        minimum = 8_000
    elif period == "half":
        key = (
            pl.col("timestamp").dt.year().cast(pl.Utf8)
            + pl.lit("-H")
            + pl.when(pl.col("timestamp").dt.month() <= 6)
            .then(pl.lit("1"))
            .otherwise(pl.lit("2"))
        ).alias("key")
        minimum = 4_000
    else:
        raise ValueError(f"unknown period: {period}")
    return (
        frame.with_columns(key)
        .group_by("key")
        .agg(
            (pl.col("portfolio_return") + 1.0).product().sub(1.0).alias("return"),
            pl.len().alias("periods"),
        )
        .filter(pl.col("periods") >= minimum)["return"]
        .to_list()
    )


def monthly_returns(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(
            (pl.col("portfolio_return") + 1.0).product().sub(1.0).alias("return"),
            pl.len().alias("periods"),
        )
        .sort("month")
    )


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    development_end = datetime.fromisoformat(args.development_end).replace(tzinfo=UTC)
    candles = MarketDataClient(data_root=args.data_root).candles(
        "ETH-USDT-SWAP", "1h", start, end
    )
    frames = {
        (fast, slow): indicators(candles, fast=fast, slow=slow)
        for fast, slow in ((8, 24), (12, 48), (24, 72), (24, 168), (48, 168), (72, 336))
    }
    rows: list[dict[str, object]] = []
    portfolios: dict[tuple[int, int, float, float, bool], pl.DataFrame] = {}
    for (fast, slow), threshold_bps, atr_min_bps, use_kdj in itertools.product(
        frames,
        (0.0, 10.0, 25.0, 50.0),
        (0.0, 25.0, 50.0),
        (False, True),
    ):
        portfolio = simulate(
            frames[(fast, slow)],
            threshold_bps=threshold_bps,
            atr_min_bps=atr_min_bps,
            use_kdj=use_kdj,
        )
        key = (fast, slow, threshold_bps, atr_min_bps, use_kdj)
        portfolios[key] = portfolio
        development = portfolio.filter(pl.col("timestamp") < development_end)
        out_of_sample = portfolio.filter(pl.col("timestamp") >= development_end)
        years = calendar_returns(development, period="year")
        halves = calendar_returns(development, period="half")
        rows.append(
            {
                "fast_period": fast,
                "slow_period": slow,
                "threshold_bps": threshold_bps,
                "atr_min_bps": atr_min_bps,
                "use_kdj": use_kdj,
                **{f"dev_{name}": value for name, value in performance(development).items()},
                "dev_worst_year_return": min(years),
                "dev_positive_year_rate": sum(value > 0 for value in years) / len(years),
                "dev_worst_half_return": min(halves),
                "dev_positive_half_rate": sum(value > 0 for value in halves) / len(halves),
                **{f"oos_{name}": value for name, value in performance(out_of_sample).items()},
                "position_changes": int((portfolio["turnover"] > 0).sum()),
            }
        )
    screen = pl.DataFrame(rows).sort(
        [
            "dev_positive_year_rate",
            "dev_worst_year_return",
            "dev_positive_half_rate",
            "dev_sharpe",
        ],
        descending=True,
    )
    eligible = screen.filter(
        (pl.col("dev_positive_year_rate") == 1.0)
        & (pl.col("dev_worst_year_return") > 0)
        & (pl.col("dev_annual_return") >= 0.15)
        & (pl.col("dev_max_drawdown") >= -0.35)
    )
    gates_passed = not eligible.is_empty()
    selected = (eligible if gates_passed else screen).row(0, named=True)
    key = (
        int(selected["fast_period"]),
        int(selected["slow_period"]),
        float(selected["threshold_bps"]),
        float(selected["atr_min_bps"]),
        bool(selected["use_kdj"]),
    )
    portfolio = portfolios[key]
    monthly = monthly_returns(portfolio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen.write_csv(args.output_dir / "development_screen.csv")
    portfolio.write_csv(args.output_dir / "portfolio_returns.csv")
    monthly.write_csv(args.output_dir / "monthly_returns.csv")
    summary = {
        "research": "eth_hourly_trend_timing",
        "selection_uses_dates_before": args.development_end,
        "selection_does_not_use_oos_columns": True,
        "cost": {"fee_bps_per_side": 5.0, "slippage_bps_per_side": 3.0},
        "signal_timing": "bar-close indicators execute at next hourly open",
        "selection_rule": (
            "require every complete development year positive, development annual "
            "return >= 15%, and drawdown <= 35%; maximize worst development year, "
            "then positive half-year rate and development Sharpe"
        ),
        "development_gates_passed": gates_passed,
        "selected": selected,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    print("top development candidates")
    print(screen.head(20))
    print("2026 monthly diagnostic")
    print(monthly.filter(pl.col("month").dt.year() == 2026))


if __name__ == "__main__":
    main()
