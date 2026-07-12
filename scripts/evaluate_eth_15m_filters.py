from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

try:
    from scripts.evaluate_eth_filters import (
        DEFAULT_PARQUET,
        Result,
        add_indicators,
        backtest,
        evaluate_ma_pairs,
        parse_ma_pairs,
        print_results,
        spread_confirm_signals,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root, to sys.path.
    from evaluate_eth_filters import (  # type: ignore[no-redef]
        DEFAULT_PARQUET,
        Result,
        add_indicators,
        backtest,
        evaluate_ma_pairs,
        parse_ma_pairs,
        print_results,
        spread_confirm_signals,
    )

DEFAULT_MA_PAIRS = "5:20,6:24,8:20,8:24,10:30"
DEFAULT_SPREAD_THRESHOLDS = "0.001,0.0015,0.002,0.0025,0.003,0.0035"
DEFAULT_ATR_THRESHOLDS = "0.0015,0.002,0.0025,0.003,0.004,0.005"


def parse_thresholds(value: str) -> list[float]:
    try:
        thresholds = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated decimals") from exc
    if not thresholds or any(item < 0 for item in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be non-negative")
    return thresholds


def load_15min(parquet_path: Path) -> pd.DataFrame:
    """Aggregate source candles to closed-left 15-minute OHLCV bars."""
    frame = pl.read_parquet(parquet_path)
    candles = (
        frame.sort("ts")
        .group_by_dynamic("ts", every="15m", closed="left")
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        )
        .drop_nulls(["open", "high", "low", "close"])
        .sort("ts")
    )
    data = candles.to_pandas()
    add_indicators(data)
    return data


def expand_time_equivalent_pairs(
    pairs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Convert hourly MA periods to equivalent lookback lengths on 15m bars."""
    return [(fast * 4, slow * 4) for fast, slow in pairs]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare MA cross strategies on ETH 15-minute candles."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--start", help="Optional inclusive ISO timestamp.")
    parser.add_argument("--end", help="Optional exclusive ISO timestamp.")
    parser.add_argument(
        "--ma-pairs",
        type=parse_ma_pairs,
        default=parse_ma_pairs(DEFAULT_MA_PAIRS),
        help=(
            "MA pairs in fast:slow format. Defaults to "
            f"{DEFAULT_MA_PAIRS}."
        ),
    )
    parser.add_argument(
        "--include-time-equivalent",
        action="store_true",
        help=(
            "Also test periods multiplied by four, preserving the hourly "
            "strategies' wall-clock lookback on 15-minute candles."
        ),
    )
    parser.add_argument(
        "--filter-grid",
        action="store_true",
        help="Explore every spread/ATR threshold combination for each MA pair.",
    )
    parser.add_argument(
        "--spread-thresholds",
        type=parse_thresholds,
        default=parse_thresholds(DEFAULT_SPREAD_THRESHOLDS),
        help=f"Decimal spread thresholds; default {DEFAULT_SPREAD_THRESHOLDS}.",
    )
    parser.add_argument(
        "--atr-thresholds",
        type=parse_thresholds,
        default=parse_thresholds(DEFAULT_ATR_THRESHOLDS),
        help=f"Decimal ATR/close thresholds; default {DEFAULT_ATR_THRESHOLDS}.",
    )
    return parser


def selected_pairs(
    pairs: list[tuple[int, int]], *, include_time_equivalent: bool
) -> list[tuple[int, int]]:
    selected = list(pairs)
    if include_time_equivalent:
        selected.extend(expand_time_equivalent_pairs(pairs))
    return list(dict.fromkeys(selected))


def evaluate_filter_grid(
    data: pd.DataFrame,
    pairs: list[tuple[int, int]],
    spread_thresholds: list[float],
    atr_thresholds: list[float],
    *,
    starting_balance: float,
    fee_rate: float,
) -> list[Result]:
    results: list[Result] = []
    for fast_period, slow_period in pairs:
        candidate = data.copy()
        add_indicators(candidate, fast_period=fast_period, slow_period=slow_period)
        for spread_threshold in spread_thresholds:
            spread_signals = spread_confirm_signals(candidate, spread_threshold)
            for atr_threshold in atr_thresholds:
                signals = spread_signals.where(
                    candidate["atr_pct"] >= atr_threshold,
                    0,
                )
                name = (
                    f"ma{fast_period}/ma{slow_period}"
                    f"_spread>{spread_threshold:.2%}+atr>={atr_threshold:.2%}"
                )
                results.append(
                    backtest(
                        candidate,
                        signals,
                        name,
                        starting_balance=starting_balance,
                        fee_rate=fee_rate,
                    )
                )
    return results


def run(args: argparse.Namespace) -> list[Result]:
    data = load_15min(args.parquet)
    if args.start:
        data = data[data["ts"] >= pd.to_datetime(args.start, utc=True)]
    if args.end:
        data = data[data["ts"] < pd.to_datetime(args.end, utc=True)]
    data = data.reset_index(drop=True)
    if data.empty:
        raise ValueError("no candles in the selected time range")
    pairs = selected_pairs(
        args.ma_pairs,
        include_time_equivalent=args.include_time_equivalent,
    )
    if args.filter_grid:
        return evaluate_filter_grid(
            data,
            pairs,
            args.spread_thresholds,
            args.atr_thresholds,
            starting_balance=args.starting_balance,
            fee_rate=args.fee_rate,
        )
    return evaluate_ma_pairs(
        data, pairs, starting_balance=args.starting_balance, fee_rate=args.fee_rate
    )


def main() -> None:
    args = build_parser().parse_args()
    print_results(run(args), limit=args.limit)


if __name__ == "__main__":
    main()



# 对于15分钟策略，如果使用ma5/ma20，指标过于灵敏。会反复下单，亏空手续费；

# uv run python scripts/evaluate_eth_15m_filters.py --ma-pairs 24:80,28:80,32:80 --filter-grid --spread-thresholds 0.001,0.0015,0.002,0.0025,0.003 --atr-thresholds 0.003,0.004,0.005 --start 2026-01-01 --end 2026-07-08 --limit 12 | awk -F, '{print $1, $5, $4}' | column -t