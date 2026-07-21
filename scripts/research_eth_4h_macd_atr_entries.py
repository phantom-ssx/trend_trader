from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research_eth_4h_macd_extrema import (
    DEFAULT_DATA_DIR,
    add_macd,
    load_4h_history,
    period_results,
    run_backtest,
)


@dataclass(frozen=True)
class MacdParameters:
    fast: int
    slow: int
    signal: int

    @property
    def name(self) -> str:
        return f"MACD({self.fast},{self.slow},{self.signal})"


PARAMETER_SETS = (
    MacdParameters(12, 26, 9),
    MacdParameters(24, 52, 18),
)


def add_atr_indicators(
    data: pd.DataFrame,
    *,
    atr_period: int = 14,
    percentile_lookback: int = 6 * 180,
    expansion_lookback: int = 6 * 30,
) -> pd.DataFrame:
    """Add causal 4h ATR regime and scale indicators."""
    if atr_period <= 1 or percentile_lookback <= 1 or expansion_lookback <= 1:
        raise ValueError("ATR and lookback periods must be greater than one")
    result = data.copy()
    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr14"] = true_range.ewm(
        alpha=1 / atr_period,
        adjust=False,
        min_periods=atr_period,
    ).mean()
    result["atr_pct"] = result["atr14"] / result["close"]
    result["atr_percentile"] = (
        result["atr_pct"]
        .rolling(
            percentile_lookback,
            min_periods=min(6 * 30, percentile_lookback),
        )
        .rank(pct=True)
    )
    result["atr_expansion"] = (
        result["atr_pct"]
        / result["atr_pct"]
        .rolling(
            expansion_lookback,
            min_periods=min(6 * 7, expansion_lookback),
        )
        .median()
    )
    return result


def macd_extrema_entry_targets(
    data: pd.DataFrame,
    entry_condition: pd.Series,
) -> pd.Series:
    """Exit at every opposite MACD extremum, opening only when ATR allows entry.

    A peak/trough at t-1 is confirmed at t. ``entry_condition`` is evaluated at t,
    and any resulting target is executed by the backtester at t+1 open.
    """
    if len(data) != len(entry_condition):
        raise ValueError("data and entry_condition must have equal length")
    macd = data["macd"]
    peak = (macd.shift(1) > macd.shift(2)) & (macd.shift(1) > macd)
    trough = (macd.shift(1) < macd.shift(2)) & (macd.shift(1) < macd)
    allowed = entry_condition.fillna(False).astype(bool)

    changes = pd.Series(np.nan, index=data.index, dtype=float)
    changes.loc[peak | trough] = 0.0
    changes.loc[peak & allowed] = -1.0
    changes.loc[trough & allowed] = 1.0
    return changes.ffill().fillna(0.0).astype(int)


def build_entry_conditions(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Build a small fixed grid of causal ATR entry rules."""
    conditions: dict[str, pd.Series] = {
        "baseline_no_atr": pd.Series(True, index=data.index),
    }
    for threshold in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]:
        suffix = int(threshold * 100)
        conditions[f"atr_percentile_ge_{suffix}"] = data["atr_percentile"] >= threshold
        conditions[f"atr_percentile_le_{suffix}"] = data["atr_percentile"] <= threshold

    macd_atr_at_extremum = data["macd"].shift(1).abs() / data["atr14"].shift(1)
    for threshold in [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]:
        conditions[f"macd_atr_ge_{threshold:.2f}"] = macd_atr_at_extremum >= threshold

    for threshold in [0.90, 1.00, 1.10, 1.25]:
        conditions[f"atr_expansion_ge_{threshold:.2f}"] = data["atr_expansion"] >= threshold

    for strength in [0.50, 1.00]:
        for percentile in [0.50, 0.75]:
            conditions[
                f"macd_atr_ge_{strength:.2f}_and_atr_percentile_ge_{int(percentile * 100)}"
            ] = (macd_atr_at_extremum >= strength) & (data["atr_percentile"] >= percentile)
    return conditions


def _evaluate_segment(
    data: pd.DataFrame,
    condition: pd.Series,
    *,
    starting_balance: float,
    fee_rate: float,
) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    segment = data.reset_index(drop=True)
    segment_condition = condition.reset_index(drop=True)
    targets = macd_extrema_entry_targets(segment, segment_condition)
    return run_backtest(
        segment,
        targets,
        starting_balance=starting_balance,
        fee_rate=fee_rate,
    )


def evaluate_grid(
    candles: pd.DataFrame,
    *,
    starting_balance: float,
    fee_rate: float,
    train_end: str = "2024-01-01T00:00:00Z",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    yearly_frames: list[pd.DataFrame] = []
    split = pd.Timestamp(train_end)

    for parameters in PARAMETER_SETS:
        data = add_atr_indicators(
            add_macd(
                candles,
                fast=parameters.fast,
                slow=parameters.slow,
                signal=parameters.signal,
            )
        )
        conditions = build_entry_conditions(data)
        train_mask = data["ts"] < split
        test_mask = ~train_mask

        for rule, condition in conditions.items():
            full_summary, full_equity, _ = _evaluate_segment(
                data,
                condition,
                starting_balance=starting_balance,
                fee_rate=fee_rate,
            )
            train_summary, _, _ = _evaluate_segment(
                data.loc[train_mask],
                condition.loc[train_mask],
                starting_balance=starting_balance,
                fee_rate=fee_rate,
            )
            test_summary, _, _ = _evaluate_segment(
                data.loc[test_mask],
                condition.loc[test_mask],
                starting_balance=starting_balance,
                fee_rate=fee_rate,
            )
            train_score = train_summary.cagr_pct / max(train_summary.max_drawdown_pct, 1e-9)
            rows.append(
                {
                    "parameter": parameters.name,
                    "rule": rule,
                    "full_return_pct": full_summary.total_return_pct,
                    "full_final_equity": full_summary.final_equity,
                    "full_cagr_pct": full_summary.cagr_pct,
                    "full_max_drawdown_pct": full_summary.max_drawdown_pct,
                    "full_sharpe": full_summary.sharpe_ratio,
                    "full_trades": full_summary.trades,
                    "full_win_rate_pct": full_summary.win_rate_pct,
                    "full_profit_factor": full_summary.profit_factor,
                    "full_fees": full_summary.total_fees,
                    "full_long_trades": full_summary.long_trades,
                    "full_long_pnl": full_summary.long_pnl,
                    "full_short_trades": full_summary.short_trades,
                    "full_short_pnl": full_summary.short_pnl,
                    "train_return_pct": train_summary.total_return_pct,
                    "train_cagr_pct": train_summary.cagr_pct,
                    "train_max_drawdown_pct": train_summary.max_drawdown_pct,
                    "train_sharpe": train_summary.sharpe_ratio,
                    "train_trades": train_summary.trades,
                    "train_win_rate_pct": train_summary.win_rate_pct,
                    "train_profit_factor": train_summary.profit_factor,
                    "train_score": train_score,
                    "test_return_pct": test_summary.total_return_pct,
                    "test_cagr_pct": test_summary.cagr_pct,
                    "test_max_drawdown_pct": test_summary.max_drawdown_pct,
                    "test_sharpe": test_summary.sharpe_ratio,
                    "test_trades": test_summary.trades,
                    "test_win_rate_pct": test_summary.win_rate_pct,
                    "test_profit_factor": test_summary.profit_factor,
                }
            )
            yearly = period_results(
                data,
                full_equity,
                starting_balance=starting_balance,
                period="year",
            )
            yearly.insert(0, "rule", rule)
            yearly.insert(0, "parameter", parameters.name)
            yearly_frames.append(yearly)

    return pd.DataFrame(rows), pd.concat(yearly_frames, ignore_index=True)


def selected_results(grid: pd.DataFrame, *, top_n: int = 5) -> pd.DataFrame:
    """Return baseline plus rules selected only by pre-2024 train score."""
    frames: list[pd.DataFrame] = []
    for _, group in grid.groupby("parameter", sort=False):
        baseline = group.loc[group["rule"] == "baseline_no_atr"]
        candidates = group.loc[group["rule"] != "baseline_no_atr"].nlargest(
            top_n,
            "train_score",
        )
        frames.append(pd.concat([baseline, candidates], ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def print_report(grid: pd.DataFrame) -> None:
    columns = [
        "parameter",
        "rule",
        "train_return_pct",
        "train_max_drawdown_pct",
        "train_score",
        "test_return_pct",
        "test_max_drawdown_pct",
        "test_sharpe",
        "full_return_pct",
        "full_max_drawdown_pct",
        "full_trades",
        "full_profit_factor",
    ]
    print("\nBASELINE AND TOP-5 RULES SELECTED ON 2020-2023 ONLY")
    print(
        selected_results(grid)[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\nFULL-SAMPLE BEST (HINDSIGHT DIAGNOSTIC ONLY)")
    hindsight = (
        grid.sort_values("full_return_pct", ascending=False)
        .groupby("parameter", sort=False)
        .head(5)
    )
    print(
        hindsight[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research causal ATR entry filters for ETH 4h MACD-extrema strategies."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--train-end", default="2024-01-01T00:00:00Z")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candles = load_4h_history(
        args.data_dir,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    grid, yearly = evaluate_grid(
        candles,
        starting_balance=args.starting_balance,
        fee_rate=args.fee_rate,
        train_end=args.train_end,
    )
    print_report(grid)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        grid.to_csv(args.output_dir / "grid.csv", index=False)
        selected_results(grid).to_csv(args.output_dir / "train_selected.csv", index=False)
        yearly.to_csv(args.output_dir / "yearly.csv", index=False)


if __name__ == "__main__":
    main()
