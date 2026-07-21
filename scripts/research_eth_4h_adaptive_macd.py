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
    macd_extrema_targets,
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

ER_LOOKBACKS = (6, 12, 24, 48)


def efficiency_ratio(close: pd.Series, lookback: int) -> pd.Series:
    """Kaufman efficiency ratio: net displacement divided by path length."""
    if lookback <= 1:
        raise ValueError("efficiency-ratio lookback must be greater than one")
    direction = (close - close.shift(lookback)).abs()
    path = close.diff().abs().rolling(lookback, min_periods=lookback).sum()
    return (direction / path.replace(0.0, np.nan)).clip(0.0, 1.0)


def variable_ema(values: pd.Series, alphas: pd.Series) -> pd.Series:
    """Calculate a causal EMA whose smoothing coefficient varies at every bar."""
    if len(values) != len(alphas):
        raise ValueError("values and alphas must have equal length")
    value_array = values.to_numpy(dtype=float)
    alpha_array = alphas.to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(value_array) & np.isfinite(alpha_array))
    if len(valid) == 0:
        return pd.Series(output, index=values.index, dtype=float)

    first = int(valid[0])
    output[first] = float(np.nanmean(value_array[: first + 1]))
    for index in range(first + 1, len(values)):
        value = value_array[index]
        alpha = alpha_array[index]
        if not np.isfinite(value) or not np.isfinite(alpha):
            output[index] = output[index - 1]
            continue
        if not 0.0 < alpha <= 1.0:
            raise ValueError("all finite EMA smoothing coefficients must be in (0, 1]")
        output[index] = output[index - 1] + alpha * (value - output[index - 1])
    return pd.Series(output, index=values.index, dtype=float)


def _kama_alpha(er: pd.Series, fastest_period: int, slowest_period: int) -> pd.Series:
    if not 0 < fastest_period < slowest_period:
        raise ValueError("KAMA periods must satisfy 0 < fastest < slowest")
    fastest = 2.0 / (fastest_period + 1.0)
    slowest = 2.0 / (slowest_period + 1.0)
    return (er * (fastest - slowest) + slowest).pow(2)


def adaptive_macd(
    data: pd.DataFrame,
    parameters: MacdParameters,
    *,
    er_lookback: int,
    model: str,
) -> pd.DataFrame:
    """Return a frame whose ``macd`` column uses a causal ER-adaptive construction."""
    result = data.copy()
    er = efficiency_ratio(result["close"], er_lookback)
    if model == "er_scaled_balanced":
        multiplier = 0.50 + er
        fast_alpha = (2.0 / (parameters.fast + 1.0) * multiplier).clip(upper=1.0)
        slow_alpha = (2.0 / (parameters.slow + 1.0) * multiplier).clip(upper=1.0)
    elif model == "er_scaled_wide":
        multiplier = 0.25 + 1.75 * er
        fast_alpha = (2.0 / (parameters.fast + 1.0) * multiplier).clip(upper=1.0)
        slow_alpha = (2.0 / (parameters.slow + 1.0) * multiplier).clip(upper=1.0)
    elif model == "kama_separate_bounds":
        fast_alpha = _kama_alpha(er, 2, parameters.fast)
        slow_alpha = _kama_alpha(er, parameters.fast, parameters.slow)
    elif model == "kama_shared_fast_bound":
        fast_alpha = _kama_alpha(er, 2, parameters.fast)
        slow_alpha = _kama_alpha(er, 2, parameters.slow)
    else:
        raise ValueError(f"Unknown adaptive MACD model: {model}")

    result["efficiency_ratio"] = er
    result["adaptive_fast"] = variable_ema(result["close"], fast_alpha)
    result["adaptive_slow"] = variable_ema(result["close"], slow_alpha)
    result["macd"] = result["adaptive_fast"] - result["adaptive_slow"]
    result["macd_signal"] = (
        result["macd"]
        .ewm(
            span=parameters.signal,
            adjust=False,
            min_periods=parameters.signal,
        )
        .mean()
    )
    result["macd_histogram"] = result["macd"] - result["macd_signal"]
    return result


def build_variants(
    candles: pd.DataFrame,
    parameters: MacdParameters,
) -> dict[str, pd.DataFrame]:
    variants = {
        "fixed": add_macd(
            candles,
            fast=parameters.fast,
            slow=parameters.slow,
            signal=parameters.signal,
        )
    }
    for lookback in ER_LOOKBACKS:
        for model in [
            "er_scaled_balanced",
            "er_scaled_wide",
            "kama_separate_bounds",
            "kama_shared_fast_bound",
        ]:
            variants[f"{model}_er{lookback}"] = adaptive_macd(
                candles,
                parameters,
                er_lookback=lookback,
                model=model,
            )
    return variants


def _evaluate_segment(
    data: pd.DataFrame,
    *,
    starting_balance: float,
    fee_rate: float,
) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    segment = data.reset_index(drop=True)
    targets = macd_extrema_targets(segment)
    return run_backtest(
        segment,
        targets,
        starting_balance=starting_balance,
        fee_rate=fee_rate,
    )


def evaluate_variants(
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
        for variant, data in build_variants(candles, parameters).items():
            train_mask = data["ts"] < split
            test_mask = ~train_mask
            full_summary, full_equity, _ = _evaluate_segment(
                data,
                starting_balance=starting_balance,
                fee_rate=fee_rate,
            )
            train_summary, _, _ = _evaluate_segment(
                data.loc[train_mask],
                starting_balance=starting_balance,
                fee_rate=fee_rate,
            )
            test_summary, _, _ = _evaluate_segment(
                data.loc[test_mask],
                starting_balance=starting_balance,
                fee_rate=fee_rate,
            )
            train_score = train_summary.cagr_pct / max(train_summary.max_drawdown_pct, 1e-9)
            rows.append(
                {
                    "parameter": parameters.name,
                    "variant": variant,
                    "full_return_pct": full_summary.total_return_pct,
                    "full_cagr_pct": full_summary.cagr_pct,
                    "full_max_drawdown_pct": full_summary.max_drawdown_pct,
                    "full_sharpe": full_summary.sharpe_ratio,
                    "full_trades": full_summary.trades,
                    "full_win_rate_pct": full_summary.win_rate_pct,
                    "full_profit_factor": full_summary.profit_factor,
                    "full_fees": full_summary.total_fees,
                    "full_long_pnl": full_summary.long_pnl,
                    "full_short_pnl": full_summary.short_pnl,
                    "train_return_pct": train_summary.total_return_pct,
                    "train_cagr_pct": train_summary.cagr_pct,
                    "train_max_drawdown_pct": train_summary.max_drawdown_pct,
                    "train_sharpe": train_summary.sharpe_ratio,
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
            yearly.insert(0, "variant", variant)
            yearly.insert(0, "parameter", parameters.name)
            yearly_frames.append(yearly)

    return pd.DataFrame(rows), pd.concat(yearly_frames, ignore_index=True)


def selected_results(results: pd.DataFrame, *, top_n: int = 5) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in results.groupby("parameter", sort=False):
        baseline = group.loc[group["variant"] == "fixed"]
        adaptive = group.loc[group["variant"] != "fixed"].nlargest(top_n, "train_score")
        frames.append(pd.concat([baseline, adaptive], ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def print_report(results: pd.DataFrame) -> None:
    columns = [
        "parameter",
        "variant",
        "train_return_pct",
        "train_max_drawdown_pct",
        "train_score",
        "test_return_pct",
        "test_max_drawdown_pct",
        "test_sharpe",
        "full_return_pct",
        "full_max_drawdown_pct",
        "full_sharpe",
        "full_trades",
        "full_profit_factor",
    ]
    print("\nBASELINE AND TOP-5 ADAPTIVE VARIANTS SELECTED ON 2020-2023")
    print(
        selected_results(results)[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\nFULL-SAMPLE BEST (HINDSIGHT DIAGNOSTIC ONLY)")
    hindsight = (
        results.loc[results["variant"] != "fixed"]
        .sort_values("full_return_pct", ascending=False)
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
        description="Research price-only ER/KAMA Adaptive MACD on ETH 4h extrema signals."
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
    results, yearly = evaluate_variants(
        candles,
        starting_balance=args.starting_balance,
        fee_rate=args.fee_rate,
        train_end=args.train_end,
    )
    print_report(results)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_dir / "results.csv", index=False)
        selected_results(results).to_csv(args.output_dir / "train_selected.csv", index=False)
        yearly.to_csv(args.output_dir / "yearly.csv", index=False)


if __name__ == "__main__":
    main()
