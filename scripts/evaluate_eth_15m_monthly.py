from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from scripts.evaluate_eth_15m_filters import load_15min
    from scripts.evaluate_eth_filters import (
        DEFAULT_PARQUET,
        Result,
        add_indicators,
        backtest,
        spread_confirm_signals,
    )
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from evaluate_eth_15m_filters import load_15min
    from evaluate_eth_filters import (
        DEFAULT_PARQUET,
        Result,
        add_indicators,
        backtest,
        spread_confirm_signals,
    )


@dataclass(frozen=True)
class Strategy:
    name: str
    fast_period: int
    slow_period: int
    spread_threshold: float
    atr_threshold: float


FOCUS_STRATEGIES = [
    Strategy("high_return", 24, 80, 0.0030, 0.0050),
    Strategy("risk_adjusted", 32, 80, 0.0010, 0.0040),
    Strategy("stable_split", 28, 80, 0.0015, 0.0040),
]


def monthly_results(
    data: pd.DataFrame,
    strategies: list[Strategy],
    *,
    starting_balance: float,
    fee_rate: float,
) -> list[tuple[str, Result]]:
    rows: list[tuple[str, Result]] = []
    months = data["ts"].dt.strftime("%Y-%m").drop_duplicates().tolist()
    for strategy in strategies:
        candidate = data.copy()
        add_indicators(
            candidate,
            fast_period=strategy.fast_period,
            slow_period=strategy.slow_period,
        )
        signals = spread_confirm_signals(
            candidate, strategy.spread_threshold
        ).where(candidate["atr_pct"] >= strategy.atr_threshold, 0)
        full_name = (
            f"{strategy.name}:ma{strategy.fast_period}/ma{strategy.slow_period}"
            f"+spread>{strategy.spread_threshold:.2%}"
            f"+atr>={strategy.atr_threshold:.2%}"
        )
        for month in months:
            mask = candidate["ts"].dt.strftime("%Y-%m") == month
            month_data = candidate.loc[mask].reset_index(drop=True)
            month_signals = signals.loc[mask].reset_index(drop=True)
            rows.append(
                (
                    month,
                    backtest(
                        month_data,
                        month_signals,
                        full_name,
                        starting_balance=starting_balance,
                        fee_rate=fee_rate,
                    ),
                )
            )
    return rows


def print_monthly_results(rows: list[tuple[str, Result]]) -> None:
    print(
        "month,strategy,return_pct,max_dd_pct,sharpe_ratio,net_pnl,fees,trades,wins,losses,"
        "win_rate_pct,profit_loss_ratio,avg_win,avg_loss,max_win,min_win,"
        "win_variance,max_loss,min_loss,loss_variance"
    )
    for month, result in rows:
        print(
            f"{month},{result.name},{result.return_pct:.2f},"
            f"{result.max_drawdown_pct:.2f},{result.sharpe_ratio:.3f},"
            f"{result.net_pnl:.2f},"
            f"{result.total_fees:.2f},{result.trades},{result.winning_trades},"
            f"{result.losing_trades},{result.win_rate_pct:.2f},"
            f"{result.profit_loss_ratio:.3f},{result.avg_win:.2f},"
            f"{result.avg_loss:.2f},{result.max_win:.2f},{result.min_win:.2f},"
            f"{result.win_variance:.2f},{result.max_loss:.2f},"
            f"{result.min_loss:.2f},{result.loss_variance:.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the three selected ETH 15-minute strategies by month."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = monthly_results(
        load_15min(args.parquet),
        FOCUS_STRATEGIES,
        starting_balance=args.starting_balance,
        fee_rate=args.fee_rate,
    )
    print_monthly_results(rows)


if __name__ == "__main__":
    main()
