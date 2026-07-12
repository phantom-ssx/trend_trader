from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd

try:
    from scripts.evaluate_eth_15m_filters import load_15min, parse_thresholds
    from scripts.evaluate_eth_15m_monthly import FOCUS_STRATEGIES, Strategy
    from scripts.evaluate_eth_filters import (
        DEFAULT_PARQUET,
        Result,
        add_indicators,
        backtest,
        print_results,
        spread_confirm_signals,
    )
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from evaluate_eth_15m_filters import load_15min, parse_thresholds
    from evaluate_eth_15m_monthly import FOCUS_STRATEGIES, Strategy
    from evaluate_eth_filters import (
        DEFAULT_PARQUET,
        Result,
        add_indicators,
        backtest,
        print_results,
        spread_confirm_signals,
    )


DEFAULT_STOP_LOSSES = "0.005,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12"


def backtest_with_stop_loss(
    data: pd.DataFrame,
    signals: pd.Series,
    name: str,
    *,
    starting_balance: float,
    fee_rate: float,
    stop_loss_pct: float | None,
) -> Result:
    cash = starting_balance
    position = 0.0
    entry_price: float | None = None
    trade_start_equity: float | None = None
    trade_pnls: list[float] = []
    peak = starting_balance
    max_dd = 0.0
    fees = 0.0
    events = 0
    long_entries = 0
    short_entries = 0

    def close_position(price: float) -> None:
        nonlocal cash, position, entry_price, trade_start_equity, fees, events
        equity_before_close = cash + position * price
        fee = abs(position) * price * fee_rate
        if trade_start_equity is not None:
            trade_pnls.append(equity_before_close - fee - trade_start_equity)
        cash += position * price - fee
        position = 0.0
        entry_price = None
        trade_start_equity = None
        fees += fee
        events += 1

    for row, signal in zip(data.itertuples(index=False), signals, strict=True):
        close = float(row.close)

        if stop_loss_pct is not None and entry_price is not None:
            if position > 0:
                stop_price = entry_price * (1 - stop_loss_pct)
                if float(row.low) <= stop_price:
                    close_position(min(float(row.open), stop_price))
            elif position < 0:
                stop_price = entry_price * (1 + stop_loss_pct)
                if float(row.high) >= stop_price:
                    close_position(max(float(row.open), stop_price))

        equity = cash + position * close
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)

        if signal not in (1, -1):
            continue
        direction = 1 if position > 0 else -1 if position < 0 else 0
        if direction == signal:
            continue
        equity = cash + position * close
        target_position = signal * equity / close if equity > 0 else 0.0
        delta = target_position - position
        fee = abs(delta) * close * fee_rate
        if direction != 0:
            close_fee = abs(position) * close * fee_rate
            if trade_start_equity is not None:
                trade_pnls.append(equity - close_fee - trade_start_equity)
            trade_start_equity = equity - close_fee
        else:
            trade_start_equity = equity
        cash -= delta * close + fee
        position = target_position
        entry_price = close
        fees += fee
        events += 1
        long_entries += int(signal == 1)
        short_entries += int(signal == -1)

    if abs(position) > 1e-12:
        close_position(float(data["close"].iloc[-1]))

    final_equity = cash
    net_pnl = final_equity - starting_balance
    return_pct = net_pnl / starting_balance * 100
    wins = [pnl for pnl in trade_pnls if pnl > 0]
    losses = [pnl for pnl in trade_pnls if pnl < 0]
    trades = len(trade_pnls)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_loss_ratio = avg_win / abs(avg_loss) if losses else float("inf") if wins else 0.0
    return Result(
        name=name,
        final_equity=final_equity,
        net_pnl=net_pnl,
        return_pct=return_pct,
        max_drawdown_pct=max_dd,
        total_fees=fees,
        events=events,
        long_entries=long_entries,
        short_entries=short_entries,
        trades=trades,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate_pct=len(wins) / trades * 100 if trades else 0.0,
        profit_loss_ratio=profit_loss_ratio,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_win=max(wins, default=0.0),
        min_win=min(wins, default=0.0),
        win_variance=statistics.pvariance(wins) if len(wins) > 1 else 0.0,
        max_loss=max(losses, default=0.0),
        min_loss=min(losses, default=0.0),
        loss_variance=statistics.pvariance(losses) if len(losses) > 1 else 0.0,
        score=return_pct / max(max_dd, 1e-9),
    )


def evaluate_stop_losses(
    data: pd.DataFrame,
    strategies: list[Strategy],
    stop_losses: list[float],
    *,
    starting_balance: float,
    fee_rate: float,
) -> list[Result]:
    results: list[Result] = []
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
        for stop_loss in [None, *stop_losses]:
            stop_name = "none" if stop_loss is None else f"{stop_loss:.2%}"
            result = (
                backtest(
                    candidate,
                    signals,
                    f"{strategy.name}_stop={stop_name}",
                    starting_balance=starting_balance,
                    fee_rate=fee_rate,
                )
                if stop_loss is None
                else backtest_with_stop_loss(
                    candidate,
                    signals,
                    f"{strategy.name}_stop={stop_name}",
                    starting_balance=starting_balance,
                    fee_rate=fee_rate,
                    stop_loss_pct=stop_loss,
                )
            )
            results.append(result)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare fixed stop losses for selected ETH 15-minute strategies."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--start", help="Optional inclusive ISO timestamp.")
    parser.add_argument("--end", help="Optional exclusive ISO timestamp.")
    parser.add_argument(
        "--stop-losses",
        type=parse_thresholds,
        default=parse_thresholds(DEFAULT_STOP_LOSSES),
        help=f"Decimal stop-loss percentages; default {DEFAULT_STOP_LOSSES}.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_15min(args.parquet)
    if args.start:
        data = data[data["ts"] >= pd.to_datetime(args.start, utc=True)]
    if args.end:
        data = data[data["ts"] < pd.to_datetime(args.end, utc=True)]
    data = data.reset_index(drop=True)
    if data.empty:
        raise ValueError("no candles in the selected time range")
    results = evaluate_stop_losses(
        data,
        FOCUS_STRATEGIES,
        args.stop_losses,
        starting_balance=args.starting_balance,
        fee_rate=args.fee_rate,
    )
    print_results(results, args.limit)


if __name__ == "__main__":
    main()
