from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

from trend_trader.backtest.metrics import annualized_sharpe_ratio
from trend_trader.config.models import load_backtest_config
from trend_trader.data.schema import legacy_candle_view
from trend_trader.strategies.demo_ema_cross import DemoEmaCrossSignal

console = Console()


@dataclass(frozen=True)
class Trade:
    ts: str
    side: str
    price: float
    position: float
    equity: float


@dataclass(frozen=True)
class SmaCrossBacktestResult:
    trades: list[Trade]
    bars: int
    starting_balance: float
    final_equity: float
    net_pnl: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_fees: float
    long_entries: int
    short_entries: int


def run_strategy_demo(
    df: pl.DataFrame,
    fast_period: int,
    slow_period: int,
    starting_balance: float,
) -> list[Trade]:
    data = df.sort("ts")
    signal = DemoEmaCrossSignal(fast_period=fast_period, slow_period=slow_period)
    trades: list[Trade] = []
    position = 0
    entry_price = 0.0
    realized = 0.0

    for row in data.iter_rows(named=True):
        price = float(row["close"])
        side = signal.on_price(price)
        if side is None:
            continue
        next_position = 1 if side == "BUY" else -1
        if position != 0:
            realized += position * (price - entry_price)
        position = next_position
        entry_price = price
        equity = starting_balance + realized
        trades.append(Trade(str(row["ts"]), side, price, position, equity))
    return trades


def resample_ohlcv(df: pl.DataFrame, every: str | None) -> pl.DataFrame:
    data = df.sort("ts")
    if not every:
        return data
    return (
        data.group_by_dynamic("ts", every=every, closed="left")
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


def run_sma_cross_backtest(
    df: pl.DataFrame,
    fast_period: int,
    slow_period: int,
    starting_balance: float,
    trade_size: float,
    fee_rate: float,
    sizing: str = "fixed",
) -> SmaCrossBacktestResult:
    from trend_trader.strategies.sma_cross import SmaCrossSignal

    if sizing not in {"fixed", "all-in"}:
        raise ValueError("sizing must be either 'fixed' or 'all-in'")
    if sizing == "fixed" and trade_size <= 0:
        raise ValueError("trade_size must be positive")
    if fee_rate < 0:
        raise ValueError("fee_rate must not be negative")

    data = df.sort("ts")
    signal = SmaCrossSignal(fast_period=fast_period, slow_period=slow_period)
    trades: list[Trade] = []
    position = 0.0
    entry_price = 0.0
    realized = 0.0
    total_fees = 0.0
    peak_equity = starting_balance
    max_drawdown_pct = 0.0
    long_entries = 0
    short_entries = 0
    last_ts = ""
    last_price = 0.0
    equity_curve: list[float] = []

    if sizing == "all-in":
        cash = starting_balance
        for row in data.iter_rows(named=True):
            last_ts = str(row["ts"])
            last_price = float(row["close"])
            equity = cash + position * last_price
            equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                max_drawdown_pct = max(
                    max_drawdown_pct,
                    (peak_equity - equity) / peak_equity * 100.0,
                )

            side = signal.on_price(last_price)
            if side is None:
                continue

            target_direction = 1.0 if side == "BUY" else -1.0
            target_position = target_direction * equity / last_price if equity > 0 else 0.0
            delta = target_position - position
            if delta == 0:
                continue

            fee = abs(delta) * last_price * fee_rate
            cash -= delta * last_price + fee
            total_fees += fee
            position = target_position
            equity_after_trade = cash + position * last_price
            if side == "BUY":
                long_entries += 1
            else:
                short_entries += 1
            trades.append(Trade(last_ts, side, last_price, position, equity_after_trade))

            peak_equity = max(peak_equity, equity_after_trade)
            if peak_equity > 0:
                max_drawdown_pct = max(
                    max_drawdown_pct,
                    (peak_equity - equity_after_trade) / peak_equity * 100.0,
                )

        if position != 0:
            fee = abs(position) * last_price * fee_rate
            cash += position * last_price - fee
            total_fees += fee
            position = 0.0
            trades.append(Trade(last_ts, "CLOSE", last_price, position, cash))
        final_equity = cash
        if equity_curve:
            equity_curve[-1] = final_equity
        net_pnl = final_equity - starting_balance
        return SmaCrossBacktestResult(
            trades=trades,
            bars=data.height,
            starting_balance=starting_balance,
            final_equity=final_equity,
            net_pnl=net_pnl,
            return_pct=net_pnl / starting_balance * 100.0,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=annualized_sharpe_ratio(data["ts"], equity_curve),
            total_fees=total_fees,
            long_entries=long_entries,
            short_entries=short_entries,
        )

    for row in data.iter_rows(named=True):
        last_ts = str(row["ts"])
        last_price = float(row["close"])
        side = signal.on_price(last_price)
        if side is not None:
            target_position = trade_size if side == "BUY" else -trade_size
            if target_position != position:
                if position != 0:
                    realized += position * (last_price - entry_price)
                    total_fees += abs(position) * last_price * fee_rate
                position = target_position
                entry_price = last_price
                total_fees += abs(position) * last_price * fee_rate
                if side == "BUY":
                    long_entries += 1
                else:
                    short_entries += 1
                equity = starting_balance + realized - total_fees
                trades.append(Trade(last_ts, side, last_price, position, equity))

        unrealized = position * (last_price - entry_price) if position else 0.0
        equity = starting_balance + realized + unrealized - total_fees
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_drawdown_pct = max(max_drawdown_pct, (peak_equity - equity) / peak_equity * 100.0)

    if position != 0:
        realized += position * (last_price - entry_price)
        total_fees += abs(position) * last_price * fee_rate
        position = 0.0
        final_equity = starting_balance + realized - total_fees
        trades.append(Trade(last_ts, "CLOSE", last_price, position, final_equity))
        peak_equity = max(peak_equity, final_equity)
        if peak_equity > 0:
            max_drawdown_pct = max(
                max_drawdown_pct,
                (peak_equity - final_equity) / peak_equity * 100.0,
            )
    else:
        final_equity = starting_balance + realized - total_fees

    net_pnl = final_equity - starting_balance
    if equity_curve:
        equity_curve[-1] = final_equity
    return SmaCrossBacktestResult(
        trades=trades,
        bars=data.height,
        starting_balance=starting_balance,
        final_equity=final_equity,
        net_pnl=net_pnl,
        return_pct=net_pnl / starting_balance * 100.0,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=annualized_sharpe_ratio(data["ts"], equity_curve),
        total_fees=total_fees,
        long_entries=long_entries,
        short_entries=short_entries,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local calculation backtests.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--engine",
        choices=["demo", "sma-cross"],
        default="sma-cross",
        help="Local backtest implementation to use.",
    )
    parser.add_argument("--resample", help="Optional Polars interval before backtest, e.g. 1h")
    parser.add_argument("--fast-period", type=int, help="Override strategy fast period")
    parser.add_argument("--slow-period", type=int, help="Override strategy slow period")
    parser.add_argument("--trade-size", type=float, help="Override trade size in base units")
    parser.add_argument("--fee-rate", type=float, default=0.0005, help="Fee rate per order")
    parser.add_argument(
        "--sizing",
        choices=["fixed", "all-in"],
        default="fixed",
        help="Position sizing for sma-cross. fixed uses --trade-size; all-in targets 1x equity.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_backtest_config(args.config)
    df = legacy_candle_view(pl.read_parquet(config.data.parquet_path))
    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Parquet file is missing columns: {sorted(missing)}")

    fast_period = args.fast_period or config.strategy.fast_period
    slow_period = args.slow_period or config.strategy.slow_period
    trade_size = args.trade_size or config.strategy.trade_size
    df = resample_ohlcv(df, args.resample)

    if args.engine == "sma-cross":
        output = run_sma_cross_backtest(
            df,
            fast_period=fast_period,
            slow_period=slow_period,
            starting_balance=config.starting_balance,
            trade_size=trade_size,
            fee_rate=args.fee_rate,
            sizing=args.sizing,
        )
        print_sma_cross_result(
            output,
            parquet_path=config.data.parquet_path,
            resample=args.resample,
            fast_period=fast_period,
            slow_period=slow_period,
            trade_size=trade_size,
            fee_rate=args.fee_rate,
            sizing=args.sizing,
        )
        return

    trades = run_strategy_demo(
        df,
        fast_period=fast_period,
        slow_period=slow_period,
        starting_balance=config.starting_balance,
    )
    print_demo_result(trades, bars=df.height, parquet_path=config.data.parquet_path)


def print_demo_result(trades: list[Trade], *, bars: int, parquet_path: Path) -> None:
    table = Table(title="Demo EMA Cross Local Backtest")
    table.add_column("Time")
    table.add_column("Side")
    table.add_column("Price", justify="right")
    table.add_column("Position", justify="right")
    table.add_column("Equity", justify="right")
    for trade in trades[-20:]:
        table.add_row(
            trade.ts,
            trade.side,
            f"{trade.price:.4f}",
            str(trade.position),
            f"{trade.equity:.2f}",
        )
    console.print(table)
    console.print(f"Loaded {bars} bars from {parquet_path}")
    console.print(f"Generated {len(trades)} demo trades")
    console.print("Signal logic: trend_trader.strategies.demo_ema_cross.DemoEmaCrossSignal")


def print_sma_cross_result(
    output: SmaCrossBacktestResult,
    *,
    parquet_path: Path,
    resample: str | None,
    fast_period: int,
    slow_period: int,
    trade_size: float,
    fee_rate: float,
    sizing: str,
) -> None:
    table = Table(title=f"SMA Cross Local Backtest MA{fast_period}/MA{slow_period}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    rows = [
        ("Loaded bars", str(output.bars)),
        ("Resample", resample or "none"),
        ("Sizing", sizing),
        ("Trade size", f"{trade_size:g}" if sizing == "fixed" else "all equity, 1x"),
        ("Fee rate", f"{fee_rate:.6f}"),
        ("Long entries", str(output.long_entries)),
        ("Short entries", str(output.short_entries)),
        ("Trade events", str(len(output.trades))),
        ("Starting balance", f"{output.starting_balance:.2f}"),
        ("Final equity", f"{output.final_equity:.2f}"),
        ("Net PnL", f"{output.net_pnl:.2f}"),
        ("Return", f"{output.return_pct:.2f}%"),
        ("Max drawdown", f"{output.max_drawdown_pct:.2f}%"),
        ("Sharpe ratio", f"{output.sharpe_ratio:.3f}"),
        ("Total fees", f"{output.total_fees:.2f}"),
    ]
    for key, value in rows:
        table.add_row(key, value)

    trades_table = Table(title="Last Trades")
    trades_table.add_column("Time")
    trades_table.add_column("Side")
    trades_table.add_column("Price", justify="right")
    trades_table.add_column("Position", justify="right")
    trades_table.add_column("Equity", justify="right")
    for trade in output.trades[-10:]:
        trades_table.add_row(
            trade.ts,
            trade.side,
            f"{trade.price:.4f}",
            f"{trade.position:g}",
            f"{trade.equity:.2f}",
        )

    console.print(table)
    console.print(trades_table)
    console.print(f"Loaded data from {parquet_path}")
    console.print("Strategy: trend_trader.strategies.sma_cross.SmaCrossSignal")


if __name__ == "__main__":
    main()
