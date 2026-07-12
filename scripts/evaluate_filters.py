from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
from rich.console import Console
from rich.table import Table

from trend_trader.backtest.metrics import annualized_sharpe_ratio
from trend_trader.config.models import load_backtest_config

console = Console()


@dataclass(frozen=True)
class FilterBacktestResult:
    name: str
    bars: int
    final_equity: float
    net_pnl: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_fees: float
    events: int
    long_entries: int
    short_entries: int
    score: float


def load_hourly_data(parquet_path: Path, resample: str = "1h") -> pd.DataFrame:
    df = pl.read_parquet(parquet_path)
    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Parquet file is missing columns: {sorted(missing)}")

    hourly = (
        df.sort("ts")
        .group_by_dynamic("ts", every=resample, closed="left")
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
    return add_indicators(hourly.to_pandas())


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["ma5"] = data["close"].rolling(5).mean()
    data["ma20"] = data["close"].rolling(20).mean()
    data["spread"] = data["ma5"] - data["ma20"]
    data["spread_pct"] = data["spread"] / data["ma20"]
    data["volume_ma20"] = data["volume"].rolling(20).mean()
    data["ma20_slope_6h"] = (data["ma20"] - data["ma20"].shift(6)) / data["ma20"].shift(6)

    prev_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["atr14"] = atr14
    data["atr_pct"] = atr14 / data["close"]

    up_move = data["high"] - data["high"].shift(1)
    down_move = data["low"].shift(1) - data["low"]
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    data["adx14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return data


def run_all_in_backtest(
    data: pd.DataFrame,
    signals: pd.Series,
    name: str,
    *,
    starting_balance: float,
    fee_rate: float,
) -> FilterBacktestResult:
    cash = starting_balance
    position = 0.0
    peak_equity = starting_balance
    max_drawdown_pct = 0.0
    total_fees = 0.0
    events = 0
    long_entries = 0
    short_entries = 0
    equity_curve: list[float] = []

    for row, signal in zip(data.itertuples(index=False), signals, strict=True):
        price = float(row.close)
        equity = cash + position * price
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_drawdown_pct = max(
                max_drawdown_pct,
                (peak_equity - equity) / peak_equity * 100.0,
            )

        if signal not in (1, -1):
            continue

        target_position = signal * equity / price if equity > 0 else 0.0
        delta = target_position - position
        if abs(delta) < 1e-12:
            continue

        fee = abs(delta) * price * fee_rate
        cash -= delta * price + fee
        position = target_position
        total_fees += fee
        events += 1
        long_entries += int(signal == 1)
        short_entries += int(signal == -1)

    last_price = float(data["close"].iloc[-1])
    if abs(position) > 1e-12:
        fee = abs(position) * last_price * fee_rate
        cash += position * last_price - fee
        total_fees += fee
        events += 1

    final_equity = cash
    net_pnl = final_equity - starting_balance
    return_pct = net_pnl / starting_balance * 100.0
    score = return_pct / max(max_drawdown_pct, 1e-9)
    return FilterBacktestResult(
        name=name,
        bars=len(data),
        final_equity=final_equity,
        net_pnl=net_pnl,
        return_pct=return_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=annualized_sharpe_ratio(data["ts"], equity_curve),
        total_fees=total_fees,
        events=events,
        long_entries=long_entries,
        short_entries=short_entries,
        score=score,
    )


def base_cross_signals(data: pd.DataFrame, mask: pd.Series | None = None) -> pd.Series:
    spread = data["spread"]
    previous_spread = spread.shift(1)
    buy = (previous_spread <= 0) & (spread > 0)
    sell = (previous_spread >= 0) & (spread < 0)
    if mask is not None:
        buy &= mask
        sell &= mask
    signals = pd.Series(0, index=data.index)
    signals[buy] = 1
    signals[sell] = -1
    return signals


def spread_confirm_signals(data: pd.DataFrame, threshold: float) -> pd.Series:
    spread = data["spread_pct"]
    previous_spread = spread.shift(1)
    buy = (previous_spread <= threshold) & (spread > threshold)
    sell = (previous_spread >= -threshold) & (spread < -threshold)
    signals = pd.Series(0, index=data.index)
    signals[buy] = 1
    signals[sell] = -1
    return signals


def strategy_signals(data: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "baseline": base_cross_signals(data),
        "spread_0.30%": spread_confirm_signals(data, 0.003),
        "spread_0.35%": spread_confirm_signals(data, 0.0035),
        "spread_0.35%+ATR": spread_confirm_signals(data, 0.0035).where(
            data["atr_pct"] >= 0.005,
            0,
        ),
        "spread_0.40%+ADX20": spread_confirm_signals(data, 0.004).where(
            data["adx14"] >= 20,
            0,
        ),
        "volume_1.2x": base_cross_signals(
            data,
            data["volume"] >= data["volume_ma20"] * 1.2,
        ),
    }


def evaluate_strategies(
    data: pd.DataFrame,
    *,
    starting_balance: float,
    fee_rate: float,
) -> list[FilterBacktestResult]:
    return [
        run_all_in_backtest(
            data,
            signals,
            name,
            starting_balance=starting_balance,
            fee_rate=fee_rate,
        )
        for name, signals in strategy_signals(data).items()
    ]


def monthly_results(
    data: pd.DataFrame,
    *,
    starting_balance: float,
    fee_rate: float,
) -> list[tuple[str, FilterBacktestResult]]:
    dated = data.copy()
    dated["month"] = pd.to_datetime(dated["ts"]).dt.strftime("%Y-%m")
    rows: list[tuple[str, FilterBacktestResult]] = []
    for month in sorted(dated["month"].unique()):
        monthly = dated[dated["month"] == month].copy().reset_index(drop=True)
        if len(monthly) < 30:
            continue
        for result in evaluate_strategies(
            monthly,
            starting_balance=starting_balance,
            fee_rate=fee_rate,
        ):
            rows.append((month, result))
    return rows


def print_overall_table(results: list[FilterBacktestResult]) -> None:
    table = Table(title="Filter Strategy Overall Comparison")
    table.add_column("Strategy")
    table.add_column("Return", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Net PnL", justify="right")
    table.add_column("Fees", justify="right")
    table.add_column("Events", justify="right")
    table.add_column("Score", justify="right")
    for result in sorted(results, key=lambda item: (item.score, item.return_pct), reverse=True):
        table.add_row(
            result.name,
            f"{result.return_pct:.2f}%",
            f"{result.max_drawdown_pct:.2f}%",
            f"{result.sharpe_ratio:.3f}",
            f"{result.net_pnl:.2f}",
            f"{result.total_fees:.2f}",
            str(result.events),
            f"{result.score:.3f}",
        )
    console.print(table)


def print_monthly_table(rows: list[tuple[str, FilterBacktestResult]]) -> None:
    table = Table(title="Filter Strategy Monthly Comparison")
    table.add_column("Month")
    table.add_column("Strategy")
    table.add_column("Bars", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Net PnL", justify="right")
    table.add_column("Fees", justify="right")
    table.add_column("Events", justify="right")
    for month, result in rows:
        table.add_row(
            month,
            result.name,
            str(result.bars),
            f"{result.return_pct:.2f}%",
            f"{result.max_drawdown_pct:.2f}%",
            f"{result.sharpe_ratio:.3f}",
            f"{result.net_pnl:.2f}",
            f"{result.total_fees:.2f}",
            str(result.events),
        )
    console.print(table)


def print_overall_csv(results: list[FilterBacktestResult]) -> None:
    print("strategy,bars,return_pct,max_dd_pct,sharpe_ratio,net_pnl,fees,events,longs,shorts,score")
    for result in results:
        print(
            f"{result.name},{result.bars},{result.return_pct:.2f},"
            f"{result.max_drawdown_pct:.2f},{result.sharpe_ratio:.3f},"
            f"{result.net_pnl:.2f},"
            f"{result.total_fees:.2f},{result.events},{result.long_entries},"
            f"{result.short_entries},{result.score:.3f}"
        )


def print_monthly_csv(rows: list[tuple[str, FilterBacktestResult]]) -> None:
    print("month,strategy,bars,return_pct,max_dd_pct,sharpe_ratio,net_pnl,fees,events")
    for month, result in rows:
        print(
            f"{month},{result.name},{result.bars},{result.return_pct:.2f},"
            f"{result.max_drawdown_pct:.2f},{result.sharpe_ratio:.3f},{result.net_pnl:.2f},"
            f"{result.total_fees:.2f},{result.events}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MA-cross filter strategies.")
    parser.add_argument("--config", type=Path, required=True, help="Backtest config TOML path")
    parser.add_argument("--resample", default="1h", help="Polars resample interval, default 1h")
    parser.add_argument(
        "--mode",
        choices=["overall", "monthly"],
        default="monthly",
        help="Comparison mode",
    )
    parser.add_argument("--fee-rate", type=float, default=0.0005, help="Fee rate per order")
    parser.add_argument(
        "--format",
        choices=["table", "csv"],
        default="table",
        help="Output format",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_backtest_config(args.config)
    data = load_hourly_data(config.data.parquet_path, resample=args.resample)

    if args.mode == "overall":
        results = evaluate_strategies(
            data,
            starting_balance=config.starting_balance,
            fee_rate=args.fee_rate,
        )
        if args.format == "csv":
            print_overall_csv(results)
        else:
            print_overall_table(results)
        return

    rows = monthly_results(
        data,
        starting_balance=config.starting_balance,
        fee_rate=args.fee_rate,
    )
    if args.format == "csv":
        print_monthly_csv(rows)
    else:
        print_monthly_table(rows)


if __name__ == "__main__":
    main()
