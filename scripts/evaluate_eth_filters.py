from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl

DEFAULT_PARQUET = Path(
    "data/clean/okx/ETH-USDT-SWAP/"
    "ETH-USDT-SWAP_1m_20260101T000000Z_20260707T124753Z.parquet"
)


@dataclass(frozen=True)
class Result:
    name: str
    final_equity: float
    net_pnl: float
    return_pct: float
    max_drawdown_pct: float
    total_fees: float
    events: int
    long_entries: int
    short_entries: int
    score: float


def load_hourly(parquet_path: Path) -> pd.DataFrame:
    df = pl.read_parquet(parquet_path)
    hourly = (
        df.sort("ts")
        .group_by_dynamic("ts", every="1h", closed="left")
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
    data = hourly.to_pandas()
    add_indicators(data)
    return data


def add_indicators(data: pd.DataFrame) -> None:
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
    data["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["atr_pct"] = data["atr14"] / data["close"]

    up_move = data["high"] - data["high"].shift(1)
    down_move = data["low"].shift(1) - data["low"]
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    data["adx14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


def backtest(
    data: pd.DataFrame,
    signals: pd.Series,
    name: str,
    *,
    starting_balance: float,
    fee_rate: float,
) -> Result:
    cash = starting_balance
    position = 0.0
    peak = starting_balance
    max_dd = 0.0
    fees = 0.0
    events = 0
    long_entries = 0
    short_entries = 0

    for row, signal in zip(data.itertuples(index=False), signals, strict=True):
        price = float(row.close)
        equity = cash + position * price
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)

        if signal not in (1, -1):
            continue

        target_position = signal * equity / price if equity > 0 else 0.0
        delta = target_position - position
        if abs(delta) < 1e-12:
            continue

        fee = abs(delta) * price * fee_rate
        cash -= delta * price + fee
        position = target_position
        fees += fee
        events += 1
        long_entries += int(signal == 1)
        short_entries += int(signal == -1)

    last_price = float(data["close"].iloc[-1])
    if abs(position) > 1e-12:
        fee = abs(position) * last_price * fee_rate
        cash += position * last_price - fee
        fees += fee
        events += 1

    final_equity = cash
    net_pnl = final_equity - starting_balance
    return_pct = net_pnl / starting_balance * 100
    score = return_pct / max(max_dd, 1e-9)
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
        score=score,
    )


def base_cross_signals(data: pd.DataFrame, mask: pd.Series | None = None) -> pd.Series:
    spread = data["spread"]
    prev = spread.shift(1)
    buy = (prev <= 0) & (spread > 0)
    sell = (prev >= 0) & (spread < 0)
    if mask is not None:
        buy &= mask
        sell &= mask

    signals = pd.Series(0, index=data.index)
    signals[buy] = 1
    signals[sell] = -1
    return signals


def spread_confirm_signals(data: pd.DataFrame, threshold: float) -> pd.Series:
    spread = data["spread_pct"]
    prev = spread.shift(1)
    buy = (prev <= threshold) & (spread > threshold)
    sell = (prev >= -threshold) & (spread < -threshold)

    signals = pd.Series(0, index=data.index)
    signals[buy] = 1
    signals[sell] = -1
    return signals


def slope_filter_signals(data: pd.DataFrame, threshold: float) -> pd.Series:
    spread = data["spread"]
    prev = spread.shift(1)
    slope = data["ma20_slope_6h"]
    buy = (prev <= 0) & (spread > 0) & (slope > threshold)
    sell = (prev >= 0) & (spread < 0) & (slope < -threshold)

    signals = pd.Series(0, index=data.index)
    signals[buy] = 1
    signals[sell] = -1
    return signals


def evaluate_filters(
    data: pd.DataFrame,
    *,
    starting_balance: float,
    fee_rate: float,
) -> list[Result]:
    results: list[Result] = []

    def evaluate(name: str, signals: pd.Series) -> None:
        results.append(
            backtest(
                data,
                signals,
                name,
                starting_balance=starting_balance,
                fee_rate=fee_rate,
            )
        )

    evaluate("baseline_cross", base_cross_signals(data))

    for threshold in [0.0005, 0.001, 0.002, 0.003, 0.0035, 0.005, 0.0075, 0.01]:
        evaluate(f"spread_confirm>{threshold:.4%}", spread_confirm_signals(data, threshold))
    for threshold in [15, 20, 25, 30, 35]:
        evaluate(f"cross_adx14>={threshold}", base_cross_signals(data, data["adx14"] >= threshold))
    for threshold in [0.005, 0.0075, 0.01, 0.015, 0.02]:
        evaluate(
            f"cross_atr_pct>={threshold:.2%}",
            base_cross_signals(data, data["atr_pct"] >= threshold),
        )
    for multiplier in [1.0, 1.2, 1.5, 2.0]:
        evaluate(
            f"cross_volume>={multiplier:.1f}x_ma20",
            base_cross_signals(data, data["volume"] >= data["volume_ma20"] * multiplier),
        )
    for threshold in [0.0, 0.001, 0.002, 0.003, 0.005]:
        evaluate(f"cross_ma20_slope6h>{threshold:.2%}", slope_filter_signals(data, threshold))

    for threshold in [0.003, 0.0035, 0.004, 0.005]:
        signals = spread_confirm_signals(data, threshold).where(data["atr_pct"] >= 0.005, 0)
        evaluate(f"spread>{threshold:.2%}+atr>=0.50%", signals)

    return results


def print_results(results: list[Result], limit: int) -> None:
    ranked = sorted(results, key=lambda r: (r.score, r.return_pct), reverse=True)
    print("name,return_pct,max_dd_pct,net_pnl,fees,events,longs,shorts,score")
    for result in ranked[:limit]:
        print(
            f"{result.name},{result.return_pct:.2f},{result.max_drawdown_pct:.2f},"
            f"{result.net_pnl:.2f},{result.total_fees:.2f},{result.events},"
            f"{result.long_entries},{result.short_entries},{result.score:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare simple MA cross filters on ETH hourly candles."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--limit", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_hourly(args.parquet)
    results = evaluate_filters(
        data,
        starting_balance=args.starting_balance,
        fee_rate=args.fee_rate,
    )
    print_results(results, limit=args.limit)


if __name__ == "__main__":
    main()
