from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trend_trader.backtest.metrics import annualized_sharpe_ratio, timestamps_or_daily_index

try:
    from scripts.evaluate_eth_filters import Result, add_indicators, load_hourly
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from evaluate_eth_filters import Result, add_indicators, load_hourly


DEFAULT_DATA_DIR = Path("data/clean/okx/ETH-USDT-SWAP")


@dataclass(frozen=True)
class AdaptiveParameters:
    name: str
    spread_atr: float
    atr_percentile: float
    efficiency: float


DEFAULT_TEMPLATES = [
    AdaptiveParameters("adaptive_low_efficiency", 0.35, 0.35, 0.10),
    AdaptiveParameters("adaptive_balanced", 0.35, 0.35, 0.15),
    AdaptiveParameters("adaptive_high_spread", 0.45, 0.35, 0.15),
]


def add_adaptive_indicators(
    data: pd.DataFrame,
    *,
    atr_lookback: int = 24 * 90,
    efficiency_lookback: int = 24,
) -> None:
    """Add causal, scale-free regime indicators to an hourly candle frame."""
    if atr_lookback <= 1 or efficiency_lookback <= 1:
        raise ValueError("adaptive lookbacks must be greater than one")
    min_atr_periods = min(24 * 30, atr_lookback)
    data["atr_percentile"] = data["atr_pct"].rolling(
        atr_lookback,
        min_periods=min_atr_periods,
    ).rank(pct=True)
    path = data["close"].diff().abs().rolling(efficiency_lookback).sum()
    displacement = (data["close"] - data["close"].shift(efficiency_lookback)).abs()
    data["efficiency_ratio"] = displacement / path.replace(0.0, float("nan"))
    data["spread_atr"] = data["spread"].abs() / data["atr14"]


def adaptive_targets(
    data: pd.DataFrame,
    parameters: AdaptiveParameters,
) -> pd.Series:
    """Accept normalized spread crossings only in a qualified market regime."""
    signed_spread_atr = data["spread"] / data["atr14"]
    previous = signed_spread_atr.shift(1)
    regime = (data["atr_percentile"] >= parameters.atr_percentile) & (
        data["efficiency_ratio"] >= parameters.efficiency
    )
    signals = pd.Series(0.0, index=data.index)
    signals[
        (previous <= parameters.spread_atr)
        & (signed_spread_atr > parameters.spread_atr)
        & regime
    ] = 1
    signals[
        (previous >= -parameters.spread_atr)
        & (signed_spread_atr < -parameters.spread_atr)
        & regime
    ] = -1
    return signals.replace(0, float("nan")).ffill().fillna(0).astype(int)


def fixed_targets(
    data: pd.DataFrame,
    *,
    spread_threshold: float = 0.0035,
    atr_threshold: float = 0.005,
) -> pd.Series:
    """Represent the existing fixed filter as a continuously evaluated regime."""
    spread = data["spread_pct"]
    previous = spread.shift(1)
    signals = pd.Series(0.0, index=data.index)
    signals[
        (previous <= spread_threshold)
        & (spread > spread_threshold)
        & (data["atr_pct"] >= atr_threshold)
    ] = 1
    signals[
        (previous >= -spread_threshold)
        & (spread < -spread_threshold)
        & (data["atr_pct"] >= atr_threshold)
    ] = -1
    return signals.replace(0, float("nan")).ffill().fillna(0).astype(int)


def direction_consensus_targets(
    data: pd.DataFrame,
    targets: pd.Series,
    *,
    fast_days: int = 30,
    slow_days: int = 60,
) -> pd.Series:
    """Allow positions only when price agrees with both long-term averages."""
    if fast_days <= 0 or fast_days >= slow_days:
        raise ValueError("direction periods must satisfy 0 < fast_days < slow_days")
    fast_bars = fast_days * 24
    slow_bars = slow_days * 24
    fast_average = data["close"].rolling(fast_bars, min_periods=fast_bars).mean()
    slow_average = data["close"].rolling(slow_bars, min_periods=slow_bars).mean()
    allow_long = (data["close"] > fast_average) & (data["close"] > slow_average)
    allow_short = (data["close"] < fast_average) & (data["close"] < slow_average)
    allowed = ((targets > 0) & allow_long) | ((targets < 0) & allow_short)
    return targets.where(allowed, 0).astype(int)


def backtest_next_open(
    data: pd.DataFrame,
    targets: pd.Series,
    name: str,
    *,
    starting_balance: float,
    fee_rate: float,
) -> Result:
    """Trade a target decided at bar close at the following bar open."""
    cash = starting_balance
    position = 0.0
    peak = starting_balance
    max_dd = 0.0
    fees = 0.0
    events = 0
    long_entries = 0
    short_entries = 0
    trade_start_equity: float | None = None
    trade_pnls: list[float] = []
    desired = 0
    equity_curve: list[float] = []

    def trade_to(target: int, price: float) -> None:
        nonlocal cash, position, fees, events, long_entries, short_entries
        nonlocal trade_start_equity
        current = 1 if position > 0 else -1 if position < 0 else 0
        if target == current:
            return
        equity = cash + position * price
        if current != 0 and trade_start_equity is not None:
            close_fee = abs(position) * price * fee_rate
            trade_pnls.append(equity - close_fee - trade_start_equity)
        target_position = target * equity / price if target and equity > 0 else 0.0
        delta = target_position - position
        fee = abs(delta) * price * fee_rate
        cash -= delta * price + fee
        position = target_position
        fees += fee
        events += 1
        if target:
            trade_start_equity = cash + position * price
            long_entries += int(target == 1)
            short_entries += int(target == -1)
        else:
            trade_start_equity = None

    for row, target in zip(data.itertuples(index=False), targets, strict=True):
        trade_to(desired, float(row.open))
        desired = int(target)
        equity = cash + position * float(row.close)
        equity_curve.append(equity)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)

    if position:
        trade_to(0, float(data["close"].iloc[-1]))

    final_equity = cash
    if equity_curve:
        equity_curve[-1] = final_equity
    net_pnl = final_equity - starting_balance
    return_pct = net_pnl / starting_balance * 100
    wins = [pnl for pnl in trade_pnls if pnl > 0]
    losses = [pnl for pnl in trade_pnls if pnl < 0]
    trades = len(trade_pnls)
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = statistics.mean(losses) if losses else 0.0
    profit_loss_ratio = avg_win / abs(avg_loss) if losses else float("inf") if wins else 0.0
    return Result(
        name=name,
        final_equity=final_equity,
        net_pnl=net_pnl,
        return_pct=return_pct,
        max_drawdown_pct=max_dd,
        sharpe_ratio=annualized_sharpe_ratio(timestamps_or_daily_index(data), equity_curve),
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


def load_history(data_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for year in range(start_year, end_year + 1):
        path = data_dir / f"ETH-USDT-SWAP_1m_{year}.parquet"
        frame = load_hourly(path, fast_period=8, slow_period=20)
        frames.append(frame[["ts", "open", "high", "low", "close", "volume"]])
    data = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
    add_indicators(data, fast_period=8, slow_period=20)
    add_adaptive_indicators(data)
    return data


def annual_results(
    data: pd.DataFrame,
    templates: list[AdaptiveParameters],
    *,
    starting_balance: float,
    fee_rate: float,
) -> list[tuple[int, Result]]:
    rows: list[tuple[int, Result]] = []
    targets = {"fixed": fixed_targets(data)}
    targets.update({item.name: adaptive_targets(data, item) for item in templates})
    balanced = next(
        (item for item in templates if item.name == "adaptive_balanced"),
        None,
    )
    if balanced is not None:
        targets["adaptive_balanced_direction_30d_60d"] = direction_consensus_targets(
            data,
            targets[balanced.name],
        )
    for year in data["ts"].dt.year.drop_duplicates():
        mask = data["ts"].dt.year == year
        year_data = data.loc[mask].reset_index(drop=True)
        for name, full_targets in targets.items():
            year_targets = full_targets.loc[mask].reset_index(drop=True)
            rows.append(
                (
                    int(year),
                    backtest_next_open(
                        year_data,
                        year_targets,
                        name,
                        starting_balance=starting_balance,
                        fee_rate=fee_rate,
                    ),
                )
            )
    return rows


def print_annual_results(rows: list[tuple[int, Result]]) -> None:
    print("year,strategy,return_pct,max_dd_pct,sharpe_ratio,trades,win_rate_pct,pl_ratio,fees")
    for year, result in rows:
        print(
            f"{year},{result.name},{result.return_pct:.2f},"
            f"{result.max_drawdown_pct:.2f},{result.sharpe_ratio:.3f},"
            f"{result.trades},"
            f"{result.win_rate_pct:.2f},{result.profit_loss_ratio:.3f},"
            f"{result.total_fees:.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate causal adaptive MA8/20 filters on ETH hourly candles."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_history(args.data_dir, args.start_year, args.end_year)
    print_annual_results(
        annual_results(
            data,
            DEFAULT_TEMPLATES,
            starting_balance=args.starting_balance,
            fee_rate=args.fee_rate,
        )
    )


if __name__ == "__main__":
    main()
