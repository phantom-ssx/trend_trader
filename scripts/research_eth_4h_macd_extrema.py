from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from trend_trader.backtest.metrics import annualized_sharpe_ratio

DEFAULT_DATA_DIR = Path("data/clean/okx/ETH-USDT-SWAP")


@dataclass(frozen=True)
class Trade:
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    holding_bars: int


@dataclass(frozen=True)
class BacktestSummary:
    start: str
    end: str
    bars: int
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: int
    win_rate_pct: float
    profit_factor: float
    total_fees: float
    long_trades: int
    long_pnl: float
    short_trades: int
    short_pnl: float


def load_4h_history(
    data_dir: Path,
    *,
    start_year: int = 2020,
    end_year: int = 2026,
) -> pd.DataFrame:
    """Load complete annual minute files and causally aggregate them into UTC 4h bars."""
    paths = [
        data_dir / f"ETH-USDT-SWAP_1m_{year}.parquet" for year in range(start_year, end_year + 1)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing annual ETH candle files: {missing}")

    frames: list[pl.LazyFrame] = []
    for path in paths:
        schema = pl.read_parquet_schema(path)
        timestamp = "ts" if "ts" in schema else "timestamp"
        frames.append(
            pl.scan_parquet(path).select(
                pl.col(timestamp).alias("ts"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            )
        )

    bars = (
        pl.concat(frames)
        .sort("ts")
        .unique(subset=["ts"], keep="last", maintain_order=True)
        .group_by_dynamic("ts", every="4h", closed="left", label="left")
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.len().alias("minute_count"),
        )
        .filter(pl.col("minute_count") == 240)
        .drop("minute_count")
        .collect(engine="streaming")
        .sort("ts")
    )
    data = bars.to_pandas()
    data["ts"] = pd.to_datetime(data["ts"], utc=True)
    return data


def add_macd(
    data: pd.DataFrame,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Return a copy with standard close-based MACD columns."""
    if not 0 < fast < slow or signal <= 0:
        raise ValueError("MACD periods must satisfy 0 < fast < slow and signal > 0")
    result = data.copy()
    fast_ema = result["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = result["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    result["macd"] = fast_ema - slow_ema
    result["macd_signal"] = (
        result["macd"]
        .ewm(
            span=signal,
            adjust=False,
            min_periods=signal,
        )
        .mean()
    )
    result["macd_histogram"] = result["macd"] - result["macd_signal"]
    return result


def macd_extrema_targets(
    data: pd.DataFrame,
    *,
    column: str = "macd",
    require_expected_sign: bool = False,
) -> pd.Series:
    """Fade confirmed local MACD extrema, retaining the position until the opposite extremum.

    An extremum at bar t-1 is only recognized when bar t has closed. The returned target is
    therefore stamped at t; the backtester executes it at t+1 open.
    """
    value = data[column]
    peak_confirmed = (value.shift(1) > value.shift(2)) & (value.shift(1) > value)
    trough_confirmed = (value.shift(1) < value.shift(2)) & (value.shift(1) < value)
    if require_expected_sign:
        peak_confirmed &= value.shift(1) > 0
        trough_confirmed &= value.shift(1) < 0
    changes = pd.Series(np.nan, index=data.index, dtype=float)
    changes.loc[peak_confirmed] = -1.0
    changes.loc[trough_confirmed] = 1.0
    return changes.ffill().fillna(0.0).astype(int)


def _max_drawdown_pct(equities: pd.Series, starting_balance: float) -> float:
    values = pd.concat(
        [pd.Series([starting_balance], dtype=float), equities.reset_index(drop=True)],
        ignore_index=True,
    )
    drawdowns = values / values.cummax() - 1.0
    return float(-drawdowns.min() * 100.0)


def run_backtest(
    data: pd.DataFrame,
    targets: pd.Series,
    *,
    starting_balance: float = 10_000.0,
    fee_rate: float = 0.0005,
) -> tuple[BacktestSummary, pd.DataFrame, pd.DataFrame]:
    """Run a 1x notional, next-open backtest with explicit close/open fees on reversals."""
    if len(data) != len(targets):
        raise ValueError("data and targets must have equal length")
    if starting_balance <= 0 or fee_rate < 0:
        raise ValueError("starting_balance must be positive and fee_rate non-negative")

    cash = starting_balance
    quantity = 0.0
    desired = 0
    fees = 0.0
    entry_time: pd.Timestamp | None = None
    entry_price = 0.0
    entry_equity = 0.0
    entry_bar = 0
    trades: list[Trade] = []
    curve: list[dict[str, object]] = []

    def close_position(price: float, timestamp: pd.Timestamp, bar_number: int) -> None:
        nonlocal cash, quantity, fees, entry_time
        if quantity == 0.0 or entry_time is None:
            return
        side = "long" if quantity > 0 else "short"
        close_fee = abs(quantity) * price * fee_rate
        cash += quantity * price - close_fee
        fees += close_fee
        pnl = cash - entry_equity
        trades.append(
            Trade(
                side=side,
                entry_time=entry_time,
                exit_time=timestamp,
                entry_price=entry_price,
                exit_price=price,
                pnl=pnl,
                return_pct=pnl / entry_equity * 100.0,
                holding_bars=bar_number - entry_bar,
            )
        )
        quantity = 0.0
        entry_time = None

    def open_position(target: int, price: float, timestamp: pd.Timestamp, bar_number: int) -> None:
        nonlocal cash, quantity, fees, entry_time, entry_price, entry_equity, entry_bar
        if target == 0:
            return
        if cash <= 0:
            raise RuntimeError("Strategy equity is non-positive; cannot open a new position")
        entry_equity = cash
        quantity = target * cash / price
        open_fee = abs(quantity) * price * fee_rate
        cash -= quantity * price + open_fee
        fees += open_fee
        entry_time = timestamp
        entry_price = price
        entry_bar = bar_number

    for bar_number, (row, target) in enumerate(
        zip(data.itertuples(index=False), targets, strict=True)
    ):
        timestamp = pd.Timestamp(row.ts)
        current = 1 if quantity > 0 else -1 if quantity < 0 else 0
        if desired != current:
            close_position(float(row.open), timestamp, bar_number)
            open_position(desired, float(row.open), timestamp, bar_number)

        equity = cash + quantity * float(row.close)
        curve.append(
            {
                "ts": timestamp,
                "equity": equity,
                "target": desired,
                "close": float(row.close),
            }
        )
        desired = int(target)

    if quantity != 0.0:
        last = data.iloc[-1]
        close_position(float(last["close"]), pd.Timestamp(last["ts"]), len(data) - 1)
        curve[-1]["equity"] = cash
        curve[-1]["target"] = 0

    equity_curve = pd.DataFrame(curve)
    trade_frame = pd.DataFrame(asdict(trade) for trade in trades)
    final_equity = float(equity_curve["equity"].iloc[-1])
    elapsed_years = max(
        (pd.Timestamp(data["ts"].iloc[-1]) - pd.Timestamp(data["ts"].iloc[0])).total_seconds()
        / (365.25 * 24 * 3600),
        1 / 365.25,
    )
    cagr = (final_equity / starting_balance) ** (1.0 / elapsed_years) - 1.0
    wins = trade_frame.loc[trade_frame["pnl"] > 0, "pnl"] if not trade_frame.empty else []
    losses = trade_frame.loc[trade_frame["pnl"] < 0, "pnl"] if not trade_frame.empty else []
    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss else math.inf if gross_profit else 0.0
    long_mask = trade_frame["side"].eq("long") if not trade_frame.empty else pd.Series(dtype=bool)
    short_mask = trade_frame["side"].eq("short") if not trade_frame.empty else pd.Series(dtype=bool)
    summary = BacktestSummary(
        start=str(pd.Timestamp(data["ts"].iloc[0])),
        end=str(pd.Timestamp(data["ts"].iloc[-1])),
        bars=len(data),
        final_equity=final_equity,
        total_return_pct=(final_equity / starting_balance - 1.0) * 100.0,
        cagr_pct=cagr * 100.0,
        max_drawdown_pct=_max_drawdown_pct(equity_curve["equity"], starting_balance),
        sharpe_ratio=annualized_sharpe_ratio(
            equity_curve["ts"].to_list(),
            equity_curve["equity"].to_list(),
        ),
        trades=len(trade_frame),
        win_rate_pct=float((trade_frame["pnl"] > 0).mean() * 100.0)
        if not trade_frame.empty
        else 0.0,
        profit_factor=profit_factor,
        total_fees=fees,
        long_trades=int(long_mask.sum()),
        long_pnl=float(trade_frame.loc[long_mask, "pnl"].sum()) if not trade_frame.empty else 0.0,
        short_trades=int(short_mask.sum()),
        short_pnl=float(trade_frame.loc[short_mask, "pnl"].sum()) if not trade_frame.empty else 0.0,
    )
    return summary, equity_curve, trade_frame


def period_results(
    data: pd.DataFrame,
    equity_curve: pd.DataFrame,
    *,
    starting_balance: float,
    period: str,
) -> pd.DataFrame:
    """Calculate independently chained calendar-period returns and price regimes."""
    if period not in {"year", "month"}:
        raise ValueError("period must be 'year' or 'month'")
    frame = data[["ts", "open", "close"]].copy()
    frame["equity"] = equity_curve["equity"].to_numpy()
    frame["period"] = frame["ts"].dt.strftime("%Y" if period == "year" else "%Y-%m")
    previous_equity = starting_balance
    rows: list[dict[str, float | int | str]] = []
    for label, group in frame.groupby("period", sort=True):
        end_equity = float(group["equity"].iloc[-1])
        strategy_return = end_equity / previous_equity - 1.0
        eth_return = float(group["close"].iloc[-1] / group["open"].iloc[0] - 1.0)
        close_returns = group["close"].pct_change(fill_method=None).dropna()
        path = float(close_returns.abs().sum())
        efficiency = abs(eth_return) / path if path > 0 else 0.0
        period_equity = pd.concat(
            [pd.Series([previous_equity]), group["equity"].reset_index(drop=True)],
            ignore_index=True,
        )
        drawdown = float(-(period_equity / period_equity.cummax() - 1.0).min())
        realized_vol = (
            float(close_returns.std(ddof=1) * math.sqrt(6 * 365)) if len(close_returns) > 1 else 0.0
        )
        rows.append(
            {
                "period": label,
                "strategy_return_pct": strategy_return * 100.0,
                "end_equity": end_equity,
                "max_drawdown_pct": drawdown * 100.0,
                "eth_return_pct": eth_return * 100.0,
                "path_efficiency": efficiency,
                "realized_vol_pct": realized_vol * 100.0,
                "bars": len(group),
            }
        )
        previous_equity = end_equity
    return pd.DataFrame(rows)


def regime_results(monthly: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Summarize monthly performance across ex-post price-path regimes."""
    data = monthly.copy()
    data["efficiency_regime"] = pd.qcut(
        data["path_efficiency"].rank(method="first"),
        3,
        labels=["choppy_low", "mixed_mid", "trending_high"],
    )
    data["direction_regime"] = pd.cut(
        data["eth_return_pct"],
        bins=[-math.inf, -10.0, 10.0, math.inf],
        labels=["down_below_-10pct", "range_-10_to_10pct", "up_above_10pct"],
    )
    rows: list[dict[str, float | int | str]] = []
    for dimension in ["efficiency_regime", "direction_regime"]:
        for label, group in data.groupby(dimension, observed=True):
            rows.append(
                {
                    "dimension": dimension,
                    "regime": str(label),
                    "months": len(group),
                    "profitable_months": int((group["strategy_return_pct"] > 0).sum()),
                    "avg_strategy_return_pct": float(group["strategy_return_pct"].mean()),
                    "median_strategy_return_pct": float(group["strategy_return_pct"].median()),
                    "avg_eth_return_pct": float(group["eth_return_pct"].mean()),
                    "avg_path_efficiency": float(group["path_efficiency"].mean()),
                    "avg_realized_vol_pct": float(group["realized_vol_pct"].mean()),
                }
            )
    correlations = {
        "strategy_vs_path_efficiency": float(
            data["strategy_return_pct"].corr(data["path_efficiency"])
        ),
        "strategy_vs_abs_eth_return": float(
            data["strategy_return_pct"].corr(data["eth_return_pct"].abs())
        ),
        "strategy_vs_realized_vol": float(
            data["strategy_return_pct"].corr(data["realized_vol_pct"])
        ),
    }
    return pd.DataFrame(rows), correlations


def diagnostic_variant_results(
    data: pd.DataFrame,
    *,
    starting_balance: float,
    fee_rate: float,
) -> pd.DataFrame:
    """Run fixed, non-optimized variants which help attribute baseline behavior."""
    baseline = macd_extrema_targets(data)
    variants = {
        "macd_line_both_sides": baseline,
        "macd_line_long_only": baseline.where(baseline > 0, 0),
        "macd_line_short_only": baseline.where(baseline < 0, 0),
        "macd_histogram_both_sides": macd_extrema_targets(data, column="macd_histogram"),
        "macd_line_zero_side_extrema": macd_extrema_targets(
            data,
            require_expected_sign=True,
        ),
    }
    rows: list[dict[str, float | int | str]] = []
    for name, targets in variants.items():
        result, _, _ = run_backtest(
            data,
            targets,
            starting_balance=starting_balance,
            fee_rate=fee_rate,
        )
        no_fee_result, _, _ = run_backtest(
            data,
            targets,
            starting_balance=starting_balance,
            fee_rate=0.0,
        )
        rows.append(
            {
                "variant": name,
                "return_pct": result.total_return_pct,
                "no_fee_return_pct": no_fee_result.total_return_pct,
                "cagr_pct": result.cagr_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "trades": result.trades,
                "win_rate_pct": result.win_rate_pct,
                "profit_factor": result.profit_factor,
                "total_fees": result.total_fees,
                "long_pnl": result.long_pnl,
                "short_pnl": result.short_pnl,
            }
        )
    return pd.DataFrame(rows)


def _print_frame(frame: pd.DataFrame) -> None:
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


def print_report(
    summary: BacktestSummary,
    no_fee_summary: BacktestSummary,
    yearly: pd.DataFrame,
    monthly: pd.DataFrame,
    regimes: pd.DataFrame,
    variants: pd.DataFrame,
    correlations: dict[str, float],
    *,
    buy_hold_return_pct: float,
) -> None:
    print("\nOVERALL")
    _print_frame(
        pd.DataFrame(
            [
                {
                    **asdict(summary),
                    "buy_hold_return_pct": buy_hold_return_pct,
                    "no_fee_return_pct": no_fee_summary.total_return_pct,
                }
            ]
        )
    )
    print("\nYEARLY")
    _print_frame(yearly)
    print("\nBEST 12 MONTHS")
    _print_frame(monthly.nlargest(12, "strategy_return_pct"))
    print("\nWORST 12 MONTHS")
    _print_frame(monthly.nsmallest(12, "strategy_return_pct"))
    print("\nREGIMES")
    _print_frame(regimes)
    print("\nFIXED DIAGNOSTIC VARIANTS")
    _print_frame(variants)
    print("\nCORRELATIONS")
    _print_frame(pd.DataFrame([correlations]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research a causal ETH 4h MACD local-extrema mean-reversion strategy."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--fast", type=int, default=12)
    parser.add_argument("--slow", type=int, default=26)
    parser.add_argument("--signal", type=int, default=9)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = add_macd(
        load_4h_history(
            args.data_dir,
            start_year=args.start_year,
            end_year=args.end_year,
        ),
        fast=args.fast,
        slow=args.slow,
        signal=args.signal,
    )
    targets = macd_extrema_targets(data)
    summary, equity_curve, trades = run_backtest(
        data,
        targets,
        starting_balance=args.starting_balance,
        fee_rate=args.fee_rate,
    )
    no_fee_summary, _, _ = run_backtest(
        data,
        targets,
        starting_balance=args.starting_balance,
        fee_rate=0.0,
    )
    yearly = period_results(
        data,
        equity_curve,
        starting_balance=args.starting_balance,
        period="year",
    )
    monthly = period_results(
        data,
        equity_curve,
        starting_balance=args.starting_balance,
        period="month",
    )
    regimes, correlations = regime_results(monthly)
    variants = diagnostic_variant_results(
        data,
        starting_balance=args.starting_balance,
        fee_rate=args.fee_rate,
    )
    buy_hold_return_pct = (data["close"].iloc[-1] / data["open"].iloc[0] - 1.0) * 100.0
    print_report(
        summary,
        no_fee_summary,
        yearly,
        monthly,
        regimes,
        variants,
        correlations,
        buy_hold_return_pct=buy_hold_return_pct,
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    **asdict(summary),
                    "buy_hold_return_pct": buy_hold_return_pct,
                    "no_fee_return_pct": no_fee_summary.total_return_pct,
                    **correlations,
                }
            ]
        ).to_csv(
            args.output_dir / "summary.csv",
            index=False,
        )
        yearly.to_csv(args.output_dir / "yearly.csv", index=False)
        monthly.to_csv(args.output_dir / "monthly.csv", index=False)
        regimes.to_csv(args.output_dir / "regimes.csv", index=False)
        variants.to_csv(args.output_dir / "diagnostic_variants.csv", index=False)
        trades.to_csv(args.output_dir / "trades.csv", index=False)
        equity_curve.to_csv(args.output_dir / "equity_curve.csv", index=False)


if __name__ == "__main__":
    main()
