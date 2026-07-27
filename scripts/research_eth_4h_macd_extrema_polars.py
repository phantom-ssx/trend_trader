from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

DEFAULT_DATA_DIR = Path("data/clean/okx/ETH-USDT-SWAP")


@dataclass(frozen=True)
class Trade:
    side: str
    entry_time: datetime
    exit_time: datetime
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
) -> pl.DataFrame:
    """Load complete annual minute files and causally aggregate them into UTC 4h bars."""
    paths = [
        data_dir / f"ETH-USDT-SWAP_1m_{year}.parquet"
        for year in range(start_year, end_year + 1)
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
                pl.col("open"),
                pl.col("close"),
            )
        )

    return (
        pl.concat(frames)
        .sort("ts")
        .unique(subset=["ts"], keep="last", maintain_order=True)
        .group_by_dynamic("ts", every="4h", closed="left", label="left")
        .agg(
            pl.col("open").first(),
            pl.col("close").last(),
            pl.len().alias("minute_count"),
        )
        .filter(pl.col("minute_count") == 240)
        .drop("minute_count")
        .collect(engine="streaming")
    )


def add_macd(
    data: pl.DataFrame,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.DataFrame:
    """Return a new frame with close-based DIF, DEA, and doubled MACD histogram."""
    if not 0 < fast < slow or signal <= 0:
        raise ValueError("MACD periods must satisfy 0 < fast < slow and signal > 0")

    result = data.with_columns(
        (
            pl.col("close").ewm_mean(span=fast, adjust=False, min_samples=fast)
            - pl.col("close").ewm_mean(span=slow, adjust=False, min_samples=slow)
        ).alias("dif")
    )
    return result.with_columns(
        pl.col("dif")
        .ewm_mean(span=signal, adjust=False, min_samples=signal)
        .alias("dea")
    ).with_columns(((pl.col("dif") - pl.col("dea")) * 2.0).alias("macd"))


def macd_extrema_targets(
    data: pl.DataFrame,
    *,
    column: str = "dif",
    require_expected_sign: bool = False,
) -> pl.Series:
    """Fade confirmed local MACD extrema, retaining the position until the opposite extremum.

    An extremum at bar t-1 is only recognized when bar t has closed. The returned target is
    therefore stamped at t; the backtester executes it at t+1 open.
    """
    value = pl.col(column)
    previous = value.shift(1)
    older = value.shift(2)
    peak_confirmed = (previous > older) & (previous > value)
    trough_confirmed = (previous < older) & (previous < value)
    if require_expected_sign:
        peak_confirmed &= previous > 0
        trough_confirmed &= previous < 0

    return data.select(
        pl.when(peak_confirmed)
        .then(-1)
        .when(trough_confirmed)
        .then(1)
        .otherwise(None)
        .forward_fill()
        .fill_null(0)
        .alias("target")
    ).get_column("target")


def _max_drawdown_pct(equities: pl.Series, starting_balance: float) -> float:
    values = pl.concat(
        [
            pl.Series("equity", [starting_balance], dtype=pl.Float64),
            equities,
        ]
    )
    drawdowns = values / values.cum_max() - 1.0
    return float(-drawdowns.min() * 100.0)


def _annualized_sharpe_ratio(
    timestamps: pl.Series,
    equities: pl.Series,
    *,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe from daily equity returns with a zero risk-free rate."""
    if len(timestamps) != len(equities):
        raise ValueError("timestamps and equities must have equal length")
    if len(equities) < 2:
        return 0.0

    daily = (
        pl.DataFrame(
            {
                "ts": timestamps,
                "equity": equities,
            }
        )
        .sort("ts")
        .unique(subset=["ts"], keep="last", maintain_order=True)
        .group_by_dynamic("ts", every="1d", closed="left", label="left")
        .agg(pl.col("equity").last())
        .sort("ts")
        .upsample(time_column="ts", every="1d")
        .with_columns(pl.col("equity").forward_fill())
        .select(pl.col("equity").pct_change().drop_nulls().alias("return"))
        .get_column("return")
    )
    if len(daily) < 2:
        return 0.0
    volatility = float(daily.std(ddof=1))
    if not math.isfinite(volatility) or volatility <= 0:
        return 0.0
    sharpe = float(daily.mean()) / volatility * math.sqrt(periods_per_year)
    return sharpe if math.isfinite(sharpe) else 0.0


def _empty_trade_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "side": pl.String,
            "entry_time": pl.Datetime(time_zone="UTC"),
            "exit_time": pl.Datetime(time_zone="UTC"),
            "entry_price": pl.Float64,
            "exit_price": pl.Float64,
            "pnl": pl.Float64,
            "return_pct": pl.Float64,
            "holding_bars": pl.Int64,
        }
    )


def run_backtest(
    data: pl.DataFrame,
    targets: pl.Series,
    *,
    starting_balance: float = 10_000.0,
    fee_rate: float = 0.0005,
) -> tuple[BacktestSummary, pl.DataFrame, pl.DataFrame]:
    """Run a 1x notional, next-open backtest with explicit close/open fees on reversals."""
    if len(data) != len(targets):
        raise ValueError("data and targets must have equal length")
    if data.is_empty():
        raise ValueError("data must contain at least one bar")
    if starting_balance <= 0 or fee_rate < 0:
        raise ValueError("starting_balance must be positive and fee_rate non-negative")

    cash = starting_balance
    quantity = 0.0
    desired = 0
    fees = 0.0
    entry_time: datetime | None = None
    entry_price = 0.0
    entry_equity = 0.0
    entry_bar = 0
    trades: list[Trade] = []
    curve: list[dict[str, object]] = []

    def close_position(price: float, timestamp: datetime, bar_number: int) -> None:
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

    def open_position(
        target: int, price: float, timestamp: datetime, bar_number: int
    ) -> None:
        nonlocal cash, quantity, fees, entry_time, entry_price, entry_equity, entry_bar
        if target == 0:
            return
        if cash <= 0:
            raise RuntimeError(
                "Strategy equity is non-positive; cannot open a new position"
            )
        entry_equity = cash
        quantity = target * cash / price
        open_fee = abs(quantity) * price * fee_rate
        cash -= quantity * price + open_fee
        fees += open_fee
        entry_time = timestamp
        entry_price = price
        entry_bar = bar_number

    for bar_number, (row, target) in enumerate(
        zip(data.iter_rows(named=True), targets, strict=True)
    ):
        timestamp = row["ts"]
        if not isinstance(timestamp, datetime):
            raise TypeError("data column 'ts' must contain datetime values")
        current = 1 if quantity > 0 else -1 if quantity < 0 else 0
        if desired != current:
            close_position(row["open"], timestamp, bar_number)
            open_position(desired, row["open"], timestamp, bar_number)

        close_price = row["close"]
        equity = cash + quantity * close_price
        curve.append(
            {
                "ts": timestamp,
                "equity": equity,
                "target": desired,
                "close": close_price,
            }
        )
        desired = int(target)

    if quantity != 0.0:
        last = data.row(-1, named=True)
        close_position(last["close"], last["ts"], len(data) - 1)
        curve[-1]["equity"] = cash
        curve[-1]["target"] = 0

    equity_curve = pl.from_dicts(curve)
    trade_frame = (
        pl.from_dicts([asdict(trade) for trade in trades])
        if trades
        else _empty_trade_frame()
    )
    final_equity = equity_curve.get_column("equity")[-1]
    elapsed_years = max(
        (data.get_column("ts")[-1] - data.get_column("ts")[0]).total_seconds()
        / (365.25 * 24 * 3600),
        1 / 365.25,
    )
    cagr = (final_equity / starting_balance) ** (1.0 / elapsed_years) - 1.0

    pnl = trade_frame.get_column("pnl")
    gross_profit = pnl.filter(pnl > 0).sum()
    gross_loss = -pnl.filter(pnl < 0).sum()
    profit_factor = (
        gross_profit / gross_loss if gross_loss else math.inf if gross_profit else 0.0
    )
    long_trades = trade_frame.filter(pl.col("side") == "long")
    short_trades = trade_frame.filter(pl.col("side") == "short")
    summary = BacktestSummary(
        start=str(data.get_column("ts")[0]),
        end=str(data.get_column("ts")[-1]),
        bars=len(data),
        final_equity=final_equity,
        total_return_pct=(final_equity / starting_balance - 1.0) * 100.0,
        cagr_pct=cagr * 100.0,
        max_drawdown_pct=_max_drawdown_pct(
            equity_curve.get_column("equity"),
            starting_balance,
        ),
        sharpe_ratio=_annualized_sharpe_ratio(
            equity_curve.get_column("ts"),
            equity_curve.get_column("equity"),
        ),
        trades=len(trade_frame),
        win_rate_pct=(
            float((pnl > 0).mean() * 100.0) if not trade_frame.is_empty() else 0.0
        ),
        profit_factor=profit_factor,
        total_fees=fees,
        long_trades=len(long_trades),
        long_pnl=long_trades.get_column("pnl").sum(),
        short_trades=len(short_trades),
        short_pnl=short_trades.get_column("pnl").sum(),
    )
    return summary, equity_curve, trade_frame


def period_results(
    data: pl.DataFrame,
    equity_curve: pl.DataFrame,
    *,
    starting_balance: float,
    period: str,
) -> pl.DataFrame:
    """Calculate independently chained calendar-period returns and price regimes."""
    if period not in {"year", "month"}:
        raise ValueError("period must be 'year' or 'month'")

    frame = data.select("ts", "open", "close").with_columns(
        equity_curve.get_column("equity"),
        pl.col("ts").dt.strftime("%Y" if period == "year" else "%Y-%m").alias("period"),
    )
    previous_equity = starting_balance
    rows: list[dict[str, float | int | str]] = []
    for group in frame.partition_by("period", maintain_order=True):
        label = group.get_column("period")[0]
        end_equity = float(group.get_column("equity")[-1])
        strategy_return = end_equity / previous_equity - 1.0
        eth_return = float(
            group.get_column("close")[-1] / group.get_column("open")[0] - 1.0
        )
        close_returns = group.get_column("close").pct_change().drop_nulls()
        path = float(close_returns.abs().sum())
        efficiency = abs(eth_return) / path if path > 0 else 0.0
        realized_vol = (
            float(close_returns.std(ddof=1) * math.sqrt(6 * 365))
            if len(close_returns) > 1
            else 0.0
        )
        rows.append(
            {
                "period": label,
                "strategy_return_pct": strategy_return * 100.0,
                "end_equity": end_equity,
                "max_drawdown_pct": _max_drawdown_pct(
                    group.get_column("equity"),
                    previous_equity,
                ),
                "eth_return_pct": eth_return * 100.0,
                "path_efficiency": efficiency,
                "realized_vol_pct": realized_vol * 100.0,
                "bars": len(group),
            }
        )
        previous_equity = end_equity
    return pl.from_dicts(rows)


def regime_results(monthly: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    """Summarize monthly performance across ex-post price-path regimes."""
    data = monthly.with_columns(
        pl.col("path_efficiency")
        .rank(method="ordinal")
        .qcut(
            [1 / 3, 2 / 3],
            labels=["choppy_low", "mixed_mid", "trending_high"],
        )
        .alias("efficiency_regime"),
        pl.col("eth_return_pct")
        .cut(
            [-10.0, 10.0],
            labels=["down_below_-10pct", "range_-10_to_10pct", "up_above_10pct"],
        )
        .alias("direction_regime"),
    )

    rows: list[dict[str, float | int | str]] = []
    for dimension in ["efficiency_regime", "direction_regime"]:
        for group in data.partition_by(dimension, maintain_order=True):
            rows.append(
                {
                    "dimension": dimension,
                    "regime": str(group.get_column(dimension)[0]),
                    "months": len(group),
                    "profitable_months": int(
                        (group.get_column("strategy_return_pct") > 0).sum()
                    ),
                    "avg_strategy_return_pct": float(
                        group.get_column("strategy_return_pct").mean()
                    ),
                    "median_strategy_return_pct": float(
                        group.get_column("strategy_return_pct").median()
                    ),
                    "avg_eth_return_pct": float(
                        group.get_column("eth_return_pct").mean()
                    ),
                    "avg_path_efficiency": float(
                        group.get_column("path_efficiency").mean()
                    ),
                    "avg_realized_vol_pct": float(
                        group.get_column("realized_vol_pct").mean()
                    ),
                }
            )

    def correlation(other: str | pl.Expr) -> float:
        return float(
            data.select(
                pl.corr("strategy_return_pct", other).alias("correlation")
            ).item()
        )

    correlations = {
        "strategy_vs_path_efficiency": correlation("path_efficiency"),
        "strategy_vs_abs_eth_return": correlation(pl.col("eth_return_pct").abs()),
        "strategy_vs_realized_vol": correlation("realized_vol_pct"),
    }
    return pl.from_dicts(rows), correlations


def diagnostic_variant_results(
    data: pl.DataFrame,
    *,
    starting_balance: float,
    fee_rate: float,
) -> pl.DataFrame:
    """Run fixed, non-optimized variants which help attribute baseline behavior."""
    baseline = macd_extrema_targets(data)
    variants = {
        "dif_line_both_sides": baseline,
        "dif_line_long_only": baseline.clip(lower_bound=0),
        "dif_line_short_only": baseline.clip(upper_bound=0),
        "macd_histogram_both_sides": macd_extrema_targets(data, column="macd"),
        "dif_line_zero_side_extrema": macd_extrema_targets(
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
    return pl.from_dicts(rows)


def _print_frame(frame: pl.DataFrame) -> None:
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=2):
        print(frame)


def print_report(
    summary: BacktestSummary,
    no_fee_summary: BacktestSummary,
    yearly: pl.DataFrame,
    monthly: pl.DataFrame,
    regimes: pl.DataFrame,
    variants: pl.DataFrame,
    correlations: dict[str, float],
    *,
    buy_hold_return_pct: float,
) -> None:
    print("\nOVERALL")
    _print_frame(
        pl.DataFrame(
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
    _print_frame(monthly.top_k(12, by="strategy_return_pct"))
    print("\nWORST 12 MONTHS")
    _print_frame(monthly.bottom_k(12, by="strategy_return_pct"))
    print("\nREGIMES")
    _print_frame(regimes)
    print("\nFIXED DIAGNOSTIC VARIANTS")
    _print_frame(variants)
    print("\nCORRELATIONS")
    _print_frame(pl.DataFrame([correlations]))


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
    buy_hold_return_pct = (
        data.get_column("close")[-1] / data.get_column("open")[0] - 1.0
    ) * 100.0
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
        pl.DataFrame(
            [
                {
                    **asdict(summary),
                    "buy_hold_return_pct": buy_hold_return_pct,
                    "no_fee_return_pct": no_fee_summary.total_return_pct,
                    **correlations,
                }
            ]
        ).write_csv(args.output_dir / "summary.csv")
        for filename, frame in {
            "yearly.csv": yearly,
            "monthly.csv": monthly,
            "regimes.csv": regimes,
            "diagnostic_variants.csv": variants,
            "trades.csv": trades,
            "equity_curve.csv": equity_curve,
        }.items():
            frame.write_csv(args.output_dir / filename)


if __name__ == "__main__":
    main()
