from __future__ import annotations
from dataclasses import dataclass, asdict
from collections.abc import Sequence
import polars as pl
from pathlib import Path
import math
import datetime

DEFAULT_DATA_DIR = Path("data/clean/okx/ETH-USDT-SWAP")


@dataclass(frozen=True)
class Trade:
    side: str
    entry_time: datetime.datetime
    exit_time: datetime.datetime
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
    return bars


def add_macd(
    data: pl.DataFrame,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.DataFrame:
    """Return a copy with standard close-based MACD columns."""
    if not 0 < fast < slow or signal <= 0:
        raise ValueError("MACD periods must satisfy 0 < fast < slow and signal > 0")
    fast_ema = pl.col("close").ewm_mean(com=fast)
    slow_ema = pl.col("close").ewm_mean(com=slow)
    diff = fast_ema - slow_ema
    dea = diff.ewm_mean(com=signal)
    macd = diff - dea
    data = data.with_columns(
        fast_ema.alias("fast_ema"),
        slow_ema.alias("slow_ema"),
        diff.alias("diff"),
        dea.alias("dea"),
        macd.alias("macd"),
    )
    return data


def macd_extrema_targets(
    data: pl.DataFrame,
    *,
    column: str = "macd",
    require_expected_sign: bool = False,
) -> pl.Series:
    """Fade confirmed local MACD extrema, retaining the position until the opposite extremum.

    An extremum at bar t-1 is only recognized when bar t has closed. The returned target is
    therefore stamped at t; the backtester executes it at t+1 open.
    """
    # value = data[column]
    # peak_confirmed = (value.shift(1) > value.shift(2)) & (value.shift(1) > value) # value.shift(1)是极大值
    # trough_confirmed = (value.shift(1) < value.shift(2)) & (value.shift(1) < value) # value.shift(1)是极小值
    # if require_expected_sign:
    #     peak_confirmed &= value.shift(1) > 0 #绝对值值为正
    #     trough_confirmed &= value.shift(1) < 0 #绝对值为负
    col = pl.col(column)
    peak_confirmed = (col.shift(1) > col.shift(2)) & (col.shift(1) > col)
    trough_confirmed = (col.shift(1) < col.shift(2)) & (col.shift(1) < col)
    if require_expected_sign:
        peak_confirmed &= col.shift(1) > 0
        trough_confirmed &= col.shift(1) < 0
    state = data.select(
        pl.when(col.is_null())
        .then(float("nan"))
        .when(peak_confirmed)
        .then(1.0)
        .when(trough_confirmed)
        .then(-1.0)
        .otherwise(0.0)
    ).to_series()
    return state.forward_fill().fill_nan(0.0).cast(pl.Int64)


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
    if starting_balance <= 0 or fee_rate < 0:
        raise ValueError("starting_balance must be positive and fee_rate non-negative")

    cash = starting_balance
    quantity = 0.0
    desired = 0
    fees = 0.0
    entry_time: pl.Datetime | None = None
    entry_price = 0.0
    entry_equity = 0.0
    entry_bar = 0
    trades: list[Trade] = []
    curve: list[dict[str, object]] = []

    def close_position(
        price: float, timestamp: datetime.datetime, bar_number: int
    ) -> None:
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
        target: int, price: float, timestamp: datetime.datetime, bar_number: int
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
        current = 1 if quantity > 0 else -1 if quantity < 0 else 0
        if desired != current:
            close_position(float(row["open"]), timestamp, bar_number)
            open_position(desired, float(row["open"]), timestamp, bar_number)

        equity = cash + quantity * float(row["close"])
        curve.append(
            {
                "ts": timestamp,
                "equity": equity,
                "target": desired,
                "close": float(row["close"]),
            }
        )
        desired = int(target)

    if quantity != 0.0:
        last = data.tail(1)
        close_position(float(last["close"]), pl.Datetime(last["ts"]), len(data) - 1)
        curve[-1]["equity"] = cash
        curve[-1]["target"] = 0

    equity_curve = pl.DataFrame(curve)
    trade_frame = pl.DataFrame(asdict(trade) for trade in trades)
    final_equity = float(equity_curve.select(pl.col("equity")).to_series()[-1])
    elapsed_years = max(
        (
            data.select(pl.col("ts")).to_series()[-1]
            - data.select(pl.col("ts")).to_series()[0]
        ).total_seconds()
        / (365.25 * 24 * 3600),
        1 / 365.25,
    )
    cagr = (final_equity / starting_balance) ** (1.0 / elapsed_years) - 1.0
    wins = (
        trade_frame.select(pl.col("pnl") > 0).to_series().to_list()
        if len(trade_frame) > 0
        else []
    )
    losses = (
        trade_frame.select(pl.col("pnl") < 0).to_series().to_list()
        if len(trade_frame) > 0
        else []
    )
    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))
    profit_factor = (
        gross_profit / gross_loss if gross_loss else math.inf if gross_profit else 0.0
    )
    long_mask = trade_frame.select("pnl", pl.col("side") == "long")
    short_mask = trade_frame.select("pnl", pl.col("side") == "short")
    summary = BacktestSummary(
        start=str(data.select(pl.col("ts")).to_series()[0]),
        end=str(data.select(pl.col("ts")).to_series()[-1]),
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
        win_rate_pct=(
            float((trade_frame["pnl"] > 0).mean() * 100.0)
            if len(trade_frame) > 0
            else 0.0
        ),
        profit_factor=profit_factor,
        total_fees=fees,
        long_trades=int(len(long_mask)),
        long_pnl=float(long_mask.to_series().sum()),
        short_trades=int(len(short_mask)),
        short_pnl=float(short_mask.to_series().sum()),
    )
    return summary, equity_curve, trade_frame


def _max_drawdown_pct(equities: pl.Series, starting_balance: float) -> float:
    values = pl.concat([pl.Series([starting_balance], dtype=float), equities])
    drawdowns = values / values.cum_max() - 1.0
    return float(-drawdowns.min() * 100.0)


import math
from collections.abc import Sequence

import polars as pl


import math
from collections.abc import Sequence

import polars as pl


def annualized_sharpe_ratio(
    timestamps: Sequence[object],
    equities: Sequence[float],
    *,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe from daily equity returns with a zero risk-free rate."""
    if len(timestamps) != len(equities):
        raise ValueError("timestamps and equities must have equal length")

    if len(equities) < 2:
        return 0.0

    curve = (
        pl.DataFrame({"ts": timestamps, "equity": equities})
        .with_columns(
            pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            pl.col("equity").cast(pl.Float64),
        )
        .sort("ts")
        .unique(subset="ts", keep="last", maintain_order=True)
    )

    daily = (
        curve.group_by_dynamic("ts", every="1d")
        .agg(pl.col("equity").last())
        .upsample(time_column="ts", every="1d")
        .with_columns(pl.col("equity").forward_fill())
    )

    returns = (
        daily.with_columns(pl.col("equity").pct_change().alias("return"))
        .get_column("return")
        .drop_nulls()
    )

    if len(returns) < 2:
        return 0.0

    volatility = returns.std(ddof=1)
    if volatility is None or not math.isfinite(volatility) or volatility <= 0:
        return 0.0

    mean_return = returns.mean()
    if mean_return is None:
        return 0.0

    sharpe = mean_return / volatility * math.sqrt(periods_per_year)
    return float(sharpe) if math.isfinite(sharpe) else 0.0


def main():
    print("start custom strategy")
    df = load_4h_history(DEFAULT_DATA_DIR)
    df = add_macd(df)
    targets = macd_extrema_targets(df)
    summary, _, _ = run_backtest(df, targets=targets)
    print("summary:", summary)


if __name__ == "__main__":
    main()
