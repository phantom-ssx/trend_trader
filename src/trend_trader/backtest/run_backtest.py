from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

from trend_trader.backtest.nautilus_engine import (
    NautilusBacktestOutput,
    available_nautilus_strategy_names,
    run_nautilus_backtest,
)
from trend_trader.config.models import load_backtest_config
from trend_trader.io.csv_export import CsvColumn, CsvExporter, int_sort_value, unix_nanos_to_iso

console = Console()

ORDER_CSV_EXPORTER = CsvExporter(
    [
        CsvColumn("ts_init_iso", value=lambda order: unix_nanos_to_iso(order.get("ts_init"))),
        CsvColumn("ts_last_iso", value=lambda order: unix_nanos_to_iso(order.get("ts_last"))),
        "status",
        "instrument_id",
        "side",
        "type",
        "quantity",
        "filled_qty",
        "avg_px",
        "slippage",
        "liquidity_side",
        "time_in_force",
        "is_reduce_only",
        "is_quote_quantity",
        "client_order_id",
        "venue_order_id",
        "position_id",
        "account_id",
        "last_trade_id",
        "commissions",
        "equity_after_order_usdt",
        "ts_init",
        "ts_last",
    ],
    sort_key=int_sort_value("ts_init"),
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a NautilusTrader OKX candle backtest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resample", help="Optional Polars interval before backtest, e.g. 1h")
    parser.add_argument("--fast-period", type=int, help="Override strategy fast period")
    parser.add_argument("--slow-period", type=int, help="Override strategy slow period")
    parser.add_argument("--trade-size", type=float, help="Override trade size in base units")
    parser.add_argument(
        "--strategy",
        "--nautilus-strategy",
        choices=available_nautilus_strategy_names(),
        help="Nautilus strategy to load; defaults to strategy.name in config.",
    )
    parser.add_argument(
        "--spread-threshold",
        type=float,
        help="MA spread entry threshold; defaults to strategy config.",
    )
    parser.add_argument(
        "--atr-pct-min",
        type=float,
        help="Minimum ATR/close filter; defaults to strategy config.",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        help="Symmetric MA spread exit threshold; defaults to strategy config.",
    )
    parser.add_argument(
        "--cooldown-bars",
        type=int,
        help="Bars to wait after exiting; defaults to strategy config.",
    )
    parser.add_argument(
        "--min-order-notional",
        type=float,
        help="Skip smaller orders; defaults to strategy config.",
    )
    parser.add_argument(
        "--sizing",
        choices=["fixed", "all-in"],
        help="Position sizing; defaults to strategy config.",
    )
    parser.add_argument(
        "--orders-csv",
        type=Path,
        help=(
            "Write Nautilus order details to this directory or file parent. "
            "The filename includes execution time, strategy, and instrument."
        ),
    )
    return parser
 

def main() -> None:
    run_started_at = datetime.now().astimezone()
    args = build_parser().parse_args()
    config = load_backtest_config(args.config)
    strategy_name = args.strategy or config.strategy.name
    if strategy_name not in available_nautilus_strategy_names():
        supported = ", ".join(available_nautilus_strategy_names())
        raise ValueError(f"strategy.name must be one of: {supported}")
    df = pl.read_parquet(config.data.parquet_path)
    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Parquet file is missing columns: {sorted(missing)}")

    fast_period = args.fast_period if args.fast_period is not None else config.strategy.fast_period
    slow_period = args.slow_period if args.slow_period is not None else config.strategy.slow_period
    if strategy_name == "best-filter":
        fast_period = args.fast_period or 5
        slow_period = args.slow_period or 20
    trade_size = args.trade_size if args.trade_size is not None else config.strategy.trade_size
    bar_interval = args.resample or config.strategy.bar_interval or config.data.bar
    spread_threshold = (
        args.spread_threshold
        if args.spread_threshold is not None
        else config.strategy.spread_threshold
    )
    atr_pct_min = args.atr_pct_min if args.atr_pct_min is not None else config.strategy.atr_pct_min
    exit_threshold = (
        args.exit_threshold if args.exit_threshold is not None else config.strategy.exit_threshold
    )
    cooldown_bars = (
        args.cooldown_bars if args.cooldown_bars is not None else config.strategy.cooldown_bars
    )
    min_order_notional = (
        args.min_order_notional
        if args.min_order_notional is not None
        else config.strategy.min_order_notional
    )
    sizing = args.sizing or config.strategy.sizing
    df = resample_ohlcv(df, bar_interval if bar_interval != config.data.bar else None)

    output = run_nautilus_backtest(
        config,
        df,
        strategy_name=strategy_name,
        bar_interval=bar_interval,
        fast_period=fast_period,
        slow_period=slow_period,
        trade_size=trade_size,
        sizing=sizing,
        leverage=config.strategy.leverage,
        spread_threshold=spread_threshold,
        atr_period=config.strategy.atr_period,
        atr_pct_min=atr_pct_min,
        min_order_notional=min_order_notional,
        exit_threshold=exit_threshold,
        cooldown_bars=cooldown_bars,
    )
    print_nautilus_result(output, config.data.parquet_path)
    if args.orders_csv:
        orders_csv_path = build_orders_csv_path(
            args.orders_csv,
            run_started_at=run_started_at,
            strategy_name=output.strategy_name,
            instrument_id=str(output.instrument_id),
        )
        ORDER_CSV_EXPORTER.export_rows(
            orders_with_equity_after_order(
                output.orders,
                starting_balance=Decimal(str(config.starting_balance)),
            ),
            orders_csv_path,
        )
        console.print(f"Order details written to {orders_csv_path}")


def orders_with_equity_after_order(
    orders: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    starting_balance: Decimal,
) -> list[dict[str, object]]:
    cash = starting_balance
    position = Decimal("0")
    enriched_orders: list[dict[str, object]] = []

    for order in sorted(orders, key=lambda item: int(item.get("ts_init") or 0)):
        row = dict(order)
        side = str(row.get("side", "")).upper()
        filled_qty = parse_decimal(row.get("filled_qty"))
        avg_px = parse_decimal(row.get("avg_px"))
        commission = parse_commission_amount(row.get("commissions"))

        if side in {"BUY", "SELL"} and filled_qty is not None and avg_px is not None:
            signed_delta = filled_qty if side == "BUY" else -filled_qty
            cash -= signed_delta * avg_px
            cash -= commission
            position += signed_delta
            equity = cash + position * avg_px
            row["equity_after_order_usdt"] = f"{equity:.8f}"
        else:
            row["equity_after_order_usdt"] = ""

        enriched_orders.append(row)

    return enriched_orders


def parse_decimal(value: object) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def parse_commission_amount(value: object) -> Decimal:
    if value in (None, ""):
        return Decimal("0")
    values = value if isinstance(value, list | tuple) else [value]
    total = Decimal("0")
    for item in values:
        amount = str(item).split(maxsplit=1)[0]
        try:
            total += Decimal(amount)
        except InvalidOperation:
            continue
    return total


def build_orders_csv_path(
    target: Path,
    *,
    run_started_at: datetime,
    strategy_name: str,
    instrument_id: str,
) -> Path:
    directory = target.parent if target.suffix.lower() == ".csv" else target
    timestamp = run_started_at.strftime("%Y%m%dT%H%M%S%z")
    instrument_symbol = instrument_id.split(".", maxsplit=1)[0]
    filename = (
        f"orders_{timestamp}_"
        f"{safe_filename_part(strategy_name)}_"
        f"{safe_filename_part(instrument_symbol)}.csv"
    )
    return directory / filename


def safe_filename_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return safe.strip("-_") or "unknown"


def print_nautilus_result(output: NautilusBacktestOutput, parquet_path: Path) -> None:
    result = output.result
    table = Table(title="NautilusTrader Backtest")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    summary = result.summary
    rows = [
        ("Instrument", str(output.instrument_id)),
        ("Bar type", str(output.bar_type)),
        ("Strategy", output.strategy_name),
        ("Loaded bars", str(output.bars_loaded)),
        ("Iterations", summary.get("iterations", str(result.iterations))),
        ("Orders total", summary.get("orders.total", str(result.total_orders))),
        ("Orders closed", summary.get("orders.closed", "")),
        ("Positions total", summary.get("positions.total", str(result.total_positions))),
        ("Positions open", summary.get("positions.open", "")),
        ("Positions closed", summary.get("positions.closed", "")),
        ("Total events", summary.get("total_events", str(result.total_events))),
        ("Final equity", f"{output.final_equity:.2f}"),
        ("Net position", f"{output.final_net_position:f}"),
        ("Last price", f"{output.last_price:.2f}"),
        ("Unrealized PnL", f"{output.unrealized_pnl:.2f}"),
        ("Est. equity after final close", f"{output.estimated_liquidation_equity:.2f}"),
        ("Sharpe ratio", f"{result.stats_returns.get('Sharpe Ratio (252 days)', 0.0):.3f}"),
    ]
    rows.extend(
        (key, value)
        for key, value in summary.items()
        if key.startswith("account.") and key.endswith((".total", ".free", ".locked"))
    )
    for key, value in rows:
        table.add_row(key, value)

    console.print(table)
    console.print(f"Loaded data from {parquet_path}")
    console.print(f"Strategy: {output.strategy_class_path}")


if __name__ == "__main__":
    main()


'''
uv run nt-okx-backtest \
  --config configs/backtest.eth-15m-ma25-ma80.toml \
  --resample 15m \
  --strategy best-filter \
  --sizing all-in \
  --spread-threshold 0.0025 \
  --atr-pct-min 0.0050
'''
