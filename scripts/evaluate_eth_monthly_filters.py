from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scripts.evaluate_filters import (
    FilterBacktestResult,
    load_hourly_data,
    monthly_results,
)
from trend_trader.config.models import load_backtest_config

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ETH MA-cross filter strategies by calendar month.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/backtest.eth-2026.toml"),
        help="Backtest config TOML path",
    )
    parser.add_argument("--resample", default="1h", help="Polars resample interval")
    parser.add_argument("--fee-rate", type=float, default=0.0005, help="Fee rate per order")
    parser.add_argument(
        "--section",
        choices=["all", "monthly", "summary", "best"],
        default="all",
        help="Which result section to print",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv"],
        default="table",
        help="Output format",
    )
    return parser


def summarize_by_strategy(
    rows: list[tuple[str, FilterBacktestResult]],
) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[FilterBacktestResult]] = defaultdict(list)
    for _, result in rows:
        grouped[result.name].append(result)

    summary: list[dict[str, float | int | str]] = []
    for strategy, results in grouped.items():
        returns = [result.return_pct for result in results]
        drawdowns = [result.max_drawdown_pct for result in results]
        sharpes = [result.sharpe_ratio for result in results]
        fees = [result.total_fees for result in results]
        events = [result.events for result in results]
        summary.append(
            {
                "strategy": strategy,
                "months": len(results),
                "win_months": sum(return_pct > 0 for return_pct in returns),
                "avg_return_pct": sum(returns) / len(returns),
                "sum_return_pct": sum(returns),
                "worst_month_pct": min(returns),
                "best_month_pct": max(returns),
                "avg_drawdown_pct": sum(drawdowns) / len(drawdowns),
                "avg_sharpe_ratio": sum(sharpes) / len(sharpes),
                "avg_fees": sum(fees) / len(fees),
                "avg_events": sum(events) / len(events),
            }
        )
    return sorted(summary, key=lambda item: float(item["avg_return_pct"]), reverse=True)


def best_by_month(
    rows: list[tuple[str, FilterBacktestResult]],
) -> list[tuple[str, FilterBacktestResult]]:
    grouped: dict[str, list[FilterBacktestResult]] = defaultdict(list)
    for month, result in rows:
        grouped[month].append(result)
    return [
        (month, max(results, key=lambda result: result.return_pct))
        for month, results in sorted(grouped.items())
    ]


def print_monthly_table(rows: list[tuple[str, FilterBacktestResult]]) -> None:
    table = Table(title="Monthly Strategy Comparison")
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


def print_summary_table(summary: list[dict[str, float | int | str]]) -> None:
    table = Table(title="Strategy Monthly Summary")
    table.add_column("Strategy")
    table.add_column("Avg Return", justify="right")
    table.add_column("Win Months", justify="right")
    table.add_column("Worst", justify="right")
    table.add_column("Best", justify="right")
    table.add_column("Avg DD", justify="right")
    table.add_column("Avg Sharpe", justify="right")
    table.add_column("Avg Fees", justify="right")
    table.add_column("Avg Events", justify="right")
    for row in summary:
        table.add_row(
            str(row["strategy"]),
            f"{float(row['avg_return_pct']):.2f}%",
            f"{int(row['win_months'])}/{int(row['months'])}",
            f"{float(row['worst_month_pct']):.2f}%",
            f"{float(row['best_month_pct']):.2f}%",
            f"{float(row['avg_drawdown_pct']):.2f}%",
            f"{float(row['avg_sharpe_ratio']):.3f}",
            f"{float(row['avg_fees']):.2f}",
            f"{float(row['avg_events']):.1f}",
        )
    console.print(table)


def print_best_table(rows: list[tuple[str, FilterBacktestResult]]) -> None:
    table = Table(title="Best Strategy By Month")
    table.add_column("Month")
    table.add_column("Strategy")
    table.add_column("Return", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Fees", justify="right")
    table.add_column("Events", justify="right")
    for month, result in rows:
        table.add_row(
            month,
            result.name,
            f"{result.return_pct:.2f}%",
            f"{result.max_drawdown_pct:.2f}%",
            f"{result.sharpe_ratio:.3f}",
            f"{result.total_fees:.2f}",
            str(result.events),
        )
    console.print(table)


def print_monthly_csv(rows: list[tuple[str, FilterBacktestResult]]) -> None:
    print("month,strategy,bars,return_pct,max_dd_pct,sharpe_ratio,net_pnl,fees,events")
    for month, result in rows:
        print(
            f"{month},{result.name},{result.bars},{result.return_pct:.2f},"
            f"{result.max_drawdown_pct:.2f},{result.sharpe_ratio:.3f},{result.net_pnl:.2f},"
            f"{result.total_fees:.2f},{result.events}"
        )


def print_summary_csv(summary: list[dict[str, float | int | str]]) -> None:
    print(
        "strategy,avg_return_pct,sum_return_pct,win_months,months,"
        "worst_month_pct,best_month_pct,avg_drawdown_pct,avg_sharpe_ratio,avg_fees,avg_events"
    )
    for row in summary:
        print(
            f"{row['strategy']},{float(row['avg_return_pct']):.2f},"
            f"{float(row['sum_return_pct']):.2f},{int(row['win_months'])},"
            f"{int(row['months'])},{float(row['worst_month_pct']):.2f},"
            f"{float(row['best_month_pct']):.2f},{float(row['avg_drawdown_pct']):.2f},"
            f"{float(row['avg_sharpe_ratio']):.3f},"
            f"{float(row['avg_fees']):.2f},{float(row['avg_events']):.1f}"
        )


def print_best_csv(rows: list[tuple[str, FilterBacktestResult]]) -> None:
    print("month,strategy,return_pct,max_dd_pct,sharpe_ratio,net_pnl,fees,events")
    for month, result in rows:
        print(
            f"{month},{result.name},{result.return_pct:.2f},"
            f"{result.max_drawdown_pct:.2f},{result.sharpe_ratio:.3f},{result.net_pnl:.2f},"
            f"{result.total_fees:.2f},{result.events}"
        )


def main() -> None:
    args = build_parser().parse_args()
    config = load_backtest_config(args.config)
    data = load_hourly_data(config.data.parquet_path, resample=args.resample)
    rows = monthly_results(
        data,
        starting_balance=config.starting_balance,
        fee_rate=args.fee_rate,
    )
    summary = summarize_by_strategy(rows)
    best_rows = best_by_month(rows)

    if args.format == "csv":
        if args.section in {"all", "monthly"}:
            print_monthly_csv(rows)
        if args.section in {"all", "summary"}:
            print_summary_csv(summary)
        if args.section in {"all", "best"}:
            print_best_csv(best_rows)
        return

    if args.section in {"all", "monthly"}:
        print_monthly_table(rows)
    if args.section in {"all", "summary"}:
        print_summary_table(summary)
    if args.section in {"all", "best"}:
        print_best_table(best_rows)


if __name__ == "__main__":
    main()
