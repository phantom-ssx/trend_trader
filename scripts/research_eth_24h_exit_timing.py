"""Screen fixed stop exits for the ETH 24h contrarian sleeve on development data."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from trend_trader.data import MarketDataClient

try:
    from scripts.research_eth_24h_entry_timing import (
        build_signal_dataset,
        candidate_portfolio,
        half_year_returns,
        performance,
    )
except ModuleNotFoundError:
    from research_eth_24h_entry_timing import (
        build_signal_dataset,
        candidate_portfolio,
        half_year_returns,
        performance,
    )

ONE_WAY_COST = 0.0008  # 5bp fee + 3bp slippage.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--development-end", default="2025-01-01")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def simulate_stop(
    portfolio: pl.DataFrame,
    candles: pl.DataFrame,
    *,
    stop_loss: float | None,
) -> pl.DataFrame:
    candle_rows = candles.sort("timestamp").partition_by("timestamp", as_dict=True)
    actual_position = 0.0
    trade_entry_price: float | None = None
    rows: list[dict[str, object]] = []
    for row in portfolio.sort("timestamp").iter_rows(named=True):
        entry = row["timestamp"]
        exit_time = row["exit_time"]
        desired_position = float(row["position"])
        entry_candle = candle_rows[(entry,)]
        interval_entry = float(entry_candle["open"][0])
        cost = ONE_WAY_COST * abs(desired_position - actual_position)
        if desired_position > 0 and actual_position == 0:
            trade_entry_price = interval_entry
        elif desired_position == 0:
            trade_entry_price = None
        actual_position = desired_position
        exit_price = float(
            candle_rows[(exit_time,)]["open"][0]
            if (exit_time,) in candle_rows
            else interval_entry * (1.0 + float(row["benchmark_return"]))
        )
        stopped = False
        if actual_position > 0 and stop_loss is not None and trade_entry_price is not None:
            stop_price = trade_entry_price * (1.0 - stop_loss)
            timestamp = entry
            while timestamp < exit_time:
                candle = candle_rows[(timestamp,)]
                open_price = float(candle["open"][0])
                if open_price <= stop_price:
                    exit_price = open_price
                    stopped = True
                    break
                if float(candle["low"][0]) <= stop_price:
                    exit_price = stop_price
                    stopped = True
                    break
                timestamp = timestamp.replace() + (exit_time - entry) / 24
        gross_return = actual_position * (exit_price / interval_entry - 1.0)
        if stopped:
            cost += ONE_WAY_COST
            actual_position = 0.0
            trade_entry_price = None
        rows.append(
            {
                **row,
                "position": desired_position,
                "gross_portfolio_return": gross_return,
                "transaction_cost": cost,
                "portfolio_return": gross_return - cost,
                "stopped": stopped,
            }
        )
    result = pl.DataFrame(rows)
    return result.with_columns(
        (pl.col("portfolio_return") + 1.0).cum_prod().alias("wealth"),
        (pl.col("position").diff().fill_null(pl.col("position")).abs() * 0.5).alias(
            "turnover"
        ),
    )


def main() -> None:
    args = parse_args()
    development_end = datetime.fromisoformat(args.development_end).replace(tzinfo=UTC)
    data = MarketDataClient(data_root=args.data_root)
    config, dataset = build_signal_dataset(args.config, data=data, workdir=Path.cwd())
    phases = {
        offset: candidate_portfolio(
            config,
            dataset,
            offset_bars=offset,
            threshold_bps=2.0,
            smoothing_periods=3,
            trend_bars=None,
            trend_min_bps=0.0,
        )
        for offset in (0, 6, 12, 18)
    }
    start = min(frame["timestamp"].min() for frame in phases.values())
    end = max(frame["exit_time"].max() for frame in phases.values())
    candles = data.candles("ETH-USDT-SWAP", "1h", start, end)
    rows: list[dict[str, object]] = []
    portfolios: dict[tuple[float | None, int], pl.DataFrame] = {}
    for stop_loss in (None, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
        for offset, phase in phases.items():
            frame = simulate_stop(phase, candles, stop_loss=stop_loss)
            portfolios[(stop_loss, offset)] = frame
            development = frame.filter(pl.col("timestamp") < development_end)
            out_of_sample = frame.filter(pl.col("timestamp") >= development_end)
            halves = half_year_returns(development)
            rows.append(
                {
                    "stop_loss": stop_loss,
                    "offset_bars": offset,
                    **{
                        f"dev_{name}": value
                        for name, value in performance(
                            development["portfolio_return"].to_list(),
                            periods_per_year=365.25,
                        ).items()
                    },
                    "dev_worst_half_return": min(halves),
                    "dev_positive_half_rate": sum(value > 0 for value in halves)
                    / len(halves),
                    **{
                        f"oos_{name}": value
                        for name, value in performance(
                            out_of_sample["portfolio_return"].to_list(),
                            periods_per_year=365.25,
                        ).items()
                    },
                    "stops": int(frame["stopped"].sum()),
                }
            )
    phase_results = pl.DataFrame(rows)
    screen = (
        phase_results.group_by("stop_loss")
        .agg(
            (pl.col("dev_annual_return") > 0).mean().alias("dev_positive_phase_rate"),
            pl.col("dev_annual_return").min().alias("dev_annual_min"),
            pl.col("dev_annual_return").median().alias("dev_annual_median"),
            pl.col("dev_sharpe").median().alias("dev_sharpe_median"),
            pl.col("dev_max_drawdown").min().alias("dev_worst_drawdown"),
            pl.col("dev_worst_half_return").min().alias("dev_worst_half_return"),
            pl.col("dev_positive_half_rate").min().alias("dev_positive_half_rate"),
            pl.col("oos_annual_return").median().alias("oos_annual_median_diagnostic"),
            pl.col("oos_max_drawdown").min().alias("oos_worst_drawdown_diagnostic"),
            pl.col("stops").sum().alias("stops"),
        )
        .sort(
            ["dev_positive_phase_rate", "dev_positive_half_rate", "dev_sharpe_median"],
            descending=True,
        )
    )
    baseline = screen.filter(pl.col("stop_loss").is_null()).row(0, named=True)
    eligible = screen.filter(
        (pl.col("stop_loss").is_not_null())
        & (pl.col("dev_positive_phase_rate") == 1.0)
        & (pl.col("dev_annual_median") >= 0.8 * float(baseline["dev_annual_median"]))
        & (pl.col("dev_worst_drawdown") >= float(baseline["dev_worst_drawdown"]))
    )
    gates_passed = not eligible.is_empty()
    selected = (eligible if gates_passed else screen.filter(pl.col("stop_loss").is_null())).row(
        0, named=True
    )
    selected_stop = selected["stop_loss"]
    selected_portfolio = portfolios[(selected_stop, 0)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_results.write_csv(args.output_dir / "phase_results.csv")
    screen.write_csv(args.output_dir / "development_screen.csv")
    selected_portfolio.write_csv(args.output_dir / "selected_portfolio_returns.csv")
    summary = {
        "research": "eth_24h_exit_timing",
        "selection_uses_dates_before": args.development_end,
        "selection_does_not_use_oos_columns": True,
        "cost": {"fee_bps_per_side": 5.0, "slippage_bps_per_side": 3.0},
        "selection_rule": (
            "fixed stop must keep every phase profitable, retain at least 80% of "
            "baseline development median annual return, and improve worst drawdown; "
            "then maximize development phase and half-year consistency and Sharpe"
        ),
        "development_gates_passed": gates_passed,
        "baseline": baseline,
        "selected": selected,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    print(screen)


if __name__ == "__main__":
    main()
