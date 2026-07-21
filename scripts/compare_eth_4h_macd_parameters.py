from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from scripts.research_eth_4h_macd_extrema import (
    DEFAULT_DATA_DIR,
    add_macd,
    load_4h_history,
    macd_extrema_targets,
    period_results,
    regime_results,
    run_backtest,
)


@dataclass(frozen=True)
class MacdParameters:
    fast: int
    slow: int
    signal: int

    @property
    def name(self) -> str:
        return f"MACD({self.fast},{self.slow},{self.signal})"


PARAMETER_SETS = (
    MacdParameters(5, 34, 5),
    MacdParameters(12, 26, 9),
    MacdParameters(19, 39, 9),
    MacdParameters(24, 52, 18),
)


def compare_parameters(
    candles: pd.DataFrame,
    *,
    starting_balance: float,
    fee_rate: float,
    extrema_column: str = "macd",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, float | int | str]] = []
    yearly_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []
    regime_frames: list[pd.DataFrame] = []
    year_side_frames: list[pd.DataFrame] = []

    for parameters in PARAMETER_SETS:
        data = add_macd(
            candles,
            fast=parameters.fast,
            slow=parameters.slow,
            signal=parameters.signal,
        )
        targets = macd_extrema_targets(data, column=extrema_column)
        summary, equity_curve, trades = run_backtest(
            data,
            targets,
            starting_balance=starting_balance,
            fee_rate=fee_rate,
        )
        no_fee_summary, _, _ = run_backtest(
            data,
            targets,
            starting_balance=starting_balance,
            fee_rate=0.0,
        )
        yearly = period_results(
            data,
            equity_curve,
            starting_balance=starting_balance,
            period="year",
        )
        monthly = period_results(
            data,
            equity_curve,
            starting_balance=starting_balance,
            period="month",
        )
        regimes, correlations = regime_results(monthly)

        wins = trades.loc[trades["pnl"] > 0, "return_pct"]
        losses = trades.loc[trades["pnl"] < 0, "return_pct"]
        overall_rows.append(
            {
                "extrema_source": extrema_column,
                "parameter": parameters.name,
                "fast": parameters.fast,
                "slow": parameters.slow,
                "signal": parameters.signal,
                **asdict(summary),
                "no_fee_return_pct": no_fee_summary.total_return_pct,
                "profitable_months": int((monthly["strategy_return_pct"] > 0).sum()),
                "total_months": len(monthly),
                "avg_monthly_return_pct": float(monthly["strategy_return_pct"].mean()),
                "median_monthly_return_pct": float(monthly["strategy_return_pct"].median()),
                "avg_winner_pct": float(wins.mean()),
                "avg_loser_pct": float(losses.mean()),
                "mean_holding_hours": float(trades["holding_bars"].mean() * 4),
                "median_holding_hours": float(trades["holding_bars"].median() * 4),
                **correlations,
            }
        )

        yearly.insert(0, "parameter", parameters.name)
        yearly.insert(0, "extrema_source", extrema_column)
        yearly_frames.append(yearly)
        monthly.insert(0, "parameter", parameters.name)
        monthly.insert(0, "extrema_source", extrema_column)
        monthly_frames.append(monthly)
        regimes.insert(0, "parameter", parameters.name)
        regimes.insert(0, "extrema_source", extrema_column)
        regime_frames.append(regimes)

        trade_year_side = trades.copy()
        trade_year_side["year"] = pd.to_datetime(trade_year_side["exit_time"], utc=True).dt.year
        trade_year_side = (
            trade_year_side.groupby(["year", "side"], as_index=False)
            .agg(
                trades=("pnl", "size"),
                pnl=("pnl", "sum"),
                win_rate_pct=("pnl", lambda values: float((values > 0).mean() * 100.0)),
                avg_return_pct=("return_pct", "mean"),
            )
            .assign(parameter=parameters.name, extrema_source=extrema_column)
        )
        year_side_frames.append(
            trade_year_side[
                [
                    "extrema_source",
                    "parameter",
                    "year",
                    "side",
                    "trades",
                    "pnl",
                    "win_rate_pct",
                    "avg_return_pct",
                ]
            ]
        )

    return (
        pd.DataFrame(overall_rows),
        pd.concat(yearly_frames, ignore_index=True),
        pd.concat(monthly_frames, ignore_index=True),
        pd.concat(regime_frames, ignore_index=True),
        pd.concat(year_side_frames, ignore_index=True),
    )


def print_report(
    overall: pd.DataFrame,
    yearly: pd.DataFrame,
    monthly: pd.DataFrame,
    regimes: pd.DataFrame,
) -> None:
    columns = [
        "extrema_source",
        "parameter",
        "total_return_pct",
        "no_fee_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "sharpe_ratio",
        "trades",
        "win_rate_pct",
        "profit_factor",
        "total_fees",
        "long_pnl",
        "short_pnl",
        "profitable_months",
        "avg_monthly_return_pct",
        "mean_holding_hours",
    ]
    print("\nOVERALL PARAMETER COMPARISON")
    print(overall[columns].to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    for source, source_yearly in yearly.groupby("extrema_source", sort=False):
        print(f"\nYEARLY RETURNS (%) - {source}")
        yearly_pivot = source_yearly.pivot(
            index="period",
            columns="parameter",
            values="strategy_return_pct",
        )
        print(yearly_pivot.to_string(float_format=lambda value: f"{value:.2f}"))

    print("\nBEST AND WORST MONTH FOR EACH PARAMETER")
    rows: list[dict[str, float | str]] = []
    for (source, parameter), group in monthly.groupby(["extrema_source", "parameter"], sort=False):
        best = group.loc[group["strategy_return_pct"].idxmax()]
        worst = group.loc[group["strategy_return_pct"].idxmin()]
        rows.append(
            {
                "extrema_source": source,
                "parameter": parameter,
                "best_month": best["period"],
                "best_return_pct": best["strategy_return_pct"],
                "worst_month": worst["period"],
                "worst_return_pct": worst["strategy_return_pct"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    print("\nPATH-EFFICIENCY REGIMES")
    efficiency = regimes.loc[regimes["dimension"] == "efficiency_regime"]
    print(
        efficiency[
            [
                "parameter",
                "extrema_source",
                "regime",
                "months",
                "profitable_months",
                "avg_strategy_return_pct",
                "median_strategy_return_pct",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare fixed MACD parameter sets for the ETH 4h extrema strategy."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candles = load_4h_history(
        args.data_dir,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    line_results = compare_parameters(
        candles,
        starting_balance=args.starting_balance,
        fee_rate=args.fee_rate,
    )
    histogram_results = compare_parameters(
        candles,
        starting_balance=args.starting_balance,
        fee_rate=args.fee_rate,
        extrema_column="macd_histogram",
    )
    overall, yearly, monthly, regimes, year_side = tuple(
        pd.concat([line_frame, histogram_frame], ignore_index=True)
        for line_frame, histogram_frame in zip(line_results, histogram_results, strict=True)
    )
    print_report(overall, yearly, monthly, regimes)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        overall.to_csv(args.output_dir / "overall.csv", index=False)
        yearly.to_csv(args.output_dir / "yearly.csv", index=False)
        monthly.to_csv(args.output_dir / "monthly.csv", index=False)
        regimes.to_csv(args.output_dir / "regimes.csv", index=False)
        year_side.to_csv(args.output_dir / "year_side.csv", index=False)


if __name__ == "__main__":
    main()
