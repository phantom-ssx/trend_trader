from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.evaluate_eth_15m_filters import load_15min
    from scripts.evaluate_eth_filters import add_indicators
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root, to sys.path.
    from evaluate_eth_15m_filters import load_15min  # type: ignore[no-redef]
    from evaluate_eth_filters import add_indicators  # type: ignore[no-redef]

DEFAULT_DATA_DIRECTORY = Path("data/clean/okx/ETH-USDT-SWAP")


@dataclass(frozen=True)
class SearchResult:
    fast_period: int
    slow_period: int
    return_pct: float
    max_drawdown_pct: float
    total_fees: float
    events: int
    entries: int

    @property
    def score(self) -> float:
        return self.return_pct / max(self.max_drawdown_pct, 1e-9)


def parse_int_values(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be comma-separated positive integers")
    return values


def load_years(data_directory: Path, years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        path = data_directory / f"ETH-USDT-SWAP_1m_{year}.parquet"
        frame = load_15min(path)[["ts", "open", "high", "low", "close", "volume"]]
        frame["year"] = year
        frames.append(frame)
    data = (
        pd.concat(frames, ignore_index=True)
        .sort_values("ts")
        .drop_duplicates("ts")
        .reset_index(drop=True)
    )
    add_indicators(data, fast_period=25, slow_period=80)
    return data


def run_backtest(
    *,
    close: np.ndarray,
    atr_pct: np.ndarray,
    fast_period: int,
    slow_period: int,
    starting_balance: float,
    fee_rate: float,
    entry_threshold: float,
    exit_threshold: float,
    atr_pct_min: float,
    indices: np.ndarray | None = None,
) -> SearchResult:
    fast_ma = pd.Series(close).rolling(fast_period).mean().to_numpy()
    slow_ma = pd.Series(close).rolling(slow_period).mean().to_numpy()
    spread_pct = (fast_ma - slow_ma) / slow_ma
    selected = np.arange(len(close)) if indices is None else indices

    cash = starting_balance
    position = 0.0
    peak_equity = starting_balance
    max_drawdown_pct = 0.0
    total_fees = 0.0
    events = 0
    entries = 0
    state = 0
    previous_spread = np.nan

    for index in selected:
        price = close[index]
        equity = cash + position * price
        peak_equity = max(peak_equity, equity)
        max_drawdown_pct = max(
            max_drawdown_pct,
            (peak_equity - equity) / peak_equity * 100,
        )

        action = 0
        spread = spread_pct[index]
        if (
            state == 0
            and np.isfinite(previous_spread)
            and np.isfinite(spread)
            and atr_pct[index] >= atr_pct_min
        ):
            if previous_spread <= entry_threshold < spread:
                state, action = 1, 1
            elif previous_spread >= -entry_threshold > spread:
                state, action = -1, -1
        elif state == 1 and spread <= exit_threshold:
            state, action = 0, 2
        elif state == -1 and spread >= -exit_threshold:
            state, action = 0, 2

        if action:
            target_position = 0.0 if action == 2 else action * equity / price
            delta = target_position - position
            fee = abs(delta) * price * fee_rate
            cash -= delta * price + fee
            position = target_position
            total_fees += fee
            events += 1
            entries += int(action != 2)
        previous_spread = spread

    last_price = close[selected[-1]]
    final_equity = cash + position * last_price
    if abs(position) > 1e-12:
        fee = abs(position) * last_price * fee_rate
        final_equity -= fee
        total_fees += fee
        events += 1

    return SearchResult(
        fast_period=fast_period,
        slow_period=slow_period,
        return_pct=(final_equity / starting_balance - 1) * 100,
        max_drawdown_pct=max_drawdown_pct,
        total_fees=total_fees,
        events=events,
        entries=entries,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search MA periods for the 15-minute spread/ATR strategy with exits.",
    )
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--years", type=parse_int_values, default=list(range(2020, 2027)))
    parser.add_argument("--fast-periods", type=parse_int_values, default=list(range(18, 23)))
    parser.add_argument("--slow-periods", type=parse_int_values, default=list(range(68, 81)))
    parser.add_argument("--entry-threshold", type=float, default=0.0025)
    parser.add_argument("--exit-threshold", type=float, default=0.0015)
    parser.add_argument("--atr-pct-min", type=float, default=0.005)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--yearly-top",
        type=int,
        default=5,
        help="Print independent yearly returns for this many top combinations.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_years(args.data_directory, args.years)
    close = data["close"].to_numpy(float)
    atr_pct = data["atr_pct"].to_numpy(float)
    years = data["year"].to_numpy(int)

    results = [
        run_backtest(
            close=close,
            atr_pct=atr_pct,
            fast_period=fast_period,
            slow_period=slow_period,
            starting_balance=args.starting_balance,
            fee_rate=args.fee_rate,
            entry_threshold=args.entry_threshold,
            exit_threshold=args.exit_threshold,
            atr_pct_min=args.atr_pct_min,
        )
        for fast_period in args.fast_periods
        for slow_period in args.slow_periods
        if fast_period < slow_period
    ]
    results.sort(key=lambda item: (item.score, item.return_pct), reverse=True)

    print("rank,fast,slow,return_pct,max_dd_pct,fees,events,entries,score")
    for rank, result in enumerate(results[: args.limit], 1):
        print(
            f"{rank},{result.fast_period},{result.slow_period},"
            f"{result.return_pct:.2f},{result.max_drawdown_pct:.2f},"
            f"{result.total_fees:.2f},{result.events},{result.entries},{result.score:.3f}"
        )

    print("yearly: fast/slow,positive_years,worst,average,average_dd,returns")
    for result in results[: args.yearly_top]:
        annual = [
            run_backtest(
                close=close,
                atr_pct=atr_pct,
                fast_period=result.fast_period,
                slow_period=result.slow_period,
                starting_balance=args.starting_balance,
                fee_rate=args.fee_rate,
                entry_threshold=args.entry_threshold,
                exit_threshold=args.exit_threshold,
                atr_pct_min=args.atr_pct_min,
                indices=np.flatnonzero(years == year),
            )
            for year in args.years
        ]
        returns = [item.return_pct for item in annual]
        drawdowns = [item.max_drawdown_pct for item in annual]
        print(
            f"{result.fast_period}/{result.slow_period},"
            f"{sum(value > 0 for value in returns)},"
            f"{min(returns):.2f},{np.mean(returns):.2f},{np.mean(drawdowns):.2f},"
            + "|".join(f"{value:.1f}" for value in returns)
        )


if __name__ == "__main__":
    main()
