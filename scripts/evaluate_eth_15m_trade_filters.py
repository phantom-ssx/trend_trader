from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.evaluate_eth_15m_filters import load_15min
    from scripts.evaluate_eth_filters import add_indicators, parse_ma_pairs
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root, to sys.path.
    from evaluate_eth_15m_filters import load_15min  # type: ignore[no-redef]
    from evaluate_eth_filters import add_indicators, parse_ma_pairs  # type: ignore[no-redef]

DEFAULT_DATA_DIRECTORY = Path("data/clean/okx/ETH-USDT-SWAP")


@dataclass(frozen=True)
class Variant:
    name: str
    entry_threshold: float = 0.0025
    exit_threshold: float = 0.0015
    entry_confirmation_bars: int = 1
    exit_confirmation_bars: int = 1
    cooldown_bars: int = 0
    efficiency_min: float = 0.0
    slope_lookback: int = 0


@dataclass(frozen=True)
class Result:
    variant: str
    fast_period: int
    slow_period: int
    return_pct: float
    max_drawdown_pct: float
    fees: float
    events: int
    entries: int

    @property
    def score(self) -> float:
        return self.return_pct / max(self.max_drawdown_pct, 1e-9)


VARIANTS = (
    Variant("baseline"),
    Variant("entry_confirm_2", entry_confirmation_bars=2),
    Variant("entry_confirm_3", entry_confirmation_bars=3),
    Variant("entry_0.30%", entry_threshold=0.003),
    Variant("entry_0.35%", entry_threshold=0.0035),
    Variant("exit_confirm_2", exit_confirmation_bars=2),
    Variant("exit_confirm_3", exit_confirmation_bars=3),
    Variant("cooldown_4", cooldown_bars=4),
    Variant("cooldown_8", cooldown_bars=8),
    Variant("efficiency_0.20", efficiency_min=0.20),
    Variant("efficiency_0.30", efficiency_min=0.30),
    Variant("ma_slope_4", slope_lookback=4),
    Variant(
        "combined_balanced",
        entry_threshold=0.003,
        exit_threshold=0.001,
        entry_confirmation_bars=2,
        exit_confirmation_bars=2,
        efficiency_min=0.20,
    ),
    Variant(
        "combined_strict",
        entry_threshold=0.0035,
        exit_threshold=0.001,
        entry_confirmation_bars=3,
        exit_confirmation_bars=2,
        cooldown_bars=4,
        efficiency_min=0.30,
        slope_lookback=4,
    ),
)


def parse_years(value: str) -> list[int]:
    years = [int(item.strip()) for item in value.split(",")]
    if not years:
        raise argparse.ArgumentTypeError("years must not be empty")
    return years


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


def run_variant(
    data: pd.DataFrame,
    *,
    fast_period: int,
    slow_period: int,
    variant: Variant,
    starting_balance: float,
    fee_rate: float,
    atr_pct_min: float,
    efficiency_period: int,
    indices: np.ndarray | None = None,
) -> Result:
    close = data["close"].to_numpy(float)
    atr_pct = data["atr_pct"].to_numpy(float)
    fast_ma = data["close"].rolling(fast_period).mean().to_numpy()
    slow_ma = data["close"].rolling(slow_period).mean().to_numpy()
    spread_pct = (fast_ma - slow_ma) / slow_ma
    price_change = data["close"].diff().abs()
    path_length = price_change.rolling(efficiency_period).sum()
    efficiency = (
        data["close"].sub(data["close"].shift(efficiency_period)).abs() / path_length
    ).to_numpy()
    selected = np.arange(len(data)) if indices is None else indices

    cash = starting_balance
    position = 0.0
    peak_equity = starting_balance
    max_drawdown_pct = 0.0
    fees = 0.0
    events = 0
    entries = 0
    state = 0
    cooldown_remaining = 0
    entry_direction = 0
    entry_count = 0
    exit_count = 0
    previous_spread = np.nan

    for index in selected:
        price = close[index]
        equity = cash + position * price
        peak_equity = max(peak_equity, equity)
        max_drawdown_pct = max(
            max_drawdown_pct,
            (peak_equity - equity) / peak_equity * 100,
        )
        spread = spread_pct[index]
        action = 0

        if state == 0:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                entry_direction = 0
                entry_count = 0
            elif np.isfinite(previous_spread) and np.isfinite(spread):
                crossed_long = previous_spread <= variant.entry_threshold < spread
                crossed_short = previous_spread >= -variant.entry_threshold > spread
                if crossed_long:
                    entry_direction, entry_count = 1, 1
                elif crossed_short:
                    entry_direction, entry_count = -1, 1
                elif entry_direction == 1 and spread > variant.entry_threshold:
                    entry_count += 1
                elif entry_direction == -1 and spread < -variant.entry_threshold:
                    entry_count += 1
                else:
                    entry_direction, entry_count = 0, 0

                if entry_count >= variant.entry_confirmation_bars:
                    direction_ok = _direction_filter_passes(
                        direction=entry_direction,
                        index=index,
                        fast_ma=fast_ma,
                        slow_ma=slow_ma,
                        slope_lookback=variant.slope_lookback,
                    )
                    if (
                        atr_pct[index] >= atr_pct_min
                        and efficiency[index] >= variant.efficiency_min
                        and direction_ok
                    ):
                        state = entry_direction
                        action = entry_direction
                    entry_direction, entry_count = 0, 0
        elif state == 1:
            exit_count = exit_count + 1 if spread <= variant.exit_threshold else 0
            if exit_count >= variant.exit_confirmation_bars:
                state, action = 0, 2
        else:
            exit_count = exit_count + 1 if spread >= -variant.exit_threshold else 0
            if exit_count >= variant.exit_confirmation_bars:
                state, action = 0, 2

        if action:
            target_position = 0.0 if action == 2 else action * equity / price
            delta = target_position - position
            fee = abs(delta) * price * fee_rate
            cash -= delta * price + fee
            position = target_position
            fees += fee
            events += 1
            entries += int(action != 2)
            if action == 2:
                cooldown_remaining = variant.cooldown_bars
            exit_count = 0
        previous_spread = spread

    last_price = close[selected[-1]]
    final_equity = cash + position * last_price
    if abs(position) > 1e-12:
        close_fee = abs(position) * last_price * fee_rate
        final_equity -= close_fee
        fees += close_fee
        events += 1

    return Result(
        variant=variant.name,
        fast_period=fast_period,
        slow_period=slow_period,
        return_pct=(final_equity / starting_balance - 1) * 100,
        max_drawdown_pct=max_drawdown_pct,
        fees=fees,
        events=events,
        entries=entries,
    )


def _direction_filter_passes(
    *,
    direction: int,
    index: int,
    fast_ma: np.ndarray,
    slow_ma: np.ndarray,
    slope_lookback: int,
) -> bool:
    if slope_lookback == 0:
        return True
    previous = index - slope_lookback
    if previous < 0:
        return False
    if direction == 1:
        return fast_ma[index] > fast_ma[previous] and slow_ma[index] > slow_ma[previous]
    return fast_ma[index] < fast_ma[previous] and slow_ma[index] < slow_ma[previous]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare filters for the ETH 15-minute MA spread exit strategy.",
    )
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--years", type=parse_years, default=list(range(2020, 2027)))
    parser.add_argument("--ma-pairs", type=parse_ma_pairs, default=parse_ma_pairs("20:72,21:80"))
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--atr-pct-min", type=float, default=0.005)
    parser.add_argument("--efficiency-period", type=int, default=16)
    parser.add_argument("--limit", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_years(args.data_directory, args.years)
    year_values = data["year"].to_numpy(int)
    rows: list[tuple[Result, int, float, str]] = []

    for fast_period, slow_period in args.ma_pairs:
        for variant in VARIANTS:
            continuous = run_variant(
                data,
                fast_period=fast_period,
                slow_period=slow_period,
                variant=variant,
                starting_balance=args.starting_balance,
                fee_rate=args.fee_rate,
                atr_pct_min=args.atr_pct_min,
                efficiency_period=args.efficiency_period,
            )
            annual = [
                run_variant(
                    data,
                    fast_period=fast_period,
                    slow_period=slow_period,
                    variant=variant,
                    starting_balance=args.starting_balance,
                    fee_rate=args.fee_rate,
                    atr_pct_min=args.atr_pct_min,
                    efficiency_period=args.efficiency_period,
                    indices=np.flatnonzero(year_values == year),
                )
                for year in args.years
            ]
            annual_returns = [result.return_pct for result in annual]
            rows.append(
                (
                    continuous,
                    sum(value > 0 for value in annual_returns),
                    min(annual_returns),
                    "|".join(f"{value:.1f}" for value in annual_returns),
                )
            )

    rows.sort(key=lambda row: (row[0].score, row[0].return_pct), reverse=True)
    print(
        "rank,ma,variant,return_pct,max_dd_pct,fees,events,entries,score,"
        "positive_years,worst_year,yearly_returns"
    )
    for rank, (result, positive_years, worst_year, yearly_returns) in enumerate(
        rows[: args.limit],
        1,
    ):
        print(
            f"{rank},{result.fast_period}/{result.slow_period},{result.variant},"
            f"{result.return_pct:.2f},{result.max_drawdown_pct:.2f},{result.fees:.2f},"
            f"{result.events},{result.entries},{result.score:.3f},{positive_years},"
            f"{worst_year:.2f},{yearly_returns}"
        )


if __name__ == "__main__":
    main()
