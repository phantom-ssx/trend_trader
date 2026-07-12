from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.evaluate_eth_filters import add_indicators
    from scripts.evaluate_filters import load_hourly_data
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root, to sys.path.
    from evaluate_eth_filters import add_indicators  # type: ignore[no-redef]
    from evaluate_filters import load_hourly_data  # type: ignore[no-redef]

DEFAULT_DATA_DIRECTORY = Path("data/clean/okx/ETH-USDT-SWAP")


def load_years(data_directory: Path, years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        path = data_directory / f"ETH-USDT-SWAP_1m_{year}.parquet"
        frame = load_hourly_data(path, "1h")[["ts", "open", "high", "low", "close", "volume"]]
        frame["year"] = year
        frames.append(frame)
    data = (
        pd.concat(frames, ignore_index=True)
        .sort_values("ts")
        .drop_duplicates("ts")
        .reset_index(drop=True)
    )
    add_indicators(data, fast_period=5, slow_period=20)
    return data


def simulate(
    data: pd.DataFrame,
    *,
    fast_period: int = 5,
    slow_period: int = 20,
    entry_threshold: float = 0.0025,
    exit_threshold: float = 0.0,
    atr_pct_min: float = 0.005,
    cooldown_bars: int = 10,
    starting_balance: float = 10_000.0,
    fee_rate: float = 0.0005,
    slippage_bps: float = 0.0,
    indices: np.ndarray | None = None,
) -> tuple[float, float, float, int, int, float]:
    opens = data["open"].to_numpy(float)
    closes = data["close"].to_numpy(float)
    atr_pct = data["atr_pct"].to_numpy(float)
    fast_ma = data["close"].rolling(fast_period).mean().to_numpy()
    slow_ma = data["close"].rolling(slow_period).mean().to_numpy()
    spread_pct = (fast_ma - slow_ma) / slow_ma
    selected = np.arange(len(data)) if indices is None else indices

    cash = starting_balance
    position = 0.0
    peak_equity = starting_balance
    max_drawdown_pct = 0.0
    total_fees = 0.0
    events = 0
    entries = 0
    state = 0
    cooldown_remaining = 0
    pending_action: int | None = None
    previous_spread = np.nan
    slippage = slippage_bps / 10_000

    for index in selected:
        if pending_action is not None:
            mark_price = opens[index]
            equity = cash + position * mark_price
            target = 0.0 if pending_action == 2 else pending_action * equity / mark_price
            delta = target - position
            fill_price = mark_price * (1 + slippage if delta > 0 else 1 - slippage)
            fee = abs(delta) * fill_price * fee_rate
            cash -= delta * fill_price + fee
            position = target
            total_fees += fee
            events += 1
            entries += int(pending_action != 2)
            pending_action = None

        equity = cash + position * closes[index]
        peak_equity = max(peak_equity, equity)
        max_drawdown_pct = max(
            max_drawdown_pct,
            (peak_equity - equity) / peak_equity * 100,
        )
        spread = spread_pct[index]
        if state == 0:
            if cooldown_remaining:
                cooldown_remaining -= 1
            elif np.isfinite(previous_spread) and atr_pct[index] >= atr_pct_min:
                if previous_spread <= entry_threshold < spread:
                    state, pending_action = 1, 1
                elif previous_spread >= -entry_threshold > spread:
                    state, pending_action = -1, -1
        elif state == 1 and spread <= exit_threshold:
            state, pending_action, cooldown_remaining = 0, 2, cooldown_bars
        elif state == -1 and spread >= -exit_threshold:
            state, pending_action, cooldown_remaining = 0, 2, cooldown_bars
        previous_spread = spread

    last_price = closes[selected[-1]]
    final_equity = cash + position * last_price
    if abs(position) > 1e-12:
        fill_price = last_price * (1 - slippage if position > 0 else 1 + slippage)
        close_fee = abs(position) * fill_price * fee_rate
        final_equity = cash + position * fill_price - close_fee
        total_fees += close_fee
        events += 1
    return_pct = (final_equity / starting_balance - 1) * 100
    score = return_pct / max(max_drawdown_pct, 1e-9)
    return return_pct, max_drawdown_pct, total_fees, events, entries, score


def parse_years(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the ETH hourly MA exit strategy with next-open fills and costs.",
    )
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--years", type=parse_years, default=list(range(2020, 2027)))
    parser.add_argument("--fast-period", type=int, default=5)
    parser.add_argument("--slow-period", type=int, default=20)
    parser.add_argument("--entry-threshold", type=float, default=0.0025)
    parser.add_argument("--exit-threshold", type=float, default=0.0)
    parser.add_argument("--atr-pct-min", type=float, default=0.005)
    parser.add_argument("--cooldown-bars", type=int, default=10)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_years(args.data_directory, args.years)
    year_values = data["year"].to_numpy(int)
    base = {
        "fast_period": args.fast_period,
        "slow_period": args.slow_period,
        "entry_threshold": args.entry_threshold,
        "exit_threshold": args.exit_threshold,
        "atr_pct_min": args.atr_pct_min,
        "cooldown_bars": args.cooldown_bars,
        "starting_balance": args.starting_balance,
    }
    scenarios = [(0.0005, 0), (0.0005, 2), (0.0005, 5), (0.0005, 10), (0.0007, 5), (0.001, 5)]
    print("fee_rate,slippage_bps,return_pct,max_dd_pct,fees,events,score,yearly_returns")
    for fee_rate, slippage_bps in scenarios:
        result = simulate(data, fee_rate=fee_rate, slippage_bps=slippage_bps, **base)
        annual = [
            simulate(
                data,
                fee_rate=fee_rate,
                slippage_bps=slippage_bps,
                indices=np.flatnonzero(year_values == year),
                **base,
            )[0]
            for year in args.years
        ]
        print(
            f"{fee_rate:.4f},{slippage_bps},{result[0]:.2f},{result[1]:.2f},"
            f"{result[2]:.2f},{result[3]},{result[5]:.3f},"
            + "|".join(f"{value:.1f}" for value in annual)
        )


if __name__ == "__main__":
    main()
