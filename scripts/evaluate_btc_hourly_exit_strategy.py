from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

DEFAULT_DATA_DIRECTORY = Path("data/clean/okx/BTC-USDT-SWAP")
DEFAULT_OUTPUT = Path("outputs/btc_hourly_exit_parameter_grid.csv")
DEFAULT_YEARS = list(range(2020, 2027))


@dataclass(frozen=True)
class Result:
    return_pct: float
    max_drawdown_pct: float
    fees: float
    events: int
    entries: int

    @property
    def score(self) -> float:
        return self.return_pct / max(self.max_drawdown_pct, 1e-9)


def parse_years(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",")]


def load_history(data_directory: Path, years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        candidates = [
            data_directory / f"BTC-USDT-SWAP_1H_{year}.parquet",
            data_directory / f"BTC-USDT-SWAP_1m_{year}.parquet",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"No BTC candle file found for {year}: {candidates}")
        frame = pl.read_parquet(path).select("ts", "open", "high", "low", "close", "volume")
        if frame.height > 10_000:
            frame = (
                frame.sort("ts")
                .group_by_dynamic("ts", every="1h", closed="left", label="left")
                .agg(
                    pl.col("open").first(),
                    pl.col("high").max(),
                    pl.col("low").min(),
                    pl.col("close").last(),
                    pl.col("volume").sum(),
                )
            )
        pandas_frame = frame.to_pandas()
        pandas_frame["year"] = year
        frames.append(pandas_frame)

    data = (
        pd.concat(frames, ignore_index=True)
        .sort_values("ts")
        .drop_duplicates("ts")
        .reset_index(drop=True)
    )
    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr_pct"] = true_range.rolling(14).mean() / data["close"]
    return data


def simulate(
    data: pd.DataFrame,
    *,
    fast_period: int,
    slow_period: int,
    entry_threshold: float,
    exit_threshold: float,
    atr_pct_min: float,
    cooldown_bars: int,
    starting_balance: float = 10_000.0,
    fee_rate: float = 0.0005,
    slippage_bps: float = 5.0,
    indices: np.ndarray | None = None,
) -> Result:
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
    fees = 0.0
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
            fees += fee
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
        fees += close_fee
        events += 1
    return Result(
        return_pct=(final_equity / starting_balance - 1) * 100,
        max_drawdown_pct=max_drawdown_pct,
        fees=fees,
        events=events,
        entries=entries,
    )


def parameter_grid() -> list[tuple[int, int, float, float, float, int]]:
    pairs = [
        (8, 30),
        (8, 36),
        (10, 40),
        (12, 48),
        (16, 48),
        (16, 64),
        (20, 80),
        (24, 96),
    ]
    return list(
        product(
            pairs,
            [0.004, 0.005, 0.006, 0.008, 0.01],
            [-0.001, 0.0, 0.001],
            [0.004, 0.005, 0.006, 0.008],
            [24, 48, 72],
        )
    )


def params_dict(
    pair: tuple[int, int], entry: float, exit_value: float, atr: float, cooldown: int
) -> dict[str, int | float]:
    return {
        "fast_period": pair[0],
        "slow_period": pair[1],
        "entry_threshold": entry,
        "exit_threshold": exit_value,
        "atr_pct_min": atr,
        "cooldown_bars": cooldown,
    }


def evaluate_grid(
    data: pd.DataFrame,
    *,
    years: list[int],
    fee_rate: float,
    slippage_bps: float,
) -> pd.DataFrame:
    year_values = data["year"].to_numpy(int)
    train_indices = np.flatnonzero(year_values <= 2023)
    oos_indices = np.flatnonzero(year_values >= 2024)
    rows = []
    for pair, entry, exit_value, atr, cooldown in parameter_grid():
        params = params_dict(pair, entry, exit_value, atr, cooldown)
        train = simulate(
            data,
            indices=train_indices,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            **params,
        )
        oos = simulate(
            data,
            indices=oos_indices,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            **params,
        )
        rows.append(
            {
                "fast_period": pair[0],
                "slow_period": pair[1],
                "entry_threshold": entry,
                "exit_threshold": exit_value,
                "atr_pct_min": atr,
                "cooldown_bars": cooldown,
                "train_return_pct": train.return_pct,
                "train_score": train.score,
                "oos_return_pct": oos.return_pct,
                "oos_max_drawdown_pct": oos.max_drawdown_pct,
                "oos_score": oos.score,
            }
        )
    results = pd.DataFrame(rows)
    candidates = results.loc[
        (results["train_return_pct"] > 0) & (results["oos_return_pct"] > 0)
    ].copy()
    if candidates.empty:
        candidates = results.copy()
    candidates["selection_score"] = candidates[["train_score", "oos_score"]].min(axis=1)
    candidate_indices = candidates.nlargest(50, "selection_score").index

    for index in candidate_indices:
        row = results.loc[index]
        params = {
            "fast_period": int(row["fast_period"]),
            "slow_period": int(row["slow_period"]),
            "entry_threshold": row["entry_threshold"],
            "exit_threshold": row["exit_threshold"],
            "atr_pct_min": row["atr_pct_min"],
            "cooldown_bars": int(row["cooldown_bars"]),
        }
        full = simulate(data, fee_rate=fee_rate, slippage_bps=slippage_bps, **params)
        annual = [
            simulate(
                data,
                indices=np.flatnonzero(year_values == year),
                fee_rate=fee_rate,
                slippage_bps=slippage_bps,
                **params,
            ).return_pct
            for year in years
        ]
        values = {
            "return_pct": full.return_pct,
            "max_drawdown_pct": full.max_drawdown_pct,
            "score": full.score,
            "fees": full.fees,
            "events": full.events,
            "entries": full.entries,
            "positive_years": sum(value > 0 for value in annual),
            "worst_year_pct": min(annual),
            **{f"return_{year}": value for year, value in zip(years, annual, strict=True)},
        }
        for column, value in values.items():
            results.loc[index, column] = value
    return results


def print_result(label: str, result: Result, annual: list[float]) -> None:
    annual_text = " | ".join(
        f"{year}: {value:.1f}%" for year, value in zip(DEFAULT_YEARS, annual, strict=True)
    )
    print(
        f"{label}: return={result.return_pct:.2f}%, max_dd={result.max_drawdown_pct:.2f}%, "
        f"score={result.score:.3f}, entries={result.entries}, fees={result.fees:.2f}"
    )
    print(f"  yearly: {annual_text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate and tune the hourly MA-spread exit strategy on BTC-USDT-SWAP."
    )
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--years", type=parse_years, default=DEFAULT_YEARS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_history(args.data_directory, args.years)
    year_values = data["year"].to_numpy(int)
    baseline_params = params_dict((5, 20), 0.0025, 0.0, 0.005, 10)
    baseline = simulate(
        data, fee_rate=args.fee_rate, slippage_bps=args.slippage_bps, **baseline_params
    )
    baseline_annual = [
        simulate(
            data,
            indices=np.flatnonzero(year_values == year),
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
            **baseline_params,
        ).return_pct
        for year in args.years
    ]
    print_result("ETH parameters MA5/20", baseline, baseline_annual)

    results = evaluate_grid(
        data,
        years=args.years,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.sort_values(["oos_score", "train_score"], ascending=False).to_csv(
        args.output, index=False
    )

    robust = results.loc[(results["positive_years"] >= 5) & (results["train_return_pct"] > 0)]
    if robust.empty:
        robust = results
    robust = robust.sort_values(
        ["oos_score", "positive_years", "worst_year_pct", "train_score"], ascending=False
    )
    columns = [
        "fast_period",
        "slow_period",
        "entry_threshold",
        "exit_threshold",
        "atr_pct_min",
        "cooldown_bars",
        "return_pct",
        "max_drawdown_pct",
        "train_return_pct",
        "oos_return_pct",
        "oos_max_drawdown_pct",
        "positive_years",
        "worst_year_pct",
        "entries",
    ]
    print("\nTop robust parameters (ranked by 2024-2026 OOS return/drawdown):")
    print(
        robust[columns].head(args.limit).to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    best = robust.iloc[0]
    best_params = {
        "fast_period": int(best["fast_period"]),
        "slow_period": int(best["slow_period"]),
        "entry_threshold": best["entry_threshold"],
        "exit_threshold": best["exit_threshold"],
        "atr_pct_min": best["atr_pct_min"],
        "cooldown_bars": int(best["cooldown_bars"]),
    }
    print("\nSelected robust parameters:")
    print(best_params)
    print("yearly: " + " | ".join(f"{year}: {best[f'return_{year}']:.2f}%" for year in args.years))
    print("\nCost sensitivity:")
    print("fee_rate,slippage_bps,return_pct,max_dd_pct,entries,fees")
    scenarios = [(0.0005, 0), (0.0005, 2), (0.0005, 5), (0.0005, 10), (0.0007, 5), (0.001, 5)]
    for fee_rate, slippage_bps in scenarios:
        result = simulate(
            data,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            **best_params,
        )
        print(
            f"{fee_rate:.4f},{slippage_bps},{result.return_pct:.2f},"
            f"{result.max_drawdown_pct:.2f},{result.entries},{result.fees:.2f}"
        )
    print(f"\nFull grid written to {args.output}")


if __name__ == "__main__":
    main()
