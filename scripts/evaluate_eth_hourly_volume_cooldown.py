from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.validate_eth_hourly_exit_strategy import load_years
except ModuleNotFoundError:
    from validate_eth_hourly_exit_strategy import load_years  # type: ignore[no-redef]


DEFAULT_DATA_DIRECTORY = Path("data/clean/okx/ETH-USDT-SWAP")
DEFAULT_OUTPUT = Path("outputs/eth_hourly_volume_cooldown.csv")
DEFAULT_WALK_FORWARD_OUTPUT = Path("outputs/eth_hourly_volume_cooldown_walk_forward.csv")


@dataclass(frozen=True)
class CooldownRule:
    name: str
    family: str
    fixed_bars: int = 10
    lookback: int = 24
    volume_hours: float = 10.0
    high_threshold: float = 1.5
    low_threshold: float = 0.75


@dataclass(frozen=True)
class Result:
    return_pct: float
    max_drawdown_pct: float
    fees: float
    entries: int
    unlocks: int
    average_lock_bars: float

    @property
    def score(self) -> float:
        return self.return_pct / max(self.max_drawdown_pct, 1e-9)


def build_rules() -> list[CooldownRule]:
    rules = [
        CooldownRule("fixed_0h", "fixed", fixed_bars=0),
        CooldownRule("fixed_10h", "fixed", fixed_bars=10),
    ]
    for lookback in (24, 72, 168):
        for volume_hours in (5.0, 8.0, 10.0, 12.0, 15.0):
            rules.append(
                CooldownRule(
                    f"volume_clock_{lookback}h_{volume_hours:g}vh",
                    "volume_clock",
                    lookback=lookback,
                    volume_hours=volume_hours,
                )
            )
    for lookback in (24, 72, 168):
        for threshold in (1.0, 1.25, 1.5, 2.0):
            rules.append(
                CooldownRule(
                    f"high_volume_cross_{lookback}h_{threshold:.2f}",
                    "high_volume_cross",
                    lookback=lookback,
                    high_threshold=threshold,
                )
            )
    for lookback in (24, 72, 168):
        for threshold in (0.5, 0.75, 1.0):
            rules.append(
                CooldownRule(
                    f"quiet_unlock_{lookback}h_{threshold:.2f}",
                    "quiet_unlock",
                    lookback=lookback,
                    low_threshold=threshold,
                )
            )
    for low_threshold in (0.5, 0.75):
        for high_threshold in (1.25, 1.5, 2.0):
            rules.append(
                CooldownRule(
                    f"quiet_then_high_24h_{low_threshold:.2f}_{high_threshold:.2f}",
                    "quiet_then_high",
                    lookback=24,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                )
            )
    return rules


def add_relative_volume(data: pd.DataFrame, rules: list[CooldownRule]) -> None:
    volume = data["volume"].astype(float)
    for lookback in sorted({rule.lookback for rule in rules}):
        reference = volume.shift(1).rolling(lookback, min_periods=lookback).median()
        data[f"relative_volume_{lookback}"] = volume / reference.replace(0.0, np.nan)


def simulate(
    data: pd.DataFrame,
    rule: CooldownRule,
    *,
    indices: np.ndarray | None = None,
    entry_threshold: float = 0.0025,
    exit_threshold: float = 0.0,
    atr_pct_min: float = 0.005,
    starting_balance: float = 10_000.0,
    fee_rate: float = 0.0005,
    slippage_bps: float = 5.0,
) -> Result:
    opens = data["open"].to_numpy(float)
    closes = data["close"].to_numpy(float)
    spread = data["spread_pct"].to_numpy(float)
    atr_pct = data["atr_pct"].to_numpy(float)
    relative_volume = data[f"relative_volume_{rule.lookback}"].to_numpy(float)
    selected = np.arange(len(data)) if indices is None else indices
    if not len(selected):
        raise ValueError("indices must not be empty")

    cash = starting_balance
    position = 0.0
    state = 0
    pending_action: int | None = None
    previous_spread = np.nan
    locked = False
    fixed_remaining = 0
    accumulated_volume = 0.0
    quiet_seen = False
    current_lock_bars = 0
    lock_durations: list[int] = []
    peak_equity = starting_balance
    max_drawdown_pct = 0.0
    fees = 0.0
    entries = 0
    slippage = slippage_bps / 10_000

    def unlock() -> None:
        nonlocal locked, current_lock_bars
        locked = False
        lock_durations.append(current_lock_bars)
        current_lock_bars = 0

    for index in selected:
        if pending_action is not None:
            mark = opens[index]
            equity = cash + position * mark
            target = 0.0 if pending_action == 2 else pending_action * equity / mark
            delta = target - position
            fill = mark * (1 + slippage if delta > 0 else 1 - slippage)
            fee = abs(delta) * fill * fee_rate
            cash -= delta * fill + fee
            position = target
            fees += fee
            entries += int(pending_action != 2)
            pending_action = None

        equity = cash + position * closes[index]
        peak_equity = max(peak_equity, equity)
        max_drawdown_pct = max(max_drawdown_pct, (peak_equity - equity) / peak_equity * 100)

        value = spread[index]
        crossed_direction = 0
        if np.isfinite(previous_spread) and np.isfinite(value):
            if previous_spread <= entry_threshold < value:
                crossed_direction = 1
            elif previous_spread >= -entry_threshold > value:
                crossed_direction = -1

        if state == 0:
            can_enter = not locked
            ratio = relative_volume[index]
            if locked:
                current_lock_bars += 1
                if rule.family == "fixed":
                    if fixed_remaining > 0:
                        fixed_remaining -= 1
                    else:
                        unlock()
                        can_enter = True
                elif rule.family == "volume_clock":
                    if np.isfinite(ratio):
                        accumulated_volume += ratio
                    if accumulated_volume >= rule.volume_hours:
                        unlock()
                        can_enter = True
                elif rule.family == "high_volume_cross":
                    can_enter = bool(
                        crossed_direction and np.isfinite(ratio) and ratio >= rule.high_threshold
                    )
                    if can_enter:
                        unlock()
                elif rule.family == "quiet_unlock":
                    if np.isfinite(ratio) and ratio <= rule.low_threshold:
                        unlock()
                        can_enter = True
                elif rule.family == "quiet_then_high":
                    quiet_seen |= bool(np.isfinite(ratio) and ratio <= rule.low_threshold)
                    can_enter = bool(
                        quiet_seen
                        and crossed_direction
                        and np.isfinite(ratio)
                        and ratio >= rule.high_threshold
                    )
                    if can_enter:
                        unlock()
                else:
                    raise ValueError(f"unknown cooldown family: {rule.family}")

            if (
                can_enter
                and crossed_direction
                and np.isfinite(atr_pct[index])
                and atr_pct[index] >= atr_pct_min
            ):
                state = crossed_direction
                pending_action = crossed_direction
                quiet_seen = False
        elif state == 1 and value <= exit_threshold:
            state, pending_action = 0, 2
            locked = rule.family != "fixed" or rule.fixed_bars > 0
            fixed_remaining = rule.fixed_bars
            accumulated_volume = 0.0
            quiet_seen = False
            current_lock_bars = 0
        elif state == -1 and value >= -exit_threshold:
            state, pending_action = 0, 2
            locked = rule.family != "fixed" or rule.fixed_bars > 0
            fixed_remaining = rule.fixed_bars
            accumulated_volume = 0.0
            quiet_seen = False
            current_lock_bars = 0
        previous_spread = value

    last_price = closes[selected[-1]]
    final_equity = cash + position * last_price
    if abs(position) > 1e-12:
        fill = last_price * (1 - slippage if position > 0 else 1 + slippage)
        fee = abs(position) * fill * fee_rate
        final_equity = cash + position * fill - fee
        fees += fee
    return Result(
        return_pct=(final_equity / starting_balance - 1) * 100,
        max_drawdown_pct=max_drawdown_pct,
        fees=fees,
        entries=entries,
        unlocks=len(lock_durations),
        average_lock_bars=float(np.mean(lock_durations)) if lock_durations else 0.0,
    )


def evaluate(data: pd.DataFrame, rules: list[CooldownRule]) -> pd.DataFrame:
    years = data["year"].to_numpy(int)
    train_indices = np.flatnonzero(years <= 2023)
    oos_indices = np.flatnonzero(years >= 2024)
    rows: list[dict[str, float | int | str]] = []
    for rule in rules:
        full = simulate(data, rule)
        train = simulate(data, rule, indices=train_indices)
        oos = simulate(data, rule, indices=oos_indices)
        annual = {
            year: simulate(data, rule, indices=np.flatnonzero(years == year)).return_pct
            for year in sorted(set(years))
        }
        rows.append(
            {
                "name": rule.name,
                "family": rule.family,
                "return_pct": full.return_pct,
                "max_drawdown_pct": full.max_drawdown_pct,
                "score": full.score,
                "entries": full.entries,
                "fees": full.fees,
                "average_lock_bars": full.average_lock_bars,
                "train_return_pct": train.return_pct,
                "train_max_drawdown_pct": train.max_drawdown_pct,
                "train_score": train.score,
                "oos_return_pct": oos.return_pct,
                "oos_max_drawdown_pct": oos.max_drawdown_pct,
                "oos_score": oos.score,
                "positive_years": sum(value > 0 for value in annual.values()),
                "worst_year_pct": min(annual.values()),
                **{f"return_{year}_pct": value for year, value in annual.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["train_score", "positive_years", "return_pct"],
        ascending=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace the ETH hourly fixed cooldown with volume-driven re-entry rules."
    )
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--walk-forward-output",
        type=Path,
        default=DEFAULT_WALK_FORWARD_OUTPUT,
    )
    args = parser.parse_args()

    data = load_years(args.data_directory, list(range(2020, 2027)))
    rules = build_rules()
    add_relative_volume(data, rules)
    results = evaluate(data, rules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    year_values = data["year"].to_numpy(int)
    volume_rules = [rule for rule in rules if rule.family != "fixed"]
    fixed_10h = next(rule for rule in rules if rule.name == "fixed_10h")
    walk_forward_rows: list[dict[str, float | int | str]] = []
    for test_year in range(2022, 2027):
        history = np.flatnonzero(year_values < test_year)
        test = np.flatnonzero(year_values == test_year)
        selected = max(volume_rules, key=lambda rule: simulate(data, rule, indices=history).score)
        selected_result = simulate(data, selected, indices=test)
        baseline_result = simulate(data, fixed_10h, indices=test)
        walk_forward_rows.append(
            {
                "test_year": test_year,
                "selected_rule": selected.name,
                "selected_return_pct": selected_result.return_pct,
                "selected_max_drawdown_pct": selected_result.max_drawdown_pct,
                "fixed_10h_return_pct": baseline_result.return_pct,
                "fixed_10h_max_drawdown_pct": baseline_result.max_drawdown_pct,
            }
        )
    walk_forward = pd.DataFrame(walk_forward_rows)
    args.walk_forward_output.parent.mkdir(parents=True, exist_ok=True)
    walk_forward.to_csv(args.walk_forward_output, index=False)

    baselines = results[results["family"] == "fixed"]
    family_best = (
        results[results["family"] != "fixed"]
        .sort_values("train_score", ascending=False)
        .groupby("family", sort=False)
        .head(1)
    )
    display = pd.concat([baselines, family_best]).sort_values("train_score", ascending=False)
    columns = [
        "name",
        "family",
        "return_pct",
        "max_drawdown_pct",
        "entries",
        "average_lock_bars",
        "train_return_pct",
        "train_max_drawdown_pct",
        "train_score",
        "oos_return_pct",
        "oos_max_drawdown_pct",
        "oos_score",
        "positive_years",
        "worst_year_pct",
    ]
    print(display[columns].to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nWalk-forward selection")
    print(walk_forward.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nFull grid written to {args.output}")
    print(f"Walk-forward results written to {args.walk_forward_output}")


if __name__ == "__main__":
    main()
