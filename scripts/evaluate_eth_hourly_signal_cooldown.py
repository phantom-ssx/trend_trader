from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.validate_eth_hourly_exit_strategy import load_years
except ModuleNotFoundError:
    from validate_eth_hourly_exit_strategy import load_years  # type: ignore[no-redef]


DEFAULT_DATA_DIRECTORY = Path("data/clean/okx/ETH-USDT-SWAP")
DEFAULT_OUTPUT = Path("outputs/eth_hourly_signal_cooldown_grid.csv")


@dataclass(frozen=True)
class CooldownRule:
    name: str
    cooldown_bars: int
    adaptive: bool = False
    growth_window: int = 2
    growth_threshold: float = 0.0005
    strength_multiple: float = 1.5


@dataclass(frozen=True)
class Result:
    return_pct: float
    max_drawdown_pct: float
    fees: float
    entries: int
    early_entries: int

    @property
    def score(self) -> float:
        return self.return_pct / max(self.max_drawdown_pct, 1e-9)


def simulate(
    data: pd.DataFrame,
    rule: CooldownRule,
    *,
    indices: np.ndarray | None = None,
    fast_period: int = 5,
    slow_period: int = 20,
    entry_threshold: float = 0.0025,
    exit_threshold: float = 0.0,
    atr_pct_min: float = 0.005,
    starting_balance: float = 10_000.0,
    fee_rate: float = 0.0005,
    slippage_bps: float = 5.0,
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
    state = 0
    pending_action: int | None = None
    previous_spread = np.nan
    cooldown_remaining = 0
    candidate_direction = 0
    candidate_start_strength = 0.0
    candidate_age = 0
    peak_equity = starting_balance
    max_drawdown_pct = 0.0
    total_fees = 0.0
    entries = 0
    early_entries = 0
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
            entries += int(pending_action != 2)
            pending_action = None

        equity = cash + position * closes[index]
        peak_equity = max(peak_equity, equity)
        max_drawdown_pct = max(max_drawdown_pct, (peak_equity - equity) / peak_equity * 100)
        spread = spread_pct[index]

        if state == 0 and np.isfinite(previous_spread) and np.isfinite(spread):
            crossed_long = previous_spread <= entry_threshold < spread
            crossed_short = previous_spread >= -entry_threshold > spread
            crossed_direction = 1 if crossed_long else -1 if crossed_short else 0

            if cooldown_remaining <= 0:
                if candidate_direction and atr_pct[index] >= atr_pct_min:
                    state, pending_action = candidate_direction, candidate_direction
                elif crossed_direction and atr_pct[index] >= atr_pct_min:
                    state, pending_action = crossed_direction, crossed_direction
                candidate_direction = 0
            elif not rule.adaptive:
                cooldown_remaining -= 1
            else:
                cooldown_remaining -= 1
                if candidate_direction == 0 and crossed_direction:
                    candidate_direction = crossed_direction
                    candidate_start_strength = crossed_direction * spread
                    candidate_age = 0
                elif candidate_direction:
                    candidate_age += 1
                    strength = candidate_direction * spread
                    if strength <= entry_threshold:
                        # A failed signal starts a fresh protection period.
                        candidate_direction = 0
                        cooldown_remaining = rule.cooldown_bars
                    elif (
                        candidate_age <= rule.growth_window
                        and strength - candidate_start_strength >= rule.growth_threshold
                        and strength >= entry_threshold * rule.strength_multiple
                        and atr_pct[index] >= atr_pct_min
                    ):
                        state, pending_action = candidate_direction, candidate_direction
                        candidate_direction = 0
                        cooldown_remaining = 0
                        early_entries += 1
        elif state == 1 and spread <= exit_threshold:
            state, pending_action = 0, 2
            cooldown_remaining = rule.cooldown_bars
            candidate_direction = 0
        elif state == -1 and spread >= -exit_threshold:
            state, pending_action = 0, 2
            cooldown_remaining = rule.cooldown_bars
            candidate_direction = 0
        previous_spread = spread

    last_price = closes[selected[-1]]
    final_equity = cash + position * last_price
    if abs(position) > 1e-12:
        fill_price = last_price * (1 - slippage if position > 0 else 1 + slippage)
        fee = abs(position) * fill_price * fee_rate
        final_equity = cash + position * fill_price - fee
        total_fees += fee
    return Result(
        return_pct=(final_equity / starting_balance - 1) * 100,
        max_drawdown_pct=max_drawdown_pct,
        fees=total_fees,
        entries=entries,
        early_entries=early_entries,
    )


def build_rules() -> list[CooldownRule]:
    rules = [
        CooldownRule("fixed_0h", 0),
        CooldownRule("fixed_3h", 3),
        CooldownRule("fixed_5h", 5),
        CooldownRule("fixed_10h", 10),
    ]
    for window, growth, multiple in product(
        (1, 2, 3, 4),
        (0.0005, 0.001, 0.0015, 0.002, 0.003),
        (1.25, 1.5, 1.75, 2.0),
    ):
        rules.append(
            CooldownRule(
                f"adaptive_10h_w{window}_g{growth:.4f}_m{multiple:.2f}",
                10,
                adaptive=True,
                growth_window=window,
                growth_threshold=growth,
                strength_multiple=multiple,
            )
        )
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed and signal-strength-aware cooldowns for ETH hourly MA5/20.",
    )
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--years", default="2020,2021,2022,2023,2024,2025,2026")
    args = parser.parse_args()

    years = [int(value) for value in args.years.split(",")]
    data = load_years(args.data_directory, years)
    year_values = data["year"].to_numpy(int)
    train_indices = np.flatnonzero(year_values <= 2023)
    oos_indices = np.flatnonzero(year_values >= 2024)
    rows: list[dict[str, float | int | str | bool]] = []
    for rule in build_rules():
        full = simulate(data, rule)
        train = simulate(data, rule, indices=train_indices)
        oos = simulate(data, rule, indices=oos_indices)
        annual = {
            year: simulate(data, rule, indices=np.flatnonzero(year_values == year)).return_pct
            for year in years
        }
        rows.append(
            {
                "name": rule.name,
                "adaptive": rule.adaptive,
                "cooldown_bars": rule.cooldown_bars,
                "growth_window": rule.growth_window if rule.adaptive else 0,
                "growth_threshold": rule.growth_threshold if rule.adaptive else 0.0,
                "strength_multiple": rule.strength_multiple if rule.adaptive else 0.0,
                "return_pct": full.return_pct,
                "max_drawdown_pct": full.max_drawdown_pct,
                "score": full.score,
                "fees": full.fees,
                "entries": full.entries,
                "early_entries": full.early_entries,
                "positive_years": sum(value > 0 for value in annual.values()),
                "worst_year_pct": min(annual.values()),
                "train_return_pct": train.return_pct,
                "train_max_dd_pct": train.max_drawdown_pct,
                "oos_return_pct": oos.return_pct,
                "oos_max_dd_pct": oos.max_drawdown_pct,
                "robust_score": min(train.score, oos.score),
                **{f"return_{year}_pct": value for year, value in annual.items()},
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["positive_years", "robust_score", "return_pct"], ascending=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    baseline = results.loc[results["name"] == "fixed_10h"]
    adaptive = results.loc[results["adaptive"]].head(8)
    comparison = pd.concat([baseline, adaptive]).drop_duplicates("name")
    columns = [
        "name",
        "return_pct",
        "max_drawdown_pct",
        "score",
        "fees",
        "entries",
        "early_entries",
        "positive_years",
        "worst_year_pct",
        "train_return_pct",
        "oos_return_pct",
        "robust_score",
        *[f"return_{year}_pct" for year in years],
    ]
    print(comparison[columns].to_csv(index=False, float_format="%.2f"), end="")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
