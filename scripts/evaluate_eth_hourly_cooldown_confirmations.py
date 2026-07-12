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
DEFAULT_OUTPUT = Path("outputs/eth_hourly_cooldown_confirmation_grid.csv")


@dataclass(frozen=True)
class Rule:
    name: str
    family: str
    cooldown_bars: int = 10
    min_wait: int = 0
    growth_window: int = 2
    growth_threshold: float = 0.001
    strength_multiple: float = 1.5
    breakout_lookback: int = 12
    adx_min: float = 20.0
    adx_rise: float = 2.0
    volume_multiple: float = 1.5


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


def add_confirmation_features(data: pd.DataFrame, lookbacks: tuple[int, ...]) -> None:
    data["volume_ratio"] = data["volume"] / data["volume"].rolling(20).mean().shift(1)
    data["adx_rise_3h"] = data["adx14"] - data["adx14"].shift(3)
    for lookback in lookbacks:
        data[f"prior_high_{lookback}"] = data["high"].rolling(lookback).max().shift(1)
        data[f"prior_low_{lookback}"] = data["low"].rolling(lookback).min().shift(1)


def confirmed(
    rule: Rule,
    *,
    direction: int,
    index: int,
    spread: float,
    start_strength: float,
    candidate_age: int,
    close: np.ndarray,
    prior_high: dict[int, np.ndarray],
    prior_low: dict[int, np.ndarray],
    adx: np.ndarray,
    adx_rise: np.ndarray,
    volume_ratio: np.ndarray,
    entry_threshold: float,
    last_exit_direction: int,
) -> bool:
    strength = direction * spread
    strong = (
        candidate_age <= rule.growth_window
        and strength - start_strength >= rule.growth_threshold
        and strength >= entry_threshold * rule.strength_multiple
    )
    breakout = (
        close[index] > prior_high[rule.breakout_lookback][index]
        if direction == 1
        else close[index] < prior_low[rule.breakout_lookback][index]
    )
    trend_activity = (
        adx[index] >= rule.adx_min and adx_rise[index] >= rule.adx_rise
    ) or volume_ratio[index] >= rule.volume_multiple

    if rule.family == "opposite_strong":
        return direction == -last_exit_direction and strong
    if rule.family == "price_breakout":
        return breakout
    if rule.family == "adx_or_volume":
        return trend_activity
    if rule.family == "min4_structure":
        return breakout and trend_activity
    return False


def simulate(
    data: pd.DataFrame,
    rule: Rule,
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
    spread_pct = data["spread_pct"].to_numpy(float)
    atr_pct = data["atr_pct"].to_numpy(float)
    adx = data["adx14"].to_numpy(float)
    adx_rise = data["adx_rise_3h"].to_numpy(float)
    volume_ratio = data["volume_ratio"].to_numpy(float)
    lookbacks = {candidate.breakout_lookback for candidate in build_rules()}
    prior_high = {n: data[f"prior_high_{n}"].to_numpy(float) for n in lookbacks}
    prior_low = {n: data[f"prior_low_{n}"].to_numpy(float) for n in lookbacks}
    selected = np.arange(len(data)) if indices is None else indices

    cash = starting_balance
    position = 0.0
    state = 0
    pending_action: int | None = None
    previous_spread = np.nan
    cooldown_remaining = 0
    last_exit_direction = 0
    candidate_direction = 0
    candidate_start_strength = 0.0
    candidate_age = 0
    peak_equity = starting_balance
    max_drawdown_pct = 0.0
    fees = 0.0
    entries = 0
    early_entries = 0
    slippage = slippage_bps / 10_000

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
        spread = spread_pct[index]

        if state == 0 and np.isfinite(previous_spread) and np.isfinite(spread):
            crossed_long = previous_spread <= entry_threshold < spread
            crossed_short = previous_spread >= -entry_threshold > spread
            crossed_direction = 1 if crossed_long else -1 if crossed_short else 0

            if cooldown_remaining <= 0:
                if crossed_direction and atr_pct[index] >= atr_pct_min:
                    state, pending_action = crossed_direction, crossed_direction
                candidate_direction = 0
            elif rule.family == "fixed":
                cooldown_remaining -= 1
            else:
                cooldown_remaining -= 1
                elapsed = rule.cooldown_bars - cooldown_remaining
                if candidate_direction == 0 and crossed_direction:
                    candidate_direction = crossed_direction
                    candidate_start_strength = crossed_direction * spread
                    candidate_age = 0
                elif candidate_direction:
                    candidate_age += 1
                    strength = candidate_direction * spread
                    if strength <= entry_threshold:
                        candidate_direction = 0
                        cooldown_remaining = rule.cooldown_bars
                    elif (
                        elapsed >= rule.min_wait
                        and atr_pct[index] >= atr_pct_min
                        and confirmed(
                            rule,
                            direction=candidate_direction,
                            index=index,
                            spread=spread,
                            start_strength=candidate_start_strength,
                            candidate_age=candidate_age,
                            close=closes,
                            prior_high=prior_high,
                            prior_low=prior_low,
                            adx=adx,
                            adx_rise=adx_rise,
                            volume_ratio=volume_ratio,
                            entry_threshold=entry_threshold,
                            last_exit_direction=last_exit_direction,
                        )
                    ):
                        state = candidate_direction
                        pending_action = candidate_direction
                        candidate_direction = 0
                        cooldown_remaining = 0
                        early_entries += 1
                if cooldown_remaining <= 0:
                    # Preserve the original rule after the protection period: a
                    # signal which crossed during cooldown is not entered late.
                    candidate_direction = 0
        elif state == 1 and spread <= exit_threshold:
            last_exit_direction = state
            state, pending_action = 0, 2
            cooldown_remaining = rule.cooldown_bars
            candidate_direction = 0
        elif state == -1 and spread >= -exit_threshold:
            last_exit_direction = state
            state, pending_action = 0, 2
            cooldown_remaining = rule.cooldown_bars
            candidate_direction = 0
        previous_spread = spread

    last_price = closes[selected[-1]]
    final_equity = cash + position * last_price
    if abs(position) > 1e-12:
        fill = last_price * (1 - slippage if position > 0 else 1 + slippage)
        fee = abs(position) * fill * fee_rate
        final_equity = cash + position * fill - fee
        fees += fee
    return Result(
        (final_equity / starting_balance - 1) * 100,
        max_drawdown_pct,
        fees,
        entries,
        early_entries,
    )


def build_rules() -> list[Rule]:
    rules = [Rule("fixed_10h", "fixed")]
    for window, growth, multiple in product((1, 2, 3), (0.001, 0.002), (1.5, 2.0)):
        rules.append(
            Rule(
                f"opposite_w{window}_g{growth:.3f}_m{multiple:.1f}",
                "opposite_strong",
                growth_window=window,
                growth_threshold=growth,
                strength_multiple=multiple,
            )
        )
    for lookback in (6, 12, 24):
        rules.append(Rule(f"breakout_{lookback}h", "price_breakout", breakout_lookback=lookback))
    for adx_min, adx_rise, volume in product((20.0, 25.0), (1.0, 3.0), (1.5, 2.0)):
        rules.append(
            Rule(
                f"adx{adx_min:.0f}_rise{adx_rise:.0f}_or_vol{volume:.1f}",
                "adx_or_volume",
                adx_min=adx_min,
                adx_rise=adx_rise,
                volume_multiple=volume,
            )
        )
    for lookback, adx_min, volume in product((6, 12, 24), (20.0, 25.0), (1.5, 2.0)):
        rules.append(
            Rule(
                f"min4_breakout{lookback}_adx{adx_min:.0f}_vol{volume:.1f}",
                "min4_structure",
                min_wait=4,
                breakout_lookback=lookback,
                adx_min=adx_min,
                adx_rise=1.0,
                volume_multiple=volume,
            )
        )
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore four adaptive cooldown confirmations.")
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--years", default="2020,2021,2022,2023,2024,2025,2026")
    args = parser.parse_args()

    years = [int(value) for value in args.years.split(",")]
    data = load_years(args.data_directory, years)
    rules = build_rules()
    add_confirmation_features(data, tuple(sorted({rule.breakout_lookback for rule in rules})))
    year_values = data["year"].to_numpy(int)
    train_indices = np.flatnonzero(year_values <= 2023)
    oos_indices = np.flatnonzero(year_values >= 2024)
    rows: list[dict[str, float | int | str]] = []
    for rule in rules:
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
                "family": rule.family,
                "return_pct": full.return_pct,
                "max_drawdown_pct": full.max_drawdown_pct,
                "entries": full.entries,
                "early_entries": full.early_entries,
                "fees": full.fees,
                "positive_years": sum(value > 0 for value in annual.values()),
                "worst_year_pct": min(annual.values()),
                "train_return_pct": train.return_pct,
                "oos_return_pct": oos.return_pct,
                "robust_score": min(train.score, oos.score),
                **{f"return_{year}_pct": value for year, value in annual.items()},
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["robust_score", "positive_years", "return_pct"], ascending=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    baseline = results[results["family"] == "fixed"]
    family_best = results[results["family"] != "fixed"].groupby("family", sort=False).head(1)
    columns = [
        "name",
        "family",
        "return_pct",
        "max_drawdown_pct",
        "entries",
        "early_entries",
        "positive_years",
        "worst_year_pct",
        "train_return_pct",
        "oos_return_pct",
        "robust_score",
        *[f"return_{year}_pct" for year in years],
    ]
    summary = pd.concat([baseline, family_best])[columns]
    print(summary.to_csv(index=False, float_format="%.2f"), end="")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
