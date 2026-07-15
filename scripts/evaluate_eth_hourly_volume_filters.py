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
DEFAULT_OUTPUT = Path("outputs/eth_hourly_volume_filters.csv")


@dataclass(frozen=True)
class VolumeRule:
    name: str
    family: str
    lookback: int = 24
    threshold: float = 0.0
    flow_lookback: int = 12
    flow_threshold: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    return_pct: float
    max_drawdown_pct: float
    fees: float
    entries: int
    win_rate_pct: float
    profit_factor: float

    @property
    def score(self) -> float:
        return self.return_pct / max(self.max_drawdown_pct, 1e-9)


def add_volume_features(data: pd.DataFrame, rules: list[VolumeRule]) -> None:
    """Add causal, scale-free volume features used by the candidate rules."""
    volume = data["volume"].astype(float)
    for lookback in sorted({rule.lookback for rule in rules}):
        prior_median = volume.shift(1).rolling(lookback, min_periods=lookback).median()
        data[f"relative_volume_{lookback}"] = volume / prior_median.replace(0.0, np.nan)
        data[f"volume_percentile_{lookback}"] = volume.rolling(
            lookback,
            min_periods=lookback,
        ).rank(pct=True)

    signed_volume = np.sign(data["close"].diff()).fillna(0.0) * volume
    for lookback in sorted({rule.flow_lookback for rule in rules}):
        numerator = signed_volume.rolling(lookback, min_periods=lookback).sum()
        denominator = volume.rolling(lookback, min_periods=lookback).sum()
        data[f"volume_flow_{lookback}"] = numerator / denominator.replace(0.0, np.nan)


def build_rules() -> list[VolumeRule]:
    rules = [VolumeRule("baseline", "baseline")]
    for lookback in (24, 72, 168):
        for threshold in (0.75, 1.0, 1.25, 1.5, 2.0):
            rules.append(
                VolumeRule(
                    f"relative_volume_{lookback}h>={threshold:.2f}",
                    "relative_volume",
                    lookback=lookback,
                    threshold=threshold,
                )
            )
    for lookback in (72, 168, 720):
        for threshold in (0.4, 0.5, 0.6, 0.7, 0.8):
            rules.append(
                VolumeRule(
                    f"volume_percentile_{lookback}h>={threshold:.2f}",
                    "volume_percentile",
                    lookback=lookback,
                    threshold=threshold,
                )
            )
    for lookback in (6, 12, 24):
        for threshold in (0.0, 0.05, 0.10, 0.20):
            rules.append(
                VolumeRule(
                    f"directional_flow_{lookback}h>={threshold:.2f}",
                    "directional_flow",
                    flow_lookback=lookback,
                    flow_threshold=threshold,
                )
            )
    for relative_threshold in (0.75, 1.0, 1.25):
        for flow_threshold in (0.0, 0.05, 0.10):
            rules.append(
                VolumeRule(
                    f"relative_72h>={relative_threshold:.2f}+flow_12h>={flow_threshold:.2f}",
                    "relative_and_flow",
                    lookback=72,
                    threshold=relative_threshold,
                    flow_lookback=12,
                    flow_threshold=flow_threshold,
                )
            )
    return rules


def entry_allowed(data: pd.DataFrame, rule: VolumeRule, direction: int, index: int) -> bool:
    if rule.family == "baseline":
        return True
    relative = data[f"relative_volume_{rule.lookback}"].iat[index]
    percentile = data[f"volume_percentile_{rule.lookback}"].iat[index]
    flow = data[f"volume_flow_{rule.flow_lookback}"].iat[index]
    if rule.family == "relative_volume":
        return bool(np.isfinite(relative) and relative >= rule.threshold)
    if rule.family == "volume_percentile":
        return bool(np.isfinite(percentile) and percentile >= rule.threshold)
    if rule.family == "directional_flow":
        return bool(np.isfinite(flow) and direction * flow >= rule.flow_threshold)
    if rule.family == "relative_and_flow":
        return bool(
            np.isfinite(relative)
            and relative >= rule.threshold
            and np.isfinite(flow)
            and direction * flow >= rule.flow_threshold
        )
    raise ValueError(f"unknown volume rule family: {rule.family}")


def simulate(
    data: pd.DataFrame,
    rule: VolumeRule,
    *,
    indices: np.ndarray | None = None,
    entry_threshold: float = 0.0025,
    exit_threshold: float = 0.0,
    atr_pct_min: float = 0.005,
    cooldown_bars: int = 10,
    starting_balance: float = 10_000.0,
    fee_rate: float = 0.0005,
    slippage_bps: float = 5.0,
) -> BacktestResult:
    opens = data["open"].to_numpy(float)
    closes = data["close"].to_numpy(float)
    spread = data["spread_pct"].to_numpy(float)
    atr_pct = data["atr_pct"].to_numpy(float)
    selected = np.arange(len(data)) if indices is None else indices
    if not len(selected):
        raise ValueError("indices must not be empty")

    cash = starting_balance
    position = 0.0
    state = 0
    pending_action: int | None = None
    previous_spread = np.nan
    cooldown_remaining = 0
    peak_equity = starting_balance
    max_drawdown_pct = 0.0
    total_fees = 0.0
    entries = 0
    entry_equity: float | None = None
    trade_pnls: list[float] = []
    slippage = slippage_bps / 10_000

    for index in selected:
        if pending_action is not None:
            mark = opens[index]
            equity = cash + position * mark
            if pending_action == 2:
                target = 0.0
            else:
                target = pending_action * equity / mark
            delta = target - position
            fill = mark * (1 + slippage if delta > 0 else 1 - slippage)
            fee = abs(delta) * fill * fee_rate
            cash -= delta * fill + fee
            position = target
            total_fees += fee
            if pending_action == 2 and entry_equity is not None:
                trade_pnls.append(cash - entry_equity)
                entry_equity = None
            elif pending_action != 2:
                entries += 1
                entry_equity = cash + position * fill
            pending_action = None

        equity = cash + position * closes[index]
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_drawdown_pct = max(
                max_drawdown_pct,
                (peak_equity - equity) / peak_equity * 100,
            )

        value = spread[index]
        if state == 0:
            if cooldown_remaining:
                cooldown_remaining -= 1
            elif np.isfinite(previous_spread) and atr_pct[index] >= atr_pct_min:
                direction = 0
                if previous_spread <= entry_threshold < value:
                    direction = 1
                elif previous_spread >= -entry_threshold > value:
                    direction = -1
                if direction and entry_allowed(data, rule, direction, index):
                    state = direction
                    pending_action = direction
        elif state == 1 and value <= exit_threshold:
            state, pending_action, cooldown_remaining = 0, 2, cooldown_bars
        elif state == -1 and value >= -exit_threshold:
            state, pending_action, cooldown_remaining = 0, 2, cooldown_bars
        previous_spread = value

    last_price = closes[selected[-1]]
    final_equity = cash + position * last_price
    if abs(position) > 1e-12:
        fill = last_price * (1 - slippage if position > 0 else 1 + slippage)
        fee = abs(position) * fill * fee_rate
        final_equity = cash + position * fill - fee
        total_fees += fee
        if entry_equity is not None:
            trade_pnls.append(final_equity - entry_equity)

    wins = [value for value in trade_pnls if value > 0]
    losses = [value for value in trade_pnls if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return BacktestResult(
        return_pct=(final_equity / starting_balance - 1) * 100,
        max_drawdown_pct=max_drawdown_pct,
        fees=total_fees,
        entries=entries,
        win_rate_pct=len(wins) / len(trade_pnls) * 100 if trade_pnls else 0.0,
        profit_factor=gross_profit / gross_loss if gross_loss else float("inf"),
    )


def evaluate(data: pd.DataFrame, rules: list[VolumeRule]) -> pd.DataFrame:
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
                "win_rate_pct": full.win_rate_pct,
                "profit_factor": full.profit_factor,
                "fees": full.fees,
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
        description="Evaluate causal volume filters on the ETH hourly MA5/20 exit strategy."
    )
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = load_years(args.data_directory, list(range(2020, 2027)))
    rules = build_rules()
    add_volume_features(data, rules)
    results = evaluate(data, rules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    baseline = results[results["family"] == "baseline"]
    family_best = (
        results[results["family"] != "baseline"]
        .sort_values("train_score", ascending=False)
        .groupby("family", sort=False)
        .head(1)
    )
    display = pd.concat([baseline, family_best]).sort_values(
        "train_score",
        ascending=False,
    )
    columns = [
        "name",
        "family",
        "return_pct",
        "max_drawdown_pct",
        "entries",
        "win_rate_pct",
        "profit_factor",
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
    print(f"\nFull grid written to {args.output}")


if __name__ == "__main__":
    main()
