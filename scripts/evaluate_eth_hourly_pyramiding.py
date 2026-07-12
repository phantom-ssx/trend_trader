from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trend_trader.backtest.metrics import annualized_sharpe_ratio, timestamps_or_daily_index

try:
    from scripts.evaluate_eth_filters import add_indicators, load_hourly
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from evaluate_eth_filters import add_indicators, load_hourly


DEFAULT_DATA_DIR = Path("data/clean/okx/ETH-USDT-SWAP")


@dataclass(frozen=True)
class PyramidPlan:
    name: str
    fractions: tuple[float, ...]
    spread_levels: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.fractions) != len(self.spread_levels) or not self.fractions:
            raise ValueError("fractions and spread_levels must have equal non-zero length")
        if tuple(sorted(self.fractions)) != self.fractions or self.fractions[-1] != 1.0:
            raise ValueError("fractions must increase and finish at 1.0")
        if tuple(sorted(self.spread_levels)) != self.spread_levels:
            raise ValueError("spread levels must be increasing")


DEFAULT_PLANS = (
    PyramidPlan("all_in", (1.0,), (0.0035,)),
    PyramidPlan("two_stage_90_100", (0.9, 1.0), (0.0035, 0.0065)),
    PyramidPlan("two_stage_50_100", (0.5, 1.0), (0.0035, 0.0050)),
    PyramidPlan("three_stage_33_67_100", (1 / 3, 2 / 3, 1.0), (0.0035, 0.0050, 0.0065)),
    PyramidPlan("three_stage_50_75_100", (0.5, 0.75, 1.0), (0.0035, 0.0045, 0.0060)),
    PyramidPlan(
        "four_stage_25_50_75_100",
        (0.25, 0.5, 0.75, 1.0),
        (0.0035, 0.0045, 0.0055, 0.0065),
    ),
)


@dataclass(frozen=True)
class PyramidResult:
    name: str
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    fees: float
    orders: int
    reversals: int

    @property
    def score(self) -> float:
        return self.return_pct / max(self.max_drawdown_pct, 1e-9)


def load_history(data_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        path = data_dir / f"ETH-USDT-SWAP_1m_{year}.parquet"
        frame = load_hourly(path, fast_period=5, slow_period=20)
        frames.append(frame[["ts", "open", "high", "low", "close", "volume"]])
    data = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
    add_indicators(data, fast_period=5, slow_period=20)
    return data


def pyramid_targets(
    data: pd.DataFrame,
    plan: PyramidPlan,
    *,
    atr_threshold: float = 0.005,
) -> pd.Series:
    """Build causal targets; exposure can only increase until the next reversal."""
    targets: list[float] = []
    direction = 0
    fraction = 0.0
    previous_spread: float | None = None

    for spread_value, atr_value in zip(data["spread_pct"], data["atr_pct"], strict=True):
        spread = float(spread_value)
        atr = float(atr_value)
        if pd.isna(spread) or pd.isna(atr):
            targets.append(direction * fraction)
            previous_spread = None if pd.isna(spread) else spread
            continue

        entry_level = plan.spread_levels[0]
        crossed_long = (
            previous_spread is not None
            and previous_spread <= entry_level < spread
            and atr >= atr_threshold
        )
        crossed_short = (
            previous_spread is not None
            and previous_spread >= -entry_level > spread
            and atr >= atr_threshold
        )
        if crossed_long:
            direction, fraction = 1, plan.fractions[0]
        elif crossed_short:
            direction, fraction = -1, plan.fractions[0]
        elif direction:
            strength = direction * spread
            for level, candidate in zip(plan.spread_levels, plan.fractions, strict=True):
                if strength > level:
                    fraction = max(fraction, candidate)

        targets.append(direction * fraction)
        previous_spread = spread
    return pd.Series(targets, index=data.index, dtype=float)


def backtest_fractional_targets(
    data: pd.DataFrame,
    targets: pd.Series,
    name: str,
    *,
    starting_balance: float,
    fee_rate: float,
) -> PyramidResult:
    """Execute each close-derived fractional target at the following open."""
    cash = starting_balance
    position = 0.0
    target_fraction = 0.0
    peak = starting_balance
    max_drawdown = 0.0
    fees = 0.0
    orders = 0
    reversals = 0
    executed_target = 0.0
    equity_curve: list[float] = []

    for row, next_target in zip(data.itertuples(index=False), targets, strict=True):
        price = float(row.open)
        desired = target_fraction
        equity = cash + position * price
        if abs(desired - executed_target) > 1e-10 and equity > 0:
            target_position = desired * equity / price
            delta = target_position - position
            fee = abs(delta) * price * fee_rate
            if position * target_position < 0:
                reversals += 1
            cash -= delta * price + fee
            position = target_position
            fees += fee
            orders += 1
            executed_target = desired

        close_equity = cash + position * float(row.close)
        equity_curve.append(close_equity)
        peak = max(peak, close_equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - close_equity) / peak * 100)
        target_fraction = float(next_target)

    if position:
        price = float(data["close"].iloc[-1])
        fee = abs(position) * price * fee_rate
        cash += position * price - fee
        fees += fee
        orders += 1
        equity_curve[-1] = cash

    return PyramidResult(
        name=name,
        return_pct=(cash / starting_balance - 1) * 100,
        max_drawdown_pct=max_drawdown,
        sharpe_ratio=annualized_sharpe_ratio(timestamps_or_daily_index(data), equity_curve),
        fees=fees,
        orders=orders,
        reversals=reversals,
    )


def evaluate(
    data: pd.DataFrame,
    plans: tuple[PyramidPlan, ...],
    *,
    starting_balance: float,
    fee_rate: float,
) -> tuple[list[PyramidResult], list[tuple[int, PyramidResult]]]:
    targets = {plan.name: pyramid_targets(data, plan) for plan in plans}
    overall = [
        backtest_fractional_targets(
            data, targets[plan.name], plan.name,
            starting_balance=starting_balance, fee_rate=fee_rate,
        )
        for plan in plans
    ]
    annual: list[tuple[int, PyramidResult]] = []
    for year in data["ts"].dt.year.drop_duplicates():
        mask = data["ts"].dt.year == year
        yearly = data.loc[mask].reset_index(drop=True)
        for plan in plans:
            annual.append((
                int(year),
                backtest_fractional_targets(
                    yearly,
                    targets[plan.name].loc[mask].reset_index(drop=True),
                    plan.name,
                    starting_balance=starting_balance,
                    fee_rate=fee_rate,
                ),
            ))
    return overall, annual


def print_results(
    overall: list[PyramidResult], annual: list[tuple[int, PyramidResult]]
) -> None:
    print("overall,strategy,return_pct,max_dd_pct,sharpe_ratio,score,fees,orders,reversals")
    for result in sorted(overall, key=lambda item: item.score, reverse=True):
        print(
            f"overall,{result.name},{result.return_pct:.2f},{result.max_drawdown_pct:.2f},"
            f"{result.sharpe_ratio:.3f},{result.score:.3f},{result.fees:.2f},{result.orders},{result.reversals}"
        )
    print("year,strategy,return_pct,max_dd_pct,sharpe_ratio,score,fees,orders,reversals")
    for year, result in annual:
        print(
            f"{year},{result.name},{result.return_pct:.2f},{result.max_drawdown_pct:.2f},"
            f"{result.sharpe_ratio:.3f},{result.score:.3f},{result.fees:.2f},{result.orders},{result.reversals}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare staged MA5/20 entries on ETH hourly bars."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    args = parser.parse_args()
    data = load_history(args.data_dir, args.start_year, args.end_year)
    print_results(*evaluate(
        data, DEFAULT_PLANS,
        starting_balance=args.starting_balance,
        fee_rate=args.fee_rate,
    ))


if __name__ == "__main__":
    main()
