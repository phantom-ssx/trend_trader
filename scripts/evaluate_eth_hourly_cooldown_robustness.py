from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.validate_eth_hourly_exit_strategy import load_years, simulate
except ModuleNotFoundError:
    from validate_eth_hourly_exit_strategy import load_years, simulate  # type: ignore[no-redef]


DEFAULT_DATA_DIRECTORY = Path("data/clean/okx/ETH-USDT-SWAP")
DEFAULT_OUTPUT = Path("outputs/eth_hourly_cooldown_robustness.csv")
DEFAULT_WALK_FORWARD_OUTPUT = Path("outputs/eth_hourly_cooldown_walk_forward.csv")


def evaluate(
    data: pd.DataFrame,
    indices: np.ndarray,
    cooldown: int,
) -> tuple[float, float, int]:
    result = simulate(
        data,
        cooldown_bars=cooldown,
        fee_rate=0.0005,
        slippage_bps=5,
        indices=indices,
    )
    return result[0], result[1], result[4]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the ETH hourly fixed cooldown is a stable plateau "
            "or an isolated optimum."
        ),
    )
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--walk-forward-output", type=Path, default=DEFAULT_WALK_FORWARD_OUTPUT)
    parser.add_argument("--max-cooldown", type=int, default=48)
    args = parser.parse_args()

    years = list(range(2020, 2027))
    data = load_years(args.data_directory, years)
    year_values = data["year"].to_numpy(int)
    all_indices = np.arange(len(data))
    train_indices = np.flatnonzero(year_values <= 2023)
    oos_indices = np.flatnonzero(year_values >= 2024)
    rows: list[dict[str, float | int]] = []
    for cooldown in range(args.max_cooldown + 1):
        full = evaluate(data, all_indices, cooldown)
        train = evaluate(data, train_indices, cooldown)
        oos = evaluate(data, oos_indices, cooldown)
        annual = {
            year: evaluate(data, np.flatnonzero(year_values == year), cooldown)[0]
            for year in years
        }
        rows.append(
            {
                "cooldown_bars": cooldown,
                "return_pct": full[0],
                "max_drawdown_pct": full[1],
                "entries": full[2],
                "train_return_pct": train[0],
                "train_max_drawdown_pct": train[1],
                "oos_return_pct": oos[0],
                "oos_max_drawdown_pct": oos[1],
                "positive_years": sum(value > 0 for value in annual.values()),
                "worst_year_pct": min(annual.values()),
                **{f"return_{year}_pct": value for year, value in annual.items()},
            }
        )
    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    walk_forward: list[dict[str, float | int]] = []
    for test_year in range(2022, 2027):
        history = np.flatnonzero(year_values < test_year)
        test = np.flatnonzero(year_values == test_year)
        candidates = [
            (evaluate(data, history, cooldown)[0], cooldown)
            for cooldown in range(args.max_cooldown + 1)
        ]
        _, selected_cooldown = max(candidates)
        test_result = evaluate(data, test, selected_cooldown)
        fixed_10_result = evaluate(data, test, 10)
        walk_forward.append(
            {
                "test_year": test_year,
                "selected_cooldown": selected_cooldown,
                "selected_return_pct": test_result[0],
                "fixed_10h_return_pct": fixed_10_result[0],
            }
        )
    walk_forward_frame = pd.DataFrame(walk_forward)
    walk_forward_frame.to_csv(args.walk_forward_output, index=False)

    train_best = results.sort_values("train_return_pct", ascending=False).iloc[0]
    full_best = results.sort_values("return_pct", ascending=False).iloc[0]
    neighborhood = results[results["cooldown_bars"].between(5, 15)]
    print("full_best")
    print(full_best.to_frame().T.to_csv(index=False, float_format="%.2f"), end="")
    print("train_best")
    print(train_best.to_frame().T.to_csv(index=False, float_format="%.2f"), end="")
    print("cooldown_5_to_15")
    print(
        neighborhood[
            ["cooldown_bars", "return_pct", "train_return_pct", "oos_return_pct", "entries"]
        ].to_csv(index=False, float_format="%.2f"),
        end="",
    )
    print("walk_forward")
    print(walk_forward_frame.to_csv(index=False, float_format="%.2f"), end="")
    print(f"output={args.output}")
    print(f"walk_forward_output={args.walk_forward_output}")


if __name__ == "__main__":
    main()
