from __future__ import annotations

import argparse
from itertools import product

import numpy as np

try:
    from scripts.validate_eth_hourly_exit_strategy import load_years, simulate
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root, to sys.path.
    from validate_eth_hourly_exit_strategy import load_years, simulate  # type: ignore[no-redef]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search ETH 1-hour MA spread exit parameters.")
    parser.add_argument("--data-directory", default="data/clean/okx/ETH-USDT-SWAP")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from pathlib import Path

    data = load_years(Path(args.data_directory), list(range(2020, 2027)))
    years = data["year"].to_numpy(int)
    pairs = [(4, 18), (5, 18), (5, 19), (5, 20), (5, 21), (5, 22), (6, 20), (6, 22)]
    rows = []
    for (fast, slow), entry, exit_threshold, atr_min, cooldown in product(
        pairs,
        [0.002, 0.0025, 0.003, 0.0035],
        [-0.0005, 0.0, 0.0005],
        [0.004, 0.005, 0.006, 0.0075],
        [6, 8, 10, 12],
    ):
        params = {
            "fast_period": fast,
            "slow_period": slow,
            "entry_threshold": entry,
            "exit_threshold": exit_threshold,
            "atr_pct_min": atr_min,
            "cooldown_bars": cooldown,
            "fee_rate": 0.0005,
            "slippage_bps": args.slippage_bps,
        }
        result = simulate(data, **params)
        oos = simulate(data, indices=np.flatnonzero(years >= 2024), **params)[0]
        rows.append((result, oos, fast, slow, entry, exit_threshold, atr_min, cooldown))
    rows.sort(key=lambda row: (row[0][5], row[0][0]), reverse=True)
    print("rank,ma,entry,exit,atr,cooldown,return_pct,max_dd_pct,score,oos_2024_2026")
    for rank, row in enumerate(rows[: args.limit], 1):
        result, oos, fast, slow, entry, exit_threshold, atr_min, cooldown = row
        print(
            f"{rank},{fast}/{slow},{entry:.2%},{exit_threshold:.2%},{atr_min:.2%},"
            f"{cooldown},{result[0]:.2f},{result[1]:.2f},{result[5]:.3f},{oos:.2f}"
        )


if __name__ == "__main__":
    main()
