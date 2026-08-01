from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import date
from pathlib import Path

from .config import DATASETS, MARKETS, BinanceOfflineConfig
from .sync import BinanceOfflineSynchronizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trend-trader-binance-download",
        description=(
            "Download every Binance USD-M/COIN-M perpetual archive and normalize to Parquet."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "sync", "compact", "publish"):
        target = subparsers.add_parser(command)
        target.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
        target.add_argument("--market", choices=MARKETS, action="append", dest="markets")
        target.add_argument("--dataset", choices=DATASETS, action="append", dest="datasets")
        target.add_argument(
            "--symbol",
            action="append",
            dest="symbols",
            help="Limit to an exchange-native symbol; repeat for multiple symbols.",
        )
        target.add_argument("--start", type=date.fromisoformat)
        target.add_argument("--end", type=date.fromisoformat)
        target.add_argument("--download-workers", type=int, default=64)
        target.add_argument("--convert-workers", type=int, default=max(1, os.cpu_count() or 1))
    return parser


def _config(args: argparse.Namespace) -> BinanceOfflineConfig:
    return BinanceOfflineConfig(
        data_root=args.data_root,
        markets=tuple(args.markets or MARKETS),
        datasets=tuple(args.datasets or DATASETS),
        symbols=tuple(symbol.upper() for symbol in (args.symbols or ())),
        start=args.start,
        end=args.end,
        download_workers=args.download_workers,
        convert_workers=args.convert_workers,
    )


async def _run(args: argparse.Namespace) -> int:
    synchronizer = BinanceOfflineSynchronizer(_config(args))
    if args.command == "plan":
        tasks = await synchronizer.build_plan()
        total_bytes = sum(task.source.size for task in tasks)
        summary = {
            "archive_count": len(tasks),
            "download_bytes": total_bytes,
            "download_gib": round(total_bytes / 1024**3, 2),
            "by_dataset": {
                dataset: sum(task.dataset == dataset for task in tasks) for dataset in DATASETS
            },
            "by_market": {
                market.upper(): sum(task.market == market for task in tasks) for market in MARKETS
            },
        }
    elif args.command == "sync":
        summary = await synchronizer.run()
    elif args.command == "compact":
        summary = await synchronizer.run_compaction_only()
    else:
        summary = await synchronizer.run_publish_only()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
