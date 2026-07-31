from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any

from trend_trader.data.offline.catalog import OfflineCatalog
from trend_trader.data.offline.config import load_offline_sync_config
from trend_trader.data.offline.notify import send_pending_notifications, send_service_failure
from trend_trader.data.offline.sync import OfflineSynchronizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OKX all-market offline data synchronization")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/etc/trend-trader/offline-sync.toml"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("daily", "backfill"):
        child = subparsers.add_parser(command)
        child.add_argument("--dataset", action="append", default=[])

    range_parser = subparsers.add_parser("range")
    range_parser.add_argument("--start", type=date.fromisoformat, required=True)
    range_parser.add_argument("--end", type=date.fromisoformat, required=True)
    range_parser.add_argument("--dataset", action="append", default=[])

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--mode", choices=("daily", "backfill", "range"), default="daily")
    plan_parser.add_argument("--start", type=date.fromisoformat)
    plan_parser.add_argument("--end", type=date.fromisoformat)
    plan_parser.add_argument("--dataset", action="append", default=[])

    subparsers.add_parser("status")
    subparsers.add_parser("availability")
    subparsers.add_parser("invalid-identifiers")
    subparsers.add_parser("notify")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_offline_sync_config(args.config)
    synchronizer = OfflineSynchronizer(config)

    if args.command == "status":
        print(json.dumps(synchronizer.catalog.coverage_summary(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "availability":
        payload = {
            "configured_starts": {
                name: options.start.isoformat() if options.start else None
                for name, options in config.datasets.enabled().items()
            },
            "observed": synchronizer.catalog.availability_summary(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "invalid-identifiers":
        print(
            json.dumps(
                synchronizer.catalog.invalid_identifier_summary(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "notify":
        return min(send_pending_notifications(synchronizer.catalog), 1)

    datasets = set(args.dataset) or None
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    mode = getattr(args, "mode", args.command)
    if start and end and start > end:
        raise SystemExit("--start must be on or before --end")

    if args.command == "plan":
        tasks = synchronizer.plan(
            mode=mode,
            start=start,
            end=end,
            datasets=datasets,
        )
        print(
            json.dumps(
                [
                    {
                        "dataset": task.dataset,
                        "date": task.target_date.isoformat(),
                        "scope": task.scope_key,
                    }
                    for task in tasks
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    report: dict[str, Any] = asyncio.run(
        synchronizer.run(
            mode=args.command,
            start=start,
            end=end,
            datasets=datasets,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report.get("status") == "success" else 1


def notify_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send pending offline sync Bark notifications")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/etc/trend-trader/offline-sync.toml"),
    )
    args = parser.parse_args(argv)
    try:
        config = load_offline_sync_config(args.config)
        catalog = OfflineCatalog(config.offline_root)
    except Exception:
        try:
            return 0 if send_service_failure() else 1
        except Exception:
            return 1
    pending = catalog.pending_notifications()
    failures = send_pending_notifications(catalog)
    if not pending:
        try:
            send_service_failure()
        except Exception:
            return 1
    return min(failures, 1)


if __name__ == "__main__":
    raise SystemExit(main())
