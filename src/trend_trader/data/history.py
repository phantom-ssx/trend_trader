"""Resumable bulk maintenance for a current top-volume trading universe."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from trend_trader.data.models import (
    STORED_BAR_TYPES,
    DataQuery,
    DataType,
    DataUnavailableError,
    FetchRequest,
    as_utc,
    bar_minutes,
)
from trend_trader.data.query import DEFAULT_DATA_ROOT, MarketDataClient
from trend_trader.data.schema import canonicalize_frame
from trend_trader.data.store import iter_partitions

DEFAULT_START = datetime(2020, 1, 1, tzinfo=UTC)
DEFAULT_DATA_TYPES = tuple(DataType)
FUNDING_LOOKBACK = timedelta(days=92)
LIQUIDATION_LOOKBACK = timedelta(days=3)


@dataclass(slots=True)
class DownloadRecord:
    data_type: str
    instrument_id: str
    start: str
    end: str
    status: str
    rows: int = 0
    error: str | None = None


def floor_time(value: datetime, minutes: int) -> datetime:
    seconds = minutes * 60
    return datetime.fromtimestamp(int(value.timestamp()) // seconds * seconds, tz=UTC)


def ceil_time(value: datetime, minutes: int) -> datetime:
    floored = floor_time(value, minutes)
    return floored if floored == value else floored + timedelta(minutes=minutes)


def available_range(
    data_type: DataType,
    start: datetime,
    end: datetime,
) -> tuple[datetime, datetime]:
    if data_type is DataType.FUNDING_RATES:
        start = max(start, end - FUNDING_LOOKBACK)
    elif data_type is DataType.LIQUIDATIONS:
        start = max(start, end - LIQUIDATION_LOOKBACK)
    return start, end


def partition_ranges(
    data_type: DataType,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, datetime]]:
    stored_bar_type = STORED_BAR_TYPES.get(data_type)
    if stored_bar_type is not None:
        minutes = bar_minutes(stored_bar_type)
        start = ceil_time(start, minutes)
        end = floor_time(end, minutes)
    if end <= start:
        return []
    return [
        (max(start, partition_start), min(end, partition_end))
        for partition_start, partition_end in iter_partitions(data_type, start, end)
        if min(end, partition_end) > max(start, partition_start)
    ]


def target_instrument(data_type: DataType, contract_id: str) -> tuple[str, str]:
    if data_type is DataType.MARKET_CAP:
        return "GLOBAL", contract_id.split("-", maxsplit=1)[0]
    return "OKX", contract_id


class HistoryDownloader:
    """Download each dataset partition independently so interrupted runs can resume."""

    def __init__(
        self,
        *,
        data_root: Path | str = DEFAULT_DATA_ROOT,
        top_n: int = 50,
        start: datetime | str = DEFAULT_START,
        end: datetime | str | None = None,
        data_types: tuple[DataType, ...] = DEFAULT_DATA_TYPES,
        refresh_universe: bool = True,
        candle_options: dict[str, Any] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.top_n = top_n
        self.start = as_utc(start)
        self.end = as_utc(end or datetime.now(UTC))
        self.data_types = data_types
        self.refresh_universe = refresh_universe
        self.candle_options = candle_options or {
            "chunk_days": 2,
            "concurrency": 6,
            "max_requests_per_second": 9.0,
        }
        self.client = MarketDataClient(data_root=self.data_root)
        self.records: list[DownloadRecord] = []
        self.manifest_path = self.data_root / "maintenance" / "top_volume_history.json"

    def _save_manifest(self, universe: pl.DataFrame) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(UTC).isoformat(),
            "requested_start": self.start.isoformat(),
            "requested_end": self.end.isoformat(),
            "top_n": self.top_n,
            "data_types": [data_type.value for data_type in self.data_types],
            "instruments": universe.get_column("instrument_id").to_list(),
            "records": [asdict(record) for record in self.records],
        }
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, self.manifest_path)

    async def _universe(self) -> pl.DataFrame:
        as_of = datetime.now(UTC)
        if self.refresh_universe:
            await self.client.refresh_instruments_async(
                venue="OKX",
                instrument_type="SWAP",
                timestamp=as_of,
            )
        return self.client.trading_universe(
            as_of,
            name=f"okx_usdt_linear_swaps_top_{self.top_n}",
            venue="OKX",
            instrument_type="SWAP",
            settle_currency="USDT",
            contract_type="linear",
            states=("live",),
            min_listing_days=0,
            min_volume_usd_24h=0,
            min_open_interest_usd=0,
            max_spread_bps=None,
            top_n=self.top_n,
            refresh=False,
            persist=True,
        ).sort("rank")

    async def _fallback_partial(self, query: DataQuery) -> int:
        request = FetchRequest(
            data_type=query.data_type,
            venue=query.venue,
            instrument_id=query.instrument_id,
            start=query.start,
            end=query.end,
            bar_type=STORED_BAR_TYPES.get(query.data_type),
            options=query.options,
        )
        errors: list[str] = []
        for source in self.client._sources:
            if not source.supports(request):
                continue
            try:
                frame = canonicalize_frame(await source.fetch(request), query.data_type).filter(
                    (pl.col("timestamp") >= query.start)
                    & (pl.col("timestamp") < query.end)
                    & (pl.col("venue") == query.venue)
                    & (pl.col("instrument_id") == query.instrument_id)
                )
                self.client.store.write_observed(
                    frame,
                    data_type=query.data_type,
                    venue=query.venue,
                    instrument_id=query.instrument_id,
                    bar_type=STORED_BAR_TYPES.get(query.data_type),
                    source_name=f"{source.name}-partial",
                )
                return frame.height
            except Exception as exc:  # noqa: BLE001 - retain failures in the manifest
                errors.append(f"{source.name}: {exc}")
        raise DataUnavailableError("; ".join(errors) or "no source supports this request")

    async def _download_partition(self, query: DataQuery) -> DownloadRecord:
        try:
            frame = await self.client.query_async(query)
            return DownloadRecord(
                query.data_type.value,
                query.instrument_id,
                query.start.isoformat(),
                query.end.isoformat(),
                "complete",
                frame.height,
            )
        except DataUnavailableError as primary_error:
            try:
                rows = await self._fallback_partial(query)
                return DownloadRecord(
                    query.data_type.value,
                    query.instrument_id,
                    query.start.isoformat(),
                    query.end.isoformat(),
                    "partial" if rows else "unavailable",
                    rows,
                    str(primary_error),
                )
            except Exception as fallback_error:  # noqa: BLE001 - continue other datasets
                return DownloadRecord(
                    query.data_type.value,
                    query.instrument_id,
                    query.start.isoformat(),
                    query.end.isoformat(),
                    "failed",
                    error=f"{primary_error}; fallback: {fallback_error}",
                )
        except Exception as exc:  # noqa: BLE001 - continue other datasets
            return DownloadRecord(
                query.data_type.value,
                query.instrument_id,
                query.start.isoformat(),
                query.end.isoformat(),
                "failed",
                error=str(exc),
            )

    async def run(self) -> Path:
        free_gib = shutil.disk_usage(self.data_root.parent).free / 1024**3
        if free_gib < 10:
            raise RuntimeError(
                f"only {free_gib:.1f} GiB disk space remains; at least 10 GiB required"
            )

        universe = await self._universe()
        if universe.is_empty():
            raise DataUnavailableError("the current top-volume universe is empty")
        self._save_manifest(universe)
        seen_targets: set[tuple[DataType, str]] = set()

        for data_type in self.data_types:
            for row in universe.iter_rows(named=True):
                contract_id = str(row["instrument_id"])
                venue, instrument_id = target_instrument(data_type, contract_id)
                target_key = (data_type, instrument_id)
                if target_key in seen_targets:
                    continue
                seen_targets.add(target_key)

                list_time = row.get("list_time")
                dataset_start = max(
                    self.start,
                    as_utc(list_time) if isinstance(list_time, datetime) else self.start,
                )
                dataset_start, dataset_end = available_range(
                    data_type,
                    dataset_start,
                    self.end,
                )
                for start, end in partition_ranges(data_type, dataset_start, dataset_end):
                    options = self.candle_options if data_type is DataType.CANDLES else {}
                    query = DataQuery(
                        data_type,
                        instrument_id,
                        start,
                        end,
                        venue=venue,
                        bar_type=STORED_BAR_TYPES.get(data_type),
                        options=options,
                    )
                    record = await self._download_partition(query)
                    self.records.append(record)
                    self._save_manifest(universe)
                    print(
                        f"[{record.status:11}] {data_type.value:18} "
                        f"{instrument_id:24} {start.date()}..{end.date()} rows={record.rows}",
                        flush=True,
                    )
        return self.manifest_path


def parse_data_types(value: str) -> tuple[DataType, ...]:
    if value.strip().lower() == "all":
        return DEFAULT_DATA_TYPES
    try:
        return tuple(DataType(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        choices = ", ".join(data_type.value for data_type in DataType)
        raise argparse.ArgumentTypeError(
            f"data types must be 'all' or a subset of: {choices}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download all supported history for the current top-volume OKX swaps"
    )
    parser.add_argument("--start", default=DEFAULT_START.isoformat())
    parser.add_argument("--end", default=None, help="exclusive UTC end; defaults to now")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--data-types", type=parse_data_types, default=DEFAULT_DATA_TYPES)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--no-refresh-universe", action="store_true")
    parser.add_argument("--chunk-days", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-requests-per-second", type=float, default=9.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    downloader = HistoryDownloader(
        data_root=args.data_root,
        top_n=args.top_n,
        start=args.start,
        end=args.end,
        data_types=args.data_types,
        refresh_universe=not args.no_refresh_universe,
        candle_options={
            "chunk_days": args.chunk_days,
            "concurrency": args.concurrency,
            "max_requests_per_second": args.max_requests_per_second,
        },
    )
    manifest = asyncio.run(downloader.run())
    print(f"maintenance manifest: {manifest}")


if __name__ == "__main__":
    main()
