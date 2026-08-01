from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from trend_trader.data.offline.catalog import OfflineCatalog

from .client import BinancePublicClient, only_zip_files
from .config import BinanceOfflineConfig
from .models import ArchiveObject, ArchiveTask, Fragment
from .transform import transform_archive

_REMOTE_DATASET = {
    "candles": "klines",
    "funding_rates": "fundingRate",
    "mark_price_candles": "markPriceKlines",
    "index_price_candles": "indexPriceKlines",
    "aggregate_trades": "aggTrades",
}
_DATE_SUFFIX = re.compile(r"-(\d{4}-\d{2}(?:-\d{2})?)\.zip$")


class BinanceOfflineSynchronizer:
    def __init__(self, config: BinanceOfflineConfig) -> None:
        self.config = config

    async def build_plan(self) -> list[ArchiveTask]:
        async with BinancePublicClient(
            workers=self.config.download_workers,
            timeout_seconds=self.config.request_timeout_seconds,
            max_retries=self.config.max_retries,
        ) as client:
            if self.config.symbols:
                universes = [set(self.config.symbols) for _ in self.config.markets]
            else:
                universes = await asyncio.gather(
                    *(self._discover_universe(client, market) for market in self.config.markets)
                )
            requests = []
            for market, symbols in zip(self.config.markets, universes, strict=True):
                for dataset in self.config.datasets:
                    identifiers = (
                        {_index_identifier(market, symbol) for symbol in symbols}
                        if dataset == "index_price_candles"
                        else symbols
                    )
                    requests.extend(
                        self._list_symbol_archives(client, market, dataset, symbol)
                        for symbol in sorted(identifiers)
                    )
            task_groups = await asyncio.gather(*requests)
        unique = {task.source.key: task for group in task_groups for task in group}
        return sorted(unique.values(), key=lambda task: task.source.key)

    async def _discover_universe(self, client: BinancePublicClient, market: str) -> set[str]:
        historical, daily_entries = await asyncio.gather(
            client.historical_perpetual_symbols(market),
            client.list_objects(f"data/futures/{market}/daily/klines/", delimiter="/"),
        )
        daily_prefix = f"data/futures/{market}/daily/klines/"
        daily_symbols = {
            item.key.removeprefix(daily_prefix).strip("/")
            for item in daily_entries
            if item.key != daily_prefix
        }
        if market == "um":
            daily_symbols = {
                symbol for symbol in daily_symbols if not re.search(r"_\d{6}$", symbol)
            }
        else:
            daily_symbols = {symbol for symbol in daily_symbols if symbol.endswith("_PERP")}
        try:
            current = await client.current_perpetual_symbols(market)
        except Exception:
            current = set()
        return historical | daily_symbols | current

    async def _list_symbol_archives(
        self,
        client: BinancePublicClient,
        market: str,
        dataset: str,
        symbol: str,
    ) -> list[ArchiveTask]:
        remote = _REMOTE_DATASET[dataset]
        interval_suffix = (
            f"/{self.config.interval}"
            if dataset
            in {
                "candles",
                "mark_price_candles",
                "index_price_candles",
            }
            else ""
        )
        monthly_prefix = f"data/futures/{market}/monthly/{remote}/{symbol}{interval_suffix}/"
        monthly = only_zip_files(await client.list_objects(monthly_prefix))
        selected_monthly = [
            item for item in monthly if self._in_requested_range(item, monthly=True)
        ]
        tasks = [ArchiveTask(dataset, market, symbol, item, "monthly") for item in selected_monthly]
        if dataset == "funding_rates":
            return tasks

        covered_months = {
            value[:7] for item in selected_monthly if (value := _archive_period(item.key))
        }
        daily_prefix = f"data/futures/{market}/daily/{remote}/{symbol}{interval_suffix}/"
        month_prefixes = self._daily_month_prefixes(monthly, covered_months)
        daily_pages = await asyncio.gather(
            *(
                client.list_objects(daily_prefix + _daily_filename_prefix(dataset, symbol, month))
                for month in month_prefixes
            )
        )
        for item in only_zip_files(item for page in daily_pages for item in page):
            if self._in_requested_range(item, monthly=False):
                tasks.append(ArchiveTask(dataset, market, symbol, item, "daily"))
        return tasks

    def _daily_month_prefixes(
        self,
        monthly: list[ArchiveObject],
        covered_months: set[str],
    ) -> list[str]:
        end = self.config.end or datetime.now(UTC).date()
        if self.config.start:
            start = self.config.start
        else:
            periods = [period for item in monthly if (period := _archive_period(item.key))]
            if periods:
                year, month = map(int, max(periods)[:7].split("-"))
                start = date(year, month, 1)
            else:
                start = date(
                    end.year - (end.month == 1), 12 if end.month == 1 else end.month - 1, 1
                )
        months: list[str] = []
        cursor = date(start.year, start.month, 1)
        final = date(end.year, end.month, 1)
        while cursor <= final:
            value = cursor.strftime("%Y-%m")
            if value not in covered_months:
                months.append(value)
            cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
        return months

    def _in_requested_range(self, item: ArchiveObject, *, monthly: bool) -> bool:
        period = _archive_period(item.key)
        if not period:
            return False
        if monthly:
            item_start = date.fromisoformat(period + "-01")
            next_month = date(
                item_start.year + (item_start.month == 12), item_start.month % 12 + 1, 1
            )
            item_end = date.fromordinal(next_month.toordinal() - 1)
        else:
            item_start = item_end = date.fromisoformat(period)
        return not (
            (self.config.start and item_end < self.config.start)
            or (self.config.end and item_start > self.config.end)
        )

    async def run(self) -> dict[str, object]:
        started = datetime.now(UTC)
        run_id = started.strftime("%Y%m%dT%H%M%SZ")
        tasks = await self.build_plan()
        missing = [task for task in tasks if not task.raw_path(self.config.raw_root).is_file()]
        downloaded_bytes, fragments = await self._download_and_convert(tasks)
        outputs = await self._compact(fragments)
        report: dict[str, object] = {
            "run_id": run_id,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "status": "complete",
            "archive_count": len(tasks),
            "new_archive_count": len(missing),
            "downloaded_bytes": downloaded_bytes,
            "fragment_count": len(fragments),
            "normalized_file_count": len(outputs),
            "normalized_files": [str(path) for path in outputs[:100]],
            "normalized_files_truncated": len(outputs) > 100,
        }
        self._record_outputs(outputs)
        manifest = self.config.offline_root / "manifests" / "runs" / f"binance-{run_id}.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, report)
        return report

    async def run_compaction_only(self) -> dict[str, object]:
        """Resume final compaction from verified Parquet fragments without any network work."""
        started = datetime.now(UTC)
        fragments = self._discover_staged_fragments()
        outputs = await self._bulk_compact(fragments)
        self._record_outputs(outputs)
        report: dict[str, object] = {
            "run_id": started.strftime("compact-%Y%m%dT%H%M%SZ"),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "status": "complete",
            "fragment_count": len(fragments),
            "normalized_file_count": len(outputs),
            "normalized_files": [str(path) for path in outputs[:100]],
            "normalized_files_truncated": len(outputs) > 100,
        }
        manifest = (
            self.config.offline_root / "manifests" / "runs" / f"binance-{report['run_id']}.json"
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, report)
        return report

    async def run_publish_only(self) -> dict[str, object]:
        """Publish completed bulk output after an interrupted finalization step."""
        started = datetime.now(UTC)
        bulk_root = self.config.staging_root / "bulk"
        datasets = [
            dataset
            for dataset in self.config.datasets
            if dataset != "aggregate_trades" and (bulk_root / dataset).is_dir()
        ]
        if not datasets:
            raise RuntimeError(f"no completed bulk output found under {bulk_root}")

        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=len(datasets)) as pool:
            groups = await asyncio.gather(
                *(
                    loop.run_in_executor(
                        pool,
                        _publish_bulk_dataset,
                        dataset,
                        bulk_root / dataset,
                        self.config.normalized_root,
                        self.config.parquet_compression,
                        self.config.parquet_row_group_size,
                    )
                    for dataset in datasets
                )
            )
        outputs = [path for group in groups for path in group]
        self._record_outputs(outputs)
        # Publishing succeeded for every selected dataset, so the derived fragments are
        # no longer needed. Removing the tree directly avoids materializing ~170k
        # Fragment objects and is substantially faster on local filesystems.
        shutil.rmtree(self.config.staging_root / "fragments", ignore_errors=True)

        report: dict[str, object] = {
            "run_id": started.strftime("publish-%Y%m%dT%H%M%SZ"),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "status": "complete",
            "normalized_file_count": len(outputs),
            "normalized_files": [str(path) for path in outputs[:100]],
            "normalized_files_truncated": len(outputs) > 100,
        }
        manifest = (
            self.config.offline_root / "manifests" / "runs" / f"binance-{report['run_id']}.json"
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, report)
        return report

    def _discover_staged_fragments(self) -> list[Fragment]:
        root = self.config.staging_root / "fragments"
        fragments: list[Fragment] = []
        if not root.exists():
            return fragments
        allowed = set(self.config.datasets)
        for path in root.rglob("*.parquet"):
            relative = path.relative_to(root)
            if len(relative.parts) != 5:
                continue
            dataset, market, symbol, _, filename = relative.parts
            if dataset not in allowed or market not in self.config.markets:
                continue
            target_date = date.fromisoformat(filename[:10])
            if (
                self.config.start
                and target_date < self.config.start
                or self.config.end
                and target_date > self.config.end
            ):
                continue
            fragments.append(
                Fragment(
                    dataset=dataset,
                    market=market,
                    symbol=symbol,
                    target_date=target_date,
                    path=path,
                    row_count=0,
                )
            )
        return fragments

    async def _bulk_compact(self, fragments: list[Fragment]) -> list[Path]:
        by_dataset: dict[str, list[Fragment]] = defaultdict(list)
        aggregate_trades: list[Fragment] = []
        for fragment in fragments:
            if fragment.dataset == "aggregate_trades":
                aggregate_trades.append(fragment)
            else:
                by_dataset[fragment.dataset].append(fragment)

        loop = asyncio.get_running_loop()
        outputs: list[Path] = []
        dataset_count = max(1, len(by_dataset))
        threads_per_dataset = max(1, self.config.convert_workers // dataset_count)
        if by_dataset:
            with ProcessPoolExecutor(max_workers=dataset_count) as pool:
                jobs = [
                    loop.run_in_executor(
                        pool,
                        _bulk_compact_dataset,
                        dataset,
                        [str(item.path) for item in values],
                        self.config.staging_root / "bulk",
                        self.config.normalized_root,
                        self.config.parquet_compression,
                        self.config.parquet_row_group_size,
                        threads_per_dataset,
                    )
                    for dataset, values in by_dataset.items()
                ]
                for group in await asyncio.gather(*jobs):
                    outputs.extend(group)
        if aggregate_trades:
            outputs.extend(await self._compact(aggregate_trades, skip_existing=True))
        for fragment in fragments:
            fragment.path.unlink(missing_ok=True)
        return outputs

    async def _download_and_convert(self, tasks: list[ArchiveTask]) -> tuple[int, list[Fragment]]:
        """Pipeline network and CPU work so neither resource waits for the other phase."""
        downloaded_bytes = 0
        completed = 0
        lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        task_queue: asyncio.Queue[ArchiveTask] = asyncio.Queue()
        convert_queue: asyncio.Queue[ArchiveTask | None] = asyncio.Queue(
            maxsize=max(1, self.config.convert_workers * 4)
        )
        for task in tasks:
            task_queue.put_nowait(task)
        groups: list[list[Fragment]] = []
        async with BinancePublicClient(
            workers=self.config.download_workers,
            timeout_seconds=self.config.request_timeout_seconds,
            max_retries=self.config.max_retries,
        ) as client:
            with ProcessPoolExecutor(max_workers=self.config.convert_workers) as pool:

                async def download_worker() -> None:
                    nonlocal downloaded_bytes
                    while True:
                        try:
                            task = task_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        changed = await client.download_verified(
                            task.source, task.raw_path(self.config.raw_root)
                        )
                        if changed:
                            async with lock:
                                downloaded_bytes += task.source.size
                        await convert_queue.put(task)
                        task_queue.task_done()

                async def convert_worker() -> None:
                    nonlocal completed
                    while True:
                        task = await convert_queue.get()
                        if task is None:
                            convert_queue.task_done()
                            return
                        converted = await loop.run_in_executor(
                            pool,
                            partial(
                                transform_archive,
                                task,
                                task.raw_path(self.config.raw_root),
                                self.config.staging_root,
                                compression=self.config.parquet_compression,
                                row_group_size=self.config.parquet_row_group_size,
                            ),
                        )
                        groups.append(converted)
                        completed += 1
                        if completed == len(tasks) or completed % 100 == 0:
                            print(
                                f"download+convert: {completed}/{len(tasks)} archives",
                                file=sys.stderr,
                                flush=True,
                            )
                        convert_queue.task_done()

                converters = [
                    asyncio.create_task(convert_worker())
                    for _ in range(self.config.convert_workers)
                ]
                downloaders = [
                    asyncio.create_task(download_worker())
                    for _ in range(min(self.config.download_workers, max(1, len(tasks))))
                ]
                await asyncio.gather(*downloaders)
                for _ in converters:
                    await convert_queue.put(None)
                await asyncio.gather(*converters)
        fragments = [fragment for group in groups for fragment in group]
        return downloaded_bytes, self._filter_fragments(fragments)

    def _record_outputs(self, outputs: list[Path]) -> None:
        catalog = OfflineCatalog(self.config.offline_root)
        for path in outputs:
            relative = path.relative_to(self.config.normalized_root)
            dataset = relative.parts[0]
            target_date = date.fromisoformat(relative.parts[3].removeprefix("date="))
            row_count = pq.ParquetFile(path).metadata.num_rows
            if dataset == "aggregate_trades":
                stem = path.name.removesuffix(f"-aggregate_trades-{target_date}.parquet")
                market, symbol = stem.split("-", maxsplit=1)
                scope_key = f"venue=BINANCE/market={market}/instrument={symbol}"
            else:
                scope_key = "venue=BINANCE"
            catalog.record_artifact(
                path,
                dataset=dataset,
                source_url=None,
                source_sha256=None,
                metadata={
                    "dataset_kind": "offline",
                    "source_name": "binance_public_archive",
                    "schema_version": 1,
                    "scope_key": scope_key,
                },
            )
            catalog.mark_coverage(
                dataset,
                target_date,
                status="complete",
                row_count=row_count,
                scope_key=scope_key,
                artifact_path=str(path),
            )

    def _filter_fragments(self, fragments: list[Fragment]) -> list[Fragment]:
        selected: list[Fragment] = []
        for fragment in fragments:
            if (
                self.config.start
                and fragment.target_date < self.config.start
                or self.config.end
                and fragment.target_date > self.config.end
            ):
                fragment.path.unlink(missing_ok=True)
            else:
                selected.append(fragment)
        return selected

    async def _compact(
        self, fragments: list[Fragment], *, skip_existing: bool = False
    ) -> list[Path]:
        grouped: dict[tuple[str, ...], list[Fragment]] = defaultdict(list)
        for fragment in fragments:
            if fragment.dataset == "aggregate_trades":
                key = (
                    fragment.dataset,
                    fragment.target_date.isoformat(),
                    fragment.market,
                    fragment.symbol,
                )
            else:
                key = (fragment.dataset, fragment.target_date.isoformat())
            grouped[key].append(fragment)
        loop = asyncio.get_running_loop()
        jobs = []
        completed_outputs: list[Path] = []
        workers = min(self.config.convert_workers, max(1, len(grouped)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for key, values in grouped.items():
                target = _normalized_path(self.config.normalized_root, key)
                if skip_existing and target.is_file():
                    completed_outputs.append(target)
                    continue
                source_paths = [str(item.path) for item in values]
                if target.exists():
                    source_paths.append(str(target))
                jobs.append(
                    loop.run_in_executor(
                        pool,
                        _compact_group,
                        key[0],
                        source_paths,
                        target,
                        self.config.parquet_compression,
                        self.config.parquet_row_group_size,
                    )
                )
            outputs = await asyncio.gather(*jobs)
        for fragment in fragments:
            fragment.path.unlink(missing_ok=True)
        return completed_outputs + outputs


def _archive_period(key: str) -> str | None:
    match = _DATE_SUFFIX.search(key)
    return match.group(1) if match else None


def _daily_filename_prefix(dataset: str, symbol: str, month: str) -> str:
    interval = (
        "-1h"
        if dataset
        in {
            "candles",
            "mark_price_candles",
            "index_price_candles",
        }
        else "-aggTrades"
    )
    return f"{symbol}{interval}-{month}"


def _index_identifier(market: str, symbol: str) -> str:
    return symbol.removesuffix("_PERP") if market == "cm" else symbol


def _normalized_path(root: Path, key: tuple[str, ...]) -> Path:
    dataset = key[0]
    target_date = date.fromisoformat(key[1])
    directory = (
        root
        / dataset
        / "venue=BINANCE"
        / f"year={target_date.year:04d}"
        / f"date={target_date.isoformat()}"
    )
    if dataset == "aggregate_trades":
        filename = f"{key[2].upper()}-{key[3]}-aggregate_trades-{target_date}.parquet"
    else:
        filename = f"{dataset}-{target_date}.parquet"
    return directory / filename


_KEYS = {
    "candles": ("venue", "market_type", "instrument_id", "bar_type", "timestamp"),
    "funding_rates": ("venue", "market_type", "instrument_id", "funding_time"),
    "mark_price_candles": (
        "venue",
        "market_type",
        "instrument_id",
        "bar_type",
        "timestamp",
    ),
    "index_price_candles": ("venue", "market_type", "index_id", "bar_type", "timestamp"),
    "aggregate_trades": (
        "venue",
        "market_type",
        "instrument_id",
        "aggregate_trade_id",
    ),
}


def _compact_group(
    dataset: str,
    source_paths: list[str],
    target: Path,
    compression: str,
    row_group_size: int,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads TO 1")
        connection.execute("SET TimeZone='UTC'")
        connection.execute("PRAGMA disable_progress_bar")
        connection.from_parquet(source_paths, union_by_name=False).create_view("source")
        keys = ", ".join(f'"{name}"' for name in _KEYS[dataset])
        time_column = "funding_time" if dataset == "funding_rates" else "timestamp"
        escaped = str(temporary).replace("'", "''")
        downloaded_at = datetime.now(UTC).isoformat().replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT * EXCLUDE (_row_number)
                FROM (
                    SELECT *, row_number() OVER (PARTITION BY {keys}) AS _row_number
                    FROM source
                )
                WHERE _row_number = 1
                ORDER BY "{time_column}", {keys}
            ) TO '{escaped}' (
                FORMAT PARQUET,
                COMPRESSION {compression.upper()},
                ROW_GROUP_SIZE {row_group_size},
                KV_METADATA {{
                    dataset_kind: 'offline',
                    source_name: 'binance_public_archive',
                    schema_version: '1',
                    downloaded_at: '{downloaded_at}'
                }}
            )
            """
        )
    finally:
        connection.close()
    temporary.replace(target)
    return target


def _bulk_compact_dataset(
    dataset: str,
    source_paths: list[str],
    staging_root: Path,
    normalized_root: Path,
    compression: str,
    row_group_size: int,
    threads: int,
) -> list[Path]:
    """Compact one complete dataset scan and let DuckDB partition it by UTC date."""
    work_root = staging_root / dataset
    shutil.rmtree(work_root, ignore_errors=True)
    output_root = work_root / "output"
    temporary_root = work_root / "duckdb-temp"
    output_root.mkdir(parents=True)
    temporary_root.mkdir(parents=True)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads TO {threads}")
        # TIMESTAMPTZ calendar functions use the session timezone. All offline
        # partitions are UTC dates, independent of the host's local timezone.
        connection.execute("SET TimeZone='UTC'")
        connection.execute("PRAGMA disable_progress_bar")
        escaped_temp = str(temporary_root).replace("'", "''")
        connection.execute(f"SET temp_directory='{escaped_temp}'")
        connection.from_parquet(source_paths, union_by_name=False).create_view("source")
        keys = ", ".join(f'"{name}"' for name in _KEYS[dataset])
        time_column = "funding_time" if dataset == "funding_rates" else "timestamp"
        escaped_output = str(output_root).replace("'", "''")
        downloaded_at = datetime.now(UTC).isoformat().replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT * EXCLUDE (_row_number),
                       year("{time_column}") AS year,
                       strftime("{time_column}", '%Y-%m-%d') AS date
                FROM (
                    SELECT *, row_number() OVER (PARTITION BY {keys}) AS _row_number
                    FROM source
                )
                WHERE _row_number = 1
                ORDER BY "{time_column}", {keys}
            ) TO '{escaped_output}' (
                FORMAT PARQUET,
                PARTITION_BY (year, date),
                COMPRESSION {compression.upper()},
                ROW_GROUP_SIZE {row_group_size},
                KV_METADATA {{
                    dataset_kind: 'offline',
                    source_name: 'binance_public_archive',
                    schema_version: '1',
                    downloaded_at: '{downloaded_at}'
                }}
            )
            """
        )
    finally:
        connection.close()

    return _publish_bulk_dataset(
        dataset,
        work_root,
        normalized_root,
        compression,
        row_group_size,
    )


def _publish_bulk_dataset(
    dataset: str,
    work_root: Path,
    normalized_root: Path,
    compression: str,
    row_group_size: int,
) -> list[Path]:
    """Atomically publish bulk partitions, merging parallel part files when needed."""
    output_root = work_root / "output"
    if not output_root.is_dir():
        raise RuntimeError(f"bulk output is missing for {dataset}: {output_root}")

    outputs: list[Path] = []
    for date_directory in output_root.rglob("date=*"):
        files = list(date_directory.glob("*.parquet"))
        target_date = date.fromisoformat(date_directory.name.removeprefix("date="))
        target = _normalized_path(normalized_root, (dataset, target_date.isoformat()))
        target.parent.mkdir(parents=True, exist_ok=True)
        sources = [str(path) for path in files]
        if target.is_file() and files:
            sources.append(str(target))
        if len(sources) == 1:
            files[0].replace(target)
        elif len(sources) > 1:
            _compact_group(dataset, sources, target, compression, row_group_size)
        if target.is_file():
            outputs.append(target)
        else:
            raise RuntimeError(f"bulk partition has no output: {dataset}/{date_directory.name}")
    shutil.rmtree(work_root)
    return outputs


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
