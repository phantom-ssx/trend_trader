from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from trend_trader.data.offline.catalog import OfflineCatalog
from trend_trader.data.offline.client import (
    OPEN_INTEREST_PATH,
    RATIO_PATHS,
    TAKER_VOLUME_PATH,
    OkxOfflineClient,
)
from trend_trader.data.offline.config import DatasetOptions, OfflineSyncConfig
from trend_trader.data.offline.schemas import (
    DATASET_STORAGE,
    aggregate_oi_frame,
    iter_candle_archive_batches,
    iter_funding_archive_batches,
    price_candle_frame,
    private_bills_frame,
    private_fills_frame,
    private_orders_frame,
    ratio_frame,
    taker_volume_frame,
)
from trend_trader.data.offline.storage import (
    DailyParquetRepository,
    OfflineLayout,
    RawRepository,
    StreamingDailyParquetRepository,
    sha256_file,
    write_run_report,
)

HISTORICAL_MODULES = {"candles": 2, "funding_rates": 3}
RETENTION_START_DAYS: dict[str, int | dict[str, int]] = {
    "aggregate_open_interest": {"5m": 2, "1H": 30, "1D": 180},
    "private_final_orders": 90,
    "private_fills": 90,
    "private_bills": 90,
}
RATIO_STARTS = {
    "account": date(2024, 2, 1),
    "top_trader_account": date(2024, 3, 22),
    "top_trader_position": date(2024, 3, 22),
}
RATIO_TYPES = tuple(RATIO_PATHS)


@dataclass(frozen=True)
class SyncTask:
    dataset: str
    target_date: date
    scope_key: str = "all"


@dataclass
class DatasetResult:
    dataset: str
    target_date: str
    scope_key: str
    status: str
    rows: int = 0
    files: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class HistoricalRawFile:
    path: Path
    url: str
    sha256: str
    instrument_type: str


class OfflineSynchronizer:
    def __init__(
        self,
        config: OfflineSyncConfig,
        *,
        client_factory: Callable[[], OkxOfflineClient] | None = None,
    ) -> None:
        self.config = config
        self.layout = OfflineLayout(config.data_root)
        self.catalog = OfflineCatalog(self.layout.root)
        self.raw = RawRepository(self.layout)
        self.client_factory = client_factory or (lambda: OkxOfflineClient(config))

    def plan(
        self,
        *,
        mode: str,
        today: date | None = None,
        start: date | None = None,
        end: date | None = None,
        datasets: set[str] | None = None,
    ) -> list[SyncTask]:
        if mode not in {"daily", "backfill", "range"}:
            raise ValueError(f"unsupported mode: {mode}")
        current = today or datetime.now(UTC).date()
        tasks: list[SyncTask] = []
        for dataset, options in self.config.datasets.enabled().items():
            if datasets and dataset not in datasets:
                continue
            if dataset.startswith("private_"):
                scopes = [account.alias for account in self.config.private_accounts]
            elif dataset == "aggregate_open_interest":
                scopes = list(options.periods)
            elif dataset == "long_short_ratio":
                scopes = list(RATIO_TYPES)
            else:
                scopes = ["all"]
            if not scopes:
                continue
            last = min(
                end or current - timedelta(days=options.mature_lag_days),
                current - timedelta(days=options.mature_lag_days),
            )
            for scope in scopes:
                first = start or self._dataset_start(dataset, options, current, scope)
                if first > last:
                    continue
                if mode == "daily":
                    first = max(
                        first,
                        last - timedelta(days=self.config.daily_history_days_per_run - 1),
                    )
                candidates = list(_date_range(first, last))
                missing = [
                    day
                    for day in candidates
                    if not self.catalog.is_resolved(dataset, day, scope)
                ]
                if mode == "daily" and dataset.startswith("private_"):
                    overlap_start = max(first, last - timedelta(days=2))
                    missing = sorted(set(missing) | set(_date_range(overlap_start, last)))
                tasks.extend(SyncTask(dataset, day, scope) for day in missing)
        return sorted(tasks, key=lambda item: (item.target_date, item.dataset, item.scope_key))

    async def run(
        self,
        *,
        mode: str = "daily",
        start: date | None = None,
        end: date | None = None,
        datasets: set[str] | None = None,
        today: date | None = None,
    ) -> dict[str, Any]:
        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        report: dict[str, Any] = {
            "run_id": run_id,
            "mode": mode,
            "started_at": datetime.now(UTC).isoformat(),
            "dataset_kind": "offline",
            "data_root": str(self.layout.root),
            "results": [],
        }
        with self.catalog.execution_lock():
            report["removed_stale_staging_files"] = self._cleanup_staging()
            report["recovered_interrupted_runs"] = self.catalog.recover_interrupted_runs()
            self.catalog.begin_run(run_id, mode, report)
            try:
                self._check_disk()
                tasks = self.plan(
                    mode=mode,
                    start=start,
                    end=end,
                    datasets=datasets,
                    today=today,
                )
                report["planned_tasks"] = len(tasks)
                async with self.client_factory() as client:
                    instruments = await client.fetch_instruments()
                    seen_at = datetime.now(UTC)
                    for instrument in instruments:
                        self.catalog.upsert_instrument(instrument, seen_at)
                    report["current_instruments"] = len(instruments)
                    for task in tasks:
                        self._check_disk()
                        _progress(
                            f"start dataset={task.dataset} date={task.target_date} "
                            f"scope={task.scope_key}"
                        )
                        result = await self._execute(client, task)
                        report["results"].append(asdict(result))
                        _progress(
                            f"finish dataset={task.dataset} date={task.target_date} "
                            f"scope={task.scope_key} status={result.status} rows={result.rows}"
                        )
                failures = [row for row in report["results"] if row["status"] == "failed"]
                report["status"] = "partial_failure" if failures else "success"
                report["failed_tasks"] = len(failures)
                report["completed_tasks"] = len(report["results"]) - len(failures)
                exit_code = 1 if failures else 0
            except Exception as exc:
                report["status"] = "failed"
                report["fatal_error"] = f"{type(exc).__name__}: {exc}"
                exit_code = 1
            report["finished_at"] = datetime.now(UTC).isoformat()
            report_path = self.layout.run_dir() / f"{run_id}.json"
            report["report_path"] = str(report_path)
            write_run_report(self.layout, run_id, report)
            self.catalog.finish_run(
                run_id,
                status=str(report["status"]),
                exit_code=exit_code,
                report=report,
            )
        return report

    async def _execute(self, client: OkxOfflineClient, task: SyncTask) -> DatasetResult:
        try:
            if task.dataset in HISTORICAL_MODULES:
                return await self._execute_historical_file_dataset(client, task)
            if task.dataset in {"mark_price_candles", "index_price_candles"}:
                files, frame = await self._price_dataset(client, task)
            elif task.dataset == "aggregate_open_interest":
                files, frame = await self._open_interest_dataset(client, task)
            elif task.dataset == "taker_volume":
                files, frame = await self._taker_dataset(client, task)
            elif task.dataset == "long_short_ratio":
                files, frame = await self._ratio_dataset(client, task)
            elif task.dataset.startswith("private_"):
                files, frame = await self._private_dataset(client, task)
            else:
                raise ValueError(f"no executor for {task.dataset}")
            self._remember_instruments(frame, task.target_date)
            outputs = self._write_frame(task, frame)
            status = (
                "complete"
                if frame.height
                else (
                    "complete_empty"
                    if task.dataset.startswith("private_")
                    else "unavailable"
                )
            )
            artifact = str(outputs[0][1]) if outputs else None
            self.catalog.mark_coverage(
                task.dataset,
                task.target_date,
                status=status,
                row_count=frame.height,
                scope_key=task.scope_key,
                artifact_path=artifact,
            )
            if frame.height:
                first_time = _first_timestamp_text(frame)
                self.catalog.upsert_availability(
                    task.dataset,
                    task.scope_key,
                    first_event_time=first_time,
                    continuous_start=self.catalog.earliest_complete_date(
                        task.dataset, task.scope_key
                    ),
                    discovery_method="download_observation",
                    status="observed",
                )
            elif status == "unavailable":
                self.catalog.upsert_availability(
                    task.dataset,
                    task.scope_key,
                    discovery_method="empty_mature_response",
                    status="unavailable_for_date",
                )
            return DatasetResult(
                task.dataset,
                task.target_date.isoformat(),
                task.scope_key,
                status,
                rows=frame.height,
                files=[str(file) for file in [*files, *(item[1] for item in outputs)]],
            )
        except Exception as exc:
            self.catalog.mark_coverage(
                task.dataset,
                task.target_date,
                status="failed",
                row_count=0,
                scope_key=task.scope_key,
            )
            return DatasetResult(
                task.dataset,
                task.target_date.isoformat(),
                task.scope_key,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _execute_historical_file_dataset(
        self,
        client: OkxOfflineClient,
        task: SyncTask,
    ) -> DatasetResult:
        raw_files = await self._download_historical_files(client, task)
        first_event_seen: str | None = None
        parsed_rows = 0
        next_progress_rows = 250_000

        def batches() -> Iterable[pl.DataFrame]:
            nonlocal first_event_seen, next_progress_rows, parsed_rows
            iterator = (
                iter_candle_archive_batches
                if task.dataset == "candles"
                else iter_funding_archive_batches
            )
            for raw_file in raw_files:
                try:
                    for frame in iterator(
                        raw_file.path,
                        batch_size=self.config.stream_batch_rows,
                    ):
                        self._remember_instruments(frame, task.target_date)
                        first_event = _first_timestamp_text(frame)
                        if first_event and (
                            first_event_seen is None
                            or first_event < first_event_seen
                        ):
                            first_event_seen = first_event
                        parsed_rows += frame.height
                        if parsed_rows >= next_progress_rows:
                            _progress(
                                f"parse dataset={task.dataset} date={task.target_date} "
                                f"rows={parsed_rows}"
                            )
                            next_progress_rows = (
                                parsed_rows // 250_000 + 1
                            ) * 250_000
                        yield frame
                except Exception:
                    quarantine = self.layout.quarantine_dir(
                        task.dataset, task.target_date
                    )
                    quarantine.mkdir(parents=True, exist_ok=True)
                    destination = quarantine / (
                        f"{raw_file.path.stem}.bad-{uuid4().hex[:8]}"
                        f"{raw_file.path.suffix}"
                    )
                    raw_file.path.replace(destination)
                    raise

        schema, key, timestamp = DATASET_STORAGE[task.dataset]
        repository = StreamingDailyParquetRepository(
            self.layout,
            dataset=task.dataset,
            schema=schema,
            primary_key=key,
            timestamp_column=timestamp,
            sort_columns=[timestamp, *[item for item in key if item != timestamp]],
            compression=self.config.parquet_compression,
            row_group_size=self.config.parquet_row_group_size,
            source_name="okx_historical_file",
            batch_rows=self.config.stream_batch_rows,
            compaction_memory_mb=self.config.compaction_memory_mb,
            compaction_threads=self.config.compaction_threads,
        )
        outputs, input_rows = repository.write_batches(batches())
        for raw_file in raw_files:
            self.catalog.record_artifact(
                raw_file.path,
                dataset=task.dataset,
                source_url=raw_file.url,
                source_sha256=raw_file.sha256,
                metadata={
                    "source_date": task.target_date.isoformat(),
                    "date_basis": "UTC+08:00",
                    "instrument_type": raw_file.instrument_type,
                },
            )
        for target_date, path, row_count in outputs:
            self.catalog.record_artifact(
                path,
                dataset=task.dataset,
                source_url=None,
                source_sha256=None,
                metadata={
                    "utc_date": target_date.isoformat(),
                    "rows": row_count,
                    "compaction": "duckdb_external_sort",
                },
            )
        status = "complete" if input_rows else "unavailable"
        artifact = str(outputs[0][1]) if outputs else None
        self.catalog.mark_coverage(
            task.dataset,
            task.target_date,
            status=status,
            row_count=input_rows,
            scope_key=task.scope_key,
            artifact_path=artifact,
        )
        if first_event_seen:
            self.catalog.upsert_availability(
                task.dataset,
                task.scope_key,
                first_event_time=first_event_seen,
                continuous_start=self.catalog.earliest_complete_date(
                    task.dataset, task.scope_key
                ),
                discovery_method="historical_file_stream",
                status="observed",
            )
        return DatasetResult(
            task.dataset,
            task.target_date.isoformat(),
            task.scope_key,
            status,
            rows=input_rows,
            files=[
                str(item)
                for item in [
                    *(raw_file.path for raw_file in raw_files),
                    *(output[1] for output in outputs),
                ]
            ],
        )

    async def _download_historical_files(
        self,
        client: OkxOfflineClient,
        task: SyncTask,
    ) -> list[HistoricalRawFile]:
        module = HISTORICAL_MODULES[task.dataset]
        instrument_types = ("SWAP", "FUTURES") if task.dataset == "candles" else ("SWAP",)
        raw_files: list[HistoricalRawFile] = []
        for instrument_type in instrument_types:
            links = await client.historical_links(
                module=module,
                instrument_type=instrument_type,
                source_date=task.target_date,
            )
            for number, link in enumerate(links):
                url = _link_value(link, "url", "downloadUrl", "downloadLink")
                if not url:
                    continue
                filename = Path(
                    _link_value(link, "fileName", "filename", "name")
                    or f"{instrument_type.lower()}-{task.dataset}-{number}.zip"
                ).name
                destination = self.layout.raw_dir(task.dataset, task.target_date) / filename
                _progress(
                    f"download dataset={task.dataset} date={task.target_date} "
                    f"file={filename}"
                )
                downloaded, digest = await client.download(url, destination)
                _progress(
                    f"downloaded dataset={task.dataset} date={task.target_date} "
                    f"file={downloaded.name} bytes={downloaded.stat().st_size}"
                )
                raw_files.append(
                    HistoricalRawFile(
                        path=downloaded,
                        url=url,
                        sha256=digest,
                        instrument_type=instrument_type,
                    )
                )
        if not raw_files:
            raise FileNotFoundError(
                f"OKX has not published {task.dataset} for {task.target_date}"
            )
        return raw_files

    async def _price_dataset(
        self,
        client: OkxOfflineClient,
        task: SyncTask,
    ) -> tuple[list[Path], pl.DataFrame]:
        start_ms, end_ms = _day_milliseconds(task.target_date)
        catalog = self.catalog.list_instruments()
        is_index = task.dataset == "index_price_candles"
        identifiers = sorted(
            {
                str(row["index_id"] if is_index else row["instrument_id"])
                for row in catalog
                if row.get("index_id" if is_index else "instrument_id")
            }
        )
        raw: dict[str, object] = {}
        fragments: list[pl.DataFrame] = []
        for identifier in identifiers:
            rows = await client.fetch_price_candles(
                instrument_id=identifier,
                start_ms=start_ms,
                end_ms=end_ms,
                index=is_index,
            )
            raw[identifier] = rows
            self._observe_api_rows(task.dataset, identifier, rows)
            fragments.append(
                price_candle_frame(rows, instrument_id=identifier, is_index=is_index)
            )
        raw_path = self.raw.write_json_gz(task.dataset, task.target_date, raw)
        self._record_rest_raw(raw_path, task.dataset)
        return [raw_path], _concat(fragments, DATASET_STORAGE[task.dataset][0])

    async def _open_interest_dataset(
        self,
        client: OkxOfflineClient,
        task: SyncTask,
    ) -> tuple[list[Path], pl.DataFrame]:
        start_ms, end_ms = _day_milliseconds(task.target_date)
        currencies = sorted(
            {
                str(row["base_currency"])
                for row in self.catalog.list_instruments()
                if row.get("base_currency")
            }
        )
        raw: dict[str, object] = {}
        fragments: list[pl.DataFrame] = []
        period = task.scope_key
        for currency in currencies:
            rows = await client.fetch_metric(
                OPEN_INTEREST_PATH,
                params={"ccy": currency, "period": period},
                start_ms=start_ms,
                end_ms=end_ms,
            )
            raw[currency] = rows
            self._observe_api_rows(task.dataset, f"{period}:{currency}", rows)
            fragments.append(
                aggregate_oi_frame(rows, base_currency=currency, period=period)
            )
        raw_path = self.raw.write_json_gz(
            task.dataset,
            task.target_date,
            raw,
            suffix=f"-{period}",
        )
        self._record_rest_raw(raw_path, task.dataset)
        return [raw_path], _concat(fragments, DATASET_STORAGE[task.dataset][0])

    async def _taker_dataset(
        self,
        client: OkxOfflineClient,
        task: SyncTask,
    ) -> tuple[list[Path], pl.DataFrame]:
        start_ms, end_ms = _day_milliseconds(task.target_date)
        identifiers = self._instrument_ids()
        raw: dict[str, object] = {}
        fragments: list[pl.DataFrame] = []
        period = self.config.datasets.taker_volume.periods[0]
        for identifier in identifiers:
            rows = await client.fetch_metric(
                TAKER_VOLUME_PATH,
                params={"instId": identifier, "period": period},
                start_ms=start_ms,
                end_ms=end_ms,
            )
            raw[identifier] = rows
            self._observe_api_rows(task.dataset, identifier, rows)
            fragments.append(
                taker_volume_frame(rows, instrument_id=identifier, period=period)
            )
        raw_path = self.raw.write_json_gz(task.dataset, task.target_date, raw)
        self._record_rest_raw(raw_path, task.dataset)
        return [raw_path], _concat(fragments, DATASET_STORAGE[task.dataset][0])

    async def _ratio_dataset(
        self,
        client: OkxOfflineClient,
        task: SyncTask,
    ) -> tuple[list[Path], pl.DataFrame]:
        start_ms, end_ms = _day_milliseconds(task.target_date)
        identifiers = self._instrument_ids()
        period = self.config.datasets.long_short_ratio.periods[0]
        raw: dict[str, object] = {}
        fragments: list[pl.DataFrame] = []
        ratio_type = task.scope_key
        for identifier in identifiers:
            rows = await client.fetch_metric(
                RATIO_PATHS[ratio_type],
                params={"instId": identifier, "period": period},
                start_ms=start_ms,
                end_ms=end_ms,
            )
            raw[identifier] = rows
            self._observe_api_rows(
                task.dataset,
                f"{ratio_type}:{identifier}",
                rows,
            )
            fragments.append(
                ratio_frame(
                    rows,
                    instrument_id=identifier,
                    period=period,
                    ratio_type=ratio_type,
                )
            )
        raw_path = self.raw.write_json_gz(
            task.dataset,
            task.target_date,
            raw,
            suffix=f"-{ratio_type}",
        )
        self._record_rest_raw(raw_path, task.dataset)
        return [raw_path], _concat(fragments, DATASET_STORAGE[task.dataset][0])

    async def _private_dataset(
        self,
        client: OkxOfflineClient,
        task: SyncTask,
    ) -> tuple[list[Path], pl.DataFrame]:
        account = next(
            account
            for account in self.config.private_accounts
            if account.alias == task.scope_key
        )
        start_ms, end_ms = _day_milliseconds(task.target_date)
        rows = await client.fetch_private_rows(
            task.dataset,
            credentials=account.credentials(),
            start_ms=start_ms,
            end_ms=end_ms,
        )
        raw_path = self.raw.write_json_gz(
            task.dataset,
            task.target_date,
            rows,
            account_alias=account.alias,
        )
        self._record_rest_raw(raw_path, task.dataset)
        builders = {
            "private_final_orders": private_orders_frame,
            "private_fills": private_fills_frame,
            "private_bills": private_bills_frame,
        }
        return [raw_path], builders[task.dataset](rows, account.alias)

    def _write_frame(
        self,
        task: SyncTask,
        frame: pl.DataFrame,
    ) -> list[tuple[date, Path, int]]:
        schema, key, timestamp = DATASET_STORAGE[task.dataset]
        sort_columns = [timestamp, *[item for item in key if item != timestamp]]
        repository = DailyParquetRepository(
            self.layout,
            dataset=task.dataset,
            schema=schema,
            primary_key=key,
            timestamp_column=timestamp,
            sort_columns=sort_columns,
            compression=self.config.parquet_compression,
            row_group_size=self.config.parquet_row_group_size,
            account_alias=task.scope_key if task.dataset.startswith("private_") else None,
            source_name=(
                "okx_historical_file"
                if task.dataset in HISTORICAL_MODULES
                else (
                    "okx_private_rest"
                    if task.dataset.startswith("private_")
                    else "okx_public_rest"
                )
            ),
        )
        outputs = repository.write(frame)
        for target_date, path, row_count in outputs:
            self.catalog.record_artifact(
                path,
                dataset=task.dataset,
                source_url=None,
                source_sha256=None,
                metadata={"utc_date": target_date.isoformat(), "rows": row_count},
            )
        return outputs

    def _record_rest_raw(self, raw_path: Path, dataset: str) -> None:
        self.catalog.record_artifact(
            raw_path,
            dataset=dataset,
            source_url="OKX REST API",
            source_sha256=sha256_file(raw_path),
            metadata={
                "source_name": (
                    "okx_private_rest" if dataset.startswith("private_") else "okx_public_rest"
                )
            },
        )

    def _instrument_ids(self) -> list[str]:
        return sorted(
            {
                str(row["instrument_id"])
                for row in self.catalog.list_instruments()
                if row.get("instrument_id")
            }
        )

    def _remember_instruments(self, frame: pl.DataFrame, target_date: date) -> None:
        if "instrument_id" not in frame.columns or frame.is_empty():
            return
        columns = ["instrument_id"]
        if "instrument_type" in frame.columns:
            columns.append("instrument_type")
        rows = frame.select(columns).unique().iter_rows(named=True)
        seen_at = datetime.combine(target_date, time.min, tzinfo=UTC)
        for row in rows:
            self.catalog.upsert_instrument(row, seen_at)

    def _observe_api_rows(
        self,
        dataset: str,
        scope_key: str,
        rows: list[list[object]],
    ) -> None:
        timestamps = [
            int(str(row[0]))
            for row in rows
            if row and str(row[0]).lstrip("-").isdigit()
        ]
        if not timestamps:
            return
        first_event = datetime.fromtimestamp(min(timestamps) / 1000, tz=UTC).isoformat()
        self.catalog.upsert_availability(
            dataset,
            scope_key,
            first_event_time=first_event,
            discovery_method="public_rest_probe",
            status="observed",
        )

    def _dataset_start(
        self,
        dataset: str,
        options: DatasetOptions,
        today: date,
        scope_key: str,
    ) -> date:
        if dataset == "long_short_ratio":
            return RATIO_STARTS[scope_key]
        if options.start:
            return options.start
        retention = RETENTION_START_DAYS.get(dataset, 1)
        days = retention.get(scope_key, 1) if isinstance(retention, dict) else retention
        return today - timedelta(days=days)

    def _check_disk(self) -> None:
        self.layout.root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.layout.root)
        if usage.free / usage.total < self.config.min_free_disk_ratio:
            raise OSError(
                f"free disk ratio {usage.free / usage.total:.1%} is below "
                f"{self.config.min_free_disk_ratio:.1%}"
            )

    def _cleanup_staging(self) -> int:
        staging = self.layout.root / ".staging"
        if not staging.exists():
            return 0
        removed = 0
        for path in staging.iterdir():
            if path.is_file():
                path.unlink()
                removed += 1
            elif path.is_dir():
                shutil.rmtree(path)
                removed += 1
        return removed


def _date_range(first: date, last: date) -> Iterable[date]:
    current = first
    while current <= last:
        yield current
        current += timedelta(days=1)


def _day_milliseconds(day: date) -> tuple[int, int]:
    start = datetime.combine(day, time.min, tzinfo=UTC)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _concat(
    frames: list[pl.DataFrame],
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    nonempty = [frame for frame in frames if not frame.is_empty()]
    if not nonempty:
        return pl.DataFrame(schema=dict(schema))
    return pl.concat(nonempty, how="vertical_relaxed")


def _link_value(item: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _first_timestamp_text(frame: pl.DataFrame) -> str | None:
    candidates = ("timestamp", "funding_time", "updated_at", "fill_time")
    for name in candidates:
        if name in frame.columns and frame.height:
            value = frame.get_column(name).min()
            return value.isoformat() if value else None
    return None


def _progress(message: str) -> None:
    print(f"[offline-sync] {message}", file=sys.stderr, flush=True)
