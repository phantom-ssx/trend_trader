from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import polars as pl

from trend_trader.data.offline.catalog import OfflineCatalog
from trend_trader.data.offline.client import (
    CANDLES_PATH,
    INDEX_CANDLES_PATH,
    MARK_CANDLES_PATH,
    OPEN_INTEREST_PATH,
    RATIO_PATHS,
    TAKER_VOLUME_PATH,
    OkxApiError,
    OkxOfflineClient,
)
from trend_trader.data.offline.config import DatasetOptions, OfflineSyncConfig
from trend_trader.data.offline.schemas import (
    DATASET_STORAGE,
    ArchiveDataQualityError,
    ArchiveParseStats,
    aggregate_oi_frame,
    candle_frame,
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
CANDLE_ARCHIVE_START = date(2023, 7, 1)
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
PERMANENT_IDENTIFIER_ERROR_CODES = {"51001"}


@dataclass(frozen=True)
class SyncTask:
    dataset: str
    target_date: date
    scope_key: str = "all"
    identifiers: tuple[str, ...] = ()


@dataclass
class DatasetResult:
    dataset: str
    target_date: str
    scope_key: str
    status: str
    rows: int = 0
    files: list[str] = field(default_factory=list)
    error: str | None = None
    identifiers: list[str] = field(default_factory=list)


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
        identifiers: set[str] | None = None,
    ) -> list[SyncTask]:
        if mode not in {"daily", "backfill", "range"}:
            raise ValueError(f"unsupported mode: {mode}")
        current = today or datetime.now(UTC).date()
        selected_identifiers = tuple(
            sorted(identifier.strip().upper() for identifier in identifiers or set())
        )
        if selected_identifiers and datasets != {"index_price_candles"}:
            raise ValueError("--identifier requires only --dataset index_price_candles")
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
                if selected_identifiers:
                    for day in candidates:
                        pending = tuple(
                            identifier
                            for identifier in selected_identifiers
                            if not self.catalog.is_identifier_resolved(
                                dataset,
                                identifier,
                                day,
                            )
                        )
                        if pending:
                            tasks.append(SyncTask(dataset, day, scope, pending))
                    continue
                missing = [
                    day for day in candidates if not self.catalog.is_resolved(dataset, day, scope)
                ]
                if mode == "daily" and dataset.startswith("private_"):
                    overlap_start = max(first, last - timedelta(days=2))
                    missing = sorted(set(missing) | set(_date_range(overlap_start, last)))
                tasks.extend(SyncTask(dataset, day, scope) for day in missing)
                if dataset == "index_price_candles":
                    missing_dates = set(missing)
                    supplemental_indices = self._supplemental_index_windows()
                    for day in candidates:
                        if day in missing_dates:
                            continue
                        pending = tuple(
                            identifier
                            for identifier, first_observed in supplemental_indices.items()
                            if day >= first_observed
                            and not self.catalog.is_identifier_resolved(
                                dataset, identifier, day
                            )
                        )
                        if pending:
                            tasks.append(SyncTask(dataset, day, scope, pending))
        return sorted(tasks, key=lambda item: (item.target_date, item.dataset, item.scope_key))

    async def run(
        self,
        *,
        mode: str = "daily",
        start: date | None = None,
        end: date | None = None,
        datasets: set[str] | None = None,
        identifiers: set[str] | None = None,
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
                async with self.client_factory() as client:
                    instruments = await client.fetch_instruments()
                    seen_at = datetime.now(UTC)
                    for instrument in instruments:
                        self.catalog.upsert_instrument(instrument, seen_at)
                    report["current_instruments"] = len(instruments)
                    if self._should_discover_indices(mode, datasets, identifiers):
                        report["index_discovery"] = await self._discover_historical_indices(
                            client
                        )
                    tasks = self.plan(
                        mode=mode,
                        start=start,
                        end=end,
                        datasets=datasets,
                        identifiers=identifiers,
                        today=today,
                    )
                    report["planned_tasks"] = len(tasks)
                    for task in tasks:
                        self._check_disk()
                        _progress(
                            f"start dataset={task.dataset} date={task.target_date} "
                            f"scope={task.scope_key}"
                            + (
                                f" identifiers={','.join(task.identifiers)}"
                                if task.identifiers
                                else ""
                            )
                        )
                        result = await self._execute(client, task)
                        report["results"].append(asdict(result))
                        error_text = f" error={result.error}" if result.error else ""
                        _progress(
                            f"finish dataset={task.dataset} date={task.target_date} "
                            f"scope={task.scope_key} status={result.status} rows={result.rows}"
                            f"{error_text}"
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
            if task.dataset == "candles" and task.target_date < CANDLE_ARCHIVE_START:
                files, frame = await self._candle_rest_dataset(client, task)
            elif task.dataset in HISTORICAL_MODULES:
                return await self._execute_historical_file_dataset(client, task)
            elif task.dataset in {"mark_price_candles", "index_price_candles"}:
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
                else ("complete_empty" if task.dataset.startswith("private_") else "unavailable")
            )
            artifact = str(outputs[0][1]) if outputs else None
            stored_rows = next(
                (
                    row_count
                    for output_date, _, row_count in outputs
                    if output_date == task.target_date
                ),
                frame.height,
            )
            coverage_identifiers = (
                task.identifiers
                if task.identifiers
                else (
                    tuple(self._supplemental_index_windows())
                    if task.dataset == "index_price_candles"
                    else ()
                )
            )
            for identifier in coverage_identifiers:
                identifier_rows = frame.filter(pl.col("index_id") == identifier).height
                self.catalog.mark_identifier_coverage(
                    task.dataset,
                    identifier,
                    task.target_date,
                    status="complete" if identifier_rows else "unavailable",
                    row_count=identifier_rows,
                    artifact_path=artifact,
                )
            if task.identifiers:
                existing = self.catalog.coverage_record(
                    task.dataset,
                    task.target_date,
                    task.scope_key,
                )
                if existing and (frame.height or str(existing["status"]) != "unavailable"):
                    self.catalog.mark_coverage(
                        task.dataset,
                        task.target_date,
                        status="complete" if frame.height else str(existing["status"]),
                        row_count=stored_rows if frame.height else int(existing["row_count"]),
                        scope_key=task.scope_key,
                        artifact_path=artifact or str(existing.get("artifact_path") or "") or None,
                    )
            else:
                self.catalog.mark_coverage(
                    task.dataset,
                    task.target_date,
                    status=status,
                    row_count=stored_rows,
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
                identifiers=list(task.identifiers),
            )
        except Exception as exc:
            if task.identifiers:
                for identifier in task.identifiers:
                    self.catalog.mark_identifier_coverage(
                        task.dataset,
                        identifier,
                        task.target_date,
                        status="failed",
                        row_count=0,
                    )
            else:
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
                identifiers=list(task.identifiers),
            )

    async def _execute_historical_file_dataset(
        self,
        client: OkxOfflineClient,
        task: SyncTask,
    ) -> DatasetResult:
        raw_files = await self._download_historical_files(client, task)
        rest_boundary = self._rest_candle_boundary(task)
        first_event_seen: str | None = None
        parsed_rows = 0
        next_progress_rows = 250_000
        parse_stats: dict[Path, ArchiveParseStats] = {}
        instrument_windows = self._instrument_windows()

        def batches() -> Iterable[pl.DataFrame]:
            nonlocal first_event_seen, next_progress_rows, parsed_rows
            iterator = (
                iter_candle_archive_batches
                if task.dataset == "candles"
                else iter_funding_archive_batches
            )
            for raw_file in raw_files:
                file_stats = ArchiveParseStats()
                parse_stats[raw_file.path] = file_stats
                try:
                    for frame in iterator(
                        raw_file.path,
                        batch_size=self.config.stream_batch_rows,
                        source_date=task.target_date,
                        stats=file_stats,
                    ):
                        self._validate_known_instrument_windows(
                            frame,
                            instrument_windows,
                            raw_file.path,
                        )
                        if rest_boundary is not None:
                            self._validate_candle_overlap(frame, rest_boundary, raw_file.path)
                        self._remember_instruments(frame, task.target_date)
                        first_event = _first_timestamp_text(frame)
                        if first_event and (
                            first_event_seen is None or first_event < first_event_seen
                        ):
                            first_event_seen = first_event
                        parsed_rows += frame.height
                        if parsed_rows >= next_progress_rows:
                            _progress(
                                f"parse dataset={task.dataset} date={task.target_date} "
                                f"rows={parsed_rows}"
                            )
                            next_progress_rows = (parsed_rows // 250_000 + 1) * 250_000
                        yield frame
                    _progress(
                        f"quality dataset={task.dataset} date={task.target_date} "
                        f"file={raw_file.path.name} input={file_stats.input_rows} "
                        f"emitted={file_stats.emitted_rows} "
                        f"adjacent_duplicates={file_stats.adjacent_duplicate_rows} "
                        f"duplicate_ratio={file_stats.duplicate_ratio:.2%}"
                    )
                except Exception:
                    quarantine = self.layout.quarantine_dir(task.dataset, task.target_date)
                    quarantine.mkdir(parents=True, exist_ok=True)
                    destination = quarantine / (
                        f"{raw_file.path.stem}.bad-{uuid4().hex[:8]}{raw_file.path.suffix}"
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
        if task.dataset == "candles" and task.target_date == CANDLE_ARCHIVE_START:
            self._validate_candle_boundary_continuity()
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
                    "input_rows": parse_stats[raw_file.path].input_rows,
                    "emitted_rows": parse_stats[raw_file.path].emitted_rows,
                    "adjacent_duplicate_rows": (parse_stats[raw_file.path].adjacent_duplicate_rows),
                    "duplicate_ratio": parse_stats[raw_file.path].duplicate_ratio,
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
                continuous_start=self.catalog.earliest_complete_date(task.dataset, task.scope_key),
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
                url = _link_value(
                    link,
                    "url",
                    "downloadUrl",
                    "downloadLink",
                    "fileHref",
                    "href",
                )
                if not url:
                    continue
                fallback_filename = (
                    Path(urlparse(url).path).name
                    or f"{instrument_type.lower()}-{task.dataset}-{number}.zip"
                )
                filename = Path(
                    _link_value(link, "fileName", "filename", "name") or fallback_filename
                ).name
                destination = self.layout.raw_dir(task.dataset, task.target_date) / filename
                _progress(
                    f"download dataset={task.dataset} date={task.target_date} file={filename}"
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
            raise FileNotFoundError(f"OKX has not published {task.dataset} for {task.target_date}")
        return raw_files

    async def _price_dataset(
        self,
        client: OkxOfflineClient,
        task: SyncTask,
    ) -> tuple[list[Path], pl.DataFrame]:
        start_ms, end_ms = _day_milliseconds(task.target_date)
        catalog = self.catalog.list_instruments()
        is_index = task.dataset == "index_price_candles"
        identifier_field = "index_id" if is_index else "instrument_id"
        rows_by_identifier: dict[str, list[Mapping[str, object]]] = {}
        for row in catalog:
            identifier = str(row.get(identifier_field) or "")
            instrument_type = str(row.get("instrument_type") or "").upper()
            if (
                identifier
                and instrument_type in {"SWAP", "FUTURES"}
                and _instrument_active_on_date(row, task.target_date)
            ):
                rows_by_identifier.setdefault(identifier, []).append(row)
        if is_index:
            for identifier, first_observed in self._supplemental_index_windows().items():
                if task.target_date >= first_observed:
                    rows_by_identifier.setdefault(
                        identifier,
                        [_historical_index_source(identifier, first_observed)],
                    )
        if task.identifiers:
            rows_by_identifier = {
                identifier: rows_by_identifier.get(
                    identifier,
                    [_historical_index_source(identifier, None)],
                )
                for identifier in task.identifiers
            }
        raw: dict[str, object] = {}
        fragments: list[pl.DataFrame] = []
        skipped_invalid: dict[str, object] = {}
        new_invalid: dict[str, object] = {}
        endpoint = INDEX_CANDLES_PATH if is_index else MARK_CANDLES_PATH
        for identifier, source_rows in sorted(rows_by_identifier.items()):
            fingerprint = _instrument_source_fingerprint(source_rows)
            cached = self.catalog.cached_invalid_identifier(
                task.dataset,
                identifier,
                fingerprint,
            )
            if cached:
                self.catalog.note_invalid_identifier_skipped(
                    task.dataset,
                    identifier,
                    task.target_date,
                )
                skipped_invalid[identifier] = {
                    "error_code": cached["error_code"],
                    "error_message": cached["error_message"],
                    "first_failed_at": cached["first_failed_at"],
                }
                continue
            try:
                rows = await client.fetch_price_candles(
                    instrument_id=identifier,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    index=is_index,
                )
            except OkxApiError as exc:
                if exc.code not in PERMANENT_IDENTIFIER_ERROR_CODES:
                    raise
                error_message = str(exc)
                self.catalog.record_invalid_identifier(
                    task.dataset,
                    identifier,
                    source_fingerprint=fingerprint,
                    endpoint=endpoint,
                    error_code=str(exc.code),
                    error_message=error_message,
                    target_date=task.target_date,
                )
                new_invalid[identifier] = {
                    "error_code": exc.code,
                    "error_message": error_message,
                }
                _progress(
                    f"exclude-invalid dataset={task.dataset} identifier={identifier} "
                    f"code={exc.code}"
                )
                continue
            self.catalog.clear_invalid_identifier(task.dataset, identifier)
            raw[identifier] = rows
            self._observe_api_rows(task.dataset, identifier, rows)
            fragments.append(price_candle_frame(rows, instrument_id=identifier, is_index=is_index))
        raw["_request_metadata"] = {
            "eligible_identifiers": len(rows_by_identifier),
            "skipped_cached_invalid": skipped_invalid,
            "new_invalid": new_invalid,
        }
        if skipped_invalid:
            _progress(
                f"skip-cached-invalid dataset={task.dataset} "
                f"count={len(skipped_invalid)}"
            )
        suffix = ""
        if task.identifiers:
            subset_digest = hashlib.sha256(
                "\n".join(task.identifiers).encode()
            ).hexdigest()[:12]
            suffix = f"-subset-{subset_digest}"
        raw_path = self.raw.write_json_gz(
            task.dataset,
            task.target_date,
            raw,
            suffix=suffix,
        )
        self._record_rest_raw(raw_path, task.dataset)
        return [raw_path], _concat(fragments, DATASET_STORAGE[task.dataset][0])

    async def _candle_rest_dataset(
        self,
        client: OkxOfflineClient,
        task: SyncTask,
    ) -> tuple[list[Path], pl.DataFrame]:
        start_ms, end_ms = _day_milliseconds(task.target_date)
        rows_by_identifier: dict[str, list[Mapping[str, object]]] = {}
        for row in self.catalog.list_instruments():
            identifier = str(row.get("instrument_id") or "")
            instrument_type = str(row.get("instrument_type") or "").upper()
            if (
                identifier
                and instrument_type in {"SWAP", "FUTURES"}
                and _instrument_active_on_date(row, task.target_date)
            ):
                rows_by_identifier.setdefault(identifier, []).append(row)
        raw: dict[str, object] = {}
        fragments: list[pl.DataFrame] = []
        skipped_invalid: dict[str, object] = {}
        new_invalid: dict[str, object] = {}
        for identifier, source_rows in sorted(rows_by_identifier.items()):
            fingerprint = _instrument_source_fingerprint(source_rows)
            cached = self.catalog.cached_invalid_identifier(
                task.dataset,
                identifier,
                fingerprint,
            )
            if cached:
                self.catalog.note_invalid_identifier_skipped(
                    task.dataset,
                    identifier,
                    task.target_date,
                )
                skipped_invalid[identifier] = {
                    "error_code": cached["error_code"],
                    "error_message": cached["error_message"],
                    "first_failed_at": cached["first_failed_at"],
                }
                continue
            try:
                rows = await client.fetch_candles(
                    instrument_id=identifier,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            except OkxApiError as exc:
                if exc.code not in PERMANENT_IDENTIFIER_ERROR_CODES:
                    raise
                error_message = str(exc)
                self.catalog.record_invalid_identifier(
                    task.dataset,
                    identifier,
                    source_fingerprint=fingerprint,
                    endpoint=CANDLES_PATH,
                    error_code=str(exc.code),
                    error_message=error_message,
                    target_date=task.target_date,
                )
                new_invalid[identifier] = {
                    "error_code": exc.code,
                    "error_message": error_message,
                }
                _progress(
                    f"exclude-invalid dataset={task.dataset} identifier={identifier} "
                    f"code={exc.code}"
                )
                continue
            self.catalog.clear_invalid_identifier(task.dataset, identifier)
            raw[identifier] = rows
            self._observe_api_rows(task.dataset, identifier, rows)
            fragments.append(candle_frame(rows, instrument_id=identifier))
        raw["_request_metadata"] = {
            "eligible_identifiers": len(rows_by_identifier),
            "skipped_cached_invalid": skipped_invalid,
            "new_invalid": new_invalid,
        }
        if skipped_invalid:
            _progress(
                f"skip-cached-invalid dataset={task.dataset} "
                f"count={len(skipped_invalid)}"
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
            fragments.append(aggregate_oi_frame(rows, base_currency=currency, period=period))
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
            fragments.append(taker_volume_frame(rows, instrument_id=identifier, period=period))
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
            account for account in self.config.private_accounts if account.alias == task.scope_key
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
                "okx_public_rest"
                if task.dataset == "candles" and task.target_date < CANDLE_ARCHIVE_START
                else (
                    "okx_historical_file"
                    if task.dataset in HISTORICAL_MODULES
                    else (
                        "okx_private_rest"
                        if task.dataset.startswith("private_")
                        else "okx_public_rest"
                    )
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

    def _supplemental_index_windows(self) -> dict[str, date]:
        configured_start = self.config.datasets.index_price_candles.start
        fallback_start = configured_start or date.min
        result = {
            identifier: fallback_start for identifier in self.config.historical_index_ids
        }
        for row in self.catalog.historical_indices():
            identifier = str(row["identifier"])
            first_observed = date.fromisoformat(str(row["first_observed_date"]))
            result[identifier] = min(result.get(identifier, first_observed), first_observed)
        return dict(sorted(result.items()))

    def _should_discover_indices(
        self,
        mode: str,
        datasets: set[str] | None,
        identifiers: set[str] | None,
    ) -> bool:
        return bool(
            mode == "backfill"
            and self.config.discover_historical_indices
            and self.config.datasets.index_price_candles.enabled
            and (datasets is None or "index_price_candles" in datasets)
            and not identifiers
            and self.config.datasets.index_price_candles.start is not None
        )

    async def _discover_historical_indices(
        self,
        client: OkxOfflineClient,
    ) -> dict[str, object]:
        target_date = self.config.datasets.index_price_candles.start
        if target_date is None:
            return {"status": "disabled_without_start"}
        discovery_key = f"index-universe-at:{target_date.isoformat()}"
        if self.catalog.discovery_completed(discovery_key):
            return {"status": "already_complete", "target_date": target_date.isoformat()}

        instrument_rows = self.catalog.list_instruments()
        known = {
            str(row.get("index_id") or "")
            for row in instrument_rows
            if str(row.get("instrument_type") or "").upper() in {"SWAP", "FUTURES"}
            and row.get("index_id")
            and _instrument_active_on_date(row, target_date)
        }
        known.update(self._supplemental_index_windows())
        candidates = await client.fetch_index_ids()
        to_probe = [identifier for identifier in candidates if identifier not in known]
        discovered: list[str] = []
        for number, identifier in enumerate(to_probe, start=1):
            if await client.index_has_candles_on_date(identifier, target_date):
                self.catalog.upsert_historical_index(
                    identifier,
                    target_date,
                    discovery_method="index_tickers_history_probe",
                )
                discovered.append(identifier)
            if number % 100 == 0:
                _progress(
                    f"discover-index-universe date={target_date} "
                    f"checked={number}/{len(to_probe)} found={len(discovered)}"
                )
        metadata = {
            "target_date": target_date.isoformat(),
            "ticker_candidates": len(candidates),
            "known_before_probe": len(known),
            "probed": len(to_probe),
            "discovered": discovered,
        }
        self.catalog.mark_discovery_completed(discovery_key, metadata)
        _progress(
            f"discover-index-universe date={target_date} status=complete "
            f"probed={len(to_probe)} found={len(discovered)}"
        )
        return {"status": "complete", **metadata}

    def _rest_candle_boundary(self, task: SyncTask) -> pl.DataFrame | None:
        if task.dataset != "candles" or task.target_date != CANDLE_ARCHIVE_START:
            return None
        path = self.layout.normalized_path("candles", CANDLE_ARCHIVE_START - timedelta(days=1))
        if not path.exists():
            return None
        return pl.read_parquet(path)

    @staticmethod
    def _validate_candle_overlap(
        archive: pl.DataFrame,
        rest: pl.DataFrame,
        source_path: Path,
    ) -> None:
        keys = ["venue", "instrument_id", "bar_type", "timestamp"]
        values = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_ccy",
            "volume_quote",
            "confirm",
        ]
        overlap = archive.join(rest, on=keys, how="inner", suffix="_rest")
        if overlap.is_empty():
            return
        matches = pl.all_horizontal(
            [pl.col(name).eq_missing(pl.col(f"{name}_rest")) for name in values]
        )
        conflicts = overlap.filter(~matches)
        if conflicts.height:
            row = conflicts.select(*keys).row(0, named=True)
            raise ArchiveDataQualityError(
                f"{source_path.name}: REST/archive candle conflict at source boundary "
                f"instrument={row['instrument_id']} timestamp={row['timestamp'].isoformat()}"
            )

    def _validate_candle_boundary_continuity(self) -> None:
        previous_path = self.layout.normalized_path(
            "candles", CANDLE_ARCHIVE_START - timedelta(days=1)
        )
        current_path = self.layout.normalized_path("candles", CANDLE_ARCHIVE_START)
        if not previous_path.exists() or not current_path.exists():
            return
        previous = (
            pl.scan_parquet(previous_path)
            .group_by("instrument_id")
            .agg(pl.col("timestamp").max().alias("previous_timestamp"))
            .collect()
        )
        current = (
            pl.scan_parquet(current_path)
            .group_by("instrument_id")
            .agg(pl.col("timestamp").min().alias("current_timestamp"))
            .collect()
        )
        common = previous.join(current, on="instrument_id", how="inner")
        gaps = common.filter(
            pl.col("current_timestamp") - pl.col("previous_timestamp")
            != timedelta(minutes=1)
        )
        if gaps.height:
            row = gaps.row(0, named=True)
            raise ArchiveDataQualityError(
                "REST/archive candle discontinuity at source boundary "
                f"instrument={row['instrument_id']} "
                f"previous={row['previous_timestamp'].isoformat()} "
                f"current={row['current_timestamp'].isoformat()}"
            )

    def _instrument_windows(
        self,
    ) -> dict[str, tuple[datetime | None, datetime | None]]:
        result: dict[str, tuple[datetime | None, datetime | None]] = {}
        for row in self.catalog.list_instruments():
            instrument_id = str(row.get("instrument_id") or "")
            if not instrument_id:
                continue
            result[instrument_id] = (
                _optional_datetime(row.get("listing_time")),
                _optional_datetime(row.get("expiration_time")),
            )
        return result

    @staticmethod
    def _validate_known_instrument_windows(
        frame: pl.DataFrame,
        windows: Mapping[str, tuple[datetime | None, datetime | None]],
        path: Path,
    ) -> None:
        if frame.is_empty() or "instrument_id" not in frame.columns:
            return
        timestamp_column = "timestamp" if "timestamp" in frame.columns else "funding_time"
        bounds = (
            frame.group_by("instrument_id")
            .agg(
                pl.col(timestamp_column).min().alias("first_timestamp"),
                pl.col(timestamp_column).max().alias("last_timestamp"),
            )
            .iter_rows(named=True)
        )
        for row in bounds:
            instrument_id = str(row["instrument_id"])
            listing_time, expiration_time = windows.get(
                instrument_id,
                (None, None),
            )
            first_timestamp = row["first_timestamp"]
            last_timestamp = row["last_timestamp"]
            if listing_time is not None and first_timestamp < listing_time:
                raise ArchiveDataQualityError(
                    f"{path.name}: data predates instrument listing "
                    f"instrument={instrument_id} data={first_timestamp.isoformat()} "
                    f"listing={listing_time.isoformat()}"
                )
            if expiration_time is not None and last_timestamp > expiration_time:
                raise ArchiveDataQualityError(
                    f"{path.name}: data is after instrument expiration "
                    f"instrument={instrument_id} data={last_timestamp.isoformat()} "
                    f"expiration={expiration_time.isoformat()}"
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
        timestamps = [int(str(row[0])) for row in rows if row and str(row[0]).lstrip("-").isdigit()]
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


def _optional_datetime(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _instrument_active_on_date(row: Mapping[str, object], target_date: date) -> bool:
    start = datetime.combine(target_date, time.min, tzinfo=UTC)
    end = start + timedelta(days=1)
    listing_time = _optional_datetime(row.get("listing_time"))
    expiration_time = _optional_datetime(row.get("expiration_time"))
    return (listing_time is None or listing_time < end) and (
        expiration_time is None or expiration_time > start
    )


def _historical_index_source(
    identifier: str,
    listing_date: date | None,
) -> Mapping[str, object]:
    return {
        "instrument_id": f"historical-index:{identifier}",
        "instrument_type": "INDEX",
        "index_id": identifier,
        "listing_time": (
            datetime.combine(listing_date, time.min, tzinfo=UTC).isoformat()
            if listing_date
            else ""
        ),
        "expiration_time": "",
        "category": "crypto",
        "rule_type": "historical_contract_index",
    }


def _instrument_source_fingerprint(rows: Iterable[Mapping[str, object]]) -> str:
    fields = (
        "instrument_id",
        "instrument_type",
        "index_id",
        "listing_time",
        "expiration_time",
        "category",
        "rule_type",
    )
    payload = sorted(
        ({field: str(row.get(field) or "") for field in fields} for row in rows),
        key=lambda item: tuple(item[field] for field in fields),
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _progress(message: str) -> None:
    print(f"[offline-sync] {message}", file=sys.stderr, flush=True)
