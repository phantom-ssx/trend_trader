from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(frame: pl.DataFrame, schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    result = frame
    for name, dtype in schema.items():
        if name not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=dtype).alias(name))
    expressions: list[pl.Expr] = []
    for name, dtype in schema.items():
        expression = pl.col(name)
        current = result.schema[name]
        if isinstance(dtype, pl.Datetime):
            if current in {pl.Int64, pl.UInt64}:
                expression = pl.from_epoch(name, time_unit="ms").dt.replace_time_zone("UTC")
            elif isinstance(current, pl.Datetime) and current.time_zone is None:
                expression = expression.dt.replace_time_zone("UTC")
            else:
                expression = expression.cast(dtype, strict=False)
        else:
            expression = expression.cast(dtype, strict=False)
        expressions.append(expression.alias(name))
    return result.with_columns(expressions).select(*schema)


class OfflineLayout:
    def __init__(self, data_root: Path | str) -> None:
        self.data_root = Path(data_root)
        self.root = self.data_root / "offline"

    def raw_dir(self, dataset: str, source_date: date, account_alias: str | None = None) -> Path:
        base = self.root / "raw" / "okx"
        if account_alias:
            base = base / "private" / account_alias
        return base / dataset / f"source_date={source_date.isoformat()}"

    def normalized_path(
        self,
        dataset: str,
        target_date: date,
        *,
        account_alias: str | None = None,
    ) -> Path:
        base = self.root / "normalized"
        if account_alias:
            base = base / "private" / dataset / "venue=OKX" / f"account={account_alias}"
        else:
            base = base / dataset / "venue=OKX"
        return (
            base
            / f"year={target_date:%Y}"
            / f"date={target_date.isoformat()}"
            / f"{dataset}-{target_date.isoformat()}.parquet"
        )

    def run_dir(self) -> Path:
        return self.root / "manifests" / "runs"

    def quarantine_dir(self, dataset: str, source_date: date) -> Path:
        return self.root / "quarantine" / dataset / f"source_date={source_date.isoformat()}"


class RawRepository:
    def __init__(self, layout: OfflineLayout) -> None:
        self.layout = layout

    def write_json_gz(
        self,
        dataset: str,
        source_date: date,
        payload: object,
        *,
        account_alias: str | None = None,
        suffix: str = "",
    ) -> Path:
        directory = self.layout.raw_dir(dataset, source_date, account_alias)
        directory.mkdir(parents=True, exist_ok=True)
        if account_alias:
            directory.chmod(0o700)
        name = f"{dataset}-{source_date.isoformat()}{suffix}.json.gz"
        path = directory / name
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.part")
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode()
            with temporary.open("wb") as file:
                with gzip.GzipFile(fileobj=file, mode="wb", filename="", mtime=0) as compressed:
                    compressed.write(encoded)
            digest = sha256_file(temporary)
            final_path = path
            if path.exists():
                if sha256_file(path) == digest:
                    return path
                final_path = path.with_name(f"{path.stem}.rev-{digest[:12]}{path.suffix}")
            if not final_path.exists():
                os.replace(temporary, final_path)
            return final_path
        finally:
            temporary.unlink(missing_ok=True)


class DailyParquetRepository:
    """One atomic all-market Parquet per UTC date and dataset."""

    def __init__(
        self,
        layout: OfflineLayout,
        *,
        dataset: str,
        schema: Mapping[str, pl.DataType],
        primary_key: Sequence[str],
        timestamp_column: str,
        sort_columns: Sequence[str],
        compression: str = "zstd",
        row_group_size: int = 1_000_000,
        account_alias: str | None = None,
        source_name: str = "okx_offline_sync",
    ) -> None:
        self.layout = layout
        self.dataset = dataset
        self.schema = dict(schema)
        self.primary_key = list(primary_key)
        self.timestamp_column = timestamp_column
        self.sort_columns = list(sort_columns)
        self.compression = compression
        self.row_group_size = row_group_size
        self.account_alias = account_alias
        self.source_name = source_name

    @contextmanager
    def _lock(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.account_alias:
            path.parent.chmod(0o700)
        lock_path = path.with_suffix(".lock")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def write(self, frame: pl.DataFrame) -> list[tuple[date, Path, int]]:
        normalized = canonicalize(frame, self.schema)
        if normalized.is_empty():
            return []
        dates = normalized.get_column(self.timestamp_column).dt.date().unique().sort().to_list()
        results: list[tuple[date, Path, int]] = []
        for target_date in dates:
            partition = normalized.filter(
                pl.col(self.timestamp_column).dt.date() == target_date
            )
            path = self.layout.normalized_path(
                self.dataset,
                target_date,
                account_alias=self.account_alias,
            )
            with self._lock(path):
                frames = [partition]
                if path.exists():
                    frames.insert(0, canonicalize(pl.read_parquet(path), self.schema))
                merged = (
                    pl.concat(frames, how="vertical_relaxed")
                    .unique(subset=self.primary_key, keep="last")
                    .sort(self.sort_columns)
                )
                temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
                metadata = {
                    b"dataset_kind": b"offline",
                    b"dataset": self.dataset.encode(),
                    b"source_name": self.source_name.encode(),
                    b"schema_version": b"1",
                    b"written_at": datetime.now(UTC).isoformat().encode(),
                }
                try:
                    table = merged.to_arrow().replace_schema_metadata(metadata)
                    pq.write_table(
                        table,
                        temporary,
                        compression=self.compression,
                        row_group_size=self.row_group_size,
                        write_statistics=True,
                    )
                    pq.read_metadata(temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            results.append((target_date, path, merged.height))
        return results


class StreamingDailyParquetRepository(DailyParquetRepository):
    """Fast bounded-memory compaction using Parquet fragments and DuckDB spilling."""

    def __init__(
        self,
        *args: object,
        batch_rows: int = 25_000,
        compaction_memory_mb: int = 512,
        compaction_threads: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.batch_rows = batch_rows
        self.compaction_memory_mb = compaction_memory_mb
        self.compaction_threads = compaction_threads

    def write_batches(
        self,
        frames: Iterable[pl.DataFrame],
    ) -> tuple[list[tuple[date, Path, int]], int]:
        staging_root = self.layout.root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = staging_root / f"{self.dataset}-{uuid4().hex}"
        staging.mkdir()
        fragments: dict[date, list[Path]] = {}
        input_rows = 0
        source_order = 0
        try:
            for frame in frames:
                normalized = canonicalize(frame, self.schema)
                if normalized.is_empty():
                    continue
                input_rows += normalized.height
                target_dates = (
                    normalized.get_column(self.timestamp_column)
                    .dt.date()
                    .unique()
                    .sort()
                    .to_list()
                )
                for target_date in target_dates:
                    partition = normalized.filter(
                        pl.col(self.timestamp_column).dt.date() == target_date
                    ).unique(
                        subset=self.primary_key,
                        keep="last",
                    )
                    source_order += 1
                    partition = partition.with_columns(
                        pl.lit(source_order, dtype=pl.Int64).alias("__source_order")
                    )
                    fragment = staging / (
                        f"{target_date:%Y%m%d}-{source_order:08d}.parquet"
                    )
                    partition.write_parquet(
                        fragment,
                        compression=self.compression,
                        row_group_size=self.row_group_size,
                        statistics=True,
                    )
                    fragments.setdefault(target_date, []).append(fragment)
            connection = duckdb.connect()
            try:
                self._configure_duckdb(connection, staging)
                results = [
                    self._compact_date(connection, target_date, paths)
                    for target_date, paths in sorted(fragments.items())
                ]
            finally:
                connection.close()
            return results, input_rows
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _configure_duckdb(
        self,
        connection: duckdb.DuckDBPyConnection,
        staging: Path,
    ) -> None:
        temporary = staging / "duckdb-tmp"
        temporary.mkdir()
        connection.execute(
            f"SET memory_limit = {_sql_literal(f'{self.compaction_memory_mb}MB')}"
        )
        connection.execute(f"SET threads = {self.compaction_threads}")
        connection.execute(f"SET temp_directory = {_sql_literal(str(temporary))}")
        connection.execute("SET preserve_insertion_order = false")

    def _compact_date(
        self,
        connection: duckdb.DuckDBPyConnection,
        target_date: date,
        fragments: list[Path],
    ) -> tuple[date, Path, int]:
        path = self.layout.normalized_path(
            self.dataset,
            target_date,
            account_alias=self.account_alias,
        )
        columns = ", ".join(_quote(name) for name in self.schema)
        primary_key = ", ".join(_quote(name) for name in self.primary_key)
        order_by = ", ".join(_quote(name) for name in self.sort_columns)
        fragment_paths = ", ".join(_sql_literal(str(item)) for item in fragments)
        sources = [
            (
                f"SELECT {columns}, \"__source_order\" "
                f"FROM read_parquet([{fragment_paths}], union_by_name=true)"
            )
        ]
        if path.exists():
            sources.insert(
                0,
                (
                    f"SELECT {columns}, 0::BIGINT AS \"__source_order\" "
                    f"FROM read_parquet({_sql_literal(str(path))})"
                ),
            )
        union = " UNION ALL ".join(sources)
        query = f"""
            SELECT {columns}
            FROM (
                SELECT {columns},
                       ROW_NUMBER() OVER (
                           PARTITION BY {primary_key}
                           ORDER BY "__source_order" DESC
                       ) AS "__row_number"
                FROM ({union}) AS source
            ) AS ranked
            WHERE "__row_number" = 1
            ORDER BY {order_by}
        """
        metadata = {
            b"dataset_kind": b"offline",
            b"dataset": self.dataset.encode(),
            b"source_name": self.source_name.encode(),
            b"schema_version": b"1",
            b"written_at": datetime.now(UTC).isoformat().encode(),
            b"compaction": b"duckdb_external_sort",
        }
        arrow_schema = (
            pl.DataFrame(schema=self.schema).to_arrow().schema.with_metadata(metadata)
        )
        output_batch_rows = min(
            max(self.batch_rows, 25_000),
            min(self.row_group_size, 100_000),
        )
        count = 0
        with self._lock(path):
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            writer: pq.ParquetWriter | None = None
            try:
                writer = pq.ParquetWriter(
                    temporary,
                    arrow_schema,
                    compression=self.compression,
                    write_statistics=True,
                )
                reader = connection.execute(query).to_arrow_reader(
                    batch_size=output_batch_rows
                )
                for batch in reader:
                    table = pa.Table.from_batches([batch]).cast(
                        arrow_schema.remove_metadata()
                    )
                    table = table.replace_schema_metadata(metadata)
                    writer.write_table(
                        table,
                        row_group_size=self.row_group_size,
                    )
                    count += table.num_rows
                writer.close()
                writer = None
                pq.read_metadata(temporary)
                os.replace(temporary, path)
            finally:
                if writer is not None:
                    writer.close()
                temporary.unlink(missing_ok=True)
        return target_date, path, count


def _quote(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _sql_literal(value: str) -> str:
    return f"'{value.replace(chr(39), chr(39) * 2)}'"


def write_run_report(layout: OfflineLayout, run_id: str, report: Mapping[str, Any]) -> Path:
    directory = layout.run_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.json"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
