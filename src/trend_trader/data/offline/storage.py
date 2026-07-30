from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    """Bounded-memory daily compaction backed by an on-disk SQLite B-tree."""

    def __init__(
        self,
        *args: object,
        batch_rows: int = 25_000,
        sqlite_cache_mb: int = 64,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.batch_rows = batch_rows
        self.sqlite_cache_mb = sqlite_cache_mb

    def write_batches(
        self,
        frames: Iterable[pl.DataFrame],
    ) -> tuple[list[tuple[date, Path, int]], int]:
        staging = self.layout.root / ".staging"
        staging.mkdir(parents=True, exist_ok=True)
        database_path = staging / f"{self.dataset}-{uuid4().hex}.sqlite"
        initialized_dates: set[date] = set()
        input_rows = 0
        connection = sqlite3.connect(database_path)
        try:
            self._configure_sqlite(connection)
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
                    table = _date_table(target_date)
                    if target_date not in initialized_dates:
                        self._create_table(connection, table)
                        self._load_existing(connection, table, target_date)
                        initialized_dates.add(target_date)
                    partition = normalized.filter(
                        pl.col(self.timestamp_column).dt.date() == target_date
                    )
                    self._insert_frame(connection, table, partition)
                connection.commit()
            results = [
                self._write_table(connection, _date_table(target_date), target_date)
                for target_date in sorted(initialized_dates)
            ]
            return results, input_rows
        finally:
            connection.close()
            database_path.unlink(missing_ok=True)
            database_path.with_suffix(".sqlite-journal").unlink(missing_ok=True)

    def _configure_sqlite(self, connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(f"PRAGMA cache_size=-{self.sqlite_cache_mb * 1024}")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")

    def _create_table(self, connection: sqlite3.Connection, table: str) -> None:
        columns = ", ".join(
            f"{_quote(name)} {_sqlite_type(dtype)}" for name, dtype in self.schema.items()
        )
        primary_key = ", ".join(_quote(name) for name in self.primary_key)
        connection.execute(
            f"CREATE TABLE {_quote(table)} ({columns}, PRIMARY KEY ({primary_key})) "
            "WITHOUT ROWID"
        )

    def _load_existing(
        self,
        connection: sqlite3.Connection,
        table: str,
        target_date: date,
    ) -> None:
        path = self.layout.normalized_path(
            self.dataset,
            target_date,
            account_alias=self.account_alias,
        )
        if not path.exists():
            return
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=self.batch_rows):
            frame = canonicalize(pl.from_arrow(batch), self.schema)
            self._insert_frame(connection, table, frame)

    def _insert_frame(
        self,
        connection: sqlite3.Connection,
        table: str,
        frame: pl.DataFrame,
    ) -> None:
        placeholders = ", ".join("?" for _ in self.schema)
        columns = ", ".join(_quote(name) for name in self.schema)
        statement = (
            f"INSERT OR REPLACE INTO {_quote(table)} ({columns}) VALUES ({placeholders})"
        )
        dtypes = tuple(self.schema.values())
        rows = (
            tuple(_to_sqlite(value, dtype) for value, dtype in zip(row, dtypes, strict=True))
            for row in frame.iter_rows()
        )
        connection.executemany(statement, rows)

    def _write_table(
        self,
        connection: sqlite3.Connection,
        table: str,
        target_date: date,
    ) -> tuple[date, Path, int]:
        path = self.layout.normalized_path(
            self.dataset,
            target_date,
            account_alias=self.account_alias,
        )
        order_by = ", ".join(_quote(name) for name in self.sort_columns)
        columns = ", ".join(_quote(name) for name in self.schema)
        count = int(
            connection.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
        )
        metadata = {
            b"dataset_kind": b"offline",
            b"dataset": self.dataset.encode(),
            b"source_name": self.source_name.encode(),
            b"schema_version": b"1",
            b"written_at": datetime.now(UTC).isoformat().encode(),
            b"compaction": b"sqlite_streaming",
        }
        arrow_schema = (
            pl.DataFrame(schema=self.schema).to_arrow().schema.with_metadata(metadata)
        )
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
                cursor = connection.execute(
                    f"SELECT {columns} FROM {_quote(table)} ORDER BY {order_by}"
                )
                while rows := cursor.fetchmany(self.batch_rows):
                    arrays = []
                    for index, field in enumerate(arrow_schema):
                        dtype = self.schema[field.name]
                        values = [_from_sqlite(row[index], dtype) for row in rows]
                        arrays.append(pa.array(values, type=field.type))
                    writer.write_table(
                        pa.Table.from_arrays(arrays, schema=arrow_schema),
                        row_group_size=self.row_group_size,
                    )
                writer.close()
                writer = None
                pq.read_metadata(temporary)
                os.replace(temporary, path)
            finally:
                if writer is not None:
                    writer.close()
                temporary.unlink(missing_ok=True)
        return target_date, path, count


def _date_table(target_date: date) -> str:
    return f"rows_{target_date:%Y%m%d}"


def _quote(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _sqlite_type(dtype: pl.DataType) -> str:
    if isinstance(dtype, (pl.Datetime, pl.Duration)):
        return "INTEGER"
    if dtype == pl.Date:
        return "TEXT"
    if isinstance(dtype, pl.Decimal):
        return "TEXT"
    if dtype in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Boolean,
    }:
        return "INTEGER"
    if dtype in {pl.Float32, pl.Float64}:
        return "REAL"
    return "TEXT"


def _to_sqlite(value: object, dtype: pl.DataType) -> object:
    if value is None:
        return None
    if isinstance(dtype, pl.Datetime):
        if isinstance(value, datetime):
            return int(value.timestamp() * 1000)
        return int(value)
    if dtype == pl.Date:
        return value.isoformat() if isinstance(value, date) else str(value)
    if isinstance(dtype, pl.Decimal):
        return str(value)
    if dtype == pl.Boolean:
        return int(bool(value))
    return value


def _from_sqlite(value: object, dtype: pl.DataType) -> object:
    if value is None:
        return None
    if isinstance(dtype, pl.Datetime):
        return datetime.fromtimestamp(int(value) / 1000, tz=UTC)
    if dtype == pl.Date:
        return date.fromisoformat(str(value))
    if isinstance(dtype, pl.Decimal):
        return Decimal(str(value))
    if dtype == pl.Boolean:
        return bool(value)
    return value


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
