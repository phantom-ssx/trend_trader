from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
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
