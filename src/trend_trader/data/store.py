"""Partitioned Parquet storage for canonical market data."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import polars as pl

from trend_trader.data.catalog import DataCatalog
from trend_trader.data.models import STORED_BAR_TYPES, DataQuery, DataType, as_utc, bar_minutes
from trend_trader.data.schema import PRIMARY_KEYS, canonicalize_frame, empty_frame


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return datetime(value.year + 1, 1, 1, tzinfo=UTC)
    return datetime(value.year, value.month + 1, 1, tzinfo=UTC)


def _next_year(value: datetime) -> datetime:
    return datetime(value.year + 1, 1, 1, tzinfo=UTC)


def iter_partitions(
    data_type: DataType,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, datetime]]:
    if data_type in {
        DataType.CANDLES,
        DataType.CONTRACT_BASIS,
        DataType.OPEN_INTEREST,
        DataType.LONG_SHORT_RATIO,
        DataType.LIQUIDATIONS,
        DataType.TAKER_VOLUME,
    }:
        cursor = datetime(start.year, start.month, 1, tzinfo=UTC)
        advance = _next_month
    else:
        cursor = datetime(start.year, 1, 1, tzinfo=UTC)
        advance = _next_year

    partitions: list[tuple[datetime, datetime]] = []
    while cursor < end:
        partition_end = advance(cursor)
        partitions.append((cursor, partition_end))
        cursor = partition_end
    return partitions


class ParquetStore:
    """Manage canonical files; callers never need to know concrete paths."""

    def __init__(self, root: Path, catalog: DataCatalog) -> None:
        self.root = root
        self.catalog = catalog

    def partition_path(
        self,
        *,
        data_type: DataType,
        venue: str,
        instrument_id: str,
        bar_type: str | None,
        partition_start: datetime,
    ) -> Path:
        base = (
            self.root
            / data_type.value
            / f"venue={venue}"
            / f"instrument_id={instrument_id}"
        )
        if data_type in STORED_BAR_TYPES:
            base /= f"bar_type={bar_type or '1m'}"
            return (
                base
                / f"year={partition_start.year:04d}"
                / f"month={partition_start.month:02d}"
                / "data.parquet"
            )
        return base / f"year={partition_start.year:04d}" / "data.parquet"

    def read(self, query: DataQuery, *, stored_bar_type: str | None) -> pl.DataFrame:
        paths = [
            self.partition_path(
                data_type=query.data_type,
                venue=query.venue,
                instrument_id=query.instrument_id,
                bar_type=stored_bar_type,
                partition_start=partition_start,
            )
            for partition_start, _ in iter_partitions(query.data_type, query.start, query.end)
        ]
        existing = [path for path in paths if path.exists()]
        if not existing:
            return empty_frame(query.data_type)

        frame = canonicalize_frame(pl.read_parquet(existing), query.data_type)
        return (
            frame.filter(
                (pl.col("timestamp") >= query.start)
                & (pl.col("timestamp") < query.end)
                & (pl.col("venue") == query.venue)
                & (pl.col("instrument_id") == query.instrument_id)
            )
            .unique(subset=PRIMARY_KEYS[query.data_type], keep="last")
            .sort("timestamp")
        )

    def missing_partition_intervals(
        self,
        query: DataQuery,
        *,
        stored_bar_type: str | None,
    ) -> list[tuple[datetime, datetime]]:
        missing: list[tuple[datetime, datetime]] = []
        for partition_start, partition_end in iter_partitions(
            query.data_type,
            query.start,
            query.end,
        ):
            path = self.partition_path(
                data_type=query.data_type,
                venue=query.venue,
                instrument_id=query.instrument_id,
                bar_type=stored_bar_type,
                partition_start=partition_start,
            )
            if not path.exists():
                missing.append((max(query.start, partition_start), min(query.end, partition_end)))
        return missing

    def rebuild_catalog(self) -> int:
        """Re-index canonical Parquet files after a missing or replaced catalog."""

        indexed = 0
        for data_type in DataType:
            dataset_root = self.root / data_type.value
            if not dataset_root.exists():
                continue
            for path in dataset_root.rglob("data.parquet"):
                frame = canonicalize_frame(pl.read_parquet(path), data_type)
                if frame.is_empty():
                    continue
                venue = str(frame["venue"][0])
                instrument_id = str(frame["instrument_id"][0])
                bar_type = str(frame["bar_type"][0]) if data_type is DataType.CANDLES else None
                min_timestamp = frame["timestamp"].min()
                max_timestamp = frame["timestamp"].max()
                if not isinstance(min_timestamp, datetime) or not isinstance(
                    max_timestamp,
                    datetime,
                ):
                    continue
                partition_start, partition_end = iter_partitions(
                    data_type,
                    min_timestamp,
                    max_timestamp + timedelta(microseconds=1),
                )[0]
                self.catalog.record_file(
                    path=path,
                    data_type=data_type,
                    venue=venue,
                    instrument_id=instrument_id,
                    bar_type=bar_type,
                    partition_start=partition_start,
                    partition_end=partition_end,
                    min_timestamp=min_timestamp,
                    max_timestamp=max_timestamp,
                    row_count=frame.height,
                    source_name="catalog-rebuild",
                )
                self._rebuild_coverage(
                    frame,
                    data_type=data_type,
                    venue=venue,
                    instrument_id=instrument_id,
                    bar_type=bar_type,
                    source_name="catalog-rebuild",
                )
                indexed += 1
        return indexed

    def _rebuild_coverage(
        self,
        frame: pl.DataFrame,
        *,
        data_type: DataType,
        venue: str,
        instrument_id: str,
        bar_type: str | None,
        source_name: str,
    ) -> None:
        timestamps = frame.get_column("timestamp").unique().sort().to_list()
        if not timestamps:
            return
        stored_bar_type = STORED_BAR_TYPES.get(data_type)
        if stored_bar_type is None:
            intervals = [(timestamps[0], timestamps[-1] + timedelta(milliseconds=1))]
        else:
            step = timedelta(minutes=bar_minutes(stored_bar_type))
            intervals: list[tuple[datetime, datetime]] = []
            interval_start = timestamps[0]
            previous = timestamps[0]
            for timestamp in timestamps[1:]:
                if timestamp - previous != step:
                    intervals.append((interval_start, previous + step))
                    interval_start = timestamp
                previous = timestamp
            intervals.append((interval_start, previous + step))

        for start, end in intervals:
            self.catalog.record_coverage(
                data_type=data_type,
                venue=venue,
                instrument_id=instrument_id,
                bar_type=bar_type,
                start=start,
                end=end,
                source_name=source_name,
            )

    def import_legacy(self, query: DataQuery, legacy_root: Path) -> int:
        """Copy matching legacy ``data/clean`` rows into canonical partitions."""

        if query.data_type not in {DataType.CANDLES, DataType.FUNDING_RATES}:
            return 0

        instrument_root = legacy_root / query.venue.lower() / query.instrument_id
        if not instrument_root.exists():
            return 0
        candidates = list(instrument_root.glob("*.parquet"))
        imported_frames: list[pl.DataFrame] = []
        for path in candidates:
            try:
                lazy = pl.scan_parquet(path)
                columns = set(lazy.collect_schema().names())
                is_funding = "funding_rate" in columns
                if is_funding != (query.data_type is DataType.FUNDING_RATES):
                    continue
                timestamp_column = "timestamp" if "timestamp" in columns else "ts"
                if timestamp_column not in columns:
                    continue
                bounds = lazy.select(
                    pl.col(timestamp_column).min().alias("minimum"),
                    pl.col(timestamp_column).max().alias("maximum"),
                ).collect()
                minimum = bounds["minimum"][0]
                maximum = bounds["maximum"][0]
                if not isinstance(minimum, datetime) or not isinstance(maximum, datetime):
                    continue
                if as_utc(maximum) < query.start or as_utc(minimum) >= query.end:
                    continue
                frame = canonicalize_frame(pl.read_parquet(path), query.data_type).filter(
                    (pl.col("timestamp") >= query.start)
                    & (pl.col("timestamp") < query.end)
                )
                if query.data_type is DataType.CANDLES:
                    frame = frame.filter(pl.col("bar_type").str.to_lowercase() == "1m")
                if not frame.is_empty():
                    imported_frames.append(frame)
            except (OSError, ValueError, pl.exceptions.PolarsError):
                continue

        if not imported_frames:
            return 0
        imported = (
            pl.concat(imported_frames, how="vertical_relaxed")
            .unique(subset=PRIMARY_KEYS[query.data_type], keep="last")
            .sort("timestamp")
        )
        bar_type = "1m" if query.data_type is DataType.CANDLES else None
        self.write(
            imported,
            data_type=query.data_type,
            venue=query.venue,
            instrument_id=query.instrument_id,
            bar_type=bar_type,
            source_name="legacy-parquet",
        )
        self._rebuild_coverage(
            imported,
            data_type=query.data_type,
            venue=query.venue,
            instrument_id=query.instrument_id,
            bar_type=bar_type,
            source_name="legacy-parquet",
        )
        return imported.height

    def write(
        self,
        frame: pl.DataFrame,
        *,
        data_type: DataType,
        venue: str,
        instrument_id: str,
        bar_type: str | None,
        source_name: str,
    ) -> None:
        frame = canonicalize_frame(frame, data_type)
        if frame.is_empty():
            return

        start = frame["timestamp"].min()
        end = frame["timestamp"].max()
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)

        for partition_start, partition_end in iter_partitions(
            data_type,
            start,
            end + timedelta(microseconds=1),
        ):
            partition = frame.filter(
                (pl.col("timestamp") >= partition_start)
                & (pl.col("timestamp") < partition_end)
            )
            if partition.is_empty():
                continue
            path = self.partition_path(
                data_type=data_type,
                venue=venue,
                instrument_id=instrument_id,
                bar_type=bar_type,
                partition_start=partition_start,
            )
            self._merge_partition(
                path,
                partition,
                data_type=data_type,
                venue=venue,
                instrument_id=instrument_id,
                bar_type=bar_type,
                partition_start=partition_start,
                partition_end=partition_end,
                source_name=source_name,
            )

    @contextmanager
    def _partition_lock(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _merge_partition(
        self,
        path: Path,
        incoming: pl.DataFrame,
        *,
        data_type: DataType,
        venue: str,
        instrument_id: str,
        bar_type: str | None,
        partition_start: datetime,
        partition_end: datetime,
        source_name: str,
    ) -> None:
        with self._partition_lock(path):
            frames = [incoming]
            if path.exists():
                frames.insert(0, canonicalize_frame(pl.read_parquet(path), data_type))
            merged = (
                pl.concat(frames, how="vertical_relaxed")
                .unique(subset=PRIMARY_KEYS[data_type], keep="last")
                .sort("timestamp")
            )
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                merged.write_parquet(temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

            min_timestamp = merged["timestamp"].min()
            max_timestamp = merged["timestamp"].max()
            self.catalog.record_file(
                path=path,
                data_type=data_type,
                venue=venue,
                instrument_id=instrument_id,
                bar_type=bar_type,
                partition_start=partition_start,
                partition_end=partition_end,
                min_timestamp=min_timestamp if isinstance(min_timestamp, datetime) else None,
                max_timestamp=max_timestamp if isinstance(max_timestamp, datetime) else None,
                row_count=merged.height,
                source_name=source_name,
            )
