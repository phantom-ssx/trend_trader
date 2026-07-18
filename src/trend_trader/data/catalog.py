"""SQLite coverage catalog for locally persisted market data."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from trend_trader.data.models import DataType, as_utc


class DataCatalog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS coverage (
                    id INTEGER PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    instrument_id TEXT NOT NULL,
                    bar_type TEXT NOT NULL,
                    start TEXT NOT NULL,
                    end TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS coverage_lookup ON coverage (
                    data_type, venue, instrument_id, bar_type, start, end
                );
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    instrument_id TEXT NOT NULL,
                    bar_type TEXT NOT NULL,
                    partition_start TEXT NOT NULL,
                    partition_end TEXT NOT NULL,
                    min_timestamp TEXT,
                    max_timestamp TEXT,
                    row_count INTEGER NOT NULL,
                    source_name TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _key_bar_type(bar_type: str | None) -> str:
        return bar_type or ""

    def record_coverage(
        self,
        *,
        data_type: DataType,
        venue: str,
        instrument_id: str,
        bar_type: str | None,
        start: datetime,
        end: datetime,
        source_name: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO coverage (
                    data_type, venue, instrument_id, bar_type, start, end,
                    source_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data_type.value,
                    venue,
                    instrument_id,
                    self._key_bar_type(bar_type),
                    as_utc(start).isoformat(),
                    as_utc(end).isoformat(),
                    source_name,
                    datetime.now().astimezone().isoformat(),
                ),
            )

    def missing_intervals(
        self,
        *,
        data_type: DataType,
        venue: str,
        instrument_id: str,
        bar_type: str | None,
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, datetime]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT start, end FROM coverage
                WHERE data_type = ? AND venue = ? AND instrument_id = ? AND bar_type = ?
                  AND end > ? AND start < ?
                ORDER BY start
                """,
                (
                    data_type.value,
                    venue,
                    instrument_id,
                    self._key_bar_type(bar_type),
                    start.isoformat(),
                    end.isoformat(),
                ),
            ).fetchall()

        cursor = start
        missing: list[tuple[datetime, datetime]] = []
        for raw_start, raw_end in rows:
            covered_start = max(as_utc(raw_start), start)
            covered_end = min(as_utc(raw_end), end)
            if covered_end <= cursor:
                continue
            if covered_start > cursor:
                missing.append((cursor, covered_start))
            cursor = max(cursor, covered_end)
        if cursor < end:
            missing.append((cursor, end))
        return missing

    def record_file(
        self,
        *,
        path: Path,
        data_type: DataType,
        venue: str,
        instrument_id: str,
        bar_type: str | None,
        partition_start: datetime,
        partition_end: datetime,
        min_timestamp: datetime | None,
        max_timestamp: datetime | None,
        row_count: int,
        source_name: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    min_timestamp=excluded.min_timestamp,
                    max_timestamp=excluded.max_timestamp,
                    row_count=excluded.row_count,
                    source_name=excluded.source_name,
                    schema_version=excluded.schema_version,
                    updated_at=excluded.updated_at
                """,
                (
                    str(path),
                    data_type.value,
                    venue,
                    instrument_id,
                    self._key_bar_type(bar_type),
                    partition_start.isoformat(),
                    partition_end.isoformat(),
                    min_timestamp.isoformat() if min_timestamp else None,
                    max_timestamp.isoformat() if max_timestamp else None,
                    row_count,
                    source_name,
                    1,
                    datetime.now().astimezone().isoformat(),
                ),
            )

    def invalidate_coverage(
        self,
        *,
        data_type: DataType,
        venue: str,
        instrument_id: str,
        bar_type: str | None,
        start: datetime,
        end: datetime,
    ) -> None:
        """Forget coverage which can no longer be verified from local files."""

        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM coverage
                WHERE data_type = ? AND venue = ? AND instrument_id = ? AND bar_type = ?
                  AND end > ? AND start < ?
                """,
                (
                    data_type.value,
                    venue,
                    instrument_id,
                    self._key_bar_type(bar_type),
                    start.isoformat(),
                    end.isoformat(),
                ),
            )
