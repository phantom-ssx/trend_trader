from __future__ import annotations

import fcntl
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


def utc_now_text() -> str:
    return datetime.now(UTC).isoformat()


class AlreadyRunningError(RuntimeError):
    pass


class OfflineCatalog:
    """SQLite coverage catalog and single-writer lease for offline synchronization."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "catalog.sqlite"
        self.lock_path = self.root / ".sync.lock"
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    exit_code INTEGER,
                    mode TEXT NOT NULL,
                    report_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS coverage (
                    dataset TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    artifact_path TEXT,
                    source_sha256 TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (dataset, scope_key, date)
                );
                CREATE TABLE IF NOT EXISTS availability (
                    dataset TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    official_start TEXT,
                    listing_time TEXT,
                    first_event_time TEXT,
                    continuous_start TEXT,
                    discovery_method TEXT NOT NULL,
                    discovered_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    PRIMARY KEY (dataset, scope_key)
                );
                CREATE TABLE IF NOT EXISTS instruments (
                    instrument_id TEXT PRIMARY KEY,
                    instrument_type TEXT NOT NULL,
                    base_currency TEXT NOT NULL,
                    index_id TEXT,
                    listing_time TEXT,
                    expiration_time TEXT,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    category TEXT NOT NULL,
                    rule_type TEXT
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    path TEXT PRIMARY KEY,
                    dataset TEXT NOT NULL,
                    source_url TEXT,
                    source_sha256 TEXT,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS notification_outbox (
                    run_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS invalid_identifiers (
                    dataset TEXT NOT NULL,
                    identifier TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    error_code TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    first_failed_at TEXT NOT NULL,
                    last_failed_at TEXT NOT NULL,
                    last_target_date TEXT NOT NULL,
                    failure_count INTEGER NOT NULL DEFAULT 1,
                    last_skipped_at TEXT,
                    last_skipped_target_date TEXT,
                    skip_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (dataset, identifier)
                );
                """
            )

    @contextmanager
    def execution_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AlreadyRunningError("another offline sync process is running") from exc
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def begin_run(self, run_id: str, mode: str, report: Mapping[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(run_id, started_at, status, mode, report_json)
                VALUES (?, ?, 'running', ?, ?)
                """,
                (run_id, utc_now_text(), mode, json.dumps(dict(report), sort_keys=True)),
            )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        exit_code: int,
        report: Mapping[str, Any],
    ) -> None:
        payload = json.dumps(dict(report), sort_keys=True, default=str)
        now = utc_now_text()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET finished_at = ?, status = ?, exit_code = ?, report_json = ?
                WHERE run_id = ?
                """,
                (now, status, exit_code, payload, run_id),
            )
            connection.execute(
                """
                INSERT INTO notification_outbox(run_id, payload_json, status, updated_at)
                VALUES (?, ?, 'pending', ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    status='pending',
                    updated_at=excluded.updated_at
                """,
                (run_id, payload, now),
            )

    def is_complete(self, dataset: str, target_date: date, scope_key: str = "all") -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT status FROM coverage
                WHERE dataset = ? AND scope_key = ? AND date = ?
                """,
                (dataset, scope_key, target_date.isoformat()),
            ).fetchone()
        return bool(row and str(row["status"]).startswith("complete"))

    def is_resolved(self, dataset: str, target_date: date, scope_key: str = "all") -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT status FROM coverage
                WHERE dataset = ? AND scope_key = ? AND date = ?
                """,
                (dataset, scope_key, target_date.isoformat()),
            ).fetchone()
        return bool(
            row
            and (
                str(row["status"]).startswith("complete")
                or str(row["status"]) == "unavailable"
            )
        )

    def mark_coverage(
        self,
        dataset: str,
        target_date: date,
        *,
        status: str,
        row_count: int,
        scope_key: str = "all",
        artifact_path: str | None = None,
        source_sha256: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO coverage(
                    dataset, scope_key, date, status, row_count,
                    artifact_path, source_sha256, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset, scope_key, date) DO UPDATE SET
                    status=excluded.status,
                    row_count=excluded.row_count,
                    artifact_path=excluded.artifact_path,
                    source_sha256=excluded.source_sha256,
                    updated_at=excluded.updated_at
                """,
                (
                    dataset,
                    scope_key,
                    target_date.isoformat(),
                    status,
                    row_count,
                    artifact_path,
                    source_sha256,
                    utc_now_text(),
                ),
            )

    def upsert_instrument(self, row: Mapping[str, object], seen_at: datetime) -> None:
        instrument_id = str(row.get("instrument_id") or row.get("instId") or "")
        if not instrument_id:
            return
        instrument_type = str(row.get("instrument_type") or row.get("instType") or "").upper()
        base_currency = str(row.get("base_currency") or row.get("baseCcy") or "")
        if not base_currency:
            base_currency = instrument_id.split("-", maxsplit=1)[0]
        index_id = str(row.get("index_id") or row.get("uly") or "") or None
        if index_id is None:
            parts = instrument_id.split("-")
            index_id = "-".join(parts[:2]) if len(parts) >= 2 else None
        category = str(row.get("instCategory") or row.get("category") or "crypto")
        listing_time = _optional_time_text(row.get("listing_time") or row.get("listTime"))
        expiration_time = _optional_time_text(row.get("expiration_time") or row.get("expTime"))
        timestamp = seen_at.astimezone(UTC).isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO instruments(
                    instrument_id, instrument_type, base_currency, index_id,
                    listing_time, expiration_time, first_seen, last_seen, category, rule_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(instrument_id) DO UPDATE SET
                    instrument_type=excluded.instrument_type,
                    base_currency=excluded.base_currency,
                    index_id=COALESCE(excluded.index_id, instruments.index_id),
                    listing_time=COALESCE(excluded.listing_time, instruments.listing_time),
                    expiration_time=COALESCE(excluded.expiration_time, instruments.expiration_time),
                    first_seen=MIN(instruments.first_seen, excluded.first_seen),
                    last_seen=MAX(instruments.last_seen, excluded.last_seen),
                    category=excluded.category,
                    rule_type=excluded.rule_type
                """,
                (
                    instrument_id,
                    instrument_type,
                    base_currency.upper(),
                    index_id,
                    listing_time,
                    expiration_time,
                    timestamp,
                    timestamp,
                    category,
                    str(row.get("ruleType") or ""),
                ),
            )

    def list_instruments(self) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM instruments ORDER BY instrument_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def cached_invalid_identifier(
        self,
        dataset: str,
        identifier: str,
        source_fingerprint: str,
    ) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM invalid_identifiers
                WHERE dataset=? AND identifier=? AND source_fingerprint=?
                """,
                (dataset, identifier, source_fingerprint),
            ).fetchone()
        return dict(row) if row else None

    def record_invalid_identifier(
        self,
        dataset: str,
        identifier: str,
        *,
        source_fingerprint: str,
        endpoint: str,
        error_code: str,
        error_message: str,
        target_date: date,
    ) -> None:
        now = utc_now_text()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO invalid_identifiers(
                    dataset, identifier, source_fingerprint, endpoint,
                    error_code, error_message, first_failed_at, last_failed_at,
                    last_target_date, failure_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(dataset, identifier) DO UPDATE SET
                    source_fingerprint=excluded.source_fingerprint,
                    endpoint=excluded.endpoint,
                    error_code=excluded.error_code,
                    error_message=excluded.error_message,
                    last_failed_at=excluded.last_failed_at,
                    last_target_date=excluded.last_target_date,
                    failure_count=invalid_identifiers.failure_count + 1
                """,
                (
                    dataset,
                    identifier,
                    source_fingerprint,
                    endpoint,
                    error_code,
                    error_message,
                    now,
                    now,
                    target_date.isoformat(),
                ),
            )

    def clear_invalid_identifier(self, dataset: str, identifier: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM invalid_identifiers WHERE dataset=? AND identifier=?",
                (dataset, identifier),
            )

    def note_invalid_identifier_skipped(
        self,
        dataset: str,
        identifier: str,
        target_date: date,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE invalid_identifiers
                SET last_skipped_at=?, last_skipped_target_date=?, skip_count=skip_count + 1
                WHERE dataset=? AND identifier=?
                """,
                (utc_now_text(), target_date.isoformat(), dataset, identifier),
            )

    def invalid_identifier_summary(self) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT dataset, identifier, endpoint, error_code, error_message,
                       first_failed_at, last_failed_at, last_target_date, failure_count,
                       last_skipped_at, last_skipped_target_date, skip_count
                FROM invalid_identifiers
                ORDER BY dataset, identifier
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_availability(
        self,
        dataset: str,
        scope_key: str,
        *,
        official_start: date | None = None,
        listing_time: str | None = None,
        first_event_time: str | None = None,
        continuous_start: date | None = None,
        discovery_method: str,
        status: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO availability(
                    dataset, scope_key, official_start, listing_time,
                    first_event_time, continuous_start, discovery_method,
                    discovered_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset, scope_key) DO UPDATE SET
                    official_start=COALESCE(excluded.official_start, availability.official_start),
                    listing_time=COALESCE(excluded.listing_time, availability.listing_time),
                    first_event_time=CASE
                        WHEN excluded.first_event_time IS NULL
                            THEN availability.first_event_time
                        WHEN availability.first_event_time IS NULL
                            OR excluded.first_event_time < availability.first_event_time
                            THEN excluded.first_event_time
                        ELSE availability.first_event_time
                    END,
                    continuous_start=CASE
                        WHEN excluded.continuous_start IS NULL
                            THEN availability.continuous_start
                        WHEN availability.continuous_start IS NULL
                            OR excluded.continuous_start < availability.continuous_start
                            THEN excluded.continuous_start
                        ELSE availability.continuous_start
                    END,
                    discovery_method=excluded.discovery_method,
                    discovered_at=excluded.discovered_at,
                    status=excluded.status
                """,
                (
                    dataset,
                    scope_key,
                    official_start.isoformat() if official_start else None,
                    listing_time,
                    first_event_time,
                    continuous_start.isoformat() if continuous_start else None,
                    discovery_method,
                    utc_now_text(),
                    status,
                ),
            )

    def earliest_complete_date(self, dataset: str, scope_key: str = "all") -> date | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT MIN(date) AS first_date
                FROM coverage
                WHERE dataset=? AND scope_key=? AND status LIKE 'complete%'
                """,
                (dataset, scope_key),
            ).fetchone()
        value = row["first_date"] if row else None
        return date.fromisoformat(str(value)) if value else None

    def coverage_summary(self) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT dataset, scope_key, MIN(date) AS start_date, MAX(date) AS end_date,
                       COUNT(*) AS partitions, SUM(row_count) AS rows
                FROM coverage
                WHERE status LIKE 'complete%'
                GROUP BY dataset, scope_key
                ORDER BY dataset, scope_key
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def availability_summary(self) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM availability ORDER BY dataset, scope_key"
            ).fetchall()
        return [dict(row) for row in rows]

    def record_artifact(
        self,
        path: Path,
        *,
        dataset: str,
        source_url: str | None,
        source_sha256: str | None,
        metadata: Mapping[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(
                    path, dataset, source_url, source_sha256,
                    size_bytes, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    source_url=excluded.source_url,
                    source_sha256=excluded.source_sha256,
                    size_bytes=excluded.size_bytes,
                    metadata_json=excluded.metadata_json
                """,
                (
                    str(path),
                    dataset,
                    source_url,
                    source_sha256,
                    path.stat().st_size,
                    utc_now_text(),
                    json.dumps(dict(metadata), sort_keys=True, default=str),
                ),
            )

    def pending_notifications(self) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, payload_json, attempts
                FROM notification_outbox
                WHERE status = 'pending'
                ORDER BY updated_at
                """
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "payload": json.loads(row["payload_json"]),
                "attempts": row["attempts"],
            }
            for row in rows
        ]

    def recover_interrupted_runs(self) -> int:
        now = utc_now_text()
        recovered = 0
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT run_id, report_json FROM runs WHERE status='running'"
            ).fetchall()
            for row in rows:
                report = json.loads(row["report_json"])
                report.update(
                    {
                        "status": "interrupted",
                        "finished_at": now,
                        "fatal_error": "previous process ended without finalizing the run",
                    }
                )
                payload = json.dumps(report, sort_keys=True, default=str)
                connection.execute(
                    """
                    UPDATE runs
                    SET finished_at=?, status='interrupted', exit_code=1, report_json=?
                    WHERE run_id=?
                    """,
                    (now, payload, row["run_id"]),
                )
                connection.execute(
                    """
                    INSERT INTO notification_outbox(run_id, payload_json, status, updated_at)
                    VALUES (?, ?, 'pending', ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        payload_json=excluded.payload_json,
                        status='pending',
                        updated_at=excluded.updated_at
                    """,
                    (row["run_id"], payload, now),
                )
                recovered += 1
        return recovered

    def mark_notification(
        self,
        run_id: str,
        *,
        sent: bool,
        error: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE notification_outbox
                SET status=?, attempts=attempts+1, last_error=?, updated_at=?
                WHERE run_id=?
                """,
                ("sent" if sent else "pending", error, utc_now_text(), run_id),
            )


def _optional_time_text(value: object) -> str | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    text = str(value)
    try:
        milliseconds = int(text)
    except ValueError:
        return text
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).isoformat()
