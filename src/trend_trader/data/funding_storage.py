"""Schemas, in-memory state, and partitioned Parquet storage for funding data."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import polars as pl

FUNDING_SNAPSHOT_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "snapshot_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "exchange_ts": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "received_at": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "funding_rate": pl.Float64,
    "next_funding_rate": pl.Float64,
    "funding_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "next_funding_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "interest_rate": pl.Float64,
    "premium": pl.Float64,
    "method": pl.Utf8,
    "formula_type": pl.Utf8,
    "data_source": pl.Utf8,
    "data_status": pl.Utf8,
}

FUNDING_HISTORY_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "funding_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "funding_rate": pl.Float64,
    "received_at": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "method": pl.Utf8,
    "formula_type": pl.Utf8,
}

SNAPSHOT_PRIMARY_KEY = ["venue", "instrument_id", "snapshot_time"]
HISTORY_PRIMARY_KEY = ["venue", "instrument_id", "funding_time"]


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def floor_minute(value: datetime) -> datetime:
    return as_utc(value).replace(second=0, microsecond=0)


def optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def optional_okx_datetime(value: object) -> datetime | None:
    milliseconds = optional_float(value)
    if milliseconds is None or milliseconds <= 0:
        return None
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)


def _empty(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict(schema))


def canonicalize(
    frame: pl.DataFrame,
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    """Coerce a frame to an ordered schema while preserving UTC timestamps."""

    for name, dtype in schema.items():
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(name))

    expressions: list[pl.Expr] = []
    for name, dtype in schema.items():
        expression = pl.col(name)
        current = frame.schema[name]
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
    return frame.with_columns(expressions).select(*schema)


@dataclass(frozen=True, slots=True)
class FundingState:
    venue: str
    instrument_id: str
    exchange_ts: datetime | None
    received_at: datetime
    funding_rate: float | None
    next_funding_rate: float | None
    funding_time: datetime | None
    next_funding_time: datetime | None
    interest_rate: float | None
    premium: float | None
    method: str | None
    formula_type: str | None
    data_source: str

    @classmethod
    def from_okx(
        cls,
        row: Mapping[str, object],
        *,
        received_at: datetime,
        data_source: str,
    ) -> FundingState:
        instrument_id = str(row.get("instId") or "")
        if not instrument_id:
            raise ValueError("OKX funding-rate row is missing instId")
        if data_source not in {"websocket", "rest"}:
            raise ValueError(f"unsupported data source: {data_source}")
        return cls(
            venue="OKX",
            instrument_id=instrument_id,
            exchange_ts=optional_okx_datetime(row.get("ts")),
            received_at=as_utc(received_at),
            funding_rate=optional_float(row.get("fundingRate")),
            next_funding_rate=optional_float(row.get("nextFundingRate")),
            funding_time=optional_okx_datetime(row.get("fundingTime")),
            next_funding_time=optional_okx_datetime(row.get("nextFundingTime")),
            interest_rate=optional_float(row.get("interestRate")),
            premium=optional_float(row.get("premium")),
            method=str(row.get("method") or "") or None,
            formula_type=str(row.get("formulaType") or "") or None,
            data_source=data_source,
        )


def build_history_frame(
    rows: Iterable[Mapping[str, object]],
    instrument_id: str,
    *,
    received_at: datetime,
) -> pl.DataFrame:
    """Build confirmed history rows, using OKX ``realizedRate`` as the final rate."""

    captured_at = as_utc(received_at)
    normalized: list[dict[str, object]] = []
    for row in rows:
        funding_time = optional_okx_datetime(row.get("fundingTime"))
        realized_rate = optional_float(row.get("realizedRate"))
        if funding_time is None or realized_rate is None:
            continue
        normalized.append(
            {
                "venue": "OKX",
                "instrument_id": instrument_id,
                "funding_time": funding_time,
                "funding_rate": realized_rate,
                "received_at": captured_at,
                "method": str(row.get("method") or "") or None,
                "formula_type": str(row.get("formulaType") or "") or None,
            }
        )
    if not normalized:
        return _empty(FUNDING_HISTORY_SCHEMA)
    return (
        canonicalize(
            pl.DataFrame(normalized, infer_schema_length=None),
            FUNDING_HISTORY_SCHEMA,
        )
        .unique(subset=HISTORY_PRIMARY_KEY, keep="last")
        .sort("funding_time")
    )


class FundingStateCache:
    """Latest funding state for every currently live instrument."""

    def __init__(self) -> None:
        self._states: dict[str, FundingState] = {}

    def update(self, state: FundingState) -> bool:
        current = self._states.get(state.instrument_id)
        if current is not None:
            current_order = current.exchange_ts or current.received_at
            incoming_order = state.exchange_ts or state.received_at
            if incoming_order < current_order:
                return False
            if (
                incoming_order == current_order
                and current.data_source == "websocket"
                and state.data_source == "rest"
            ):
                return False
        self._states[state.instrument_id] = state
        return True

    def retain(self, instrument_ids: set[str]) -> None:
        self._states = {
            instrument_id: state
            for instrument_id, state in self._states.items()
            if instrument_id in instrument_ids
        }

    def stale_or_missing(
        self,
        instrument_ids: Iterable[str],
        *,
        now: datetime,
        stale_after: timedelta,
    ) -> list[str]:
        current_time = as_utc(now)
        return [
            instrument_id
            for instrument_id in instrument_ids
            if (
                (state := self._states.get(instrument_id)) is None
                or current_time - state.received_at > stale_after
            )
        ]

    def snapshot(
        self,
        instrument_ids: Iterable[str],
        *,
        snapshot_time: datetime,
        stale_after: timedelta,
    ) -> pl.DataFrame:
        aligned_time = floor_minute(snapshot_time)
        rows: list[dict[str, object]] = []
        for instrument_id in sorted(instrument_ids):
            state = self._states.get(instrument_id)
            if state is None:
                continue
            age = max(timedelta(0), aligned_time - state.received_at)
            rows.append(
                {
                    "venue": state.venue,
                    "instrument_id": state.instrument_id,
                    "snapshot_time": aligned_time,
                    "exchange_ts": state.exchange_ts,
                    "received_at": state.received_at,
                    "funding_rate": state.funding_rate,
                    "next_funding_rate": state.next_funding_rate,
                    "funding_time": state.funding_time,
                    "next_funding_time": state.next_funding_time,
                    "interest_rate": state.interest_rate,
                    "premium": state.premium,
                    "method": state.method,
                    "formula_type": state.formula_type,
                    "data_source": state.data_source,
                    "data_status": "fresh" if age <= stale_after else "stale",
                }
            )
        if not rows:
            return _empty(FUNDING_SNAPSHOT_SCHEMA)
        return canonicalize(
            pl.DataFrame(rows, infer_schema_length=None),
            FUNDING_SNAPSHOT_SCHEMA,
        ).sort("instrument_id")

    def states(self) -> tuple[FundingState, ...]:
        return tuple(self._states.values())


class FundingParquetRepository:
    """Atomic daily Parquet partitions for snapshots and confirmed history."""

    def __init__(self, data_root: Path | str) -> None:
        self.data_root = Path(data_root)

    def snapshot_path(self, timestamp: datetime) -> Path:
        value = as_utc(timestamp)
        return (
            self.data_root
            / "funding_snapshot"
            / f"year={value:%Y}"
            / f"date={value:%Y-%m-%d}"
            / f"funding_snapshot-{value:%Y-%m-%d}.parquet"
        )

    def history_path(self, timestamp: datetime) -> Path:
        value = as_utc(timestamp)
        return (
            self.data_root
            / "funding_history"
            / f"year={value:%Y}"
            / f"date={value:%Y-%m-%d}"
            / f"funding_history-{value:%Y-%m-%d}.parquet"
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

    def _merge(
        self,
        path: Path,
        incoming: pl.DataFrame,
        *,
        schema: Mapping[str, pl.DataType],
        primary_key: list[str],
        sort_by: list[str],
    ) -> None:
        normalized = canonicalize(incoming, schema)
        if normalized.is_empty():
            return
        with self._partition_lock(path):
            frames = [normalized]
            if path.exists():
                frames.insert(0, canonicalize(pl.read_parquet(path), schema))
            merged = (
                pl.concat(frames, how="vertical_relaxed")
                .unique(subset=primary_key, keep="last")
                .sort(sort_by)
            )
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                merged.write_parquet(temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def write_snapshots(self, frame: pl.DataFrame) -> None:
        normalized = canonicalize(frame, FUNDING_SNAPSHOT_SCHEMA)
        if normalized.is_empty():
            return
        dates = normalized.get_column("snapshot_time").dt.date().unique().to_list()
        for date_value in dates:
            partition = normalized.filter(pl.col("snapshot_time").dt.date() == date_value)
            timestamp = partition.get_column("snapshot_time")[0]
            assert isinstance(timestamp, datetime)
            self._merge(
                self.snapshot_path(timestamp),
                partition,
                schema=FUNDING_SNAPSHOT_SCHEMA,
                primary_key=SNAPSHOT_PRIMARY_KEY,
                sort_by=["snapshot_time", "venue", "instrument_id"],
            )

    def write_history(self, frame: pl.DataFrame) -> None:
        normalized = canonicalize(frame, FUNDING_HISTORY_SCHEMA)
        if normalized.is_empty():
            return
        dates = normalized.get_column("funding_time").dt.date().unique().to_list()
        for date_value in dates:
            partition = normalized.filter(pl.col("funding_time").dt.date() == date_value)
            timestamp = partition.get_column("funding_time")[0]
            assert isinstance(timestamp, datetime)
            self._merge(
                self.history_path(timestamp),
                partition,
                schema=FUNDING_HISTORY_SCHEMA,
                primary_key=HISTORY_PRIMARY_KEY,
                sort_by=["funding_time", "venue", "instrument_id"],
            )

    def latest_history_times(
        self,
        instrument_ids: Iterable[str],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[str, datetime]:
        wanted = set(instrument_ids)
        if not wanted:
            return {}
        cursor = as_utc(start).replace(hour=0, minute=0, second=0, microsecond=0)
        stop = as_utc(end)
        paths: list[Path] = []
        while cursor < stop:
            path = self.history_path(cursor)
            if path.exists():
                paths.append(path)
            cursor += timedelta(days=1)
        if not paths:
            return {}
        frame = canonicalize(pl.read_parquet(paths), FUNDING_HISTORY_SCHEMA).filter(
            pl.col("instrument_id").is_in(wanted)
        )
        if frame.is_empty():
            return {}
        return {
            str(row["instrument_id"]): row["funding_time"]
            for row in frame.group_by("instrument_id")
            .agg(pl.col("funding_time").max())
            .iter_rows(named=True)
            if isinstance(row["funding_time"], datetime)
        }
