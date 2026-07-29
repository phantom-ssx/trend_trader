"""Schemas, in-memory state, and partitioned Parquet storage for open interest."""

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

OPEN_INTEREST_SNAPSHOT_SCHEMA = {
    # 交易所代码，例如 "OKX"。
    "venue": pl.Utf8,
    # OKX 合约 ID，例如 "BTC-USD-260731" 或 "BTC-USDT-SWAP"。
    "instrument_id": pl.Utf8,
    # 合约种类，例如 "FUTURES"（交割）或 "SWAP"（永续）。
    "instrument_type": pl.Utf8,
    # OKX 合约家族，例如 "BTC-USD" 或 "BTC-USDT"。
    "instrument_family": pl.Utf8,
    # 标的币种，例如 "BTC"。
    "base_currency": pl.Utf8,
    # 结算币种，例如币本位 FUTURES 为 "BTC"，U 本位 SWAP 为 "USDT"。
    "settle_currency": pl.Utf8,
    # 合约计价类型，例如 "inverse"（反向）或 "linear"（正向）。
    "contract_type": pl.Utf8,
    # 到期时间，例如 2026-07-31T08:00:00Z；永续合约为 null。
    "expiration_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    # 对齐后的本地分钟快照时间，例如 2026-07-29T08:31:00Z。
    "snapshot_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    # OKX 返回的行情时间 ts，例如 2026-07-29T08:31:01.234Z。
    "exchange_ts": pl.Datetime(time_unit="ms", time_zone="UTC"),
    # 本地收到 WS/REST 响应的时间，例如 2026-07-29T08:31:01.410Z。
    "received_at": pl.Datetime(time_unit="ms", time_zone="UTC"),
    # OKX oi：未平仓合约张数，例如 3_247_339.0。
    "open_interest": pl.Float64,
    # OKX oiCcy：折算后的标的币数量，例如 32_473.39 BTC。
    "open_interest_ccy": pl.Float64,
    # OKX oiUsd：美元名义价值，例如 2_054_400_000.0 USD。
    "open_interest_usd": pl.Float64,
    # 当前缓存值的来源，例如 "websocket" 或 "rest"。
    "data_source": pl.Utf8,
    # 快照时的数据时效状态，例如 "fresh" 或 "stale"。
    "data_status": pl.Utf8,
}

SNAPSHOT_PRIMARY_KEY = ["venue", "instrument_id", "snapshot_time"]
SUPPORTED_INSTRUMENT_TYPES = frozenset({"SWAP", "FUTURES"})


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
class OpenInterestInstrument:
    instrument_id: str
    instrument_type: str
    instrument_family: str
    base_currency: str
    settle_currency: str
    contract_type: str
    expiration_time: datetime | None

    @classmethod
    def from_okx(cls, row: Mapping[str, object]) -> OpenInterestInstrument:
        instrument_id = str(row.get("instId") or "")
        if not instrument_id:
            raise ValueError("OKX instrument row is missing instId")
        instrument_type = str(row.get("instType") or "").upper()
        if instrument_type not in SUPPORTED_INSTRUMENT_TYPES:
            raise ValueError(f"unsupported OKX instrument type: {instrument_type!r}")
        instrument_family = str(row.get("instFamily") or "")
        base_currency = str(row.get("baseCcy") or "").upper()
        if not base_currency:
            family_or_id = instrument_family or instrument_id
            base_currency = family_or_id.split("-", maxsplit=1)[0].upper()
        return cls(
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            instrument_family=instrument_family,
            base_currency=base_currency,
            settle_currency=str(row.get("settleCcy") or "").upper(),
            contract_type=str(row.get("ctType") or "").lower(),
            expiration_time=optional_okx_datetime(row.get("expTime")),
        )


@dataclass(frozen=True, slots=True)
class OpenInterestState:
    venue: str
    instrument_id: str
    instrument_type: str
    exchange_ts: datetime | None
    received_at: datetime
    open_interest: float | None
    open_interest_ccy: float | None
    open_interest_usd: float | None
    data_source: str

    @classmethod
    def from_okx(
        cls,
        row: Mapping[str, object],
        *,
        received_at: datetime,
        data_source: str,
    ) -> OpenInterestState:
        instrument_id = str(row.get("instId") or "")
        if not instrument_id:
            raise ValueError("OKX open-interest row is missing instId")
        instrument_type = str(row.get("instType") or "").upper()
        if instrument_type not in SUPPORTED_INSTRUMENT_TYPES:
            raise ValueError(f"unsupported OKX open-interest instrument type: {instrument_type!r}")
        if data_source not in {"websocket", "rest"}:
            raise ValueError(f"unsupported data source: {data_source}")
        return cls(
            venue="OKX",
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            exchange_ts=optional_okx_datetime(row.get("ts")),
            received_at=as_utc(received_at),
            open_interest=optional_float(row.get("oi")),
            open_interest_ccy=optional_float(row.get("oiCcy")),
            open_interest_usd=optional_float(row.get("oiUsd")),
            data_source=data_source,
        )


class OpenInterestStateCache:
    """Latest open-interest state for every currently live instrument."""

    def __init__(self) -> None:
        self._states: dict[str, OpenInterestState] = {}

    def update(self, state: OpenInterestState) -> bool:
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
        instruments: Mapping[str, OpenInterestInstrument],
        *,
        snapshot_time: datetime,
        stale_after: timedelta,
    ) -> pl.DataFrame:
        aligned_time = floor_minute(snapshot_time)
        rows: list[dict[str, object]] = []
        for instrument_id in sorted(instruments):
            state = self._states.get(instrument_id)
            if state is None:
                continue
            instrument = instruments[instrument_id]
            age = max(timedelta(0), aligned_time - state.received_at)
            rows.append(
                {
                    "venue": state.venue,
                    "instrument_id": state.instrument_id,
                    "instrument_type": state.instrument_type,
                    "instrument_family": instrument.instrument_family,
                    "base_currency": instrument.base_currency,
                    "settle_currency": instrument.settle_currency,
                    "contract_type": instrument.contract_type,
                    "expiration_time": instrument.expiration_time,
                    "snapshot_time": aligned_time,
                    "exchange_ts": state.exchange_ts,
                    "received_at": state.received_at,
                    "open_interest": state.open_interest,
                    "open_interest_ccy": state.open_interest_ccy,
                    "open_interest_usd": state.open_interest_usd,
                    "data_source": state.data_source,
                    "data_status": "fresh" if age <= stale_after else "stale",
                }
            )
        if not rows:
            return _empty(OPEN_INTEREST_SNAPSHOT_SCHEMA)
        return canonicalize(
            pl.DataFrame(rows, infer_schema_length=None),
            OPEN_INTEREST_SNAPSHOT_SCHEMA,
        ).sort("instrument_id")


class OpenInterestParquetRepository:
    """Atomic daily Parquet partitions for contract-level minute snapshots."""

    def __init__(self, data_root: Path | str) -> None:
        self.data_root = Path(data_root)

    def snapshot_path(self, timestamp: datetime) -> Path:
        value = as_utc(timestamp)
        return (
            self.data_root
            / "open_interest_snapshot"
            / f"year={value:%Y}"
            / f"date={value:%Y-%m-%d}"
            / f"open_interest_snapshot-{value:%Y-%m-%d}.parquet"
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

    def _merge(self, path: Path, incoming: pl.DataFrame) -> None:
        normalized = canonicalize(incoming, OPEN_INTEREST_SNAPSHOT_SCHEMA)
        if normalized.is_empty():
            return
        with self._partition_lock(path):
            frames = [normalized]
            if path.exists():
                frames.insert(
                    0,
                    canonicalize(pl.read_parquet(path), OPEN_INTEREST_SNAPSHOT_SCHEMA),
                )
            merged = (
                pl.concat(frames, how="vertical_relaxed")
                .unique(subset=SNAPSHOT_PRIMARY_KEY, keep="last")
                .sort(["snapshot_time", "venue", "instrument_id"])
            )
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                merged.write_parquet(temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def write_snapshots(self, frame: pl.DataFrame) -> None:
        normalized = canonicalize(frame, OPEN_INTEREST_SNAPSHOT_SCHEMA)
        if normalized.is_empty():
            return
        dates = normalized.get_column("snapshot_time").dt.date().unique().to_list()
        for date_value in dates:
            partition = normalized.filter(pl.col("snapshot_time").dt.date() == date_value)
            timestamp = partition.get_column("snapshot_time")[0]
            assert isinstance(timestamp, datetime)
            self._merge(self.snapshot_path(timestamp), partition)
