"""Schemas, in-memory state, and partitioned Parquet storage for long/short ratios."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

import polars as pl

from trend_trader.data.open_interest_storage import (
    OpenInterestInstrument,
    as_utc,
    canonicalize,
    optional_float,
)

LONG_SHORT_RATIO_SNAPSHOT_SCHEMA = {
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
    # 对齐后的本地 5 分钟采集截面，例如 2026-07-29T08:35:00Z。
    "snapshot_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    # OKX 指标记录自带的 5 分钟时间戳。
    "exchange_ts": pl.Datetime(time_unit="ms", time_zone="UTC"),
    # 本地收到 REST 响应的时间。
    "received_at": pl.Datetime(time_unit="ms", time_zone="UTC"),
    # 原生指标周期；当前 OKX 采集固定为 "5m"。
    "bar_type": pl.Utf8,
    # 指标类别：全市场账户、大户账户或大户持仓量多空比。
    "ratio_type": pl.Utf8,
    # 多头 / 空头的比值。账户类按账户数，持仓类按持仓价值计算。
    "long_short_ratio": pl.Float64,
    # 当前仅为 "rest"；保留来源字段便于后续接入其他采集通道。
    "data_source": pl.Utf8,
    # 按 exchange_ts 判断的时效状态，例如 "fresh" 或 "stale"。
    "data_status": pl.Utf8,
}

SNAPSHOT_PRIMARY_KEY = ["venue", "instrument_id", "ratio_type", "snapshot_time"]
NATIVE_BAR_TYPE = "5m"


class LongShortRatioType(StrEnum):
    ALL_ACCOUNT = "all_account"
    TOP_TRADER_ACCOUNT = "top_trader_account"
    TOP_TRADER_POSITION = "top_trader_position"


SUPPORTED_RATIO_TYPES = tuple(LongShortRatioType)


def floor_five_minutes(value: datetime) -> datetime:
    current = as_utc(value)
    return current.replace(
        minute=current.minute - current.minute % 5,
        second=0,
        microsecond=0,
    )


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=LONG_SHORT_RATIO_SNAPSHOT_SCHEMA)


def canonicalize_snapshots(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize snapshots and treat pre-``ratio_type`` rows as all-account data."""

    normalized = canonicalize(frame, LONG_SHORT_RATIO_SNAPSHOT_SCHEMA).with_columns(
        pl.col("ratio_type").fill_null(LongShortRatioType.ALL_ACCOUNT.value)
    )
    unsupported = set(normalized.get_column("ratio_type").unique().to_list()) - {
        ratio_type.value for ratio_type in SUPPORTED_RATIO_TYPES
    }
    if unsupported:
        raise ValueError(f"unsupported long/short ratio types: {sorted(unsupported)}")
    return normalized


@dataclass(frozen=True, slots=True)
class LongShortRatioState:
    venue: str
    instrument_id: str
    instrument_type: str
    ratio_type: LongShortRatioType
    exchange_ts: datetime
    received_at: datetime
    long_short_ratio: float
    data_source: str = "rest"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratio_type", LongShortRatioType(self.ratio_type))

    @classmethod
    def from_okx(
        cls,
        row: Sequence[object],
        *,
        instrument_id: str,
        instrument_type: str,
        ratio_type: LongShortRatioType | str,
        received_at: datetime,
    ) -> LongShortRatioState:
        if len(row) < 2:
            raise ValueError("OKX long/short-ratio row must contain timestamp and ratio")
        timestamp_ms = optional_float(row[0])
        ratio = optional_float(row[1])
        if timestamp_ms is None or timestamp_ms <= 0:
            raise ValueError("OKX long/short-ratio row has an invalid timestamp")
        if ratio is None or ratio < 0:
            raise ValueError("OKX long/short-ratio row has an invalid ratio")
        normalized_type = instrument_type.upper()
        if normalized_type not in {"SWAP", "FUTURES"}:
            raise ValueError(
                f"unsupported OKX long/short-ratio instrument type: {instrument_type!r}"
            )
        normalized_ratio_type = LongShortRatioType(ratio_type)
        return cls(
            venue="OKX",
            instrument_id=instrument_id.upper(),
            instrument_type=normalized_type,
            ratio_type=normalized_ratio_type,
            exchange_ts=datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC),
            received_at=as_utc(received_at),
            long_short_ratio=ratio,
        )


class LongShortRatioStateCache:
    """Latest native 5-minute value for every instrument and ratio type."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, LongShortRatioType], LongShortRatioState] = {}

    def update(self, state: LongShortRatioState) -> bool:
        key = (state.instrument_id, state.ratio_type)
        current = self._states.get(key)
        if current is not None and state.exchange_ts < current.exchange_ts:
            return False
        self._states[key] = state
        return True

    def retain(self, instrument_ids: set[str]) -> None:
        self._states = {
            key: state for key, state in self._states.items() if key[0] in instrument_ids
        }

    def stale_or_missing(
        self,
        instrument_ids: Iterable[str],
        *,
        now: datetime,
        stale_after: timedelta,
    ) -> list[tuple[str, LongShortRatioType]]:
        current_time = as_utc(now)
        return [
            (instrument_id, ratio_type)
            for instrument_id in instrument_ids
            for ratio_type in SUPPORTED_RATIO_TYPES
            if (
                (state := self._states.get((instrument_id, ratio_type))) is None
                or current_time - state.exchange_ts > stale_after
            )
        ]

    def snapshot(
        self,
        instruments: Mapping[str, OpenInterestInstrument],
        *,
        snapshot_time: datetime,
        stale_after: timedelta,
    ) -> pl.DataFrame:
        aligned_time = floor_five_minutes(snapshot_time)
        rows: list[dict[str, object]] = []
        for instrument_id in sorted(instruments):
            instrument = instruments[instrument_id]
            for ratio_type in SUPPORTED_RATIO_TYPES:
                state = self._states.get((instrument_id, ratio_type))
                if state is None:
                    continue
                age = max(timedelta(0), aligned_time - state.exchange_ts)
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
                        "bar_type": NATIVE_BAR_TYPE,
                        "ratio_type": state.ratio_type.value,
                        "long_short_ratio": state.long_short_ratio,
                        "data_source": state.data_source,
                        "data_status": "fresh" if age <= stale_after else "stale",
                    }
                )
        if not rows:
            return _empty()
        return canonicalize_snapshots(pl.DataFrame(rows, infer_schema_length=None)).sort(
            ["instrument_id", "ratio_type"]
        )


class LongShortRatioParquetRepository:
    """Atomic daily Parquet partitions for contract-level 5-minute snapshots."""

    def __init__(self, data_root: Path | str) -> None:
        self.data_root = Path(data_root)

    def snapshot_path(self, timestamp: datetime) -> Path:
        value = as_utc(timestamp)
        return (
            self.data_root
            / "long_short_ratio_snapshot"
            / f"year={value:%Y}"
            / f"date={value:%Y-%m-%d}"
            / f"long_short_ratio_snapshot-{value:%Y-%m-%d}.parquet"
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
        normalized = canonicalize_snapshots(incoming)
        if normalized.is_empty():
            return
        with self._partition_lock(path):
            frames = [normalized]
            if path.exists():
                frames.insert(
                    0,
                    canonicalize_snapshots(pl.read_parquet(path)),
                )
            merged = (
                pl.concat(frames, how="vertical_relaxed")
                .unique(subset=SNAPSHOT_PRIMARY_KEY, keep="last")
                .sort(["snapshot_time", "venue", "instrument_id", "ratio_type"])
            )
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                merged.write_parquet(temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def write_snapshots(self, frame: pl.DataFrame) -> None:
        normalized = canonicalize_snapshots(frame)
        if normalized.is_empty():
            return
        dates = normalized.get_column("snapshot_time").dt.date().unique().to_list()
        for date_value in dates:
            partition = normalized.filter(pl.col("snapshot_time").dt.date() == date_value)
            timestamp = partition.get_column("snapshot_time")[0]
            assert isinstance(timestamp, datetime)
            self._merge(self.snapshot_path(timestamp), partition)
