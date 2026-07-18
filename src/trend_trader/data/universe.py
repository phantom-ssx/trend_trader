"""Point-in-time instrument metadata and tradable-universe maintenance."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import httpx
import polars as pl

from trend_trader.data.models import as_utc

OKX_REST_BASE_URL = "https://www.okx.com"
OKX_INSTRUMENTS_PATH = "/api/v5/public/instruments"
OKX_TICKERS_PATH = "/api/v5/market/tickers"
OKX_OPEN_INTEREST_PATH = "/api/v5/public/open-interest"


INSTRUMENT_SNAPSHOT_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "instrument_type": pl.Utf8,
    "instrument_family": pl.Utf8,
    "base_currency": pl.Utf8,
    "quote_currency": pl.Utf8,
    "settle_currency": pl.Utf8,
    "contract_type": pl.Utf8,
    "state": pl.Utf8,
    "list_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "expiration_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "contract_value": pl.Float64,
    "contract_value_currency": pl.Utf8,
    "tick_size": pl.Float64,
    "lot_size": pl.Float64,
    "min_size": pl.Float64,
    "last_price": pl.Float64,
    "bid_price": pl.Float64,
    "ask_price": pl.Float64,
    "spread_bps": pl.Float64,
    "volume_24h": pl.Float64,
    "volume_currency_24h": pl.Float64,
    "volume_usd_24h": pl.Float64,
    "open_interest_usd": pl.Float64,
}

INSTRUMENT_LIFECYCLE_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "instrument_type": pl.Utf8,
    "base_currency": pl.Utf8,
    "quote_currency": pl.Utf8,
    "settle_currency": pl.Utf8,
    "contract_type": pl.Utf8,
    "valid_from": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "valid_to": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "first_seen": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "last_seen": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "valid_from_source": pl.Utf8,
    "valid_to_source": pl.Utf8,
    "confidence": pl.Utf8,
}

UNIVERSE_SNAPSHOT_SCHEMA = {
    "universe_name": pl.Utf8,
    "venue": pl.Utf8,
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "source_snapshot_timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "instrument_id": pl.Utf8,
    "rank": pl.UInt32,
    "list_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "volume_usd_24h": pl.Float64,
    "open_interest_usd": pl.Float64,
    "spread_bps": pl.Float64,
    "config_hash": pl.Utf8,
}


def _empty(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _canonicalize(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    for name, dtype in schema.items():
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(name))
    expressions: list[pl.Expr] = []
    for name, dtype in schema.items():
        current = frame.schema[name]
        expression = pl.col(name)
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


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _optional_datetime(value: object) -> datetime | None:
    number = _optional_float(value)
    if number is None or number <= 0:
        return None
    return datetime.fromtimestamp(number / 1000, tz=UTC)


def _currencies(instrument: dict[str, object]) -> tuple[str, str]:
    instrument_id = str(instrument.get("instId") or "")
    parts = instrument_id.split("-")
    base = str(instrument.get("baseCcy") or (parts[0] if parts else "")).upper()
    quote = str(
        instrument.get("quoteCcy")
        or (parts[1] if len(parts) >= 2 else instrument.get("settleCcy") or "")
    ).upper()
    return base, quote


def build_okx_instrument_snapshot(
    instruments: list[dict[str, object]],
    tickers: list[dict[str, object]],
    open_interest: list[dict[str, object]],
    timestamp: datetime | str,
) -> pl.DataFrame:
    """Normalize the three OKX cross-sectional endpoints into one snapshot."""

    captured_at = as_utc(timestamp)
    ticker_by_id = {str(row.get("instId")): row for row in tickers}
    oi_by_id = {str(row.get("instId")): row for row in open_interest}
    rows: list[dict[str, object]] = []
    for instrument in instruments:
        instrument_id = str(instrument.get("instId") or "")
        if not instrument_id:
            continue
        ticker = ticker_by_id.get(instrument_id, {})
        interest = oi_by_id.get(instrument_id, {})
        base_currency, quote_currency = _currencies(instrument)
        last_price = _optional_float(ticker.get("last"))
        bid_price = _optional_float(ticker.get("bidPx"))
        ask_price = _optional_float(ticker.get("askPx"))
        middle = (
            (bid_price + ask_price) / 2
            if bid_price is not None and ask_price is not None
            else None
        )
        spread_bps = (
            (ask_price - bid_price) / middle * 10_000
            if middle is not None and middle > 0
            else None
        )
        volume_currency = _optional_float(ticker.get("volCcy24h"))
        settle_currency = str(instrument.get("settleCcy") or quote_currency).upper()
        volume_usd = (
            volume_currency * last_price
            if volume_currency is not None
            and last_price is not None
            and settle_currency in {"USD", "USDT", "USDC"}
            else None
        )
        rows.append(
            {
                "venue": "OKX",
                "instrument_id": instrument_id,
                "timestamp": captured_at,
                "instrument_type": instrument.get("instType") or "",
                "instrument_family": instrument.get("instFamily") or "",
                "base_currency": base_currency,
                "quote_currency": quote_currency,
                "settle_currency": settle_currency,
                "contract_type": instrument.get("ctType") or "",
                "state": instrument.get("state") or "",
                "list_time": _optional_datetime(instrument.get("listTime")),
                "expiration_time": _optional_datetime(instrument.get("expTime")),
                "contract_value": _optional_float(instrument.get("ctVal")),
                "contract_value_currency": instrument.get("ctValCcy") or "",
                "tick_size": _optional_float(instrument.get("tickSz")),
                "lot_size": _optional_float(instrument.get("lotSz")),
                "min_size": _optional_float(instrument.get("minSz")),
                "last_price": last_price,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "spread_bps": spread_bps,
                "volume_24h": _optional_float(ticker.get("vol24h")),
                "volume_currency_24h": volume_currency,
                "volume_usd_24h": volume_usd,
                "open_interest_usd": _optional_float(interest.get("oiUsd")),
            }
        )
    if not rows:
        return _empty(INSTRUMENT_SNAPSHOT_SCHEMA)
    return _canonicalize(
        pl.DataFrame(rows, infer_schema_length=None), INSTRUMENT_SNAPSHOT_SCHEMA
    ).sort("instrument_id")


class InstrumentSource(Protocol):
    name: str

    def supports(self, venue: str) -> bool: ...

    async def fetch_snapshot(
        self,
        *,
        venue: str,
        instrument_type: str,
        timestamp: datetime,
    ) -> pl.DataFrame: ...


class OkxInstrumentSource:
    """OKX public instrument catalog enriched with ticker and open-interest data."""

    name = "okx-instruments"

    def supports(self, venue: str) -> bool:
        return venue.upper() == "OKX"

    async def fetch_snapshot(
        self,
        *,
        venue: str,
        instrument_type: str,
        timestamp: datetime,
    ) -> pl.DataFrame:
        if not self.supports(venue):
            raise ValueError(f"unsupported venue: {venue}")
        if abs(datetime.now(tz=UTC) - timestamp) > timedelta(minutes=5):
            raise ValueError(
                "OKX instrument endpoints are current-only and cannot be backdated; "
                "use stored snapshots or rebuild lifecycle from historical candles"
            )
        params = {"instType": instrument_type.upper()}
        async with httpx.AsyncClient(base_url=OKX_REST_BASE_URL, timeout=30) as client:
            responses = await asyncio.gather(
                client.get(OKX_INSTRUMENTS_PATH, params=params),
                client.get(OKX_TICKERS_PATH, params=params),
                client.get(OKX_OPEN_INTEREST_PATH, params=params),
            )
        payloads: list[list[dict[str, object]]] = []
        for response in responses:
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != "0":
                raise RuntimeError(f"OKX API error {payload.get('code')}: {payload.get('msg')}")
            data = payload.get("data", [])
            payloads.append([row for row in data if isinstance(row, dict)])
        return build_okx_instrument_snapshot(*payloads, timestamp)


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    name: str = "okx_usdt_linear_swaps"
    venue: str = "OKX"
    instrument_type: str = "SWAP"
    settle_currency: str = "USDT"
    contract_type: str = "linear"
    states: tuple[str, ...] = ("live",)
    min_listing_days: int = 30
    min_volume_usd_24h: float = 20_000_000
    min_open_interest_usd: float = 0
    max_spread_bps: float | None = 50
    top_n: int = 30

    def __post_init__(self) -> None:
        if self.min_listing_days < 0:
            raise ValueError("min_listing_days cannot be negative")
        if self.min_volume_usd_24h < 0 or self.min_open_interest_usd < 0:
            raise ValueError("liquidity thresholds cannot be negative")
        if self.max_spread_bps is not None and self.max_spread_bps < 0:
            raise ValueError("max_spread_bps cannot be negative")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        object.__setattr__(self, "venue", self.venue.upper())
        object.__setattr__(self, "instrument_type", self.instrument_type.upper())
        object.__setattr__(self, "settle_currency", self.settle_currency.upper())
        object.__setattr__(self, "contract_type", self.contract_type.lower())
        object.__setattr__(self, "states", tuple(state.lower() for state in self.states))

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class InstrumentRepository:
    """Atomic Parquet storage for instrument, lifecycle, and universe snapshots."""

    def __init__(self, data_root: Path | str) -> None:
        self.data_root = Path(data_root)

    def _instrument_path(self, venue: str, timestamp: datetime) -> Path:
        return (
            self.data_root
            / "instruments"
            / f"venue={venue.upper()}"
            / f"date={timestamp:%Y-%m-%d}"
            / "data.parquet"
        )

    def _lifecycle_path(self, venue: str) -> Path:
        return (
            self.data_root
            / "instrument_lifecycle"
            / f"venue={venue.upper()}"
            / "data.parquet"
        )

    def _universe_path(self, name: str, venue: str, timestamp: datetime) -> Path:
        return (
            self.data_root
            / "universes"
            / f"name={name}"
            / f"venue={venue.upper()}"
            / f"date={timestamp:%Y-%m-%d}"
            / "data.parquet"
        )

    @staticmethod
    def _write_atomic(
        path: Path,
        incoming: pl.DataFrame,
        *,
        schema: dict[str, pl.DataType],
        primary_key: list[str],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                frames = [_canonicalize(incoming, schema)]
                if path.exists():
                    frames.insert(0, _canonicalize(pl.read_parquet(path), schema))
                merged = (
                    pl.concat(frames, how="vertical_relaxed")
                    .unique(subset=primary_key, keep="last")
                    .sort(primary_key)
                )
                temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
                try:
                    merged.write_parquet(temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _replace_atomic(
        path: Path,
        frame: pl.DataFrame,
        *,
        schema: dict[str, pl.DataType],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
                try:
                    _canonicalize(frame, schema).write_parquet(temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def save_instrument_snapshot(self, frame: pl.DataFrame) -> None:
        frame = _canonicalize(frame, INSTRUMENT_SNAPSHOT_SCHEMA)
        if frame.is_empty():
            raise ValueError("cannot save an empty instrument snapshot")
        for (venue, _day), partition in frame.with_columns(
            pl.col("timestamp").dt.date().alias("_date")
        ).group_by("venue", "_date"):
            timestamp = partition["timestamp"][0]
            assert isinstance(timestamp, datetime)
            self._write_atomic(
                self._instrument_path(str(venue), timestamp),
                partition.drop("_date"),
                schema=INSTRUMENT_SNAPSHOT_SCHEMA,
                primary_key=["venue", "instrument_id", "timestamp"],
            )

    def read_instrument_snapshots(self, venue: str) -> pl.DataFrame:
        root = self.data_root / "instruments" / f"venue={venue.upper()}"
        paths = sorted(root.glob("date=*/data.parquet"))
        if not paths:
            return _empty(INSTRUMENT_SNAPSHOT_SCHEMA)
        return _canonicalize(pl.read_parquet(paths), INSTRUMENT_SNAPSHOT_SCHEMA).sort(
            "timestamp", "instrument_id"
        )

    def instrument_snapshot_at(self, venue: str, timestamp: datetime | str) -> pl.DataFrame:
        as_of = as_utc(timestamp)
        snapshots = self.read_instrument_snapshots(venue).filter(pl.col("timestamp") <= as_of)
        if snapshots.is_empty():
            return snapshots
        latest = snapshots["timestamp"].max()
        return snapshots.filter(pl.col("timestamp") == latest).sort("instrument_id")

    def read_lifecycle(self, venue: str) -> pl.DataFrame:
        path = self._lifecycle_path(venue)
        if not path.exists():
            return _empty(INSTRUMENT_LIFECYCLE_SCHEMA)
        return _canonicalize(pl.read_parquet(path), INSTRUMENT_LIFECYCLE_SCHEMA).sort(
            "instrument_id"
        )

    def _candle_lifetimes(self, venue: str) -> dict[str, tuple[datetime, datetime]]:
        root = self.data_root / "candles" / f"venue={venue.upper()}"
        result: dict[str, tuple[datetime, datetime]] = {}
        if not root.exists():
            return result
        for instrument_root in root.glob("instrument_id=*"):
            instrument_id = instrument_root.name.split("=", maxsplit=1)[1]
            paths = list(instrument_root.glob("bar_type=1m/year=*/month=*/data.parquet"))
            if not paths:
                continue
            bounds = (
                pl.scan_parquet(paths)
                .select(
                    pl.col("timestamp").min().alias("start"),
                    pl.col("timestamp").max().alias("end"),
                )
                .collect()
            )
            start, end = bounds.row(0)
            if isinstance(start, datetime) and isinstance(end, datetime):
                result[instrument_id] = (as_utc(start), as_utc(end) + timedelta(minutes=1))
        return result

    def rebuild_lifecycle(self, venue: str, *, include_candles: bool = True) -> pl.DataFrame:
        venue = venue.upper()
        snapshots = self.read_instrument_snapshots(venue)
        snapshot_times = sorted(snapshots["timestamp"].unique().to_list())
        records: dict[str, dict[str, object]] = {}
        if not snapshots.is_empty():
            for instrument_id in snapshots["instrument_id"].unique().to_list():
                history = snapshots.filter(pl.col("instrument_id") == instrument_id).sort(
                    "timestamp"
                )
                first = history.row(0, named=True)
                last = history.row(-1, named=True)
                list_times = [value for value in history["list_time"].to_list() if value]
                expiration_times = [
                    value for value in history["expiration_time"].to_list() if value
                ]
                first_seen = as_utc(first["timestamp"])
                last_seen = as_utc(last["timestamp"])
                valid_from = min(list_times) if list_times else first_seen
                if expiration_times:
                    valid_to = min(expiration_times)
                    valid_to_source = "exchange_expiration_time"
                elif snapshot_times and last_seen < snapshot_times[-1]:
                    valid_to = next(value for value in snapshot_times if value > last_seen)
                    valid_to_source = "first_missing_snapshot"
                else:
                    valid_to = None
                    valid_to_source = ""
                records[str(instrument_id)] = {
                    "venue": venue,
                    "instrument_id": instrument_id,
                    "instrument_type": last["instrument_type"],
                    "base_currency": last["base_currency"],
                    "quote_currency": last["quote_currency"],
                    "settle_currency": last["settle_currency"],
                    "contract_type": last["contract_type"],
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "valid_from_source": "exchange_list_time" if list_times else "first_seen",
                    "valid_to_source": valid_to_source,
                    "confidence": (
                        "exact"
                        if list_times and valid_to_source != "first_missing_snapshot"
                        else "inferred"
                    ),
                }

        if include_candles:
            for instrument_id, (start, end) in self._candle_lifetimes(venue).items():
                existing = records.get(instrument_id)
                if existing is None:
                    parts = instrument_id.split("-")
                    records[instrument_id] = {
                        "venue": venue,
                        "instrument_id": instrument_id,
                        "instrument_type": "SWAP" if instrument_id.endswith("-SWAP") else "",
                        "base_currency": parts[0] if parts else "",
                        "quote_currency": parts[1] if len(parts) >= 2 else "",
                        "settle_currency": parts[1] if len(parts) >= 2 else "",
                        "contract_type": (
                            "linear"
                            if len(parts) >= 2 and parts[1] in {"USDT", "USDC"}
                            else ""
                        ),
                        "valid_from": start,
                        "valid_to": end,
                        "first_seen": start,
                        "last_seen": end - timedelta(minutes=1),
                        "valid_from_source": "first_candle",
                        "valid_to_source": "last_candle",
                        "confidence": "inferred",
                    }
                else:
                    if (
                        existing["valid_from_source"] != "exchange_list_time"
                        and start < existing["valid_from"]
                    ):
                        existing["valid_from"] = start
                        existing["valid_from_source"] = "first_candle"
                    existing["first_seen"] = min(existing["first_seen"], start)
                    if (
                        existing["valid_to"] is not None
                        and existing["valid_to_source"] == "first_missing_snapshot"
                    ):
                        existing["valid_to"] = max(existing["valid_to"], end)
                    existing["last_seen"] = max(
                        existing["last_seen"], end - timedelta(minutes=1)
                    )

        if not records:
            return _empty(INSTRUMENT_LIFECYCLE_SCHEMA)
        lifecycle = _canonicalize(
            pl.DataFrame(list(records.values()), infer_schema_length=None),
            INSTRUMENT_LIFECYCLE_SCHEMA,
        ).sort("instrument_id")
        self._replace_atomic(
            self._lifecycle_path(venue),
            lifecycle,
            schema=INSTRUMENT_LIFECYCLE_SCHEMA,
        )
        return lifecycle

    def save_universe(self, frame: pl.DataFrame) -> None:
        frame = _canonicalize(frame, UNIVERSE_SNAPSHOT_SCHEMA)
        if frame.is_empty():
            return
        for (name, venue, _day), partition in frame.with_columns(
            pl.col("timestamp").dt.date().alias("_date")
        ).group_by("universe_name", "venue", "_date"):
            timestamp = partition["timestamp"][0]
            assert isinstance(timestamp, datetime)
            self._write_atomic(
                self._universe_path(str(name), str(venue), timestamp),
                partition.drop("_date"),
                schema=UNIVERSE_SNAPSHOT_SCHEMA,
                primary_key=["universe_name", "venue", "timestamp", "instrument_id"],
            )

    def read_universe(
        self,
        name: str,
        venue: str,
        timestamp: datetime | str,
        *,
        config_hash: str | None = None,
    ) -> pl.DataFrame:
        root = self.data_root / "universes" / f"name={name}" / f"venue={venue.upper()}"
        paths = sorted(root.glob("date=*/data.parquet"))
        if not paths:
            return _empty(UNIVERSE_SNAPSHOT_SCHEMA)
        as_of = as_utc(timestamp)
        frame = _canonicalize(pl.read_parquet(paths), UNIVERSE_SNAPSHOT_SCHEMA).filter(
            pl.col("timestamp") <= as_of
        )
        if config_hash is not None:
            frame = frame.filter(pl.col("config_hash") == config_hash)
        if frame.is_empty():
            return frame
        latest = frame["timestamp"].max()
        return frame.filter(pl.col("timestamp") == latest).sort("rank")


class UniverseSelector:
    def __init__(self, repository: InstrumentRepository) -> None:
        self.repository = repository

    def _lifecycle_candidates(self, config: UniverseConfig, timestamp: datetime) -> pl.DataFrame:
        lifecycle = self.repository.read_lifecycle(config.venue).filter(
            (pl.col("valid_from") <= timestamp)
            & (pl.col("valid_to").is_null() | (pl.col("valid_to") > timestamp))
        )
        if lifecycle.is_empty():
            return _empty(INSTRUMENT_SNAPSHOT_SCHEMA)
        return _canonicalize(
            lifecycle.select(
                "venue",
                "instrument_id",
                pl.lit(None).alias("timestamp"),
                "instrument_type",
                pl.lit("").alias("instrument_family"),
                "base_currency",
                "quote_currency",
                "settle_currency",
                "contract_type",
                pl.lit("live").alias("state"),
                pl.col("valid_from").alias("list_time"),
            ),
            INSTRUMENT_SNAPSHOT_SCHEMA,
        )

    def _add_local_volume(
        self, candidates: pl.DataFrame, timestamp: datetime
    ) -> pl.DataFrame:
        if candidates.is_empty() or candidates["volume_usd_24h"].is_not_null().any():
            return candidates
        volumes: dict[str, float] = {}
        start = timestamp - timedelta(days=1)
        for instrument_id in candidates["instrument_id"].to_list():
            root = (
                self.repository.data_root
                / "candles"
                / f"venue={candidates['venue'][0]}"
                / f"instrument_id={instrument_id}"
                / "bar_type=1m"
            )
            paths = list(root.glob("year=*/month=*/data.parquet"))
            if not paths:
                continue
            frame = (
                pl.scan_parquet(paths)
                .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < timestamp))
                .select(pl.col("volume_quote").sum())
                .collect()
            )
            value = frame["volume_quote"][0]
            if value is not None:
                volumes[str(instrument_id)] = float(value)
        if not volumes:
            return candidates
        metrics = pl.DataFrame(
            {"instrument_id": list(volumes), "_local_volume": list(volumes.values())}
        )
        return (
            candidates.join(metrics, on="instrument_id", how="left")
            .with_columns(
                pl.coalesce("volume_usd_24h", "_local_volume").alias("volume_usd_24h")
            )
            .drop("_local_volume")
        )

    def select(self, config: UniverseConfig, timestamp: datetime | str) -> pl.DataFrame:
        as_of = as_utc(timestamp)
        candidates = self.repository.instrument_snapshot_at(config.venue, as_of)
        if candidates.is_empty():
            candidates = self._lifecycle_candidates(config, as_of)
        candidates = self._add_local_volume(candidates, as_of)
        if candidates.is_empty():
            return _empty(UNIVERSE_SNAPSHOT_SCHEMA)
        source_timestamp = candidates["timestamp"].max()
        minimum_list_time = as_of - timedelta(days=config.min_listing_days)
        predicate = (
            (pl.col("instrument_type") == config.instrument_type)
            & (pl.col("settle_currency") == config.settle_currency)
            & (pl.col("contract_type").str.to_lowercase() == config.contract_type)
            & (pl.col("state").str.to_lowercase().is_in(config.states))
            & pl.col("list_time").is_not_null()
            & (pl.col("list_time") <= minimum_list_time)
            & (
                pl.col("expiration_time").is_null()
                | (pl.col("expiration_time") > as_of)
            )
            & pl.col("volume_usd_24h").fill_null(0).ge(config.min_volume_usd_24h)
            & pl.col("open_interest_usd").fill_null(0).ge(config.min_open_interest_usd)
        )
        if config.max_spread_bps is not None:
            predicate &= pl.col("spread_bps").is_null() | (
                pl.col("spread_bps") <= config.max_spread_bps
            )
        selected = (
            candidates.filter(predicate)
            .sort(
                ["volume_usd_24h", "open_interest_usd", "instrument_id"],
                descending=[True, True, False],
                nulls_last=True,
            )
            .head(config.top_n)
            .with_row_index("rank", offset=1)
        )
        if selected.is_empty():
            return _empty(UNIVERSE_SNAPSHOT_SCHEMA)
        return _canonicalize(
            selected.select(
                pl.lit(config.name).alias("universe_name"),
                "venue",
                pl.lit(as_of).alias("timestamp"),
                pl.lit(source_timestamp).alias("source_snapshot_timestamp"),
                "instrument_id",
                "rank",
                "list_time",
                "volume_usd_24h",
                "open_interest_usd",
                "spread_bps",
                pl.lit(config.fingerprint).alias("config_hash"),
            ),
            UNIVERSE_SNAPSHOT_SCHEMA,
        ).sort("rank")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh and persist a point-in-time universe")
    parser.add_argument("--venue", default="OKX")
    parser.add_argument("--instrument-type", default="SWAP")
    parser.add_argument("--settle-currency", default="USDT")
    parser.add_argument("--contract-type", default="linear")
    parser.add_argument("--name", default="okx_usdt_linear_swaps")
    parser.add_argument("--min-listing-days", type=int, default=30)
    parser.add_argument("--min-volume-usd-24h", type=float, default=20_000_000)
    parser.add_argument("--min-open-interest-usd", type=float, default=0)
    parser.add_argument("--max-spread-bps", type=float, default=50)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    args = parser.parse_args()

    from trend_trader.data.query import MarketDataClient

    client = MarketDataClient(data_root=args.data_root)
    frame = client.maintain_universe(
        name=args.name,
        venue=args.venue,
        instrument_type=args.instrument_type,
        settle_currency=args.settle_currency,
        contract_type=args.contract_type,
        min_listing_days=args.min_listing_days,
        min_volume_usd_24h=args.min_volume_usd_24h,
        min_open_interest_usd=args.min_open_interest_usd,
        max_spread_bps=args.max_spread_bps,
        top_n=args.top_n,
    )
    print(frame)


if __name__ == "__main__":
    main()
