"""Shared contracts for the market-data query layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

import polars as pl


class DataType(StrEnum):
    CANDLES = "candles"
    FUNDING_RATES = "funding_rates"
    CONTRACT_BASIS = "contract_basis"
    OPEN_INTEREST = "open_interest"
    LONG_SHORT_RATIO = "long_short_ratio"
    MARKET_CAP = "market_cap"
    LIQUIDATIONS = "liquidations"
    TAKER_VOLUME = "taker_volume"


STORED_BAR_TYPES: dict[DataType, str] = {
    DataType.CANDLES: "1m",
    DataType.CONTRACT_BASIS: "1m",
    DataType.OPEN_INTEREST: "5m",
    DataType.LONG_SHORT_RATIO: "5m",
    DataType.MARKET_CAP: "1d",
    DataType.TAKER_VOLUME: "5m",
}


def as_utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def bar_minutes(bar_type: str) -> int:
    normalized = bar_type.strip().lower()
    if len(normalized) < 2 or not normalized[:-1].isdigit():
        raise ValueError(f"unsupported bar_type: {bar_type!r}")
    size = int(normalized[:-1])
    unit = normalized[-1]
    multipliers = {"m": 1, "h": 60, "d": 24 * 60}
    if size <= 0 or unit not in multipliers:
        raise ValueError(f"unsupported bar_type: {bar_type!r}")
    return size * multipliers[unit]


def normalize_bar_type(bar_type: str) -> str:
    bar_minutes(bar_type)
    normalized = bar_type.strip().lower()
    return f"{int(normalized[:-1])}{normalized[-1]}"


@dataclass(frozen=True, slots=True)
class DataQuery:
    """A source-independent query using a UTC ``[start, end)`` time range."""

    data_type: DataType | str
    instrument_id: str
    start: datetime | str
    end: datetime | str
    venue: str = "OKX"
    bar_type: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data_type = DataType(self.data_type)
        start = as_utc(self.start)
        end = as_utc(self.end)
        if end <= start:
            raise ValueError("end must be after start")
        stored_bar_type = STORED_BAR_TYPES.get(data_type)
        if stored_bar_type is not None:
            if self.bar_type is None:
                raise ValueError(f"bar_type is required for {data_type.value} queries")
            normalized_bar_type = normalize_bar_type(self.bar_type)
            requested_minutes = bar_minutes(normalized_bar_type)
            stored_minutes = bar_minutes(stored_bar_type)
            if requested_minutes < stored_minutes or requested_minutes % stored_minutes:
                raise ValueError(
                    f"bar_type must be a multiple of the stored {stored_bar_type} interval"
                )
            duration_seconds = bar_minutes(normalized_bar_type) * 60
            if int(start.timestamp()) % duration_seconds != 0:
                raise ValueError("start must align with bar_type boundaries in UTC")
            if int(end.timestamp()) % duration_seconds != 0:
                raise ValueError("end must align with bar_type boundaries in UTC")
            object.__setattr__(self, "bar_type", normalized_bar_type)
        elif self.bar_type is not None:
            raise ValueError(f"bar_type is not valid for {data_type.value} queries")

        object.__setattr__(self, "data_type", data_type)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "venue", self.venue.upper())
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True, slots=True)
class FetchRequest:
    data_type: DataType
    venue: str
    instrument_id: str
    start: datetime
    end: datetime
    bar_type: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


class DataSource(Protocol):
    name: str

    def supports(self, request: FetchRequest) -> bool: ...

    async def fetch(self, request: FetchRequest) -> pl.DataFrame: ...


class DataUnavailableError(RuntimeError):
    """Raised when local and remote sources cannot fully satisfy a query."""
