"""Unified Python API for querying market data."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import polars as pl


class DataType(StrEnum):
    """Market datasets exposed by the query layer."""

    CANDLES = "candles"
    FUNDING_RATES = "funding_rates"


REQUIRED_COLUMNS: dict[DataType, frozenset[str]] = {
    DataType.CANDLES: frozenset({"ts", "open", "high", "low", "close", "volume"}),
    DataType.FUNDING_RATES: frozenset({"ts", "funding_rate"}),
}


def _as_utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class DataQuery:
    """A source-independent market-data query.

    Time ranges are consistently interpreted as ``[start, end)`` in UTC.
    ``options`` is reserved for source-specific tuning such as OKX concurrency.
    """

    data_type: DataType | str
    inst_id: str
    start: datetime | str
    end: datetime | str
    exchange: str = "OKX"
    bar: str | None = None
    path: Path | str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data_type = DataType(self.data_type)
        start = _as_utc(self.start)
        end = _as_utc(self.end)
        if end <= start:
            raise ValueError("end must be after start")
        if data_type is DataType.CANDLES and not self.bar:
            raise ValueError("bar is required for candle queries")
        if data_type is DataType.FUNDING_RATES and self.bar is not None:
            raise ValueError("bar is only valid for candle queries")

        object.__setattr__(self, "data_type", data_type)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "exchange", self.exchange.upper())
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "options", dict(self.options))


class DataSource(Protocol):
    """Adapter contract implemented by local and remote data sources."""

    name: str

    def supports(self, query: DataQuery) -> bool: ...

    async def load(self, query: DataQuery) -> pl.DataFrame: ...


class ParquetDataSource:
    """Load either supported dataset from a local Parquet file."""

    name = "parquet"

    def supports(self, query: DataQuery) -> bool:
        return query.path is not None

    async def load(self, query: DataQuery) -> pl.DataFrame:
        if query.path is None:
            raise ValueError("path is required for the parquet source")
        if query.options:
            names = ", ".join(sorted(query.options))
            raise ValueError(f"parquet source does not accept options: {names}")

        lazy = pl.scan_parquet(query.path)
        columns = set(lazy.collect_schema().names())
        if "ts" not in columns:
            raise ValueError("Parquet file is missing columns: ['ts']")

        predicate = (pl.col("ts") >= query.start) & (pl.col("ts") < query.end)
        if "exchange" in columns:
            predicate &= pl.col("exchange").str.to_uppercase() == query.exchange
        if "inst_id" in columns:
            predicate &= pl.col("inst_id") == query.inst_id
        if query.bar is not None and "bar" in columns:
            predicate &= pl.col("bar") == query.bar
        return lazy.filter(predicate).sort("ts").collect()


class OkxRestDataSource:
    """Load supported datasets from the public OKX REST API."""

    name = "okx-rest"

    def supports(self, query: DataQuery) -> bool:
        return query.exchange == "OKX" and query.path is None

    async def load(self, query: DataQuery) -> pl.DataFrame:
        if query.data_type is DataType.CANDLES:
            from trend_trader.data.okx_candles import fetch_okx_history_candles_chunked

            allowed = {"chunk_days", "concurrency", "max_requests_per_second"}
            unknown = set(query.options).difference(allowed)
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"unsupported OKX candle options: {names}")
            return await fetch_okx_history_candles_chunked(
                query.inst_id,
                query.bar or "",
                query.start,
                query.end,
                **query.options,
            )

        from trend_trader.data.okx_funding_rates import fetch_funding_rates

        if query.options:
            names = ", ".join(sorted(query.options))
            raise ValueError(f"OKX funding-rate source does not accept options: {names}")
        return await fetch_funding_rates(query.inst_id, query.start, query.end)


class MarketDataClient:
    """Registry-backed facade providing one API for all market datasets."""

    def __init__(self, sources: list[DataSource] | None = None) -> None:
        self._sources: dict[str, DataSource] = {}
        for source in sources or [ParquetDataSource(), OkxRestDataSource()]:
            self.register(source)

    def register(self, source: DataSource, *, replace: bool = False) -> None:
        if source.name in self._sources and not replace:
            raise ValueError(f"data source already registered: {source.name}")
        self._sources[source.name] = source

    async def query_async(self, query: DataQuery, *, source: str | None = None) -> pl.DataFrame:
        adapter = self._resolve_source(query, source)
        frame = await adapter.load(query)
        self._validate_result(query, frame)
        return frame.sort("ts")

    def query(self, query: DataQuery, *, source: str | None = None) -> pl.DataFrame:
        """Execute a query from synchronous Python code.

        Async applications should call :meth:`query_async` instead.
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.query_async(query, source=source))
        raise RuntimeError("query() cannot run inside an event loop; use await query_async()")

    def candles(
        self,
        inst_id: str,
        bar: str,
        start: datetime | str,
        end: datetime | str,
        *,
        exchange: str = "OKX",
        source: str | None = None,
        path: Path | str | None = None,
        **options: Any,
    ) -> pl.DataFrame:
        return self.query(
            DataQuery(
                DataType.CANDLES,
                inst_id,
                start,
                end,
                exchange=exchange,
                bar=bar,
                path=path,
                options=options,
            ),
            source=source,
        )

    def funding_rates(
        self,
        inst_id: str,
        start: datetime | str,
        end: datetime | str,
        *,
        exchange: str = "OKX",
        source: str | None = None,
        path: Path | str | None = None,
        **options: Any,
    ) -> pl.DataFrame:
        return self.query(
            DataQuery(
                DataType.FUNDING_RATES,
                inst_id,
                start,
                end,
                exchange=exchange,
                path=path,
                options=options,
            ),
            source=source,
        )

    def _resolve_source(self, query: DataQuery, source: str | None) -> DataSource:
        if source is not None:
            try:
                adapter = self._sources[source]
            except KeyError as exc:
                available = ", ".join(sorted(self._sources))
                raise ValueError(f"unknown data source {source!r}; available: {available}") from exc
            if not adapter.supports(query):
                raise ValueError(f"data source {source!r} does not support this query")
            return adapter

        candidates = [adapter for adapter in self._sources.values() if adapter.supports(query)]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError("no registered data source supports this query")
        names = ", ".join(adapter.name for adapter in candidates)
        raise ValueError(f"multiple data sources support this query; choose source from: {names}")

    @staticmethod
    def _validate_result(query: DataQuery, frame: pl.DataFrame) -> None:
        required = REQUIRED_COLUMNS[query.data_type]
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{query.data_type.value} result is missing columns: {missing}")


default_client = MarketDataClient()


def query(query: DataQuery, *, source: str | None = None) -> pl.DataFrame:
    """Query market data through the process-wide default client."""

    return default_client.query(query, source=source)


async def query_async(query: DataQuery, *, source: str | None = None) -> pl.DataFrame:
    """Asynchronously query market data through the process-wide default client."""

    return await default_client.query_async(query, source=source)
