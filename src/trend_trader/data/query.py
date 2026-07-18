"""Local-first, source-independent market-data query API."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from trend_trader.data.catalog import DataCatalog
from trend_trader.data.coingecko import CoinGeckoDataSource
from trend_trader.data.models import (
    STORED_BAR_TYPES,
    DataQuery,
    DataSource,
    DataType,
    DataUnavailableError,
    FetchRequest,
    as_utc,
    bar_minutes,
)
from trend_trader.data.schema import canonicalize_frame, validate_frame
from trend_trader.data.sources import OkxRestDataSource
from trend_trader.data.store import ParquetStore, iter_partitions
from trend_trader.data.universe import (
    InstrumentRepository,
    InstrumentSource,
    OkxInstrumentSource,
    UniverseConfig,
    UniverseSelector,
)

DEFAULT_DATA_ROOT = Path("data/market/v1")
DEFAULT_LEGACY_DATA_ROOT = Path("data/clean")


class MarketDataClient:
    """Read-through local cache for all supported market datasets."""

    def __init__(
        self,
        *,
        data_root: Path | str = DEFAULT_DATA_ROOT,
        sources: list[DataSource] | None = None,
        instrument_sources: list[InstrumentSource] | None = None,
        legacy_data_root: Path | str | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        catalog_path = self.data_root / "catalog.sqlite"
        catalog_existed = catalog_path.exists()
        self.catalog = DataCatalog(catalog_path)
        self.store = ParquetStore(self.data_root, self.catalog)
        self.instruments_repository = InstrumentRepository(self.data_root)
        self.universe_selector = UniverseSelector(self.instruments_repository)
        if not catalog_existed:
            self.store.rebuild_catalog()
        if legacy_data_root is None and self.data_root == DEFAULT_DATA_ROOT:
            self.legacy_data_root: Path | None = DEFAULT_LEGACY_DATA_ROOT
        else:
            self.legacy_data_root = Path(legacy_data_root) if legacy_data_root else None
        self._sources = (
            list(sources)
            if sources is not None
            else [OkxRestDataSource(), CoinGeckoDataSource()]
        )
        self._instrument_sources = (
            list(instrument_sources)
            if instrument_sources is not None
            else [OkxInstrumentSource()]
        )

    def register(self, source: DataSource, *, prepend: bool = False) -> None:
        if any(candidate.name == source.name for candidate in self._sources):
            raise ValueError(f"data source already registered: {source.name}")
        if prepend:
            self._sources.insert(0, source)
        else:
            self._sources.append(source)

    def register_instrument_source(
        self, source: InstrumentSource, *, prepend: bool = False
    ) -> None:
        if any(candidate.name == source.name for candidate in self._instrument_sources):
            raise ValueError(f"instrument source already registered: {source.name}")
        if prepend:
            self._instrument_sources.insert(0, source)
        else:
            self._instrument_sources.append(source)

    async def refresh_instruments_async(
        self,
        *,
        venue: str = "OKX",
        instrument_type: str = "SWAP",
        timestamp: datetime | str | None = None,
        include_candle_history: bool = True,
    ) -> pl.DataFrame:
        """Fetch and atomically persist a complete point-in-time instrument snapshot."""

        captured_at = as_utc(timestamp or datetime.now(tz=UTC))
        normalized_venue = venue.upper()
        candidates = [
            source
            for source in self._instrument_sources
            if source.supports(normalized_venue)
        ]
        if not candidates:
            raise DataUnavailableError(
                f"no instrument source supports {normalized_venue}/{instrument_type.upper()}"
            )
        errors: list[str] = []
        for source in candidates:
            try:
                frame = await source.fetch_snapshot(
                    venue=normalized_venue,
                    instrument_type=instrument_type.upper(),
                    timestamp=captured_at,
                )
                if frame.is_empty():
                    raise ValueError("source returned an empty instrument snapshot")
                self.instruments_repository.save_instrument_snapshot(frame)
                self.instruments_repository.rebuild_lifecycle(
                    normalized_venue,
                    include_candles=include_candle_history,
                )
                return frame.sort("instrument_id")
            except Exception as exc:  # noqa: BLE001 - try the next provider
                errors.append(f"{source.name}: {exc}")
        raise DataUnavailableError(
            f"unable to refresh instruments for {normalized_venue}: {'; '.join(errors)}"
        )

    def refresh_instruments(
        self,
        *,
        venue: str = "OKX",
        instrument_type: str = "SWAP",
        timestamp: datetime | str | None = None,
        include_candle_history: bool = True,
    ) -> pl.DataFrame:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.refresh_instruments_async(
                    venue=venue,
                    instrument_type=instrument_type,
                    timestamp=timestamp,
                    include_candle_history=include_candle_history,
                )
            )
        raise RuntimeError(
            "refresh_instruments() cannot run inside an event loop; "
            "use await refresh_instruments_async()"
        )

    def instruments(
        self,
        timestamp: datetime | str | None = None,
        *,
        venue: str = "OKX",
        refresh: bool = False,
        instrument_type: str = "SWAP",
    ) -> pl.DataFrame:
        """Return the latest locally known complete snapshot at or before ``timestamp``."""

        as_of = as_utc(timestamp or datetime.now(tz=UTC))
        if refresh:
            self.refresh_instruments(
                venue=venue,
                instrument_type=instrument_type,
                timestamp=as_of,
            )
        return self.instruments_repository.instrument_snapshot_at(venue, as_of)

    def instrument_lifecycle(
        self,
        *,
        venue: str = "OKX",
        rebuild: bool = False,
        include_candle_history: bool = True,
    ) -> pl.DataFrame:
        """Return point-in-time listing intervals, optionally rebuilding from local data."""

        if rebuild:
            return self.instruments_repository.rebuild_lifecycle(
                venue,
                include_candles=include_candle_history,
            )
        return self.instruments_repository.read_lifecycle(venue)

    def saved_universe(
        self,
        timestamp: datetime | str,
        *,
        name: str = "okx_usdt_linear_swaps",
        venue: str = "OKX",
        config_hash: str | None = None,
    ) -> pl.DataFrame:
        """Read the most recent persisted universe at or before ``timestamp``."""

        return self.instruments_repository.read_universe(
            name,
            venue,
            timestamp,
            config_hash=config_hash,
        )

    def trading_universe(
        self,
        timestamp: datetime | str | None = None,
        *,
        name: str = "okx_usdt_linear_swaps",
        venue: str = "OKX",
        instrument_type: str = "SWAP",
        settle_currency: str = "USDT",
        contract_type: str = "linear",
        states: tuple[str, ...] = ("live",),
        min_listing_days: int = 30,
        min_volume_usd_24h: float = 20_000_000,
        min_open_interest_usd: float = 0,
        max_spread_bps: float | None = 50,
        top_n: int = 30,
        refresh: bool = False,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Select a survivorship-safe universe using only data known by ``timestamp``."""

        as_of = as_utc(timestamp or datetime.now(tz=UTC))
        config = UniverseConfig(
            name=name,
            venue=venue,
            instrument_type=instrument_type,
            settle_currency=settle_currency,
            contract_type=contract_type,
            states=states,
            min_listing_days=min_listing_days,
            min_volume_usd_24h=min_volume_usd_24h,
            min_open_interest_usd=min_open_interest_usd,
            max_spread_bps=max_spread_bps,
            top_n=top_n,
        )
        if refresh:
            self.refresh_instruments(
                venue=config.venue,
                instrument_type=config.instrument_type,
                timestamp=as_of,
            )
        result = self.universe_selector.select(config, as_of)
        if persist:
            self.instruments_repository.save_universe(result)
        return result

    def maintain_universe(self, **options: Any) -> pl.DataFrame:
        """Refresh raw instrument metadata, rebuild lifecycle, select, and persist."""

        return self.trading_universe(refresh=True, persist=True, **options)

    async def query_async(self, query: DataQuery) -> pl.DataFrame:
        stored_bar_type = STORED_BAR_TYPES.get(query.data_type)
        if self.legacy_data_root is not None:
            initial_missing = self.catalog.missing_intervals(
                data_type=query.data_type,
                venue=query.venue,
                instrument_id=query.instrument_id,
                bar_type=stored_bar_type,
                start=query.start,
                end=query.end,
            )
            if initial_missing:
                self.store.import_legacy(query, self.legacy_data_root)
        if stored_bar_type is not None:
            for missing_start, missing_end in self.store.missing_partition_intervals(
                query,
                stored_bar_type=stored_bar_type,
            ):
                self.catalog.invalidate_coverage(
                    data_type=query.data_type,
                    venue=query.venue,
                    instrument_id=query.instrument_id,
                    bar_type=stored_bar_type,
                    start=missing_start,
                    end=missing_end,
                )
        await self._fill_missing(query, stored_bar_type=stored_bar_type)

        frame = self.store.read(query, stored_bar_type=stored_bar_type)
        validate_frame(frame, query.data_type)
        if stored_bar_type is not None:
            request = FetchRequest(
                data_type=query.data_type,
                venue=query.venue,
                instrument_id=query.instrument_id,
                start=query.start,
                end=query.end,
                bar_type=stored_bar_type,
            )
            try:
                self._validate_periodic_coverage(frame, request)
            except ValueError:
                self.catalog.invalidate_coverage(
                    data_type=query.data_type,
                    venue=query.venue,
                    instrument_id=query.instrument_id,
                    bar_type=stored_bar_type,
                    start=query.start,
                    end=query.end,
                )
                await self._fill_missing(query, stored_bar_type=stored_bar_type)
                frame = self.store.read(query, stored_bar_type=stored_bar_type)
                try:
                    self._validate_periodic_coverage(frame, request)
                except ValueError as exc:
                    message = f"local {stored_bar_type} data is incomplete: {exc}"
                    raise DataUnavailableError(message) from exc

        if stored_bar_type is not None and query.bar_type != stored_bar_type:
            frame = self._aggregate_periodic(frame, query)
        return frame.sort("timestamp")

    async def _fill_missing(self, query: DataQuery, *, stored_bar_type: str | None) -> None:
        missing = self.catalog.missing_intervals(
            data_type=query.data_type,
            venue=query.venue,
            instrument_id=query.instrument_id,
            bar_type=stored_bar_type,
            start=query.start,
            end=query.end,
        )
        for gap_start, gap_end in missing:
            for chunk_start, chunk_end in self._split_fetch_range(
                query.data_type,
                gap_start,
                gap_end,
            ):
                await self._download_gap(
                    query,
                    stored_bar_type=stored_bar_type,
                    start=chunk_start,
                    end=chunk_end,
                )

        remaining = self.catalog.missing_intervals(
            data_type=query.data_type,
            venue=query.venue,
            instrument_id=query.instrument_id,
            bar_type=stored_bar_type,
            start=query.start,
            end=query.end,
        )
        if remaining:
            raise DataUnavailableError(self._format_missing(query, remaining))

    def query(self, query: DataQuery) -> pl.DataFrame:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.query_async(query))
        raise RuntimeError("query() cannot run inside an event loop; use await query_async()")

    def candles(
        self,
        instrument_id: str,
        bar_type: str,
        start: datetime | str,
        end: datetime | str,
        *,
        venue: str = "OKX",
        **options: Any,
    ) -> pl.DataFrame:
        return self.query(
            DataQuery(
                DataType.CANDLES,
                instrument_id,
                start,
                end,
                venue=venue,
                bar_type=bar_type,
                options=options,
            )
        )

    def funding_rates(
        self,
        instrument_id: str,
        start: datetime | str,
        end: datetime | str,
        *,
        venue: str = "OKX",
        **options: Any,
    ) -> pl.DataFrame:
        return self.query(
            DataQuery(
                DataType.FUNDING_RATES,
                instrument_id,
                start,
                end,
                venue=venue,
                options=options,
            )
        )

    def contract_basis(
        self,
        instrument_id: str,
        bar_type: str,
        start: datetime | str,
        end: datetime | str,
        *,
        venue: str = "OKX",
        **options: Any,
    ) -> pl.DataFrame:
        return self._query_periodic(
            DataType.CONTRACT_BASIS,
            instrument_id,
            bar_type,
            start,
            end,
            venue=venue,
            options=options,
        )

    def open_interest(
        self,
        instrument_id: str,
        bar_type: str,
        start: datetime | str,
        end: datetime | str,
        *,
        venue: str = "OKX",
        **options: Any,
    ) -> pl.DataFrame:
        return self._query_periodic(
            DataType.OPEN_INTEREST,
            instrument_id,
            bar_type,
            start,
            end,
            venue=venue,
            options=options,
        )

    def long_short_ratio(
        self,
        instrument_id: str,
        bar_type: str,
        start: datetime | str,
        end: datetime | str,
        *,
        venue: str = "OKX",
        **options: Any,
    ) -> pl.DataFrame:
        return self._query_periodic(
            DataType.LONG_SHORT_RATIO,
            instrument_id,
            bar_type,
            start,
            end,
            venue=venue,
            options=options,
        )

    def market_cap(
        self,
        instrument_id: str,
        start: datetime | str,
        end: datetime | str,
        *,
        bar_type: str = "1d",
        venue: str = "GLOBAL",
        **options: Any,
    ) -> pl.DataFrame:
        return self._query_periodic(
            DataType.MARKET_CAP,
            instrument_id,
            bar_type,
            start,
            end,
            venue=venue,
            options=options,
        )

    def liquidations(
        self,
        instrument_id: str,
        start: datetime | str,
        end: datetime | str,
        *,
        venue: str = "OKX",
        **options: Any,
    ) -> pl.DataFrame:
        return self.query(
            DataQuery(
                DataType.LIQUIDATIONS,
                instrument_id,
                start,
                end,
                venue=venue,
                options=options,
            )
        )

    def taker_volume(
        self,
        instrument_id: str,
        bar_type: str,
        start: datetime | str,
        end: datetime | str,
        *,
        venue: str = "OKX",
        **options: Any,
    ) -> pl.DataFrame:
        return self._query_periodic(
            DataType.TAKER_VOLUME,
            instrument_id,
            bar_type,
            start,
            end,
            venue=venue,
            options=options,
        )

    def _query_periodic(
        self,
        data_type: DataType,
        instrument_id: str,
        bar_type: str,
        start: datetime | str,
        end: datetime | str,
        *,
        venue: str,
        options: dict[str, Any],
    ) -> pl.DataFrame:
        return self.query(
            DataQuery(
                data_type,
                instrument_id,
                start,
                end,
                venue=venue,
                bar_type=bar_type,
                options=options,
            )
        )

    async def _download_gap(
        self,
        query: DataQuery,
        *,
        stored_bar_type: str | None,
        start: datetime,
        end: datetime,
    ) -> None:
        request = FetchRequest(
            data_type=query.data_type,
            venue=query.venue,
            instrument_id=query.instrument_id,
            start=start,
            end=end,
            bar_type=stored_bar_type,
            options=query.options,
        )
        candidates = [source for source in self._sources if source.supports(request)]
        if not candidates:
            raise DataUnavailableError(
                f"no remote source supports {query.data_type.value} for {query.venue}; "
                f"missing interval [{request.start.isoformat()}, {request.end.isoformat()})"
            )

        errors: list[str] = []
        for source in candidates:
            try:
                frame = canonicalize_frame(await source.fetch(request), query.data_type)
                frame = frame.filter(
                    (pl.col("timestamp") >= request.start)
                    & (pl.col("timestamp") < request.end)
                    & (pl.col("venue") == query.venue)
                    & (pl.col("instrument_id") == query.instrument_id)
                )
                if stored_bar_type is not None:
                    frame = frame.filter(pl.col("bar_type") == stored_bar_type)
                    self._validate_periodic_coverage(frame, request)
                self.store.write(
                    frame,
                    data_type=query.data_type,
                    venue=query.venue,
                    instrument_id=query.instrument_id,
                    bar_type=stored_bar_type,
                    source_name=source.name,
                )
                self.catalog.record_coverage(
                    data_type=query.data_type,
                    venue=query.venue,
                    instrument_id=query.instrument_id,
                    bar_type=stored_bar_type,
                    start=request.start,
                    end=request.end,
                    source_name=source.name,
                )
                return
            except Exception as exc:  # noqa: BLE001 - try the next registered provider
                errors.append(f"{source.name}: {exc}")

        details = "; ".join(errors)
        raise DataUnavailableError(
            f"unable to fill [{request.start.isoformat()}, {request.end.isoformat()}): {details}"
        )

    @staticmethod
    def _split_fetch_range(
        data_type: DataType,
        start: datetime,
        end: datetime,
    ):
        for partition_start, partition_end in iter_partitions(data_type, start, end):
            yield max(start, partition_start), min(end, partition_end)

    @staticmethod
    def _validate_periodic_coverage(frame: pl.DataFrame, request: FetchRequest) -> None:
        if request.bar_type is None:
            return
        if request.data_type is DataType.CANDLES:
            unconfirmed = frame.filter(pl.col("confirm") != 1).height
            if unconfirmed:
                raise ValueError(f"received {unconfirmed} unconfirmed candles")
        step = timedelta(minutes=bar_minutes(request.bar_type))
        expected = int((request.end - request.start) / step)
        timestamps = (
            frame.select("timestamp")
            .unique()
            .sort("timestamp")
            .get_column("timestamp")
            .to_list()
        )
        if len(timestamps) != expected:
            raise ValueError(
                f"expected {expected} {request.bar_type} rows, received {len(timestamps)}"
            )
        if timestamps and (
            timestamps[0] != request.start
            or timestamps[-1] != request.end - step
            or any(
                right - left != step
                for left, right in zip(timestamps, timestamps[1:], strict=False)
            )
        ):
            raise ValueError(f"{request.bar_type} timestamps are not continuous")

    @staticmethod
    def _aggregate_periodic(frame: pl.DataFrame, query: DataQuery) -> pl.DataFrame:
        assert query.bar_type is not None
        keys = [
            pl.lit(query.venue).alias("venue"),
            pl.lit(query.instrument_id).alias("instrument_id"),
            pl.lit(query.bar_type).alias("bar_type"),
        ]
        if query.data_type is DataType.CANDLES:
            aggregations = [
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("volume_ccy").sum(),
                pl.col("volume_quote").sum(),
                pl.col("confirm").min(),
            ]
        elif query.data_type is DataType.TAKER_VOLUME:
            aggregations = [
                pl.col("buy_volume").sum(),
                pl.col("sell_volume").sum(),
                pl.col("net_buy_volume").sum(),
            ]
        else:
            value_columns = [
                column
                for column in frame.columns
                if column not in {"venue", "instrument_id", "bar_type", "timestamp"}
            ]
            aggregations = [pl.col(column).last() for column in value_columns]
        aggregated = (
            frame.group_by_dynamic("timestamp", every=query.bar_type, closed="left", label="left")
            .agg(*keys, *aggregations)
            .sort("timestamp")
        )
        return canonicalize_frame(aggregated, query.data_type)

    @staticmethod
    def _format_missing(
        query: DataQuery,
        intervals: list[tuple[datetime, datetime]],
    ) -> str:
        formatted = ", ".join(f"[{start}, {end})" for start, end in intervals)
        return (
            f"local data remains incomplete for {query.venue}/{query.instrument_id} "
            f"{query.data_type.value}: {formatted}"
        )


_default_client: MarketDataClient | None = None


def _get_default_client() -> MarketDataClient:
    global _default_client
    if _default_client is None:
        _default_client = MarketDataClient()
    return _default_client


def query(query: DataQuery) -> pl.DataFrame:
    return _get_default_client().query(query)


async def query_async(query: DataQuery) -> pl.DataFrame:
    return await _get_default_client().query_async(query)
