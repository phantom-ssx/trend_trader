from __future__ import annotations

import asyncio
import hashlib
import json
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
import pyarrow.parquet as pq
import pytest

from trend_trader.data.offline.catalog import OfflineCatalog
from trend_trader.data.offline.client import OkxApiError, OkxOfflineClient
from trend_trader.data.offline.config import (
    DatasetOptions,
    DatasetsConfig,
    OfflineSyncConfig,
    PrivateAccountConfig,
)
from trend_trader.data.offline.notify import format_run_notification
from trend_trader.data.offline.schemas import (
    CANDLE_SCHEMA,
    ArchiveDataQualityError,
    ArchiveParseStats,
    candle_frame,
    iter_candle_archive_batches,
    parse_candle_archive,
)
from trend_trader.data.offline.storage import (
    DailyParquetRepository,
    OfflineLayout,
    RawRepository,
    StreamingDailyParquetRepository,
)
from trend_trader.data.offline.sync import OfflineSynchronizer


def _config(tmp_path: Path) -> OfflineSyncConfig:
    config = OfflineSyncConfig(
        data_root=tmp_path,
        requests_per_second=100_000,
        daily_history_days_per_run=14,
    )
    for name in type(config.datasets).model_fields:
        getattr(config.datasets, name).enabled = False
    return config


def test_deferred_large_datasets_cannot_be_enabled() -> None:
    with pytest.raises(ValueError, match="deferred"):
        DatasetsConfig(public_trades=DatasetOptions(enabled=True))


def test_instrument_catalog_prefers_official_underlying_for_index_id(tmp_path: Path) -> None:
    catalog = OfflineCatalog(tmp_path)

    catalog.upsert_instrument(
        {
            "instId": "AAVE-USD_UM_XPERP-310704",
            "instType": "FUTURES",
            "baseCcy": "AAVE",
            "uly": "AAVE-USD",
        },
        datetime(2026, 7, 31, tzinfo=UTC),
    )

    assert catalog.list_instruments()[0]["index_id"] == "AAVE-USD"


def test_permanent_okx_api_error_is_not_retried_and_keeps_details(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.max_retries = 4
    request_count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={"code": "51001", "msg": "Instrument ID doesn't exist", "data": []},
        )

    async def run() -> None:
        async with OkxOfflineClient(
            config,
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(OkxApiError, match="51001") as caught:
                await client.fetch_price_candles(
                    instrument_id="AAVE-USD_UM_XPERP",
                    start_ms=0,
                    end_ms=60_000,
                    index=True,
                )
            assert caught.value.code == "51001"

    asyncio.run(run())
    assert request_count == 1


def test_retryable_okx_api_error_uses_configured_retries(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.max_retries = 3
    request_count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={"code": "50011", "msg": "Rate limit reached", "data": []},
        )

    async def run() -> None:
        async with OkxOfflineClient(
            config,
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(OkxApiError, match="50011") as caught:
                await client.fetch_price_candles(
                    instrument_id="BTC-USDT",
                    start_ms=0,
                    end_ms=60_000,
                    index=True,
                )
            assert caught.value.retryable is True

    asyncio.run(run())
    assert request_count == 3


def test_daily_plan_only_scans_recent_mature_window(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.datasets.candles.enabled = True
    config.datasets.candles.start = date(2023, 7, 1)
    sync = OfflineSynchronizer(config)

    tasks = sync.plan(
        mode="daily",
        today=date(2026, 7, 30),
        datasets={"candles"},
    )

    assert len(tasks) == 14
    assert tasks[0].target_date == date(2026, 7, 15)
    assert tasks[-1].target_date == date(2026, 7, 28)


def test_default_candle_backfill_starts_at_rest_history_boundary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.datasets.candles.enabled = True
    sync = OfflineSynchronizer(config)

    tasks = sync.plan(
        mode="backfill",
        today=date(2020, 1, 5),
        datasets={"candles"},
    )

    assert tasks[0].target_date == date(2020, 1, 2)


def test_oi_retention_is_planned_separately_by_period(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.datasets.aggregate_open_interest.enabled = True
    sync = OfflineSynchronizer(config)

    tasks = sync.plan(
        mode="daily",
        today=date(2026, 7, 30),
        datasets={"aggregate_open_interest"},
    )

    counts = {
        period: sum(task.scope_key == period for task in tasks) for period in ("5m", "1H", "1D")
    }
    assert counts == {"5m": 2, "1H": 14, "1D": 14}


def test_private_daily_plan_always_refetches_three_day_overlap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.private_accounts = (
        PrivateAccountConfig(
            alias="main",
            api_key_env="TEST_KEY",
            secret_key_env="TEST_SECRET",
            passphrase_env="TEST_PASSPHRASE",
        ),
    )
    config.datasets.private_bills.enabled = True
    sync = OfflineSynchronizer(config)
    today = date(2026, 7, 30)
    for day in (date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)):
        sync.catalog.mark_coverage(
            "private_bills",
            day,
            status="complete",
            row_count=1,
            scope_key="main",
        )

    tasks = sync.plan(mode="daily", today=today, datasets={"private_bills"})

    recent = [task.target_date for task in tasks if task.target_date >= date(2026, 7, 27)]
    assert recent == [date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)]


def test_archive_parser_and_daily_file_keep_all_instruments(tmp_path: Path) -> None:
    archive_path = tmp_path / "candles.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "candles.csv",
            "instId,ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm\n"
            "BTC-USDT-SWAP,1722384000000,1,2,0.5,1.5,10,2,15,1\n"
            "ETH-USDT-SWAP,1722384000000,3,4,2.5,3.5,20,4,70,1\n",
        )
    frame = parse_candle_archive(archive_path)
    repository = DailyParquetRepository(
        OfflineLayout(tmp_path),
        dataset="candles",
        schema=CANDLE_SCHEMA,
        primary_key=("venue", "instrument_id", "bar_type", "timestamp"),
        timestamp_column="timestamp",
        sort_columns=("timestamp", "instrument_id"),
    )

    outputs = repository.write(frame)

    assert len(outputs) == 1
    stored = pl.read_parquet(outputs[0][1])
    assert stored.get_column("instrument_id").to_list() == [
        "BTC-USDT-SWAP",
        "ETH-USDT-SWAP",
    ]
    assert stored.schema["timestamp"] == pl.Datetime("ms", "UTC")
    assert "offline/normalized/candles/venue=OKX" in str(outputs[0][1])


def test_archive_parser_yields_fixed_batches_without_loading_member(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "candles.zip"
    rows = "\n".join(f"{1722384000000 + index * 60000},1,2,0.5,1.5,10,2,15,1" for index in range(5))
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "BTC-USD-260101.csv",
            "ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm\n" + rows,
        )

    batches = list(iter_candle_archive_batches(archive_path, batch_size=2))

    assert [batch.height for batch in batches] == [2, 2, 1]
    assert {
        value for batch in batches for value in batch.get_column("instrument_id").unique().to_list()
    } == {"BTC-USD-260101"}


def test_archive_parser_reads_instrument_name_and_drops_adjacent_duplicates(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "candles.zip"
    row = "BTC-USD-SWAP,30070.4,30070.4,30057.3,30066.4,341,None,None,1688140800000,1\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "candles.csv",
            "instrument_name,open,high,low,close,vol,vol_ccy,vol_quote,"
            "open_time,confirm\n" + row * 3,
        )
    stats = ArchiveParseStats()

    batches = list(
        iter_candle_archive_batches(
            archive_path,
            source_date=date(2023, 7, 1),
            stats=stats,
        )
    )

    assert sum(batch.height for batch in batches) == 1
    assert batches[0]["instrument_id"][0] == "BTC-USD-SWAP"
    assert stats.input_rows == 3
    assert stats.emitted_rows == 1
    assert stats.adjacent_duplicate_rows == 2
    assert stats.duplicate_ratio == pytest.approx(2 / 3)


def test_archive_parser_accepts_and_reports_high_official_duplicate_ratio(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "candles.zip"
    row = "BTC-USDT-SWAP,1688140800000,1,2,0.5,1.5,10,2,15,1\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "candles.csv",
            "instId,ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm\n" + row * 4,
        )
    stats = ArchiveParseStats()

    batches = list(
        iter_candle_archive_batches(
            archive_path,
            source_date=date(2023, 7, 1),
            stats=stats,
        )
    )

    assert sum(batch.height for batch in batches) == 1
    assert stats.input_rows == 4
    assert stats.adjacent_duplicate_rows == 3
    assert stats.duplicate_ratio == pytest.approx(0.75)


def test_archive_parser_rejects_conflicting_adjacent_primary_key(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "candles.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "candles.csv",
            "instId,ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm\n"
            "BTC-USDT-SWAP,1688140800000,1,2,0.5,1.5,10,2,15,1\n"
            "BTC-USDT-SWAP,1688140800000,1,2,0.5,1.6,10,2,15,1\n",
        )

    with pytest.raises(ArchiveDataQualityError, match="conflicting candle rows"):
        list(iter_candle_archive_batches(archive_path))


def test_archive_parser_rejects_implausible_historical_futures_name(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "allfutures-candles.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "candles.csv",
            "instrument_name,open,high,low,close,vol,vol_ccy,vol_quote,"
            "open_time,confirm\n"
            "BTC-USD-251017,30070.4,30070.4,30057.3,30066.4,"
            "341,None,None,1688140800000,1\n",
        )

    with pytest.raises(ArchiveDataQualityError, match="implausible standard futures expiry"):
        list(
            iter_candle_archive_batches(
                archive_path,
                source_date=date(2023, 7, 1),
            )
        )


def test_archive_parser_accepts_observed_long_dated_standard_future(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "allfutures-candles.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "candles.csv",
            "instrument_name,open,high,low,close,vol,vol_ccy,vol_quote,"
            "open_time,confirm\n"
            "BTC-USD-270924,1,2,0.5,1.5,10,None,None,1785168000000,1\n",
        )

    batches = list(
        iter_candle_archive_batches(
            archive_path,
            source_date=date(2026, 7, 28),
        )
    )

    assert sum(batch.height for batch in batches) == 1


def test_streaming_repository_compacts_cross_day_batches_and_upserts(
    tmp_path: Path,
) -> None:
    layout = OfflineLayout(tmp_path)
    repository = StreamingDailyParquetRepository(
        layout,
        dataset="candles",
        schema=CANDLE_SCHEMA,
        primary_key=("venue", "instrument_id", "bar_type", "timestamp"),
        timestamp_column="timestamp",
        sort_columns=("timestamp", "instrument_id"),
        batch_rows=2,
        compaction_memory_mb=128,
        compaction_threads=1,
    )
    first_day = int(datetime(2026, 7, 28, tzinfo=UTC).timestamp() * 1000)
    second_day = int(datetime(2026, 7, 29, tzinfo=UTC).timestamp() * 1000)

    def candle_frame(rows: list[tuple[str, int, float]]) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "venue": "OKX",
                    "instrument_id": instrument,
                    "instrument_type": "SWAP",
                    "bar_type": "1m",
                    "timestamp": timestamp,
                    "open": 1.0,
                    "high": 2.0,
                    "low": 0.5,
                    "close": close,
                    "volume": 1.0,
                    "volume_ccy": 1.0,
                    "volume_quote": 1.0,
                    "confirm": 1,
                }
                for instrument, timestamp, close in rows
            ]
        )

    outputs, input_rows = repository.write_batches(
        iter(
            [
                candle_frame(
                    [
                        ("BTC-USDT-SWAP", first_day, 1.0),
                        ("ETH-USDT-SWAP", first_day, 2.0),
                    ]
                ),
                candle_frame(
                    [
                        ("BTC-USDT-SWAP", first_day, 9.0),
                        ("BTC-USDT-SWAP", second_day, 3.0),
                    ]
                ),
            ]
        )
    )

    assert input_rows == 4
    assert [item[0] for item in outputs] == [date(2026, 7, 28), date(2026, 7, 29)]
    first_path = layout.normalized_path("candles", date(2026, 7, 28))
    first = pl.read_parquet(first_path)
    assert first.height == 2
    assert first.filter(pl.col("instrument_id") == "BTC-USDT-SWAP")["close"][0] == 9.0

    repository.write_batches(iter([candle_frame([("BTC-USDT-SWAP", first_day, 10.0)])]))
    revised = pl.read_parquet(first_path)
    assert revised.height == 2
    assert revised.filter(pl.col("instrument_id") == "BTC-USDT-SWAP")["close"][0] == 10.0
    assert revised.schema["timestamp"] == pl.Datetime("ms", "UTC")


def test_raw_rest_revisions_are_immutable(tmp_path: Path) -> None:
    raw = RawRepository(OfflineLayout(tmp_path))
    first = raw.write_json_gz("taker_volume", date(2026, 7, 28), {"value": 1})
    repeated = raw.write_json_gz("taker_volume", date(2026, 7, 28), {"value": 1})
    revised = raw.write_json_gz("taker_volume", date(2026, 7, 28), {"value": 2})

    assert repeated == first
    assert revised != first
    assert ".rev-" in revised.name
    assert first.exists() and revised.exists()


def test_client_flattens_official_historical_link_response(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v5/public/market-data-history"
        assert request.url.params["module"] == "2"
        assert request.url.params["instType"] == "SWAP"
        assert request.url.params["dateAggrType"] == "daily"
        assert request.url.params["begin"] == "1785168000000"
        assert request.url.params["end"] == "1785168000000"
        return httpx.Response(
            200,
            json={
                "code": "0",
                "data": [
                    {
                        "fileName": "allswap.zip",
                        "fileHref": "https://static.okx.com/allswap.zip",
                    }
                ],
            },
        )

    config = _config(tmp_path)
    transport = httpx.MockTransport(handler)

    async def run() -> list[dict[str, object]]:
        async with OkxOfflineClient(config, transport=transport) as client:
            return await client.historical_links(
                module=2,
                instrument_type="SWAP",
                source_date=date(2026, 7, 28),
            )

    links = asyncio.run(run())
    assert links[0]["fileName"] == "allswap.zip"


def test_client_fetches_rest_candles_with_backward_pagination(tmp_path: Path) -> None:
    start = int(datetime(2020, 1, 2, tzinfo=UTC).timestamp() * 1000)
    end = start + 86_400_000
    requested_cursors: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v5/market/history-candles"
        assert request.url.params["instId"] == "BTC-USDT-SWAP"
        assert request.url.params["bar"] == "1m"
        assert request.url.params["limit"] == "300"
        cursor = int(request.url.params["after"])
        requested_cursors.append(cursor)
        timestamp = end - 60_000 if len(requested_cursors) == 1 else start
        return httpx.Response(
            200,
            json={
                "code": "0",
                "data": [[timestamp, "1", "2", "0.5", "1.5", "10", "2", "15", "1"]],
            },
        )

    config = _config(tmp_path)
    transport = httpx.MockTransport(handler)

    async def run() -> list[list[object]]:
        async with OkxOfflineClient(config, transport=transport) as client:
            return await client.fetch_candles(
                instrument_id="BTC-USDT-SWAP",
                start_ms=start,
                end_ms=end,
            )

    rows = asyncio.run(run())
    assert [int(str(row[0])) for row in rows] == [end - 60_000, start]
    assert requested_cursors == [end, end - 60_001]


def test_mark_price_range_run_writes_one_all_market_file(tmp_path: Path) -> None:
    target = date(2026, 7, 28)
    timestamp = int(datetime(2026, 7, 28, tzinfo=UTC).timestamp() * 1000)
    config = _config(tmp_path)
    config.datasets.mark_price_candles.enabled = True
    config.datasets.mark_price_candles.start = target

    class FakeClient:
        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def fetch_instruments(self) -> list[dict[str, object]]:
            return [
                {
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "baseCcy": "BTC",
                },
                {
                    "instId": "ETH-USDT-SWAP",
                    "instType": "SWAP",
                    "baseCcy": "ETH",
                },
            ]

        async def fetch_price_candles(
            self,
            *,
            instrument_id: str,
            start_ms: int,
            end_ms: int,
            index: bool,
        ) -> list[list[object]]:
            assert start_ms <= timestamp < end_ms
            return [[timestamp, "1", "2", "0.5", "1.5", "1"]]

    synchronizer = OfflineSynchronizer(config, client_factory=FakeClient)
    synchronizer._check_disk = lambda: None
    report = asyncio.run(
        synchronizer.run(
            mode="range",
            start=target,
            end=target,
            datasets={"mark_price_candles"},
            today=target + timedelta(days=1),
        )
    )

    assert report["status"] == "success", json.dumps(report, indent=2)
    output = OfflineLayout(tmp_path).normalized_path("mark_price_candles", target)
    assert output.exists()
    assert pl.read_parquet(output).height == 2


def test_price_run_filters_listing_window_and_caches_invalid_identifier(
    tmp_path: Path,
) -> None:
    first = date(2020, 1, 11)
    second = first + timedelta(days=1)
    config = _config(tmp_path)
    config.datasets.index_price_candles.enabled = True
    config.datasets.index_price_candles.start = first
    calls: list[tuple[str, date]] = []

    class FakeClient:
        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def fetch_instruments(self) -> list[dict[str, object]]:
            return [
                {
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "baseCcy": "BTC",
                    "uly": "BTC-USDT",
                    "listTime": "1577836800000",
                },
                {
                    "instId": "AAVE-USD_UM_XPERP-310704",
                    "instType": "FUTURES",
                    "baseCcy": "AAVE",
                    "uly": "AAVE-USD",
                    "listTime": "1577836800000",
                },
                {
                    "instId": "FUTURE-USDT-SWAP",
                    "instType": "SWAP",
                    "baseCcy": "FUTURE",
                    "uly": "FUTURE-USDT",
                    "listTime": "1767225600000",
                },
            ]

        async def fetch_price_candles(
            self,
            *,
            instrument_id: str,
            start_ms: int,
            end_ms: int,
            index: bool,
        ) -> list[list[object]]:
            assert index is True
            target = datetime.fromtimestamp(start_ms / 1000, tz=UTC).date()
            calls.append((instrument_id, target))
            if instrument_id == "AAVE-USD":
                raise OkxApiError(
                    "OKX API index: 51001 Instrument ID doesn't exist",
                    code="51001",
                )
            return [[start_ms, "1", "2", "0.5", "1.5", "1"]]

    synchronizer = OfflineSynchronizer(config, client_factory=FakeClient)
    synchronizer._check_disk = lambda: None
    report = asyncio.run(
        synchronizer.run(
            mode="range",
            start=first,
            end=second,
            datasets={"index_price_candles"},
            today=second + timedelta(days=1),
        )
    )

    assert report["status"] == "success", json.dumps(report, indent=2)
    listed_day = date(2026, 1, 2)
    listed_report = asyncio.run(
        synchronizer.run(
            mode="range",
            start=listed_day,
            end=listed_day,
            datasets={"index_price_candles"},
            today=listed_day + timedelta(days=1),
        )
    )
    assert listed_report["status"] == "success", json.dumps(listed_report, indent=2)
    assert calls == [
        ("AAVE-USD", first),
        ("BTC-USDT", first),
        ("BTC-USDT", second),
        ("BTC-USDT", listed_day),
        ("FUTURE-USDT", listed_day),
    ]
    invalid = synchronizer.catalog.invalid_identifier_summary()
    assert len(invalid) == 1
    assert invalid[0]["identifier"] == "AAVE-USD"
    assert invalid[0]["failure_count"] == 1
    assert invalid[0]["skip_count"] == 2
    assert invalid[0]["last_skipped_target_date"] == listed_day.isoformat()


def test_pre_archive_candles_use_rest_and_active_instrument_windows(tmp_path: Path) -> None:
    target = date(2020, 1, 2)
    timestamp = int(datetime(2020, 1, 2, tzinfo=UTC).timestamp() * 1000)
    config = _config(tmp_path)
    config.datasets.candles.enabled = True
    requested: list[str] = []

    class FakeRestClient:
        async def __aenter__(self) -> FakeRestClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def fetch_instruments(self) -> list[dict[str, object]]:
            return [
                {
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "baseCcy": "BTC",
                    "listTime": str(timestamp - 86_400_000),
                },
                {
                    "instId": "ETH-USDT-SWAP",
                    "instType": "SWAP",
                    "baseCcy": "ETH",
                    "listTime": str(timestamp + 86_400_000),
                },
                {
                    "instId": "BTC-USD-191227",
                    "instType": "FUTURES",
                    "baseCcy": "BTC",
                    "expTime": str(timestamp - 86_400_000),
                },
            ]

        async def fetch_candles(
            self,
            *,
            instrument_id: str,
            start_ms: int,
            end_ms: int,
        ) -> list[list[object]]:
            requested.append(instrument_id)
            assert start_ms <= timestamp < end_ms
            return [[timestamp, "1", "2", "0.5", "1.5", "10", "2", "15", "1"]]

    synchronizer = OfflineSynchronizer(config, client_factory=FakeRestClient)
    synchronizer._check_disk = lambda: None
    report = asyncio.run(
        synchronizer.run(
            mode="range",
            start=target,
            end=target,
            datasets={"candles"},
            today=target + timedelta(days=2),
        )
    )

    assert report["status"] == "success", json.dumps(report, indent=2)
    assert requested == ["BTC-USDT-SWAP"]
    output = OfflineLayout(tmp_path).normalized_path("candles", target)
    stored = pl.read_parquet(output)
    assert stored.get_column("instrument_id").to_list() == ["BTC-USDT-SWAP"]
    assert pq.read_metadata(output).metadata[b"source_name"] == b"okx_public_rest"
    raw_files = list(OfflineLayout(tmp_path).raw_dir("candles", target).glob("*.json.gz"))
    assert len(raw_files) == 1


def test_rest_archive_overlap_must_have_identical_candles(tmp_path: Path) -> None:
    timestamp = int(datetime(2023, 6, 30, 23, 59, tzinfo=UTC).timestamp() * 1000)
    rest = candle_frame(
        [[timestamp, "1", "2", "0.5", "1.5", "10", "2", "15", "1"]],
        instrument_id="BTC-USDT-SWAP",
    )

    OfflineSynchronizer._validate_candle_overlap(rest, rest, tmp_path / "archive.zip")

    conflicting = rest.with_columns(pl.lit(1.6).alias("close"))
    with pytest.raises(ArchiveDataQualityError, match="REST/archive candle conflict"):
        OfflineSynchronizer._validate_candle_overlap(
            conflicting,
            rest,
            tmp_path / "archive.zip",
        )


def test_historical_candles_run_uses_streaming_compaction(tmp_path: Path) -> None:
    target = date(2026, 7, 28)
    timestamp = int(datetime(2026, 7, 28, tzinfo=UTC).timestamp() * 1000)
    archive_path = tmp_path / "all-swap.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "candles.csv",
            "instId,ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm\n"
            f"BTC-USDT-SWAP,{timestamp},1,2,0.5,1.5,10,2,15,1\n"
            f"ETH-USDT-SWAP,{timestamp},3,4,2.5,3.5,20,4,70,1\n",
        )
    config = _config(tmp_path)
    config.datasets.candles.enabled = True
    config.datasets.candles.start = target
    config.stream_batch_rows = 1_000
    config.compaction_memory_mb = 128
    config.compaction_threads = 1

    class FakeHistoricalClient:
        async def __aenter__(self) -> FakeHistoricalClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def fetch_instruments(self) -> list[dict[str, object]]:
            return []

        async def historical_links(
            self,
            *,
            module: int,
            instrument_type: str,
            source_date: date,
        ) -> list[dict[str, object]]:
            if instrument_type == "FUTURES":
                return []
            return [{"fileName": archive_path.name, "url": "https://example.test/archive.zip"}]

        async def download(self, url: str, destination: Path) -> tuple[Path, str]:
            digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            return archive_path, digest

    synchronizer = OfflineSynchronizer(config, client_factory=FakeHistoricalClient)
    synchronizer._check_disk = lambda: None
    report = asyncio.run(
        synchronizer.run(
            mode="range",
            start=target,
            end=target,
            datasets={"candles"},
            today=target + timedelta(days=2),
        )
    )

    assert report["status"] == "success", json.dumps(report, indent=2)
    output = OfflineLayout(tmp_path).normalized_path("candles", target)
    assert pl.read_parquet(output).height == 2
    metadata = pl.read_parquet_schema(output)
    assert metadata["timestamp"] == pl.Datetime("ms", "UTC")


def test_bark_summary_contains_failed_dataset() -> None:
    title, body = format_run_notification(
        {
            "run_id": "run-1",
            "mode": "daily",
            "status": "partial_failure",
            "results": [
                {
                    "dataset": "candles",
                    "target_date": "2026-07-28",
                    "status": "failed",
                    "rows": 0,
                }
            ],
        }
    )

    assert "⚠️" in title
    assert "candles:2026-07-28" in body
