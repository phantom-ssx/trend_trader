import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl

from trend_trader.data.long_short_ratio_storage import (
    LONG_SHORT_RATIO_SNAPSHOT_SCHEMA,
    LongShortRatioParquetRepository,
    LongShortRatioState,
    LongShortRatioStateCache,
    LongShortRatioType,
    floor_five_minutes,
)
from trend_trader.data.okx_long_short_ratio_collector import (
    LongShortRatioCollectorConfig,
    OkxLongShortRatioCollector,
    OkxLongShortRatioRestClient,
)
from trend_trader.data.open_interest_storage import OpenInterestInstrument


def _instrument(
    instrument_id: str = "BTC-USDT-SWAP",
    *,
    instrument_type: str = "SWAP",
) -> OpenInterestInstrument:
    is_future = instrument_type == "FUTURES"
    return OpenInterestInstrument(
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        instrument_family="BTC-USD" if is_future else "BTC-USDT",
        base_currency="BTC",
        settle_currency="BTC" if is_future else "USDT",
        contract_type="inverse" if is_future else "linear",
        expiration_time=(datetime(2026, 7, 31, 8, tzinfo=UTC) if is_future else None),
    )


def _state(
    instrument_id: str = "BTC-USDT-SWAP",
    *,
    instrument_type: str = "SWAP",
    exchange_ts: datetime | None = None,
    received_at: datetime | None = None,
    ratio: float = 1.5,
    ratio_type: LongShortRatioType = LongShortRatioType.ALL_ACCOUNT,
) -> LongShortRatioState:
    return LongShortRatioState(
        venue="OKX",
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        ratio_type=ratio_type,
        exchange_ts=exchange_ts or datetime(2026, 7, 29, 8, 30, tzinfo=UTC),
        received_at=received_at or datetime(2026, 7, 29, 8, 31, tzinfo=UTC),
        long_short_ratio=ratio,
    )


def test_state_parses_native_timestamp_and_ratio() -> None:
    state = LongShortRatioState.from_okx(
        ["1785313800000", "1.5133426780566502"],
        instrument_id="BTC-USDT-SWAP",
        instrument_type="SWAP",
        ratio_type=LongShortRatioType.TOP_TRADER_ACCOUNT,
        received_at=datetime(2026, 7, 29, 8, 31, 2, tzinfo=UTC),
    )

    assert state.instrument_id == "BTC-USDT-SWAP"
    assert state.instrument_type == "SWAP"
    assert state.ratio_type is LongShortRatioType.TOP_TRADER_ACCOUNT
    assert state.exchange_ts == datetime(2026, 7, 29, 8, 30, tzinfo=UTC)
    assert state.long_short_ratio == 1.5133426780566502
    assert state.data_source == "rest"


def test_cache_orders_by_exchange_time_and_uses_it_for_staleness() -> None:
    cache = LongShortRatioStateCache()
    latest = _state()
    top_account = _state(
        ratio_type=LongShortRatioType.TOP_TRADER_ACCOUNT,
        ratio=0.8,
    )
    top_position = _state(
        ratio_type=LongShortRatioType.TOP_TRADER_POSITION,
        ratio=1.2,
    )
    older = _state(
        exchange_ts=latest.exchange_ts - timedelta(minutes=5),
        received_at=latest.received_at + timedelta(minutes=1),
        ratio=2.0,
    )

    assert cache.update(latest)
    assert cache.update(top_account)
    assert cache.update(top_position)
    assert not cache.update(older)

    fresh = cache.snapshot(
        {"BTC-USDT-SWAP": _instrument()},
        snapshot_time=datetime(2026, 7, 29, 8, 39, 59, tzinfo=UTC),
        stale_after=timedelta(minutes=10),
    )
    stale = cache.snapshot(
        {"BTC-USDT-SWAP": _instrument()},
        snapshot_time=datetime(2026, 7, 29, 8, 45, tzinfo=UTC),
        stale_after=timedelta(minutes=10),
    )

    assert floor_five_minutes(datetime(2026, 7, 29, 8, 39, 59, tzinfo=UTC)) == datetime(
        2026, 7, 29, 8, 35, tzinfo=UTC
    )
    assert fresh["snapshot_time"].unique().to_list() == [datetime(2026, 7, 29, 8, 35, tzinfo=UTC)]
    assert fresh["data_status"].unique().to_list() == ["fresh"]
    assert stale["data_status"].unique().to_list() == ["stale"]
    assert fresh["bar_type"].unique().to_list() == ["5m"]
    assert fresh["ratio_type"].to_list() == [
        "all_account",
        "top_trader_account",
        "top_trader_position",
    ]
    assert fresh["long_short_ratio"].to_list() == [1.5, 0.8, 1.2]
    assert fresh["instrument_family"].unique().to_list() == ["BTC-USDT"]


def test_repository_partitions_and_idempotently_merges(tmp_path) -> None:
    repository = LongShortRatioParquetRepository(tmp_path)
    cache = LongShortRatioStateCache()
    cache.update(_state())
    cache.update(
        _state(
            ratio_type=LongShortRatioType.TOP_TRADER_ACCOUNT,
            ratio=0.8,
        )
    )
    cache.update(
        _state(
            ratio_type=LongShortRatioType.TOP_TRADER_POSITION,
            ratio=1.2,
        )
    )
    first = cache.snapshot(
        {"BTC-USDT-SWAP": _instrument()},
        snapshot_time=datetime(2026, 7, 29, 8, 39, tzinfo=UTC),
        stale_after=timedelta(minutes=10),
    )
    updated = first.with_columns(
        pl.when(pl.col("ratio_type") == LongShortRatioType.ALL_ACCOUNT.value)
        .then(pl.lit(1.75))
        .otherwise(pl.col("long_short_ratio"))
        .alias("long_short_ratio")
    )

    repository.write_snapshots(first)
    repository.write_snapshots(updated)
    snapshot_path = (
        tmp_path
        / "long_short_ratio_snapshot"
        / "year=2026"
        / "date=2026-07-29"
        / "long_short_ratio_snapshot-2026-07-29.parquet"
    )
    stored = pl.read_parquet(snapshot_path)

    assert snapshot_path.exists()
    assert stored.height == 3
    assert stored["long_short_ratio"].to_list() == [1.75, 0.8, 1.2]
    assert stored.schema == LONG_SHORT_RATIO_SNAPSHOT_SCHEMA


def test_repository_migrates_legacy_rows_to_all_account(tmp_path) -> None:
    repository = LongShortRatioParquetRepository(tmp_path)
    cache = LongShortRatioStateCache()
    cache.update(_state())
    legacy = cache.snapshot(
        {"BTC-USDT-SWAP": _instrument()},
        snapshot_time=datetime(2026, 7, 29, 8, 35, tzinfo=UTC),
        stale_after=timedelta(minutes=10),
    ).drop("ratio_type")
    path = repository.snapshot_path(datetime(2026, 7, 29, 8, 35, tzinfo=UTC))
    path.parent.mkdir(parents=True)
    legacy.write_parquet(path)

    cache.update(
        _state(
            ratio_type=LongShortRatioType.TOP_TRADER_ACCOUNT,
            ratio=0.8,
        )
    )
    repository.write_snapshots(
        cache.snapshot(
            {"BTC-USDT-SWAP": _instrument()},
            snapshot_time=datetime(2026, 7, 29, 8, 35, tzinfo=UTC),
            stale_after=timedelta(minutes=10),
        ).filter(pl.col("ratio_type") == "top_trader_account")
    )
    stored = pl.read_parquet(path)

    assert stored.height == 2
    assert stored["ratio_type"].to_list() == ["all_account", "top_trader_account"]


def test_rest_client_and_collector_cover_live_swaps_and_futures(tmp_path) -> None:
    captured_at = datetime(2026, 7, 29, 8, 31, 2, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/instruments"):
            instrument_type = request.url.params["instType"]
            live_id = "BTC-USDT-SWAP" if instrument_type == "SWAP" else "BTC-USD-260731"
            return httpx.Response(
                200,
                json={
                    "code": "0",
                    "msg": "",
                    "data": [
                        {
                            "instId": live_id,
                            "instType": instrument_type,
                            "instFamily": ("BTC-USDT" if instrument_type == "SWAP" else "BTC-USD"),
                            "baseCcy": "BTC",
                            "settleCcy": ("USDT" if instrument_type == "SWAP" else "BTC"),
                            "ctType": ("linear" if instrument_type == "SWAP" else "inverse"),
                            "expTime": ("" if instrument_type == "SWAP" else "1785484800000"),
                            "state": "live",
                        },
                        {
                            "instId": f"OLD-{instrument_type}",
                            "instType": instrument_type,
                            "state": "suspend",
                        },
                    ],
                },
            )
        if "long-short" in request.url.path:
            assert request.url.params["period"] == "5m"
            assert request.url.params["limit"] == "2"
            base_ratio = 1.5 if request.url.params["instId"].endswith("SWAP") else 0.75
            if request.url.path.endswith("/long-short-account-ratio-contract"):
                ratio = base_ratio
            elif request.url.path.endswith("/long-short-account-ratio-contract-top-trader"):
                ratio = base_ratio + 1
            elif request.url.path.endswith("/long-short-position-ratio-contract-top-trader"):
                ratio = base_ratio + 2
            else:
                raise AssertionError(f"unexpected ratio endpoint: {request.url.path}")
            return httpx.Response(
                200,
                json={
                    "code": "0",
                    "msg": "",
                    "data": [
                        ["1785313800000", str(ratio)],
                        ["1785313500000", "9.9"],
                    ],
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    async def run_scenario() -> tuple[dict[str, OpenInterestInstrument], int]:
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://www.okx.com",
        )
        rest = OkxLongShortRatioRestClient(
            client=http_client,
            requests_per_second=1_000_000,
            now=lambda: captured_at,
        )
        repository = LongShortRatioParquetRepository(tmp_path)
        collector = OkxLongShortRatioCollector(
            LongShortRatioCollectorConfig(data_root=tmp_path),
            rest_client=rest,
            repository=repository,
            now=lambda: captured_at,
        )
        try:
            live = await rest.fetch_live_instruments(("SWAP", "FUTURES"))
            rows = await collector.run_once()
            return live, rows
        finally:
            await http_client.aclose()

    live, rows = asyncio.run(run_scenario())

    assert set(live) == {"BTC-USD-260731", "BTC-USDT-SWAP"}
    assert rows == 6
    stored = pl.read_parquet(
        tmp_path
        / "long_short_ratio_snapshot"
        / "year=2026"
        / "date=2026-07-29"
        / "long_short_ratio_snapshot-2026-07-29.parquet"
    )
    assert stored.group_by("instrument_type").len().sort("instrument_type").to_dict(
        as_series=False
    ) == {"instrument_type": ["FUTURES", "SWAP"], "len": [3, 3]}
    assert stored["ratio_type"].unique().sort().to_list() == [
        "all_account",
        "top_trader_account",
        "top_trader_position",
    ]
    assert stored["long_short_ratio"].sort().to_list() == [
        0.75,
        1.5,
        1.75,
        2.5,
        2.75,
        3.5,
    ]
    assert stored["exchange_ts"].unique().to_list() == [datetime(2026, 7, 29, 8, 30, tzinfo=UTC)]
