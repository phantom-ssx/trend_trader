import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl

from trend_trader.data.okx_open_interest_collector import (
    OkxOpenInterestCollector,
    OkxOpenInterestRestClient,
    OpenInterestCollectorConfig,
    parse_ws_open_interest_states,
)
from trend_trader.data.open_interest_storage import (
    OPEN_INTEREST_SNAPSHOT_SCHEMA,
    OpenInterestInstrument,
    OpenInterestParquetRepository,
    OpenInterestState,
    OpenInterestStateCache,
)


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
    data_source: str = "websocket",
) -> OpenInterestState:
    return OpenInterestState(
        venue="OKX",
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        exchange_ts=exchange_ts or datetime(2026, 7, 28, 12, 0, 1, tzinfo=UTC),
        received_at=received_at or datetime(2026, 7, 28, 12, 0, 2, tzinfo=UTC),
        open_interest=3_244_542.98,
        open_interest_ccy=32_445.4298,
        open_interest_usd=2_058_396_468.28,
        data_source=data_source,
    )


def test_parse_ws_state_keeps_contract_type_and_all_units() -> None:
    message = json.dumps(
        {
            "arg": {"channel": "open-interest", "instId": "BTC-USD-260731"},
            "data": [
                {
                    "instId": "BTC-USD-260731",
                    "instType": "FUTURES",
                    "oi": "10020.5",
                    "oiCcy": "15.8",
                    "oiUsd": "1002050",
                    "ts": "1785240001000",
                }
            ],
        }
    )

    states = parse_ws_open_interest_states(
        message,
        received_at=datetime(2026, 7, 28, 12, 0, 2, tzinfo=UTC),
    )

    assert len(states) == 1
    state = states[0]
    assert state.instrument_type == "FUTURES"
    assert state.open_interest == 10020.5
    assert state.open_interest_ccy == 15.8
    assert state.open_interest_usd == 1002050
    assert state.data_source == "websocket"
    assert state.received_at.tzinfo is UTC


def test_cache_rejects_older_rest_state_and_marks_staleness() -> None:
    cache = OpenInterestStateCache()
    websocket_state = _state()
    older_rest_state = _state(
        exchange_ts=websocket_state.exchange_ts - timedelta(seconds=1),
        received_at=websocket_state.received_at + timedelta(seconds=5),
        data_source="rest",
    )
    same_time_rest_state = _state(
        exchange_ts=websocket_state.exchange_ts,
        received_at=websocket_state.received_at + timedelta(seconds=10),
        data_source="rest",
    )

    assert cache.update(websocket_state)
    assert not cache.update(older_rest_state)
    assert not cache.update(same_time_rest_state)

    fresh = cache.snapshot(
        {"BTC-USDT-SWAP": _instrument()},
        snapshot_time=datetime(2026, 7, 28, 12, 1, tzinfo=UTC),
        stale_after=timedelta(seconds=120),
    )
    stale = cache.snapshot(
        {"BTC-USDT-SWAP": _instrument()},
        snapshot_time=datetime(2026, 7, 28, 12, 3, tzinfo=UTC),
        stale_after=timedelta(seconds=120),
    )

    assert fresh["data_status"].to_list() == ["fresh"]
    assert stale["data_status"].to_list() == ["stale"]
    assert fresh["instrument_type"].to_list() == ["SWAP"]
    assert fresh["instrument_family"].to_list() == ["BTC-USDT"]
    assert fresh["base_currency"].to_list() == ["BTC"]
    assert fresh["open_interest_usd"].to_list() == [2_058_396_468.28]


def test_repository_partitions_and_idempotently_merges(tmp_path) -> None:
    repository = OpenInterestParquetRepository(tmp_path)
    cache = OpenInterestStateCache()
    cache.update(_state())
    first = cache.snapshot(
        {"BTC-USDT-SWAP": _instrument()},
        snapshot_time=datetime(2026, 7, 28, 12, 0, 42, tzinfo=UTC),
        stale_after=timedelta(seconds=120),
    )
    updated = first.with_columns(pl.lit(2_100_000_000.0).alias("open_interest_usd"))

    repository.write_snapshots(first)
    repository.write_snapshots(updated)
    snapshot_path = (
        tmp_path
        / "open_interest_snapshot"
        / "year=2026"
        / "date=2026-07-28"
        / "open_interest_snapshot-2026-07-28.parquet"
    )
    stored = pl.read_parquet(snapshot_path)

    assert snapshot_path.exists()
    assert stored.height == 1
    assert stored["open_interest_usd"].to_list() == [2_100_000_000.0]
    assert stored.schema == OPEN_INTEREST_SNAPSHOT_SCHEMA
    assert stored["snapshot_time"].to_list() == [datetime(2026, 7, 28, 12, 0, tzinfo=UTC)]


def test_rest_client_and_collector_cover_live_swaps_and_futures(tmp_path) -> None:
    captured_at = datetime(2026, 7, 28, 12, 0, 2, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        instrument_type = request.url.params["instType"]
        if request.url.path.endswith("/instruments"):
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
        if request.url.path.endswith("/open-interest"):
            row = (
                {
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "oi": "3244542.98",
                    "oiCcy": "32445.4298",
                    "oiUsd": "2058396468.28",
                    "ts": "1785240001000",
                }
                if instrument_type == "SWAP"
                else {
                    "instId": "BTC-USD-260731",
                    "instType": "FUTURES",
                    "oi": "10020.5",
                    "oiCcy": "15.8",
                    "oiUsd": "1002050",
                    "ts": "1785240001000",
                }
            )
            return httpx.Response(
                200,
                json={"code": "0", "msg": "", "data": [row]},
            )
        raise AssertionError(f"unexpected request: {request.url}")

    async def run_scenario() -> tuple[dict[str, OpenInterestInstrument], set[str], int]:
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://www.okx.com",
        )
        rest = OkxOpenInterestRestClient(
            client=http_client,
            requests_per_second=1_000_000,
            now=lambda: captured_at,
        )
        repository = OpenInterestParquetRepository(tmp_path)
        collector = OkxOpenInterestCollector(
            OpenInterestCollectorConfig(data_root=tmp_path),
            rest_client=rest,
            repository=repository,
            now=lambda: captured_at,
        )
        try:
            live = await rest.fetch_live_instruments(("SWAP", "FUTURES"))
            states = await rest.fetch_current(("SWAP", "FUTURES"))
            rows = await collector.run_once()
            return live, {state.instrument_type for state in states}, rows
        finally:
            await http_client.aclose()

    live, instrument_types, rows = asyncio.run(run_scenario())

    assert set(live) == {"BTC-USD-260731", "BTC-USDT-SWAP"}
    assert live["BTC-USD-260731"].instrument_type == "FUTURES"
    assert live["BTC-USD-260731"].instrument_family == "BTC-USD"
    assert live["BTC-USD-260731"].base_currency == "BTC"
    assert live["BTC-USD-260731"].expiration_time == datetime(2026, 7, 31, 8, tzinfo=UTC)
    assert instrument_types == {"SWAP", "FUTURES"}
    assert rows == 2
    stored = pl.read_parquet(
        tmp_path
        / "open_interest_snapshot"
        / "year=2026"
        / "date=2026-07-28"
        / "open_interest_snapshot-2026-07-28.parquet"
    )
    assert stored["instrument_type"].sort().to_list() == ["FUTURES", "SWAP"]
    assert stored["instrument_family"].sort().to_list() == ["BTC-USD", "BTC-USDT"]
    assert stored["base_currency"].unique().to_list() == ["BTC"]
