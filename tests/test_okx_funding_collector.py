import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl

from trend_trader.data.funding_storage import (
    FUNDING_HISTORY_SCHEMA,
    FUNDING_SNAPSHOT_SCHEMA,
    FundingParquetRepository,
    FundingState,
    FundingStateCache,
    build_history_frame,
)
from trend_trader.data.okx_funding_collector import (
    FundingCollectorConfig,
    OkxFundingRestClient,
    parse_ws_funding_states,
)


def _state(
    instrument_id: str = "BTC-USDT-SWAP",
    *,
    exchange_ts: datetime | None = None,
    received_at: datetime | None = None,
    data_source: str = "websocket",
) -> FundingState:
    return FundingState(
        venue="OKX",
        instrument_id=instrument_id,
        exchange_ts=exchange_ts or datetime(2026, 7, 28, 12, 0, 1, tzinfo=UTC),
        received_at=received_at or datetime(2026, 7, 28, 12, 0, 2, tzinfo=UTC),
        funding_rate=0.000101,
        next_funding_rate=0.000102,
        funding_time=datetime(2026, 7, 28, 16, 0, tzinfo=UTC),
        next_funding_time=datetime(2026, 7, 29, 0, 0, tzinfo=UTC),
        interest_rate=0.0001,
        premium=0.000001,
        method="next_period",
        formula_type="withRate",
        data_source=data_source,
    )


def test_default_history_window_starts_ten_days_before_utc_midnight() -> None:
    config = FundingCollectorConfig()

    assert config.initial_history_start(datetime(2026, 7, 28, 18, 35, 42, tzinfo=UTC)) == datetime(
        2026, 7, 18, 0, 0, tzinfo=UTC
    )


def test_parse_ws_state_keeps_rule_and_next_period_fields() -> None:
    message = json.dumps(
        {
            "arg": {"channel": "funding-rate", "instId": "BTC-USDT-SWAP"},
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingRate": "0.000101",
                    "nextFundingRate": "0.000102",
                    "fundingTime": "1785248000000",
                    "nextFundingTime": "1785276800000",
                    "interestRate": "0.0001",
                    "premium": "0.000001",
                    "method": "next_period",
                    "formulaType": "withRate",
                    "ts": "1785240001000",
                }
            ],
        }
    )
    states = parse_ws_funding_states(
        message,
        received_at=datetime(2026, 7, 28, 12, 0, 2, tzinfo=UTC),
    )

    assert len(states) == 1
    state = states[0]
    assert state.next_funding_rate == 0.000102
    assert state.method == "next_period"
    assert state.formula_type == "withRate"
    assert state.data_source == "websocket"
    assert state.received_at.tzinfo is UTC


def test_cache_rejects_older_rest_state_and_marks_staleness() -> None:
    cache = FundingStateCache()
    websocket_state = _state()
    older_rest_state = _state(
        exchange_ts=websocket_state.exchange_ts - timedelta(seconds=1),
        received_at=websocket_state.received_at + timedelta(seconds=5),
        data_source="rest",
    )

    assert cache.update(websocket_state)
    assert not cache.update(older_rest_state)

    fresh = cache.snapshot(
        {"BTC-USDT-SWAP"},
        snapshot_time=datetime(2026, 7, 28, 12, 1, tzinfo=UTC),
        stale_after=timedelta(seconds=120),
    )
    stale = cache.snapshot(
        {"BTC-USDT-SWAP"},
        snapshot_time=datetime(2026, 7, 28, 12, 3, tzinfo=UTC),
        stale_after=timedelta(seconds=120),
    )
    assert fresh["data_status"].to_list() == ["fresh"]
    assert stale["data_status"].to_list() == ["stale"]
    assert fresh["next_funding_rate"].to_list() == [0.000102]


def test_history_uses_realized_rate_and_skips_unconfirmed_rows() -> None:
    frame = build_history_frame(
        [
            {
                "fundingTime": "1785248000000",
                "fundingRate": "0.0002",
                "realizedRate": "0.00019",
                "method": "current_period",
                "formulaType": "withRate",
            },
            {
                "fundingTime": "1785276800000",
                "fundingRate": "0.0003",
                "realizedRate": "",
                "method": "current_period",
                "formulaType": "withRate",
            },
        ],
        "BTC-USDT-SWAP",
        received_at=datetime(2026, 7, 28, 12, tzinfo=UTC),
    )

    assert frame.height == 1
    assert frame["funding_rate"].to_list() == [0.00019]
    assert frame["method"].to_list() == ["current_period"]


def test_repository_partitions_and_idempotently_merges(tmp_path) -> None:
    repository = FundingParquetRepository(tmp_path)
    cache = FundingStateCache()
    cache.update(_state())
    first = cache.snapshot(
        {"BTC-USDT-SWAP"},
        snapshot_time=datetime(2026, 7, 28, 12, 0, 42, tzinfo=UTC),
        stale_after=timedelta(seconds=120),
    )
    updated = first.with_columns(pl.lit(0.0002).alias("funding_rate"))

    repository.write_snapshots(first)
    repository.write_snapshots(updated)
    snapshot_path = (
        tmp_path
        / "funding_snapshot"
        / "year=2026"
        / "date=2026-07-28"
        / "funding_snapshot-2026-07-28.parquet"
    )
    stored = pl.read_parquet(snapshot_path)

    assert snapshot_path.exists()
    assert stored.height == 1
    assert stored["funding_rate"].to_list() == [0.0002]
    assert stored.schema == FUNDING_SNAPSHOT_SCHEMA
    assert stored["snapshot_time"].to_list() == [datetime(2026, 7, 28, 12, 0, tzinfo=UTC)]

    history = build_history_frame(
        [
            {
                "fundingTime": "1785248000000",
                "fundingRate": "0.0002",
                "realizedRate": "0.00019",
                "method": "current_period",
                "formulaType": "withRate",
            }
        ],
        "BTC-USDT-SWAP",
        received_at=datetime(2026, 7, 28, 12, tzinfo=UTC),
    )
    repository.write_history(history)
    history_path = repository.history_path(history["funding_time"][0])
    assert "funding_history" in history_path.parts
    assert history_path.name == "funding_history-2026-07-28.parquet"
    assert pl.read_parquet(history_path).schema == FUNDING_HISTORY_SCHEMA


def test_rest_client_filters_live_swaps_and_builds_confirmed_history() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/instruments"):
            return httpx.Response(
                200,
                json={
                    "code": "0",
                    "msg": "",
                    "data": [
                        {"instId": "BTC-USDT-SWAP", "state": "live"},
                        {"instId": "OLD-USDT-SWAP", "state": "suspend"},
                    ],
                },
            )
        if request.url.path.endswith("/funding-rate-history"):
            if int(request.url.params["after"]) <= 1785248000000:
                return httpx.Response(
                    200,
                    json={"code": "0", "msg": "", "data": []},
                )
            return httpx.Response(
                200,
                json={
                    "code": "0",
                    "msg": "",
                    "data": [
                        {
                            "fundingTime": "1785248000000",
                            "fundingRate": "0.0002",
                            "realizedRate": "0.00019",
                            "method": "current_period",
                            "formulaType": "withRate",
                        }
                    ],
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    async def run_scenario() -> pl.DataFrame:
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://www.okx.com",
        )
        client = OkxFundingRestClient(
            client=http_client,
            requests_per_second=1_000_000,
        )
        try:
            assert await client.fetch_live_instruments() == ["BTC-USDT-SWAP"]
            return await client.fetch_history(
                "BTC-USDT-SWAP",
                datetime.fromtimestamp(1785247000, tz=UTC),
                datetime.fromtimestamp(1785249000, tz=UTC),
            )
        finally:
            await http_client.aclose()

    history = asyncio.run(run_scenario())

    assert history.height == 1
    assert history["funding_rate"].to_list() == [0.00019]
