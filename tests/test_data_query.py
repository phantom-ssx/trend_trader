from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trend_trader.data import DataQuery, DataType, DataUnavailableError, MarketDataClient
from trend_trader.data.models import STORED_BAR_TYPES, FetchRequest, bar_minutes
from trend_trader.data.schema import empty_frame


def minute_candles(request: FetchRequest) -> pl.DataFrame:
    timestamps = pl.datetime_range(
        request.start,
        request.end - timedelta(minutes=1),
        interval="1m",
        eager=True,
        time_zone="UTC",
    )
    size = len(timestamps)
    values = [float(index + 1) for index in range(size)]
    return pl.DataFrame(
        {
            "venue": [request.venue] * size,
            "instrument_id": [request.instrument_id] * size,
            "bar_type": ["1m"] * size,
            "timestamp": timestamps,
            "open": values,
            "high": [value + 1 for value in values],
            "low": [value - 0.5 for value in values],
            "close": [value + 0.5 for value in values],
            "volume": [1.0] * size,
            "volume_ccy": [2.0] * size,
            "volume_quote": [3.0] * size,
            "confirm": [1] * size,
        }
    )


class RecordingSource:
    name = "recording"

    def __init__(self) -> None:
        self.requests: list[FetchRequest] = []

    def supports(self, request: FetchRequest) -> bool:
        return request.venue in {"OKX", "GLOBAL"}

    async def fetch(self, request: FetchRequest) -> pl.DataFrame:
        self.requests.append(request)
        if request.data_type is DataType.CANDLES:
            return minute_candles(request)
        if request.data_type is DataType.FUNDING_RATES:
            return pl.DataFrame(
                {
                    "venue": [request.venue],
                    "instrument_id": [request.instrument_id],
                    "timestamp": [request.start],
                    "funding_rate": [0.0001],
                    "realized_rate": [0.00009],
                    "method": ["current_period"],
                    "formula_type": ["withRate"],
                }
            )
        if request.data_type is DataType.LIQUIDATIONS:
            return pl.DataFrame(
                {
                    "venue": [request.venue],
                    "instrument_id": [request.instrument_id],
                    "timestamp": [request.start],
                    "liquidation_id": ["liq-1"],
                    "side": ["sell"],
                    "position_side": ["long"],
                    "bankruptcy_price": [2000.0],
                    "size": [10.0],
                    "bankruptcy_loss": [2.0],
                }
            )
        return periodic_metric_frame(request)


def periodic_metric_frame(request: FetchRequest) -> pl.DataFrame:
    bar_type = STORED_BAR_TYPES[request.data_type]
    step = timedelta(minutes=bar_minutes(bar_type))
    timestamps = pl.datetime_range(
        request.start,
        request.end - step,
        interval=bar_type,
        eager=True,
        time_zone="UTC",
    )
    size = len(timestamps)
    common: dict[str, object] = {
        "venue": [request.venue] * size,
        "instrument_id": [request.instrument_id] * size,
        "bar_type": [bar_type] * size,
        "timestamp": timestamps,
    }
    values = [float(index + 1) for index in range(size)]
    extras: dict[DataType, dict[str, object]] = {
        DataType.CONTRACT_BASIS: {
            "mark_price": [100.0 + value for value in values],
            "index_price": [100.0] * size,
            "basis": values,
            "basis_rate": [value / 100 for value in values],
        },
        DataType.OPEN_INTEREST: {
            "open_interest_usd": values,
            "volume_usd": [value * 2 for value in values],
        },
        DataType.LONG_SHORT_RATIO: {"long_short_ratio": values},
        DataType.MARKET_CAP: {
            "market_cap_usd": values,
            "price_usd": [value * 10 for value in values],
            "volume_24h_usd": [value * 100 for value in values],
        },
        DataType.TAKER_VOLUME: {
            "buy_volume": values,
            "sell_volume": [value / 2 for value in values],
            "net_buy_volume": [value / 2 for value in values],
        },
    }
    if request.data_type not in extras:
        return empty_frame(request.data_type)
    return pl.DataFrame({**common, **extras[request.data_type]})


def test_query_downloads_one_minute_data_then_reads_local_and_aggregates(tmp_path) -> None:
    source = RecordingSource()
    client = MarketDataClient(data_root=tmp_path / "market", sources=[source])

    result = client.candles(
        "ETH-USDT-SWAP",
        "1h",
        "2024-01-01T00:00:00Z",
        "2024-01-01T02:00:00Z",
    )

    assert len(source.requests) == 1
    assert source.requests[0].bar_type == "1m"
    assert result.height == 2
    assert result.columns[:4] == ["venue", "instrument_id", "bar_type", "timestamp"]
    assert result["bar_type"].to_list() == ["1h", "1h"]
    assert result["volume"].to_list() == [60.0, 60.0]

    stored_path = (
        tmp_path
        / "market/candles/venue=OKX/instrument_id=ETH-USDT-SWAP/"
        "bar_type=1m/year=2024/month=01/data.parquet"
    )
    stored = pl.read_parquet(stored_path)
    assert stored.height == 120
    assert stored.columns[:4] == ["venue", "instrument_id", "bar_type", "timestamp"]
    assert stored["bar_type"].unique().to_list() == ["1m"]

    second = client.candles(
        "ETH-USDT-SWAP",
        "1h",
        "2024-01-01T00:00:00Z",
        "2024-01-01T02:00:00Z",
    )
    assert second.equals(result)
    assert len(source.requests) == 1

    stored_path.unlink()
    repaired = client.candles(
        "ETH-USDT-SWAP",
        "1h",
        "2024-01-01T00:00:00Z",
        "2024-01-01T02:00:00Z",
    )
    assert repaired.equals(result)
    assert len(source.requests) == 2

    local_only_client = MarketDataClient(data_root=tmp_path / "market", sources=[])
    local_result = local_only_client.candles(
        "ETH-USDT-SWAP",
        "1h",
        "2024-01-01T00:00:00Z",
        "2024-01-01T02:00:00Z",
    )
    assert local_result.equals(result)

    (tmp_path / "market/catalog.sqlite").unlink()
    rebuilt_client = MarketDataClient(data_root=tmp_path / "market", sources=[])
    rebuilt_result = rebuilt_client.candles(
        "ETH-USDT-SWAP",
        "1h",
        "2024-01-01T00:00:00Z",
        "2024-01-01T02:00:00Z",
    )
    assert rebuilt_result.equals(result)


def test_query_only_downloads_the_missing_tail(tmp_path) -> None:
    source = RecordingSource()
    client = MarketDataClient(data_root=tmp_path / "market", sources=[source])
    client.candles(
        "ETH-USDT-SWAP",
        "1h",
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
    )

    result = client.candles(
        "ETH-USDT-SWAP",
        "1h",
        "2024-01-01T00:00:00Z",
        "2024-01-01T02:00:00Z",
    )

    assert result.height == 2
    assert len(source.requests) == 2
    assert source.requests[1].start == datetime(2024, 1, 1, 1, tzinfo=UTC)
    assert source.requests[1].end == datetime(2024, 1, 1, 2, tzinfo=UTC)


def test_query_imports_matching_legacy_data_before_using_network(tmp_path) -> None:
    request = FetchRequest(
        DataType.CANDLES,
        "OKX",
        "ETH-USDT-SWAP",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 1, tzinfo=UTC),
        bar_type="1m",
    )
    legacy = minute_candles(request).rename(
        {
            "venue": "exchange",
            "instrument_id": "inst_id",
            "bar_type": "bar",
            "timestamp": "ts",
        }
    )
    legacy_root = tmp_path / "clean"
    legacy_path = legacy_root / "okx/ETH-USDT-SWAP/ETH-USDT-SWAP_1m_2024.parquet"
    legacy_path.parent.mkdir(parents=True)
    legacy.write_parquet(legacy_path)
    client = MarketDataClient(
        data_root=tmp_path / "market",
        legacy_data_root=legacy_root,
        sources=[],
    )

    result = client.candles(
        "ETH-USDT-SWAP",
        "1h",
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
    )

    assert result.height == 1
    assert result["volume"].to_list() == [60.0]
    assert (
        tmp_path
        / "market/candles/venue=OKX/instrument_id=ETH-USDT-SWAP/"
        "bar_type=1m/year=2024/month=01/data.parquet"
    ).exists()


def test_query_splits_storage_and_downloads_at_month_boundaries(tmp_path) -> None:
    source = RecordingSource()
    client = MarketDataClient(data_root=tmp_path / "market", sources=[source])

    result = client.candles(
        "ETH-USDT-SWAP",
        "1h",
        "2024-01-31T23:00:00Z",
        "2024-02-01T01:00:00Z",
    )

    assert result.height == 2
    assert len(source.requests) == 2
    instrument_root = (
        tmp_path
        / "market/candles/venue=OKX/instrument_id=ETH-USDT-SWAP/bar_type=1m/year=2024"
    )
    assert (instrument_root / "month=01/data.parquet").exists()
    assert (instrument_root / "month=02/data.parquet").exists()


def test_funding_rates_use_the_same_local_first_api(tmp_path) -> None:
    source = RecordingSource()
    client = MarketDataClient(data_root=tmp_path / "market", sources=[source])

    result = client.funding_rates(
        "ETH-USDT-SWAP",
        "2024-01-01T00:00:00Z",
        "2024-02-01T00:00:00Z",
    )

    assert result.height == 1
    assert result.columns[:3] == ["venue", "instrument_id", "timestamp"]
    assert result["funding_rate"].to_list() == [0.0001]


def test_missing_local_data_without_a_source_is_explicit(tmp_path) -> None:
    client = MarketDataClient(data_root=tmp_path / "market", sources=[])
    query = DataQuery(
        "candles",
        "ETH-USDT-SWAP",
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
        bar_type="1h",
    )

    with pytest.raises(DataUnavailableError, match="no remote source supports"):
        client.query(query)


def test_query_uses_full_names_and_requires_aligned_bars() -> None:
    query = DataQuery(
        "candles",
        "ETH-USDT-SWAP",
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
        venue="okx",
        bar_type="1H",
    )
    assert query.instrument_id == "ETH-USDT-SWAP"
    assert query.venue == "OKX"
    assert query.bar_type == "1h"

    with pytest.raises(ValueError, match="align"):
        DataQuery(
            "candles",
            "ETH-USDT-SWAP",
            "2024-01-01T00:30:00Z",
            "2024-01-01T01:00:00Z",
            bar_type="1h",
        )


def test_extended_periodic_data_uses_base_granularity_and_aggregation(tmp_path) -> None:
    source = RecordingSource()
    client = MarketDataClient(data_root=tmp_path / "market", sources=[source])

    basis = client.contract_basis(
        "ETH-USDT-SWAP",
        "5m",
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:10:00Z",
    )
    open_interest = client.open_interest(
        "ETH-USDT-SWAP",
        "10m",
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:20:00Z",
    )
    ratio = client.long_short_ratio(
        "ETH-USDT-SWAP",
        "5m",
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:10:00Z",
    )
    taker = client.taker_volume(
        "ETH-USDT-SWAP",
        "10m",
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:20:00Z",
    )
    market_cap = client.market_cap(
        "ETH",
        "2024-01-01T00:00:00Z",
        "2024-01-03T00:00:00Z",
    )

    assert basis.height == 2
    assert basis["bar_type"].to_list() == ["5m", "5m"]
    assert open_interest.height == 2
    assert open_interest["open_interest_usd"].to_list() == [2.0, 4.0]
    assert ratio["long_short_ratio"].to_list() == [1.0, 2.0]
    assert taker.height == 2
    assert taker["buy_volume"].to_list() == [3.0, 7.0]
    assert taker["net_buy_volume"].to_list() == [1.5, 3.5]
    assert market_cap["venue"].unique().to_list() == ["GLOBAL"]
    assert {request.data_type for request in source.requests} == {
        DataType.CONTRACT_BASIS,
        DataType.OPEN_INTEREST,
        DataType.LONG_SHORT_RATIO,
        DataType.TAKER_VOLUME,
        DataType.MARKET_CAP,
    }


def test_liquidations_are_event_data_with_a_stable_primary_key(tmp_path) -> None:
    source = RecordingSource()
    client = MarketDataClient(data_root=tmp_path / "market", sources=[source])

    result = client.liquidations(
        "ETH-USDT-SWAP",
        "2024-01-01T00:00:00Z",
        "2024-01-02T00:00:00Z",
    )

    assert result.height == 1
    assert result["liquidation_id"].to_list() == ["liq-1"]
    assert "bar_type" not in result.columns


def test_periodic_query_rejects_finer_than_stored_interval() -> None:
    with pytest.raises(ValueError, match="stored 5m"):
        DataQuery(
            DataType.OPEN_INTEREST,
            "ETH-USDT-SWAP",
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:05:00Z",
            bar_type="1m",
        )
