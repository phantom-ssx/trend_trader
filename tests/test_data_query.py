from datetime import UTC, datetime

import polars as pl
import pytest

from trend_trader.data import DataQuery, DataType, MarketDataClient


def candle_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
            ],
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10.0, 20.0, 30.0],
            "exchange": ["OKX", "OKX", "OKX"],
            "inst_id": ["ETH-USDT-SWAP", "ETH-USDT-SWAP", "ETH-USDT-SWAP"],
            "bar": ["1D", "1D", "1D"],
        }
    )


def test_parquet_candle_query_filters_half_open_time_range(tmp_path) -> None:
    path = tmp_path / "candles.parquet"
    candle_frame().write_parquet(path)

    result = MarketDataClient().candles(
        "ETH-USDT-SWAP",
        "1D",
        "2024-01-02T00:00:00Z",
        "2024-01-03T00:00:00Z",
        path=path,
    )

    assert result.height == 1
    assert result["close"].to_list() == [2.5]


def test_query_requires_bar_only_for_candles() -> None:
    with pytest.raises(ValueError, match="bar is required"):
        DataQuery(DataType.CANDLES, "ETH-USDT-SWAP", "2024-01-01", "2024-01-02")

    with pytest.raises(ValueError, match="only valid"):
        DataQuery(
            DataType.FUNDING_RATES,
            "ETH-USDT-SWAP",
            "2024-01-01",
            "2024-01-02",
            bar="1h",
        )


def test_custom_source_is_registered_and_validated() -> None:
    class MemorySource:
        name = "memory"

        def supports(self, query: DataQuery) -> bool:
            return True

        async def load(self, query: DataQuery) -> pl.DataFrame:
            return candle_frame()

    client = MarketDataClient(sources=[MemorySource()])
    result = client.query(
        DataQuery(
            "candles",
            "ETH-USDT-SWAP",
            "2024-01-01T00:00:00Z",
            "2024-01-04T00:00:00Z",
            bar="1D",
        )
    )

    assert result.height == 3


def test_result_schema_is_validated() -> None:
    class InvalidSource:
        name = "invalid"

        def supports(self, query: DataQuery) -> bool:
            return True

        async def load(self, query: DataQuery) -> pl.DataFrame:
            return pl.DataFrame({"ts": [datetime(2024, 1, 1, tzinfo=UTC)]})

    client = MarketDataClient(sources=[InvalidSource()])
    query = DataQuery("funding_rates", "ETH-USDT-SWAP", "2024-01-01", "2024-01-02")

    with pytest.raises(ValueError, match="funding_rate"):
        client.query(query)


def test_source_options_are_not_silently_ignored(tmp_path) -> None:
    path = tmp_path / "candles.parquet"
    candle_frame().write_parquet(path)

    with pytest.raises(ValueError, match="does not accept options: typo"):
        MarketDataClient().candles(
            "ETH-USDT-SWAP",
            "1D",
            "2024-01-01",
            "2024-01-02",
            path=path,
            typo=True,
        )
