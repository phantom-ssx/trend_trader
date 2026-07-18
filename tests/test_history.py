from datetime import UTC, datetime

from trend_trader.data.history import (
    available_range,
    ceil_time,
    floor_time,
    parse_data_types,
    partition_ranges,
    target_instrument,
)
from trend_trader.data.models import DataType


def test_time_alignment_and_month_partitions() -> None:
    value = datetime(2024, 1, 1, 0, 2, 30, tzinfo=UTC)
    assert floor_time(value, 5) == datetime(2024, 1, 1, tzinfo=UTC)
    assert ceil_time(value, 5) == datetime(2024, 1, 1, 0, 5, tzinfo=UTC)
    ranges = partition_ranges(
        DataType.CANDLES,
        datetime(2024, 1, 31, 23, 59, 30, tzinfo=UTC),
        datetime(2024, 2, 1, 0, 2, 30, tzinfo=UTC),
    )
    assert ranges == [
        (
            datetime(2024, 2, 1, tzinfo=UTC),
            datetime(2024, 2, 1, 0, 2, tzinfo=UTC),
        )
    ]


def test_short_retention_ranges_and_market_cap_target() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2026, 7, 18, tzinfo=UTC)
    funding_start, _ = available_range(DataType.FUNDING_RATES, start, end)
    liquidation_start, _ = available_range(DataType.LIQUIDATIONS, start, end)
    assert funding_start == datetime(2026, 4, 17, tzinfo=UTC)
    assert liquidation_start == datetime(2026, 7, 15, tzinfo=UTC)
    assert target_instrument(DataType.MARKET_CAP, "ETH-USDT-SWAP") == ("GLOBAL", "ETH")
    assert target_instrument(DataType.CANDLES, "ETH-USDT-SWAP") == (
        "OKX",
        "ETH-USDT-SWAP",
    )


def test_parse_data_types() -> None:
    assert parse_data_types("candles,funding_rates") == (
        DataType.CANDLES,
        DataType.FUNDING_RATES,
    )
    assert set(parse_data_types("all")) == set(DataType)
