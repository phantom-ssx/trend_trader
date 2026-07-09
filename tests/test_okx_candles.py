from datetime import UTC, datetime

from trend_trader.data.okx_candles import (
    build_frame,
    clean_candles,
    default_output_path,
    split_time_range,
    to_ms,
)


def test_build_and_clean_candles_deduplicates_and_sorts() -> None:
    rows = [
        ["1704067260000", "11", "13", "10", "12", "1", "2", "12", "1"],
        ["1704067200000", "10", "12", "9", "11", "1", "2", "11", "1"],
        ["1704067200000", "10", "12", "9", "11", "1", "2", "11", "1"],
    ]
    start = to_ms(datetime(2024, 1, 1, tzinfo=UTC))
    end = to_ms(datetime(2024, 1, 1, 0, 2, tzinfo=UTC))

    cleaned = clean_candles(build_frame(rows, "BTC-USDT-SWAP", "1m"), start, end)

    assert cleaned.height == 2
    assert cleaned["close"].to_list() == [11.0, 12.0]


def test_default_output_path_includes_time_range() -> None:
    path = default_output_path(
        "ETH-USDT-SWAP",
        "1m",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2026, 7, 7, 12, 4, 7, tzinfo=UTC),
    )

    assert str(path) == (
        "data/clean/okx/ETH-USDT-SWAP/"
        "ETH-USDT-SWAP_1m_20240101T000000Z_20260707T120407Z.parquet"
    )


def test_split_time_range() -> None:
    chunks = split_time_range(
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 10, 12, tzinfo=UTC),
        chunk_days=7,
    )

    assert chunks == [
        (
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 8, tzinfo=UTC),
        ),
        (
            datetime(2026, 1, 8, tzinfo=UTC),
            datetime(2026, 1, 10, 12, tzinfo=UTC),
        ),
    ]
