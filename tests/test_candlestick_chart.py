from datetime import UTC, datetime, timedelta

import polars as pl

from trend_trader.visualization.candlestick_chart import (
    load_candles,
    moving_average_data,
    parse_ma_periods,
    render_html,
)


def test_load_candles_filters_and_resamples(tmp_path) -> None:
    parquet_path = tmp_path / "candles.parquet"
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(10)]
    df = pl.DataFrame(
        {
            "ts": ts,
            "open": [float(i) for i in range(10)],
            "high": [float(i + 1) for i in range(10)],
            "low": [float(i - 1) for i in range(10)],
            "close": [float(i) + 0.5 for i in range(10)],
            "volume": [1.0] * 10,
        }
    )
    df.write_parquet(parquet_path)

    loaded = load_candles(
        parquet_path,
        start=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
        end=datetime(2026, 1, 1, 0, 8, tzinfo=UTC),
        resample="5m",
    )

    assert loaded.height == 2
    assert loaded["open"].to_list() == [2.0, 5.0]
    assert loaded["high"].to_list() == [5.0, 8.0]
    assert loaded["low"].to_list() == [1.0, 4.0]
    assert loaded["close"].to_list() == [4.5, 7.5]
    assert loaded["volume"].to_list() == [3.0, 3.0]


def test_moving_average_data_uses_close_prices() -> None:
    candles = [
        {"time": i, "open": float(i), "high": float(i), "low": float(i), "close": float(i)}
        for i in range(1, 6)
    ]

    assert moving_average_data(candles, 3) == [
        {"time": 3, "value": 2.0},
        {"time": 4, "value": 3.0},
        {"time": 5, "value": 4.0},
    ]


def test_parse_ma_periods_deduplicates_and_allows_empty() -> None:
    assert parse_ma_periods("5, 10, 5") == (5, 10)
    assert parse_ma_periods("") == ()


def test_render_html_includes_moving_average_series() -> None:
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(4)]
    df = pl.DataFrame(
        {
            "ts": ts,
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [0.0, 1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0, 4.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )

    html = render_html(
        "Test",
        df,
        source_path="candles.parquet",
        start=None,
        end=None,
        resample=None,
        ma_periods=(2,),
    )

    assert '"period":2' in html
    assert '"value":1.5' in html
    assert "chart.addLineSeries" in html
    assert "MA${ma.period}" in html
