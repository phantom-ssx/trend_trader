import pandas as pd
import pytest

from scripts.research_eth_4h_macd_extrema import macd_extrema_targets, run_backtest


def test_macd_extrema_are_confirmed_one_bar_after_the_turn() -> None:
    data = pd.DataFrame({"macd": [0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0]})

    targets = macd_extrema_targets(data)

    assert targets.tolist() == [0, 0, 0, -1, -1, -1, -1, 1]


def test_expected_sign_filter_ignores_peaks_below_zero_and_troughs_above_zero() -> None:
    data = pd.DataFrame({"macd": [1.0, 2.0, 1.0, -1.0, -2.0, -1.0]})

    targets = macd_extrema_targets(data, require_expected_sign=True)

    assert targets.tolist() == [0, 0, -1, -1, -1, 1]


def test_backtest_executes_confirmed_target_at_next_open() -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    data = pd.DataFrame(
        {
            "ts": timestamps,
            "open": [100.0, 110.0, 120.0, 130.0],
            "close": [105.0, 115.0, 125.0, 135.0],
        }
    )

    summary, _, trades = run_backtest(
        data,
        pd.Series([1, 1, 1, 1]),
        starting_balance=100.0,
        fee_rate=0.0,
    )

    assert summary.trades == 1
    assert trades.iloc[0]["entry_time"] == timestamps[1]
    assert trades.iloc[0]["entry_price"] == 110.0
    assert summary.final_equity == pytest.approx(100.0 * 135.0 / 110.0)


def test_round_trip_fees_are_charged_on_entry_and_exit() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    data = pd.DataFrame(
        {
            "ts": timestamps,
            "open": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
        }
    )

    summary, _, _ = run_backtest(
        data,
        pd.Series([1, 1, 1]),
        starting_balance=100.0,
        fee_rate=0.001,
    )

    assert summary.final_equity == pytest.approx(99.8)
    assert summary.total_fees == pytest.approx(0.2)
