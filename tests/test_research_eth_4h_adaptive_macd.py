import pandas as pd
import pytest

from scripts.research_eth_4h_adaptive_macd import (
    MacdParameters,
    adaptive_macd,
    efficiency_ratio,
    variable_ema,
)


def test_efficiency_ratio_distinguishes_trend_from_round_trip_noise() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])

    result = efficiency_ratio(close, lookback=2)

    assert result.iloc[2] == pytest.approx(1.0)
    assert result.iloc[4] == pytest.approx(1.0)
    assert efficiency_ratio(pd.Series([1.0, 2.0, 1.0]), 2).iloc[-1] == pytest.approx(0.0)


def test_variable_ema_does_not_change_history_when_future_data_is_appended() -> None:
    values = pd.Series([100.0, 102.0, 101.0, 105.0, 104.0])
    alphas = pd.Series([float("nan"), 0.2, 0.4, 0.3, 0.5])

    short = variable_ema(values.iloc[:4], alphas.iloc[:4])
    full = variable_ema(values, alphas)

    pd.testing.assert_series_equal(full.iloc[:4], short)


def test_adaptive_macd_is_causal() -> None:
    close = pd.Series([100.0 + index + (index % 3) for index in range(30)])
    base = pd.DataFrame({"close": close})
    parameters = MacdParameters(5, 12, 4)

    short = adaptive_macd(
        base.iloc[:25],
        parameters,
        er_lookback=6,
        model="er_scaled_balanced",
    )
    full = adaptive_macd(
        base,
        parameters,
        er_lookback=6,
        model="er_scaled_balanced",
    )

    assert full["macd"].iloc[:25].tolist() == pytest.approx(short["macd"].tolist(), nan_ok=True)
