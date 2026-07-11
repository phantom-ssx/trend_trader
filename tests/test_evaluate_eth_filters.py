import argparse

import pandas as pd
import pytest

from scripts.evaluate_eth_filters import add_indicators, backtest, parse_ma_pairs


def test_add_indicators_supports_custom_ma_periods() -> None:
    data = pd.DataFrame(
        {
            "high": range(1, 31),
            "low": range(1, 31),
            "close": range(1, 31),
            "volume": [1.0] * 30,
        }
    )

    add_indicators(data, fast_period=6, slow_period=24)

    assert data["ma_fast"].iloc[-1] == pytest.approx(27.5)
    assert data["ma_slow"].iloc[-1] == pytest.approx(18.5)
    assert data["ma6"].equals(data["ma_fast"])
    assert data["ma24"].equals(data["ma_slow"])


def test_parse_ma_pairs() -> None:
    assert parse_ma_pairs("5:20, 6:24,10:30") == [(5, 20), (6, 24), (10, 30)]


@pytest.mark.parametrize("value", ["20:5", "0:20", "5-20"])
def test_parse_ma_pairs_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_ma_pairs(value)


def test_backtest_calculates_closed_trade_statistics() -> None:
    data = pd.DataFrame(
        {
            "close": [10.0, 12.0, 10.0, 13.0],
        }
    )
    signals = pd.Series([1, -1, 0, 0])

    result = backtest(
        data,
        signals,
        "stats",
        starting_balance=100.0,
        fee_rate=0.0,
    )

    assert result.trades == 2
    assert result.winning_trades == 1
    assert result.losing_trades == 1
    assert result.win_rate_pct == pytest.approx(50.0)
    assert result.avg_win == pytest.approx(20.0)
    assert result.avg_loss == pytest.approx(-10.0)
    assert result.profit_loss_ratio == pytest.approx(2.0)
    assert result.max_win == pytest.approx(20.0)
    assert result.min_win == pytest.approx(20.0)
    assert result.win_variance == pytest.approx(0.0)
    assert result.max_loss == pytest.approx(-10.0)
    assert result.min_loss == pytest.approx(-10.0)
    assert result.loss_variance == pytest.approx(0.0)
    assert sum([result.avg_win, result.avg_loss]) == pytest.approx(result.net_pnl)


def test_trade_statistics_include_fees_and_same_direction_rebalances() -> None:
    data = pd.DataFrame({"close": [10.0, 11.0, 12.0, 11.0]})
    result = backtest(
        data,
        pd.Series([1, 1, -1, 0]),
        "fees",
        starting_balance=100.0,
        fee_rate=0.001,
    )

    pnl_from_trades = (
        result.avg_win * result.winning_trades
        + result.avg_loss * result.losing_trades
    )
    assert pnl_from_trades == pytest.approx(result.net_pnl)


def test_trade_statistics_calculate_profit_and_loss_ranges_and_variances() -> None:
    data = pd.DataFrame({"close": [10.0, 12.0, 14.0, 13.0, 11.0, 10.0]})
    result = backtest(
        data,
        pd.Series([1, -1, 1, -1, 1, 0]),
        "distribution",
        starting_balance=100.0,
        fee_rate=0.0,
    )

    assert result.max_win >= result.min_win > 0
    assert result.max_loss < 0
    assert result.min_loss <= result.max_loss
    assert result.win_variance >= 0
    assert result.loss_variance >= 0
