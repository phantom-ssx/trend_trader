from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trend_trader.backtest.run_backtest import run_sma_cross_backtest
from trend_trader.strategies.sma_cross import SmaCrossSignal


def test_sma_cross_signal_emits_only_on_real_crosses() -> None:
    signal = SmaCrossSignal(fast_period=2, slow_period=3)

    emitted = [signal.on_price(price) for price in [3, 2, 1, 2, 3, 2, 1]]

    assert emitted == [None, None, None, None, "BUY", None, "SELL"]


def test_sma_cross_backtest_flips_and_closes_final_position() -> None:
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(7)]
    df = pl.DataFrame(
        {
            "ts": ts,
            "open": [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            "high": [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            "low": [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            "close": [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            "volume": [1.0] * 7,
        }
    )

    result = run_sma_cross_backtest(
        df,
        fast_period=2,
        slow_period=3,
        starting_balance=100.0,
        trade_size=1.0,
        fee_rate=0.0,
    )

    assert [trade.side for trade in result.trades] == ["BUY", "SELL", "CLOSE"]
    assert result.long_entries == 1
    assert result.short_entries == 1
    assert result.net_pnl == -2.0
    assert result.final_equity == 98.0


def test_sma_cross_backtest_all_in_uses_current_equity() -> None:
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(6)]
    prices = [3.0, 2.0, 1.0, 2.0, 3.0, 4.0]
    df = pl.DataFrame(
        {
            "ts": ts,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1.0] * 6,
        }
    )

    result = run_sma_cross_backtest(
        df,
        fast_period=2,
        slow_period=3,
        starting_balance=100.0,
        trade_size=1.0,
        fee_rate=0.0,
        sizing="all-in",
    )

    assert [trade.side for trade in result.trades] == ["BUY", "CLOSE"]
    assert result.trades[0].position == pytest.approx(100.0 / 3.0)
    assert result.final_equity == pytest.approx(100.0 / 3.0 * 4.0)
