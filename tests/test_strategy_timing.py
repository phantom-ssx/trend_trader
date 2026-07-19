from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trend_trader.experiments.strategy.timing import (
    aligned_daily_returns,
    equal_weight_daily_wealth,
    interval_daily_wealth,
    streaming_daily_wealth,
)


def test_interval_daily_wealth_marks_open_position_and_cost() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    portfolio = pl.DataFrame(
        {
            "timestamp": [start],
            "exit_time": [start + timedelta(days=2)],
            "position": [1.0],
            "transaction_cost": [0.01],
            "portfolio_return": [0.19],
            "wealth": [1.19],
        }
    )
    candles = pl.DataFrame(
        {
            "timestamp": [start, start + timedelta(days=1), start + timedelta(days=2)],
            "open": [100.0, 110.0, 120.0],
        }
    )

    wealth = interval_daily_wealth(portfolio, candles)

    assert wealth[start] == pytest.approx(0.99)
    assert wealth[start + timedelta(days=1)] == pytest.approx(1.09)
    assert wealth[start + timedelta(days=2)] == pytest.approx(1.19)


def test_streaming_and_phase_aggregation() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    portfolio = pl.DataFrame(
        {
            "timestamp": [start, start + timedelta(days=1)],
            "horizon_bars": [1, 1],
            "portfolio_return": [0.10, -0.05],
            "transaction_cost": [0.0, 0.0],
        }
    )
    first = streaming_daily_wealth(portfolio, horizon_bars=1)
    second = {
        start + timedelta(days=1): 1.0,
        start + timedelta(days=2): 1.10,
    }

    combined = equal_weight_daily_wealth({"first": first, "second": second})
    returns = aligned_daily_returns({"first": first, "second": second})

    assert combined[start + timedelta(days=2)] == pytest.approx(1.025)
    assert returns["first"].to_list() == pytest.approx([-0.05])
    assert returns["second"].to_list() == pytest.approx([0.10])
