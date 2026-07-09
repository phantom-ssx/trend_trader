from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from scripts.evaluate_filters import (
    add_indicators,
    monthly_results,
    run_all_in_backtest,
    spread_confirm_signals,
)


def make_frame(prices: list[float], *, start_month: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [
                datetime(2026, start_month, 1, tzinfo=UTC) + timedelta(hours=index)
                for index in range(len(prices))
            ],
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [10.0] * len(prices),
        }
    )


def test_add_indicators_adds_filter_columns() -> None:
    data = add_indicators(make_frame([float(index) for index in range(1, 31)]))

    assert {"ma5", "ma20", "spread_pct", "atr_pct", "adx14"}.issubset(data.columns)
    assert data["ma5"].iloc[-1] == pytest.approx(28.0)
    assert data["ma20"].iloc[-1] == pytest.approx(20.5)


def test_spread_confirm_signals_uses_threshold_cross() -> None:
    data = add_indicators(make_frame([10.0] * 20 + [12.0] * 5 + [8.0] * 5))

    signals = spread_confirm_signals(data, 0.01)

    assert 1 in set(signals)
    assert -1 in set(signals)


def test_all_in_backtest_uses_equity_sizing() -> None:
    data = make_frame([10.0, 12.0, 14.0])
    signals = pd.Series([1, 0, 0])

    result = run_all_in_backtest(
        data,
        signals,
        "test",
        starting_balance=100.0,
        fee_rate=0.0,
    )

    assert result.final_equity == pytest.approx(140.0)
    assert result.return_pct == pytest.approx(40.0)
    assert result.events == 2


def test_monthly_results_returns_each_strategy_per_month() -> None:
    january = make_frame([10.0] * 35, start_month=1)
    february = make_frame([20.0] * 35, start_month=2)
    data = add_indicators(pd.concat([january, february], ignore_index=True))

    rows = monthly_results(data, starting_balance=100.0, fee_rate=0.0)

    assert {month for month, _ in rows} == {"2026-01", "2026-02"}
    assert len(rows) == 12
