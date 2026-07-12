import pandas as pd
import pytest

from scripts.evaluate_eth_hourly_adaptive import (
    AdaptiveParameters,
    adaptive_targets,
    add_adaptive_indicators,
    backtest_next_open,
    direction_consensus_targets,
)


def test_adaptive_indicators_are_causal_and_scale_free() -> None:
    data = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 104.0, 103.0, 106.0],
            "atr_pct": [0.01, 0.02, 0.03, 0.02, 0.04, 0.05],
            "atr14": [1.0] * 6,
            "spread": [0.5] * 6,
        }
    )
    add_adaptive_indicators(data, atr_lookback=3, efficiency_lookback=2)

    assert data["atr_percentile"].iloc[-1] == pytest.approx(1.0)
    assert data["spread_atr"].iloc[-1] == pytest.approx(0.5)
    assert data["efficiency_ratio"].iloc[-1] == pytest.approx(0.5)


def test_adaptive_targets_use_zero_as_no_trade_state() -> None:
    data = pd.DataFrame(
        {
            "spread": [0.1, 0.5, -0.5, 0.1],
            "atr14": [1.0] * 4,
            "spread_atr": [0.1, 0.5, 0.5, 0.1],
            "atr_percentile": [0.8, 0.8, 0.8, 0.1],
            "efficiency_ratio": [0.5, 0.5, 0.5, 0.5],
        }
    )
    params = AdaptiveParameters("test", 0.3, 0.5, 0.2)

    assert adaptive_targets(data, params).tolist() == [0, 1, -1, -1]


def test_backtest_executes_close_signal_at_next_open() -> None:
    data = pd.DataFrame(
        {
            "open": [100.0, 110.0, 120.0],
            "close": [105.0, 115.0, 125.0],
        }
    )
    result = backtest_next_open(
        data,
        pd.Series([1, 0, 0]),
        "next_open",
        starting_balance=100.0,
        fee_rate=0.0,
    )

    assert result.trades == 1
    assert result.net_pnl == pytest.approx(100.0 * (120.0 / 110.0 - 1))


def test_direction_consensus_waits_for_warmup_and_filters_both_sides() -> None:
    hours = 24 * 3
    close = [100.0] * (24 * 2) + [101.0 + item for item in range(24)]
    data = pd.DataFrame({"close": close})
    targets = pd.Series([1] * hours)

    filtered = direction_consensus_targets(
        data,
        targets,
        fast_days=1,
        slow_days=2,
    )

    assert filtered.iloc[: 24 * 2 - 1].eq(0).all()
    assert filtered.iloc[-1] == 1

    short_targets = pd.Series([-1] * hours)
    short_filtered = direction_consensus_targets(
        data,
        short_targets,
        fast_days=1,
        slow_days=2,
    )
    assert short_filtered.iloc[-1] == 0
