import pandas as pd
import pytest

from scripts.evaluate_eth_hourly_pyramiding import (
    PyramidPlan,
    backtest_fractional_targets,
    pyramid_targets,
)


def test_pyramid_targets_add_but_do_not_reduce_before_reversal() -> None:
    data = pd.DataFrame({
        "spread_pct": [0.003, 0.004, 0.006, 0.004, -0.004, -0.006],
        "atr_pct": [0.006] * 6,
    })
    plan = PyramidPlan("staged", (0.5, 1.0), (0.0035, 0.005))

    assert pyramid_targets(data, plan).tolist() == [0, 0.5, 1.0, 1.0, -0.5, -1.0]


def test_fractional_backtest_trades_close_target_at_next_open() -> None:
    data = pd.DataFrame({"open": [100.0, 100.0, 120.0], "close": [100.0, 120.0, 120.0]})
    targets = pd.Series([0.5, 1.0, 1.0])

    result = backtest_fractional_targets(
        data, targets, "staged", starting_balance=100.0, fee_rate=0.0
    )

    assert result.return_pct == pytest.approx(10.0)
    assert result.orders == 3
