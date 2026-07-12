import pandas as pd
import pytest

from scripts.evaluate_eth_15m_stop_losses import backtest_with_stop_loss


def test_long_stop_loss_uses_intrabar_low_and_charges_fees() -> None:
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 94.0],
            "close": [100.0, 99.0],
        }
    )
    result = backtest_with_stop_loss(
        data,
        pd.Series([1, 0]),
        "long_stop",
        starting_balance=10_000.0,
        fee_rate=0.0005,
        stop_loss_pct=0.05,
    )

    assert result.trades == 1
    assert result.losing_trades == 1
    assert result.final_equity == pytest.approx(9490.25)


def test_short_stop_loss_fills_at_worse_open_after_gap() -> None:
    data = pd.DataFrame(
        {
            "open": [100.0, 108.0],
            "high": [101.0, 110.0],
            "low": [99.0, 107.0],
            "close": [100.0, 109.0],
        }
    )
    result = backtest_with_stop_loss(
        data,
        pd.Series([-1, 0]),
        "short_gap",
        starting_balance=10_000.0,
        fee_rate=0.0,
        stop_loss_pct=0.05,
    )

    assert result.net_pnl == pytest.approx(-800.0)
