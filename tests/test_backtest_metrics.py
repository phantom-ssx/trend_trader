from __future__ import annotations

import math

import pytest

from trend_trader.backtest.metrics import annualized_sharpe_ratio


def test_annualized_sharpe_ratio_uses_daily_equity_returns() -> None:
    equities = [100.0, 101.0, 100.495, 102.5049]
    result = annualized_sharpe_ratio(
        ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        equities,
    )
    daily_returns = [0.01, -0.005, 0.02]
    mean = sum(daily_returns) / len(daily_returns)
    variance = sum((value - mean) ** 2 for value in daily_returns) / 2
    assert result == pytest.approx(mean / math.sqrt(variance) * math.sqrt(252))


def test_annualized_sharpe_ratio_returns_zero_without_variance() -> None:
    assert annualized_sharpe_ratio(
        ["2026-01-01", "2026-01-02", "2026-01-03"],
        [100.0, 100.0, 100.0],
    ) == 0.0
