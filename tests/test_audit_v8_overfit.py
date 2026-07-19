import math

import pytest

from scripts.audit_v8_overfit import bootstrap_annual_returns, performance


def test_performance_compounds_and_reports_drawdown() -> None:
    result = performance([0.10, -0.10], periods_per_year=2)

    assert result["total_return"] == pytest.approx(-0.01)
    assert result["annual_return"] == pytest.approx(-0.01)
    assert result["max_drawdown"] == pytest.approx(-0.10)
    assert math.isfinite(result["sharpe"])


def test_bootstrap_is_deterministic() -> None:
    first = bootstrap_annual_returns([0.01, 0.02, -0.01], samples=100, seed=7)
    second = bootstrap_annual_returns([0.01, 0.02, -0.01], samples=100, seed=7)

    assert first == second
    assert first["months"] == 3
    assert first["probability_positive"] > 0.5


def test_bootstrap_rejects_empty_returns() -> None:
    with pytest.raises(ValueError, match="monthly returns are empty"):
        bootstrap_annual_returns([], samples=10, seed=1)
