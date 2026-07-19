from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from scripts.optimize_v8r_allocation import allocation_returns


def test_allocation_overlay_uses_only_completed_trailing_returns_and_costs_switch() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "date": [start + timedelta(days=index) for index in range(4)],
            "minute": [0.0] * 4,
            "eth_168h_anchor": [0.0] * 4,
            "eth_24h_anchor": [0.10, -0.20, 0.10, 0.10],
        }
    )

    values = allocation_returns(
        frame,
        minute_weight=0.0,
        eth_168h_weight=0.0,
        eth_24h_weight=1.0,
        leverage=1.0,
        lookback_days=2,
        losing_scale=0.0,
        transfer_to="cash",
    )

    assert values[:2] == pytest.approx([0.10, -0.20])
    assert values[2] == pytest.approx(-0.0008)
    assert values[3] == pytest.approx(0.0)
