import pandas as pd
import pytest

from scripts.research_eth_4h_macd_atr_entries import (
    add_atr_indicators,
    macd_extrema_entry_targets,
)


def test_atr_uses_true_range_and_only_past_data() -> None:
    data = pd.DataFrame(
        {
            "high": [101.0, 110.0, 106.0, 108.0],
            "low": [99.0, 104.0, 102.0, 103.0],
            "close": [100.0, 105.0, 104.0, 107.0],
        }
    )

    result = add_atr_indicators(
        data,
        atr_period=2,
        percentile_lookback=2,
        expansion_lookback=2,
    )

    assert result["atr14"].iloc[1] == pytest.approx(6.0)
    assert result["atr14"].iloc[2] == pytest.approx(5.0)


def test_failed_atr_entry_closes_old_position_without_reversing() -> None:
    data = pd.DataFrame({"macd": [0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0]})
    condition = pd.Series([True, True, True, True, True, True, True, False])

    targets = macd_extrema_entry_targets(data, condition)

    assert targets.tolist() == [0, 0, 0, -1, -1, -1, -1, 0]
