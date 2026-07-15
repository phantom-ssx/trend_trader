import pandas as pd

from scripts.evaluate_eth_hourly_volume_filters import (
    VolumeRule,
    add_volume_features,
    entry_allowed,
)


def test_relative_volume_reference_excludes_current_bar() -> None:
    data = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [10.0, 20.0, 30.0, 100.0],
        }
    )
    rules = [VolumeRule("test", "relative_volume", lookback=3, threshold=5.0)]
    add_volume_features(data, rules)

    assert data["relative_volume_3"].iloc[-1] == 5.0
    assert entry_allowed(data, rules[0], 1, 3)


def test_directional_flow_is_checked_against_trade_direction() -> None:
    data = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 101.0],
            "volume": [10.0, 20.0, 30.0, 5.0],
        }
    )
    rule = VolumeRule(
        "test",
        "directional_flow",
        flow_lookback=3,
        flow_threshold=0.5,
    )
    add_volume_features(data, [rule])

    assert entry_allowed(data, rule, 1, 3)
    assert not entry_allowed(data, rule, -1, 3)
