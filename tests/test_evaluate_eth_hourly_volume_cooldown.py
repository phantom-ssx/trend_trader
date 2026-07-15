import pandas as pd

from scripts.evaluate_eth_hourly_volume_cooldown import (
    CooldownRule,
    add_relative_volume,
)


def test_relative_volume_uses_only_prior_bars_as_reference() -> None:
    data = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [10.0, 20.0, 30.0, 100.0],
        }
    )
    rule = CooldownRule("clock", "volume_clock", lookback=3)
    add_relative_volume(data, [rule])

    assert data["relative_volume_3"].iloc[-1] == 5.0


def test_rule_grid_contains_fixed_and_volume_replacements() -> None:
    from scripts.evaluate_eth_hourly_volume_cooldown import build_rules

    families = {rule.family for rule in build_rules()}

    assert families == {
        "fixed",
        "volume_clock",
        "high_volume_cross",
        "quiet_unlock",
        "quiet_then_high",
    }
