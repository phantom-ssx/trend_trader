from scripts.evaluate_eth_15m_filters import (
    expand_time_equivalent_pairs,
    parse_thresholds,
    selected_pairs,
)


def test_expand_time_equivalent_pairs_preserves_wall_clock_lookback() -> None:
    assert expand_time_equivalent_pairs([(5, 20), (10, 30)]) == [
        (20, 80),
        (40, 120),
    ]


def test_selected_pairs_can_include_time_equivalent_without_duplicates() -> None:
    assert selected_pairs(
        [(5, 20), (20, 80)], include_time_equivalent=True
    ) == [(5, 20), (20, 80), (80, 320)]


def test_parse_thresholds() -> None:
    assert parse_thresholds("0.0015, 0.002") == [0.0015, 0.002]
