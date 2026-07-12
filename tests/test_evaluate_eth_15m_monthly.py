from scripts.evaluate_eth_15m_monthly import FOCUS_STRATEGIES


def test_focus_strategies_match_selected_candidates() -> None:
    assert [
        (item.fast_period, item.slow_period, item.spread_threshold, item.atr_threshold)
        for item in FOCUS_STRATEGIES
    ] == [
        (24, 80, 0.003, 0.005),
        (32, 80, 0.001, 0.004),
        (28, 80, 0.0015, 0.004),
    ]
