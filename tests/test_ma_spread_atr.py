from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd

from scripts.evaluate_filters import add_indicators, spread_confirm_signals
from trend_trader.strategies.ma_spread_atr import MaSpreadAtrSignal, is_below_min_order_notional


def test_ma_spread_atr_signal_matches_filter_evaluation() -> None:
    prices = [100.0] * 20 + [102.0, 104.0, 106.0, 108.0, 110.0] + [108.0, 106.0, 103.0]
    prices += [100.0, 97.0, 94.0, 91.0, 88.0]
    data = pd.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)
                for index in range(len(prices))
            ],
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
            "volume": [10.0] * len(prices),
        }
    )
    data = add_indicators(data)
    expected = spread_confirm_signals(data, 0.0035).where(data["atr_pct"] >= 0.005, 0)

    signal = MaSpreadAtrSignal(
        fast_period=5,
        slow_period=20,
        spread_threshold=0.0035,
        atr_pct_min=0.005,
    )
    actual = []
    for row in data.itertuples(index=False):
        side = signal.on_bar(high=float(row.high), low=float(row.low), close=float(row.close))
        actual.append(1 if side == "BUY" else -1 if side == "SELL" else 0)

    assert actual == expected.to_list()


def test_min_order_notional_filters_small_rebalance_orders() -> None:
    assert is_below_min_order_notional(
        quantity=Decimal("0.00311259514"),
        price=Decimal("3211.95"),
        min_order_notional=Decimal("50"),
    )
    assert not is_below_min_order_notional(
        quantity=Decimal("0.02"),
        price=Decimal("3211.95"),
        min_order_notional=Decimal("50"),
    )
