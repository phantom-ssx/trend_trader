from datetime import UTC, datetime, timedelta

import polars as pl

from scripts.local_backtest import run_strategy_demo
from trend_trader.strategies.demo_ema_cross import DemoEmaCrossSignal


def test_demo_ema_cross_signal_emits_only_on_signal_change() -> None:
    signal = DemoEmaCrossSignal(fast_period=2, slow_period=4)

    emitted = [signal.on_price(price) for price in [10, 11, 12, 11, 10, 9]]

    assert emitted == ["SELL", "BUY", None, None, "SELL", None]


def test_backtest_uses_strategy_signal(monkeypatch) -> None:
    calls: list[float] = []
    original_on_price = DemoEmaCrossSignal.on_price

    def tracking_on_price(self, close: float):
        calls.append(close)
        return original_on_price(self, close)

    monkeypatch.setattr(DemoEmaCrossSignal, "on_price", tracking_on_price)
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(6)]
    df = pl.DataFrame(
        {
            "ts": ts,
            "open": [10.0, 11.0, 12.0, 11.0, 10.0, 9.0],
            "high": [11.0, 12.0, 13.0, 12.0, 11.0, 10.0],
            "low": [9.0, 10.0, 11.0, 10.0, 9.0, 8.0],
            "close": [10.0, 11.0, 12.0, 11.0, 10.0, 9.0],
            "volume": [1.0] * 6,
        }
    )

    trades = run_strategy_demo(df, fast_period=2, slow_period=4, starting_balance=10_000.0)

    assert calls == [10.0, 11.0, 12.0, 11.0, 10.0, 9.0]
    assert [trade.side for trade in trades] == ["SELL", "BUY", "SELL"]
