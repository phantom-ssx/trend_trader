from trend_trader.notifications.bark import BarkNotifier


def test_bark_notifier_rejects_backtest_mode() -> None:
    try:
        BarkNotifier("https://api.day.app/key", "回测")
    except ValueError as exc:
        assert "模拟盘 or 实盘" in str(exc)
    else:
        raise AssertionError("backtest mode must not be accepted")


def test_bark_notifier_trims_trailing_slash() -> None:
    notifier = BarkNotifier("https://api.day.app/key/", "模拟盘")
    assert notifier.base_url == "https://api.day.app/key"
