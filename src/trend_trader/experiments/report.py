"""Compatibility import for strategy HTML reports."""

from trend_trader.experiments.strategy.report import render_strategy_report

render_report = render_strategy_report

__all__ = ["render_report", "render_strategy_report"]
