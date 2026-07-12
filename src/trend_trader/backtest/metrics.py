from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd


def timestamps_or_daily_index(data: pd.DataFrame) -> Sequence[object]:
    """Return candle timestamps, falling back to daily spacing for minimal test frames."""
    if "ts" in data.columns:
        return data["ts"]
    return pd.date_range("2000-01-01", periods=len(data), freq="1D", tz="UTC")


def annualized_sharpe_ratio(
    timestamps: Sequence[object],
    equities: Sequence[float],
    *,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe from daily equity returns with a zero risk-free rate."""
    if len(timestamps) != len(equities):
        raise ValueError("timestamps and equities must have equal length")
    if len(equities) < 2:
        return 0.0

    index = pd.to_datetime(list(timestamps), utc=True)
    curve = pd.Series(equities, index=index, dtype=float).sort_index()
    curve = curve[~curve.index.duplicated(keep="last")]
    daily = curve.resample("1D").last().ffill()
    returns = daily.pct_change(fill_method=None).dropna()
    if len(returns) < 2:
        return 0.0
    volatility = float(returns.std(ddof=1))
    if not math.isfinite(volatility) or volatility <= 0:
        return 0.0
    sharpe = float(returns.mean()) / volatility * math.sqrt(periods_per_year)
    return sharpe if math.isfinite(sharpe) else 0.0
