"""Non-overlapping long-short portfolio construction for strategy experiments."""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta
from typing import Literal

import polars as pl

from trend_trader.data.models import bar_minutes
from trend_trader.research import ResearchDataset


def build_portfolio_returns(
    dataset: ResearchDataset,
    *,
    factor_name: str,
    timeframe: str,
    start: datetime,
    quantiles: int,
    round_trip_cost_bps: float,
    mode: Literal["long_short", "long_only", "short_only", "time_series_threshold"] = "long_short",
    long_threshold_bps: float = 0.0,
    short_threshold_bps: float | None = None,
    signal_multiplier: Literal[-1.0, 1.0] = 1.0,
    signal_smoothing_periods: int = 1,
    long_trend_filter_bars: int | None = None,
    long_trend_min_return_bps: float = 0.0,
    position_size: float = 1.0,
) -> pl.DataFrame:
    """Build equal-weight, non-overlapping portfolios from signal tails.

    Portfolio transaction costs are proportional to one-way weight turnover. The
    supplied round-trip rate is multiplied by ``0.5 * sum(abs(weight change))``.
    Long-short portfolios allocate 50% gross exposure to each side; one-sided
    portfolios allocate 100% gross exposure to the selected side.
    """

    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    if mode not in {"long_short", "long_only", "short_only", "time_series_threshold"}:
        raise ValueError(f"unsupported portfolio mode: {mode}")
    if long_threshold_bps < 0 or (short_threshold_bps is not None and short_threshold_bps < 0):
        raise ValueError("portfolio signal thresholds must not be negative")
    if signal_smoothing_periods < 1:
        raise ValueError("signal_smoothing_periods must be at least 1")
    if long_trend_filter_bars is not None and long_trend_filter_bars < 1:
        raise ValueError("long_trend_filter_bars must be at least 1")
    if not 0 < position_size <= 1:
        raise ValueError("position_size must be in (0, 1]")
    step = timedelta(minutes=bar_minutes(timeframe))
    selected_columns = [
        "label_name",
        "horizon_bars",
        "timestamp",
        "exit_time",
        "instrument_id",
        "value",
        "gross_return",
    ]
    if long_trend_filter_bars is not None:
        selected_columns.append("entry_price")
    selected = (
        dataset.frame.filter(
            pl.col("is_valid")
            & (pl.col("factor_name") == factor_name)
            & pl.col("value").is_not_null()
            & pl.col("gross_return").is_not_null()
        )
        .select(*selected_columns)
        .sort("horizon_bars", "timestamp", "value", "instrument_id")
    )
    if mode == "time_series_threshold":
        return _build_time_series_threshold_returns(
            selected,
            factor_name=factor_name,
            start=start,
            step=step,
            round_trip_cost_bps=round_trip_cost_bps,
            long_threshold_bps=long_threshold_bps,
            short_threshold_bps=short_threshold_bps,
            signal_multiplier=signal_multiplier,
            signal_smoothing_periods=signal_smoothing_periods,
            long_trend_filter_bars=long_trend_filter_bars,
            long_trend_min_return_bps=long_trend_min_return_bps,
            position_size=position_size,
            timestamp_dtype=dataset.frame.schema["timestamp"],
        )
    rows: list[dict[str, object]] = []
    for label_frame in selected.partition_by(["label_name", "horizon_bars"], maintain_order=True):
        label_name = str(label_frame["label_name"][0])
        horizon = int(label_frame["horizon_bars"][0])
        previous_weights: dict[str, float] = {}
        for cross_section in label_frame.partition_by("timestamp", maintain_order=True):
            timestamp = cross_section["timestamp"][0]
            bar_index = int((timestamp - start) / step)
            if bar_index < 0 or bar_index % horizon:
                continue
            if cross_section.height < quantiles:
                continue
            bucket_size = max(1, cross_section.height // quantiles)
            bottom = cross_section.head(bucket_size)
            top = cross_section.tail(bucket_size)
            top_ids = (
                [str(value) for value in top["instrument_id"].to_list()]
                if mode != "short_only"
                else []
            )
            bottom_ids = (
                [str(value) for value in bottom["instrument_id"].to_list()]
                if mode != "long_only"
                else []
            )
            if set(top_ids) & set(bottom_ids):
                continue
            gross_long = float(top["gross_return"].mean()) if top_ids else 0.0
            gross_short = -float(bottom["gross_return"].mean()) if bottom_ids else 0.0
            hypothetical_round_trip_cost = round_trip_cost_bps / 10_000
            if mode == "long_short":
                weights = {instrument_id: 0.5 / len(top_ids) for instrument_id in top_ids}
                weights.update(
                    {instrument_id: -0.5 / len(bottom_ids) for instrument_id in bottom_ids}
                )
                gross_return = 0.5 * (gross_long + gross_short)
            elif mode == "long_only":
                weights = {instrument_id: 1.0 / len(top_ids) for instrument_id in top_ids}
                gross_return = gross_long
            else:
                weights = {instrument_id: -1.0 / len(bottom_ids) for instrument_id in bottom_ids}
                gross_return = gross_short
            if previous_weights:
                instruments = set(previous_weights) | set(weights)
                turnover = 0.5 * sum(
                    abs(weights.get(item, 0.0) - previous_weights.get(item, 0.0))
                    for item in instruments
                )
            else:
                turnover = 0.5 * sum(abs(weight) for weight in weights.values())
            previous_weights = weights
            transaction_cost = hypothetical_round_trip_cost * turnover
            portfolio_return = gross_return - transaction_cost
            benchmark_return = float(cross_section["gross_return"].mean())
            net_exposure = sum(weights.values())
            gross_active_return = gross_return - net_exposure * benchmark_return
            portfolio_active_return = gross_active_return - transaction_cost
            net_long = (
                portfolio_return
                if mode == "long_only"
                else gross_long - hypothetical_round_trip_cost
            )
            net_short = (
                portfolio_return
                if mode == "short_only"
                else gross_short - hypothetical_round_trip_cost
            )
            rows.append(
                {
                    "factor_name": factor_name,
                    "label_name": label_name,
                    "horizon_bars": horizon,
                    "timestamp": timestamp,
                    "exit_time": cross_section["exit_time"][0],
                    "long_count": len(top_ids),
                    "short_count": len(bottom_ids),
                    "gross_long_return": gross_long,
                    "gross_short_return": gross_short,
                    "net_long_return": net_long,
                    "net_short_return": net_short,
                    "gross_portfolio_return": gross_return,
                    "benchmark_return": benchmark_return,
                    "gross_active_return": gross_active_return,
                    "transaction_cost": transaction_cost,
                    "portfolio_return": portfolio_return,
                    "portfolio_active_return": portfolio_active_return,
                    "turnover": turnover,
                }
            )
    if not rows:
        return _empty_portfolio(dataset.frame.schema["timestamp"])
    return _enrich_portfolio(pl.DataFrame(rows, infer_schema_length=None))


def _build_time_series_threshold_returns(
    selected: pl.DataFrame,
    *,
    factor_name: str,
    start: datetime,
    step: timedelta,
    round_trip_cost_bps: float,
    long_threshold_bps: float,
    short_threshold_bps: float | None,
    signal_multiplier: Literal[-1.0, 1.0],
    signal_smoothing_periods: int,
    long_trend_filter_bars: int | None,
    long_trend_min_return_bps: float,
    position_size: float,
    timestamp_dtype: pl.DataType,
) -> pl.DataFrame:
    if selected.is_empty():
        return _empty_portfolio(timestamp_dtype)
    if selected["instrument_id"].n_unique() != 1:
        raise ValueError("time_series_threshold portfolio requires exactly one instrument")
    long_threshold = long_threshold_bps / 10_000
    short_threshold = short_threshold_bps / 10_000 if short_threshold_bps is not None else None
    cost_rate = round_trip_cost_bps / 10_000
    rows: list[dict[str, object]] = []
    for label_frame in selected.partition_by(["label_name", "horizon_bars"], maintain_order=True):
        label_name = str(label_frame["label_name"][0])
        horizon = int(label_frame["horizon_bars"][0])
        if long_trend_filter_bars is not None:
            label_frame = label_frame.with_columns(
                (
                    pl.col("entry_price").shift(1)
                    / pl.col("entry_price").shift(1 + long_trend_filter_bars)
                    - 1
                ).alias("_long_trend_return")
            )
        previous_position = 0.0
        recent_signals: deque[float] = deque(maxlen=signal_smoothing_periods)
        for observation in label_frame.partition_by("timestamp", maintain_order=True):
            timestamp = observation["timestamp"][0]
            bar_index = int((timestamp - start) / step)
            if bar_index < 0 or bar_index % horizon:
                continue
            raw_signal = float(observation["value"][0])
            recent_signals.append(signal_multiplier * raw_signal)
            signal = sum(recent_signals) / len(recent_signals)
            trend_return = (
                observation["_long_trend_return"][0]
                if long_trend_filter_bars is not None
                else None
            )
            long_trend_allows_entry = long_trend_filter_bars is None or (
                trend_return is not None
                and float(trend_return) > long_trend_min_return_bps / 10_000
            )
            if signal > long_threshold and long_trend_allows_entry:
                position = position_size
            elif short_threshold is not None and signal < -short_threshold:
                position = -position_size
            else:
                position = 0.0
            turnover = 0.5 * abs(position - previous_position)
            previous_position = position
            asset_return = float(observation["gross_return"][0])
            gross_return = position * asset_return
            transaction_cost = cost_rate * turnover
            portfolio_return = gross_return - transaction_cost
            gross_active_return = gross_return - asset_return
            portfolio_active_return = gross_active_return - transaction_cost
            rows.append(
                {
                    "factor_name": factor_name,
                    "label_name": label_name,
                    "horizon_bars": horizon,
                    "timestamp": timestamp,
                    "exit_time": observation["exit_time"][0],
                    "long_count": int(position > 0),
                    "short_count": int(position < 0),
                    "gross_long_return": asset_return if position > 0 else 0.0,
                    "gross_short_return": -asset_return if position < 0 else 0.0,
                    "net_long_return": portfolio_return if position > 0 else 0.0,
                    "net_short_return": portfolio_return if position < 0 else 0.0,
                    "gross_portfolio_return": gross_return,
                    "benchmark_return": asset_return,
                    "gross_active_return": gross_active_return,
                    "transaction_cost": transaction_cost,
                    "portfolio_return": portfolio_return,
                    "portfolio_active_return": portfolio_active_return,
                    "turnover": turnover,
                    "raw_signal_value": raw_signal,
                    "signal_value": signal,
                    "long_trend_return": trend_return,
                    "position": position,
                }
            )
    if not rows:
        return _empty_portfolio(timestamp_dtype)
    return _enrich_portfolio(pl.DataFrame(rows, infer_schema_length=None))


def _enrich_portfolio(result: pl.DataFrame) -> pl.DataFrame:
    result = result.sort("horizon_bars", "timestamp")
    enriched: list[pl.DataFrame] = []
    for frame in result.partition_by(["label_name", "horizon_bars"], maintain_order=True):
        enriched.append(
            frame.with_columns(
                (pl.col("portfolio_return") + 1).cum_prod().alias("wealth"),
                (pl.col("portfolio_active_return") + 1).cum_prod().alias("active_wealth"),
                (pl.col("benchmark_return") + 1).cum_prod().alias("benchmark_wealth"),
            ).with_columns(
                (
                    pl.col("wealth") / pl.max_horizontal(pl.col("wealth").cum_max(), pl.lit(1.0))
                    - 1
                ).alias("drawdown"),
                (
                    pl.col("active_wealth")
                    / pl.max_horizontal(pl.col("active_wealth").cum_max(), pl.lit(1.0))
                    - 1
                ).alias("active_drawdown"),
                (
                    pl.col("benchmark_wealth")
                    / pl.max_horizontal(pl.col("benchmark_wealth").cum_max(), pl.lit(1.0))
                    - 1
                ).alias("benchmark_drawdown"),
            )
        )
    return pl.concat(enriched, how="vertical_relaxed").sort("horizon_bars", "timestamp")


def portfolio_metrics(portfolio: pl.DataFrame, *, timeframe: str) -> pl.DataFrame:
    """Annualized performance statistics for each label horizon."""

    if portfolio.is_empty():
        return pl.DataFrame(
            schema={
                "factor_name": pl.Utf8,
                "label_name": pl.Utf8,
                "horizon_bars": pl.Int32,
                "periods": pl.Int64,
                "total_return": pl.Float64,
                "annual_return": pl.Float64,
                "sharpe": pl.Float64,
                "max_drawdown": pl.Float64,
                "active_total_return": pl.Float64,
                "active_annual_return": pl.Float64,
                "active_sharpe": pl.Float64,
                "active_max_drawdown": pl.Float64,
                "benchmark_total_return": pl.Float64,
                "benchmark_annual_return": pl.Float64,
                "benchmark_sharpe": pl.Float64,
                "benchmark_max_drawdown": pl.Float64,
                "relative_total_return": pl.Float64,
                "turnover": pl.Float64,
                "win_rate": pl.Float64,
                "long_rate": pl.Float64,
                "short_rate": pl.Float64,
                "flat_rate": pl.Float64,
            }
        )
    base_periods_per_year = 365 * 24 * 60 / bar_minutes(timeframe)
    rows: list[dict[str, object]] = []
    for frame in portfolio.partition_by(["label_name", "horizon_bars"], maintain_order=True):
        values = [float(value) for value in frame["portfolio_return"].to_list()]
        horizon = int(frame["horizon_bars"][0])
        periods_per_year = base_periods_per_year / horizon
        count = len(values)
        mean = sum(values) / count
        std = _sample_std(values)
        wealth = math.prod(1 + value for value in values)
        annual_return = wealth ** (periods_per_year / count) - 1 if wealth > 0 else -1.0
        sharpe = mean / std * periods_per_year**0.5 if std > 0 else None
        active_values = [float(value) for value in frame["portfolio_active_return"].to_list()]
        active_mean = sum(active_values) / count
        active_std = _sample_std(active_values)
        active_wealth = math.prod(1 + value for value in active_values)
        active_annual_return = (
            active_wealth ** (periods_per_year / count) - 1 if active_wealth > 0 else -1.0
        )
        active_sharpe = active_mean / active_std * periods_per_year**0.5 if active_std > 0 else None
        benchmark_values = [float(value) for value in frame["benchmark_return"].to_list()]
        benchmark_mean = sum(benchmark_values) / count
        benchmark_std = _sample_std(benchmark_values)
        benchmark_wealth = math.prod(1 + value for value in benchmark_values)
        benchmark_annual_return = (
            benchmark_wealth ** (periods_per_year / count) - 1 if benchmark_wealth > 0 else -1.0
        )
        benchmark_sharpe = (
            benchmark_mean / benchmark_std * periods_per_year**0.5 if benchmark_std > 0 else None
        )
        turnover_values = [float(value) for value in frame["turnover"].to_list()]
        rows.append(
            {
                "factor_name": frame["factor_name"][0],
                "label_name": frame["label_name"][0],
                "horizon_bars": horizon,
                "periods": count,
                "total_return": wealth - 1,
                "annual_return": annual_return,
                "sharpe": sharpe,
                "max_drawdown": float(frame["drawdown"].min()),
                "active_total_return": active_wealth - 1,
                "active_annual_return": active_annual_return,
                "active_sharpe": active_sharpe,
                "active_max_drawdown": float(frame["active_drawdown"].min()),
                "benchmark_total_return": benchmark_wealth - 1,
                "benchmark_annual_return": benchmark_annual_return,
                "benchmark_sharpe": benchmark_sharpe,
                "benchmark_max_drawdown": float(frame["benchmark_drawdown"].min()),
                "relative_total_return": (
                    wealth / benchmark_wealth - 1 if benchmark_wealth > 0 else None
                ),
                "turnover": (
                    sum(turnover_values) / len(turnover_values) if turnover_values else 0.0
                ),
                "win_rate": sum(value > 0 for value in values) / count,
                "long_rate": float((frame["long_count"] > 0).mean()),
                "short_rate": float((frame["short_count"] > 0).mean()),
                "flat_rate": float(
                    ((frame["long_count"] == 0) & (frame["short_count"] == 0)).mean()
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort("horizon_bars")


def portfolio_yearly_metrics(portfolio: pl.DataFrame, *, timeframe: str) -> pl.DataFrame:
    """Calendar-year strategy, benchmark, cost, and exposure diagnostics."""

    schema = {
        "factor_name": pl.Utf8,
        "label_name": pl.Utf8,
        "horizon_bars": pl.Int32,
        "year": pl.Int32,
        "periods": pl.Int64,
        "gross_total_return": pl.Float64,
        "strategy_return": pl.Float64,
        "benchmark_return": pl.Float64,
        "relative_total_return": pl.Float64,
        "sharpe": pl.Float64,
        "max_drawdown": pl.Float64,
        "transaction_cost": pl.Float64,
        "turnover": pl.Float64,
        "position_changes": pl.Int64,
        "long_rate": pl.Float64,
        "short_rate": pl.Float64,
        "flat_rate": pl.Float64,
    }
    if portfolio.is_empty():
        return pl.DataFrame(schema=schema)
    base_periods_per_year = 365 * 24 * 60 / bar_minutes(timeframe)
    grouped = portfolio.with_columns(pl.col("timestamp").dt.year().alias("year"))
    rows: list[dict[str, object]] = []
    for frame in grouped.partition_by(["label_name", "horizon_bars", "year"], maintain_order=True):
        values = [float(value) for value in frame["portfolio_return"].to_list()]
        gross_values = [float(value) for value in frame["gross_portfolio_return"].to_list()]
        benchmark_values = [float(value) for value in frame["benchmark_return"].to_list()]
        horizon = int(frame["horizon_bars"][0])
        periods_per_year = base_periods_per_year / horizon
        mean = sum(values) / len(values)
        std = _sample_std(values)
        strategy_wealth = math.prod(1 + value for value in values)
        benchmark_wealth = math.prod(1 + value for value in benchmark_values)
        rows.append(
            {
                "factor_name": frame["factor_name"][0],
                "label_name": frame["label_name"][0],
                "horizon_bars": horizon,
                "year": int(frame["year"][0]),
                "periods": len(values),
                "gross_total_return": math.prod(1 + value for value in gross_values) - 1,
                "strategy_return": strategy_wealth - 1,
                "benchmark_return": benchmark_wealth - 1,
                "relative_total_return": (
                    strategy_wealth / benchmark_wealth - 1 if benchmark_wealth > 0 else None
                ),
                "sharpe": mean / std * periods_per_year**0.5 if std > 0 else None,
                "max_drawdown": _max_drawdown(values),
                "transaction_cost": float(frame["transaction_cost"].sum()),
                "turnover": float(frame["turnover"].mean()),
                "position_changes": int((frame["turnover"] > 0).sum()),
                "long_rate": float((frame["long_count"] > 0).mean()),
                "short_rate": float((frame["short_count"] > 0).mean()),
                "flat_rate": float(
                    ((frame["long_count"] == 0) & (frame["short_count"] == 0)).mean()
                ),
            }
        )
    return pl.DataFrame(rows, schema=schema, infer_schema_length=None).sort("horizon_bars", "year")


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _max_drawdown(values: list[float]) -> float:
    wealth = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in values:
        wealth *= 1 + value
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1)
    return drawdown


def _empty_portfolio(timestamp_dtype: pl.DataType) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "factor_name": pl.Utf8,
            "label_name": pl.Utf8,
            "horizon_bars": pl.Int32,
            "timestamp": timestamp_dtype,
            "exit_time": timestamp_dtype,
            "long_count": pl.Int64,
            "short_count": pl.Int64,
            "gross_long_return": pl.Float64,
            "gross_short_return": pl.Float64,
            "net_long_return": pl.Float64,
            "net_short_return": pl.Float64,
            "gross_portfolio_return": pl.Float64,
            "benchmark_return": pl.Float64,
            "gross_active_return": pl.Float64,
            "transaction_cost": pl.Float64,
            "portfolio_return": pl.Float64,
            "portfolio_active_return": pl.Float64,
            "turnover": pl.Float64,
            "wealth": pl.Float64,
            "drawdown": pl.Float64,
            "active_wealth": pl.Float64,
            "active_drawdown": pl.Float64,
            "benchmark_wealth": pl.Float64,
            "benchmark_drawdown": pl.Float64,
            "signal_value": pl.Float64,
            "raw_signal_value": pl.Float64,
            "long_trend_return": pl.Float64,
            "position": pl.Float64,
        }
    )
