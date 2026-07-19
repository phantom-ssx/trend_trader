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
    long_threshold_value: float | None = None,
    short_threshold_value: float | None = None,
    signal_multiplier: Literal[-1.0, 1.0] = 1.0,
    signal_smoothing_periods: int = 1,
    signal_standardization_periods: int | None = None,
    signal_standardization_min_periods: int | None = None,
    long_threshold_zscore: float | None = None,
    short_threshold_zscore: float | None = None,
    long_trend_filter_bars: int | None = None,
    long_trend_min_return_bps: float = 0.0,
    position_size: float = 1.0,
    fixed_holding_periods: int | None = None,
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
    if (long_threshold_value is not None and long_threshold_value < 0) or (
        short_threshold_value is not None and short_threshold_value < 0
    ):
        raise ValueError("portfolio raw signal thresholds must not be negative")
    if signal_smoothing_periods < 1:
        raise ValueError("signal_smoothing_periods must be at least 1")
    if signal_standardization_periods is not None:
        standardization_minimum = (
            signal_standardization_min_periods or signal_standardization_periods
        )
        if signal_standardization_periods < 2 or not (
            2 <= standardization_minimum <= signal_standardization_periods
        ):
            raise ValueError("invalid signal standardization window")
        if long_threshold_zscore is None:
            raise ValueError("standardized signals require long_threshold_zscore")
    else:
        standardization_minimum = None
        if long_threshold_zscore is not None or short_threshold_zscore is not None:
            raise ValueError("z-score thresholds require signal_standardization_periods")
    if (long_threshold_zscore is not None and long_threshold_zscore < 0) or (
        short_threshold_zscore is not None and short_threshold_zscore < 0
    ):
        raise ValueError("portfolio z-score thresholds must not be negative")
    if long_trend_filter_bars is not None and long_trend_filter_bars < 1:
        raise ValueError("long_trend_filter_bars must be at least 1")
    if not 0 < position_size <= 1:
        raise ValueError("position_size must be in (0, 1]")
    if fixed_holding_periods is not None and fixed_holding_periods < 1:
        raise ValueError("fixed_holding_periods must be at least 1")
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
    frame = dataset.frame
    if "label_is_valid" not in frame.columns:
        frame = frame.with_columns(pl.lit(True).alias("label_is_valid"))
    if mode == "time_series_threshold":
        valid_observation = (
            pl.col("factor_is_valid")
            if "factor_is_valid" in frame.columns
            else pl.col("value").is_not_null()
        )
    else:
        valid_observation = pl.col("is_valid")
    selected_columns.append("label_is_valid")
    selected = (
        frame.filter(
            valid_observation
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
            long_threshold_value=long_threshold_value,
            short_threshold_value=short_threshold_value,
            signal_multiplier=signal_multiplier,
            signal_smoothing_periods=signal_smoothing_periods,
            signal_standardization_periods=signal_standardization_periods,
            signal_standardization_min_periods=standardization_minimum,
            long_threshold_zscore=long_threshold_zscore,
            short_threshold_zscore=short_threshold_zscore,
            long_trend_filter_bars=long_trend_filter_bars,
            long_trend_min_return_bps=long_trend_min_return_bps,
            position_size=position_size,
            fixed_holding_periods=fixed_holding_periods,
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
    long_threshold_value: float | None,
    short_threshold_value: float | None,
    signal_multiplier: Literal[-1.0, 1.0],
    signal_smoothing_periods: int,
    signal_standardization_periods: int | None,
    signal_standardization_min_periods: int | None,
    long_threshold_zscore: float | None,
    short_threshold_zscore: float | None,
    long_trend_filter_bars: int | None,
    long_trend_min_return_bps: float,
    position_size: float,
    fixed_holding_periods: int | None,
    timestamp_dtype: pl.DataType,
) -> pl.DataFrame:
    if selected.is_empty():
        return _empty_portfolio(timestamp_dtype)
    if selected["instrument_id"].n_unique() != 1:
        raise ValueError("time_series_threshold portfolio requires exactly one instrument")
    standardized = signal_standardization_periods is not None
    long_threshold = (
        float(long_threshold_zscore)
        if standardized
        else long_threshold_value
        if long_threshold_value is not None
        else long_threshold_bps / 10_000
    )
    short_threshold = (
        float(short_threshold_zscore)
        if standardized and short_threshold_zscore is not None
        else short_threshold_value
        if not standardized and short_threshold_value is not None
        else short_threshold_bps / 10_000
        if not standardized and short_threshold_bps is not None
        else None
    )
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
        remaining_holding_periods = 0
        recent_signals: deque[float] = deque(maxlen=signal_smoothing_periods)
        standardization_values: deque[float] = deque()
        standardization_sum = 0.0
        standardization_sum_squares = 0.0
        for observation in label_frame.partition_by("timestamp", maintain_order=True):
            timestamp = observation["timestamp"][0]
            bar_index = int((timestamp - start) / step)
            if bar_index < 0 or bar_index % horizon:
                continue
            raw_signal = float(observation["value"][0])
            recent_signals.append(signal_multiplier * raw_signal)
            smoothed_signal = sum(recent_signals) / len(recent_signals)
            signal: float | None = smoothed_signal
            signal_mean: float | None = None
            signal_std: float | None = None
            if signal_standardization_periods is not None:
                if len(standardization_values) == signal_standardization_periods:
                    removed = standardization_values.popleft()
                    standardization_sum -= removed
                    standardization_sum_squares -= removed * removed
                standardization_values.append(smoothed_signal)
                standardization_sum += smoothed_signal
                standardization_sum_squares += smoothed_signal * smoothed_signal
                observations = len(standardization_values)
                if observations >= int(signal_standardization_min_periods or 0):
                    signal_mean = standardization_sum / observations
                    variance = (
                        standardization_sum_squares
                        - observations * signal_mean * signal_mean
                    ) / (observations - 1)
                    signal_std = math.sqrt(max(0.0, variance))
                    signal = (
                        (smoothed_signal - signal_mean) / signal_std
                        if signal_std > 0
                        else 0.0
                    )
                else:
                    signal = None
            trend_return = (
                observation["_long_trend_return"][0]
                if long_trend_filter_bars is not None
                else None
            )
            long_trend_allows_entry = long_trend_filter_bars is None or (
                trend_return is not None
                and float(trend_return) > long_trend_min_return_bps / 10_000
            )
            can_trade = bool(observation["label_is_valid"][0])
            if not can_trade:
                position = previous_position
            elif fixed_holding_periods is not None and previous_position != 0:
                if remaining_holding_periods > 0:
                    position = previous_position
                    remaining_holding_periods -= 1
                else:
                    position = 0.0
            elif signal is not None and signal > long_threshold and long_trend_allows_entry:
                position = position_size
                if fixed_holding_periods is not None:
                    remaining_holding_periods = fixed_holding_periods - 1
            elif signal is not None and short_threshold is not None and signal < -short_threshold:
                position = -position_size
                if fixed_holding_periods is not None:
                    remaining_holding_periods = fixed_holding_periods - 1
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
                    "smoothed_signal_value": smoothed_signal,
                    "signal_value": signal,
                    "signal_standardization_mean": signal_mean,
                    "signal_standardization_std": signal_std,
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


def portfolio_trade_log(portfolio: pl.DataFrame, *, timeframe: str) -> pl.DataFrame:
    """Reconstruct position-level trades, including entry and exit costs.

    The log is available for portfolios that expose a numeric ``position`` column.
    A trade starts when exposure changes from flat to non-zero and closes when it
    returns to flat. Direct side flips close the old trade and open the new one at
    the same timestamp, splitting that row's transaction cost between them.
    """

    timestamp_dtype = portfolio.schema.get("timestamp", pl.Datetime)
    schema = {
        "factor_name": pl.Utf8,
        "label_name": pl.Utf8,
        "horizon_bars": pl.Int32,
        "trade_id": pl.Int64,
        "side": pl.Utf8,
        "entry_time": timestamp_dtype,
        "exit_time": timestamp_dtype,
        "is_closed": pl.Boolean,
        "holding_periods": pl.Int64,
        "holding_hours": pl.Float64,
        "gross_return": pl.Float64,
        "strategy_return": pl.Float64,
        "transaction_cost": pl.Float64,
        "max_drawdown": pl.Float64,
        "entry_signal_value": pl.Float64,
    }
    if portfolio.is_empty() or "position" not in portfolio.columns:
        return pl.DataFrame(schema=schema)

    relevant = portfolio.filter(
        (pl.col("position") != 0) | (pl.col("turnover") > 0)
    ).sort("label_name", "horizon_bars", "timestamp")
    if relevant.is_empty():
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, object]] = []
    for frame in relevant.partition_by(["label_name", "horizon_bars"], maintain_order=True):
        horizon = int(frame["horizon_bars"][0])
        holding_hours_per_period = bar_minutes(timeframe) * horizon / 60
        trade_id = 0
        active: dict[str, object] | None = None
        last_exit_time = frame["exit_time"][-1]

        def start_trade(
            index: int,
            position: float,
            initial_cost: float,
            trade_frame: pl.DataFrame,
        ) -> dict[str, object]:
            gross_return = float(trade_frame["gross_portfolio_return"][index])
            net_return = gross_return - initial_cost
            signal_value = (
                trade_frame["signal_value"][index]
                if "signal_value" in trade_frame.columns
                else None
            )
            return {
                "side_sign": 1 if position > 0 else -1,
                "entry_time": trade_frame["timestamp"][index],
                "gross_values": [gross_return],
                "net_values": [net_return],
                "transaction_cost": initial_cost,
                "holding_periods": 1,
                "entry_signal_value": (
                    float(signal_value) if signal_value is not None else None
                ),
            }

        def finish_trade(
            trade: dict[str, object],
            *,
            exit_time: object,
            is_closed: bool,
            factor_name: object,
            label_name: object,
            trade_horizon: int,
            hours_per_period: float,
        ) -> None:
            nonlocal trade_id
            trade_id += 1
            gross_values = list(trade["gross_values"])
            net_values = list(trade["net_values"])
            holding_periods = int(trade["holding_periods"])
            rows.append(
                {
                    "factor_name": factor_name,
                    "label_name": label_name,
                    "horizon_bars": trade_horizon,
                    "trade_id": trade_id,
                    "side": "long" if int(trade["side_sign"]) > 0 else "short",
                    "entry_time": trade["entry_time"],
                    "exit_time": exit_time,
                    "is_closed": is_closed,
                    "holding_periods": holding_periods,
                    "holding_hours": holding_periods * hours_per_period,
                    "gross_return": math.prod(1 + value for value in gross_values) - 1,
                    "strategy_return": math.prod(1 + value for value in net_values) - 1,
                    "transaction_cost": float(trade["transaction_cost"]),
                    "max_drawdown": _max_drawdown(net_values),
                    "entry_signal_value": trade["entry_signal_value"],
                }
            )

        for index in range(frame.height):
            position = float(frame["position"][index])
            side_sign = 1 if position > 0 else -1 if position < 0 else 0
            row_cost = float(frame["transaction_cost"][index])
            if active is None:
                if side_sign:
                    active = start_trade(index, position, row_cost, frame)
                continue

            active_side = int(active["side_sign"])
            if side_sign == active_side:
                gross_return = float(frame["gross_portfolio_return"][index])
                active["gross_values"].append(gross_return)  # type: ignore[union-attr]
                active["net_values"].append(gross_return - row_cost)  # type: ignore[union-attr]
                active["transaction_cost"] = float(active["transaction_cost"]) + row_cost
                active["holding_periods"] = int(active["holding_periods"]) + 1
            elif side_sign == 0:
                active["net_values"].append(-row_cost)  # type: ignore[union-attr]
                active["transaction_cost"] = float(active["transaction_cost"]) + row_cost
                finish_trade(
                    active,
                    exit_time=frame["timestamp"][index],
                    is_closed=True,
                    factor_name=frame["factor_name"][0],
                    label_name=frame["label_name"][0],
                    trade_horizon=horizon,
                    hours_per_period=holding_hours_per_period,
                )
                active = None
            else:
                exit_cost = row_cost / 2
                active["net_values"].append(-exit_cost)  # type: ignore[union-attr]
                active["transaction_cost"] = float(active["transaction_cost"]) + exit_cost
                finish_trade(
                    active,
                    exit_time=frame["timestamp"][index],
                    is_closed=True,
                    factor_name=frame["factor_name"][0],
                    label_name=frame["label_name"][0],
                    trade_horizon=horizon,
                    hours_per_period=holding_hours_per_period,
                )
                active = start_trade(index, position, row_cost - exit_cost, frame)

        if active is not None:
            finish_trade(
                active,
                exit_time=last_exit_time,
                is_closed=False,
                factor_name=frame["factor_name"][0],
                label_name=frame["label_name"][0],
                trade_horizon=horizon,
                hours_per_period=holding_hours_per_period,
            )

    return pl.DataFrame(rows, schema=schema, infer_schema_length=None).sort(
        "horizon_bars", "entry_time"
    )


def portfolio_monthly_metrics(
    portfolio: pl.DataFrame,
    *,
    timeframe: str,
    trades: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Calendar-month return, cost, exposure, and closed-trade diagnostics."""

    schema = {
        "factor_name": pl.Utf8,
        "label_name": pl.Utf8,
        "horizon_bars": pl.Int32,
        "month": pl.Utf8,
        "periods": pl.Int64,
        "gross_total_return": pl.Float64,
        "strategy_return": pl.Float64,
        "benchmark_return": pl.Float64,
        "relative_total_return": pl.Float64,
        "sharpe": pl.Float64,
        "max_drawdown": pl.Float64,
        "transaction_cost": pl.Float64,
        "turnover": pl.Float64,
        "total_turnover": pl.Float64,
        "position_changes": pl.Int64,
        "entries": pl.Int64,
        "exits": pl.Int64,
        "closed_trades": pl.Int64,
        "winning_trades": pl.Int64,
        "trade_win_rate": pl.Float64,
        "average_trade_return": pl.Float64,
        "worst_trade_return": pl.Float64,
        "best_trade_return": pl.Float64,
        "average_holding_hours": pl.Float64,
        "long_rate": pl.Float64,
        "short_rate": pl.Float64,
        "flat_rate": pl.Float64,
    }
    if portfolio.is_empty():
        return pl.DataFrame(schema=schema)

    trades = trades if trades is not None else portfolio_trade_log(portfolio, timeframe=timeframe)
    closed_trades = trades.filter(pl.col("is_closed")) if not trades.is_empty() else trades
    trade_groups: dict[tuple[str, int, str], pl.DataFrame] = {}
    if not closed_trades.is_empty():
        closed_trades = closed_trades.with_columns(
            pl.col("exit_time").dt.strftime("%Y-%m").alias("_month")
        )
        for trade_frame in closed_trades.partition_by(
            ["label_name", "horizon_bars", "_month"], maintain_order=True
        ):
            trade_groups[
                (
                    str(trade_frame["label_name"][0]),
                    int(trade_frame["horizon_bars"][0]),
                    str(trade_frame["_month"][0]),
                )
            ] = trade_frame

    base_periods_per_year = 365 * 24 * 60 / bar_minutes(timeframe)
    if "position" in portfolio.columns:
        position = pl.col("position")
    else:
        position = (
            (pl.col("long_count") > 0).cast(pl.Int8)
            - (pl.col("short_count") > 0).cast(pl.Int8)
        )
    grouped = (
        portfolio.sort("label_name", "horizon_bars", "timestamp")
        .with_columns(position.alias("_position"))
        .with_columns(
            pl.col("_position")
            .shift(1)
            .over("label_name", "horizon_bars")
            .fill_null(0)
            .alias("_previous_position"),
            pl.col("timestamp").dt.strftime("%Y-%m").alias("month"),
        )
    )
    rows: list[dict[str, object]] = []
    for frame in grouped.partition_by(
        ["label_name", "horizon_bars", "month"], maintain_order=True
    ):
        values = [float(value) for value in frame["portfolio_return"].to_list()]
        gross_values = [float(value) for value in frame["gross_portfolio_return"].to_list()]
        benchmark_values = [float(value) for value in frame["benchmark_return"].to_list()]
        strategy_wealth = math.prod(1 + value for value in values)
        benchmark_wealth = math.prod(1 + value for value in benchmark_values)
        horizon = int(frame["horizon_bars"][0])
        periods_per_year = base_periods_per_year / horizon
        mean = sum(values) / len(values)
        std = _sample_std(values)
        month = str(frame["month"][0])
        key = (str(frame["label_name"][0]), horizon, month)
        month_trades = trade_groups.get(key)
        trade_count = month_trades.height if month_trades is not None else 0
        trade_returns = (
            [float(value) for value in month_trades["strategy_return"].to_list()]
            if month_trades is not None
            else []
        )
        current = frame["_position"]
        previous = frame["_previous_position"]
        side_changed = (current.sign() != previous.sign())
        rows.append(
            {
                "factor_name": frame["factor_name"][0],
                "label_name": frame["label_name"][0],
                "horizon_bars": horizon,
                "month": month,
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
                "total_turnover": float(frame["turnover"].sum()),
                "position_changes": int((frame["turnover"] > 0).sum()),
                "entries": int(((current != 0) & side_changed).sum()),
                "exits": int(((previous != 0) & side_changed).sum()),
                "closed_trades": trade_count,
                "winning_trades": sum(value > 0 for value in trade_returns),
                "trade_win_rate": (
                    sum(value > 0 for value in trade_returns) / trade_count
                    if trade_count
                    else None
                ),
                "average_trade_return": (
                    sum(trade_returns) / trade_count if trade_count else None
                ),
                "worst_trade_return": min(trade_returns) if trade_returns else None,
                "best_trade_return": max(trade_returns) if trade_returns else None,
                "average_holding_hours": (
                    float(month_trades["holding_hours"].mean())
                    if month_trades is not None
                    else None
                ),
                "long_rate": float((frame["long_count"] > 0).mean()),
                "short_rate": float((frame["short_count"] > 0).mean()),
                "flat_rate": float(
                    ((frame["long_count"] == 0) & (frame["short_count"] == 0)).mean()
                ),
            }
        )
    return pl.DataFrame(rows, schema=schema, infer_schema_length=None).sort(
        "horizon_bars", "month"
    )


def portfolio_monthly_summary(
    monthly_metrics: pl.DataFrame,
    *,
    trades: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Summarize monthly consistency from the first traded month onward."""

    schema = {
        "factor_name": pl.Utf8,
        "label_name": pl.Utf8,
        "horizon_bars": pl.Int32,
        "months": pl.Int64,
        "positive_month_rate": pl.Float64,
        "negative_month_rate": pl.Float64,
        "flat_month_rate": pl.Float64,
        "average_monthly_return": pl.Float64,
        "median_monthly_return": pl.Float64,
        "monthly_sharpe": pl.Float64,
        "worst_month_return": pl.Float64,
        "best_month_return": pl.Float64,
        "total_return": pl.Float64,
        "total_transaction_cost": pl.Float64,
        "average_monthly_entries": pl.Float64,
        "closed_trades": pl.Int64,
        "trade_win_rate": pl.Float64,
        "average_trade_return": pl.Float64,
        "median_trade_return": pl.Float64,
        "average_holding_hours": pl.Float64,
    }
    if monthly_metrics.is_empty():
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, object]] = []
    for full_frame in monthly_metrics.partition_by(
        ["label_name", "horizon_bars"], maintain_order=True
    ):
        first_traded = full_frame.filter(pl.col("position_changes") > 0)
        if first_traded.is_empty():
            frame = full_frame
        else:
            frame = full_frame.filter(pl.col("month") >= str(first_traded["month"][0]))
        values = [float(value) for value in frame["strategy_return"].to_list()]
        mean = sum(values) / len(values)
        std = _sample_std(values)
        horizon = int(frame["horizon_bars"][0])
        group_trades = None
        if trades is not None and not trades.is_empty():
            group_trades = trades.filter(
                (pl.col("label_name") == frame["label_name"][0])
                & (pl.col("horizon_bars") == horizon)
                & pl.col("is_closed")
            )
        trade_returns = (
            [float(value) for value in group_trades["strategy_return"].to_list()]
            if group_trades is not None
            else []
        )
        rows.append(
            {
                "factor_name": frame["factor_name"][0],
                "label_name": frame["label_name"][0],
                "horizon_bars": horizon,
                "months": len(values),
                "positive_month_rate": sum(value > 0 for value in values) / len(values),
                "negative_month_rate": sum(value < 0 for value in values) / len(values),
                "flat_month_rate": sum(value == 0 for value in values) / len(values),
                "average_monthly_return": mean,
                "median_monthly_return": _median(values),
                "monthly_sharpe": mean / std * 12**0.5 if std > 0 else None,
                "worst_month_return": min(values),
                "best_month_return": max(values),
                "total_return": math.prod(1 + value for value in values) - 1,
                "total_transaction_cost": float(frame["transaction_cost"].sum()),
                "average_monthly_entries": float(frame["entries"].mean()),
                "closed_trades": len(trade_returns),
                "trade_win_rate": (
                    sum(value > 0 for value in trade_returns) / len(trade_returns)
                    if trade_returns
                    else None
                ),
                "average_trade_return": (
                    sum(trade_returns) / len(trade_returns) if trade_returns else None
                ),
                "median_trade_return": _median(trade_returns) if trade_returns else None,
                "average_holding_hours": (
                    float(group_trades["holding_hours"].mean())
                    if group_trades is not None and not group_trades.is_empty()
                    else None
                ),
            }
        )
    return pl.DataFrame(rows, schema=schema, infer_schema_length=None).sort("horizon_bars")


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


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
            "smoothed_signal_value": pl.Float64,
            "signal_standardization_mean": pl.Float64,
            "signal_standardization_std": pl.Float64,
            "long_trend_return": pl.Float64,
            "position": pl.Float64,
        }
    )
