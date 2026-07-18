"""Non-overlapping long-short portfolio construction for strategy experiments."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

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
) -> pl.DataFrame:
    """Build equal-weight, 50/50 gross-normalized, non-overlapping portfolios.

    Portfolio transaction costs are proportional to one-way weight turnover. The
    supplied round-trip rate is multiplied by ``0.5 * sum(abs(weight change))``.
    """

    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    step = timedelta(minutes=bar_minutes(timeframe))
    selected = (
        dataset.frame.filter(
            pl.col("is_valid")
            & (pl.col("factor_name") == factor_name)
            & pl.col("value").is_not_null()
            & pl.col("gross_return").is_not_null()
        )
        .select(
            "label_name",
            "horizon_bars",
            "timestamp",
            "exit_time",
            "instrument_id",
            "value",
            "gross_return",
        )
        .sort("horizon_bars", "timestamp", "value", "instrument_id")
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
            top_ids = [str(value) for value in top["instrument_id"].to_list()]
            bottom_ids = [str(value) for value in bottom["instrument_id"].to_list()]
            if set(top_ids) & set(bottom_ids):
                continue
            gross_long = float(top["gross_return"].mean())
            gross_short = -float(bottom["gross_return"].mean())
            hypothetical_round_trip_cost = round_trip_cost_bps / 10_000
            net_long = gross_long - hypothetical_round_trip_cost
            net_short = gross_short - hypothetical_round_trip_cost
            weights = {instrument_id: 0.5 / len(top_ids) for instrument_id in top_ids}
            weights.update({instrument_id: -0.5 / len(bottom_ids) for instrument_id in bottom_ids})
            if previous_weights:
                instruments = set(previous_weights) | set(weights)
                turnover = 0.5 * sum(
                    abs(weights.get(item, 0.0) - previous_weights.get(item, 0.0))
                    for item in instruments
                )
            else:
                turnover = 0.5 * sum(abs(weight) for weight in weights.values())
            previous_weights = weights
            gross_return = 0.5 * (gross_long + gross_short)
            transaction_cost = hypothetical_round_trip_cost * turnover
            portfolio_return = gross_return - transaction_cost
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
                    "transaction_cost": transaction_cost,
                    "portfolio_return": portfolio_return,
                    "turnover": turnover,
                }
            )
    if not rows:
        return _empty_portfolio(dataset.frame.schema["timestamp"])
    result = pl.DataFrame(rows, infer_schema_length=None).sort("horizon_bars", "timestamp")
    enriched: list[pl.DataFrame] = []
    for frame in result.partition_by(["label_name", "horizon_bars"], maintain_order=True):
        enriched.append(
            frame.with_columns(
                (pl.col("portfolio_return") + 1).cum_prod().alias("wealth")
            ).with_columns(
                (
                    pl.col("wealth") / pl.max_horizontal(pl.col("wealth").cum_max(), pl.lit(1.0))
                    - 1
                ).alias("drawdown")
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
                "turnover": pl.Float64,
                "win_rate": pl.Float64,
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
                "turnover": (
                    sum(turnover_values) / len(turnover_values) if turnover_values else 0.0
                ),
                "win_rate": sum(value > 0 for value in values) / count,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort("horizon_bars")


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


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
            "transaction_cost": pl.Float64,
            "portfolio_return": pl.Float64,
            "turnover": pl.Float64,
            "wealth": pl.Float64,
            "drawdown": pl.Float64,
        }
    )
