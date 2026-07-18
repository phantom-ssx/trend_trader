"""Build point-in-time factor research datasets."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

import polars as pl

from trend_trader.data import MarketDataClient
from trend_trader.data.models import bar_minutes
from trend_trader.factors import FactorClient, FactorRequest
from trend_trader.research.labels import ExecutionReturnLabeler
from trend_trader.research.models import ExecutionReturnSpec, ResearchDataset

_JOIN_KEYS = ["venue", "instrument_id", "bar_type", "timestamp"]


class FactorResearchClient:
    """Create execution-aware datasets from factors and future labels."""

    def __init__(
        self,
        data: MarketDataClient | None = None,
        *,
        factors: FactorClient | None = None,
    ) -> None:
        if factors is not None and data is not None and factors.data is not data:
            raise ValueError("data and factors must use the same MarketDataClient")
        self.data = data or (factors.data if factors is not None else MarketDataClient())
        self.factors = factors or FactorClient(self.data)
        self.labeler = ExecutionReturnLabeler()

    def build(
        self,
        request: FactorRequest,
        labels: Sequence[ExecutionReturnSpec],
    ) -> ResearchDataset:
        specs = tuple(labels)
        if not specs:
            raise ValueError("at least one label is required")
        names = [spec.label_name for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError("research request contains duplicate label names")

        factor_result = self.factors.query(request)
        step = timedelta(minutes=bar_minutes(request.bar_type))
        label_end = request.end + step * max(spec.horizon_bars for spec in specs)
        label_frames: list[pl.DataFrame] = []
        for instrument_id in request.instrument_ids:
            candles = self.data.candles(
                instrument_id,
                request.bar_type,
                request.start,
                label_end,
                venue=request.venue,
            )
            label_frames.append(self.labeler.compute(candles, specs).frame)
        labels_frame = (
            pl.concat(label_frames, how="vertical_relaxed")
            .filter((pl.col("timestamp") >= request.start) & (pl.col("timestamp") < request.end))
            .sort("label_name", "instrument_id", "timestamp")
        )
        factor_frame = factor_result.frame.rename(
            {
                "is_valid": "factor_is_valid",
                "quality_flags": "factor_quality_flags",
            }
        )
        joined = factor_frame.join(labels_frame, on=_JOIN_KEYS, how="inner").with_columns(
            (pl.col("factor_is_valid") & pl.col("label_is_valid")).alias("is_valid")
        )
        return ResearchDataset(
            joined.select(
                *_JOIN_KEYS,
                "factor_name",
                "factor_key",
                "factor_version",
                "raw_value",
                "value",
                "factor_is_valid",
                "factor_quality_flags",
                "label_name",
                "horizon_bars",
                "round_trip_cost_bps",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "gross_return",
                "net_return",
                "label_value",
                "label_is_valid",
                "label_quality_flags",
                "is_valid",
            ).sort("label_name", "factor_name", "instrument_id", "timestamp")
        )
