"""Public models for factor research datasets and reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl


@dataclass(frozen=True, slots=True)
class ExecutionReturnSpec:
    """Open-to-open return available after a factor signal.

    Factor timestamps already represent the first tradable bar open. A horizon of
    four therefore labels ``open(T + 4 bars) / open(T) - 1``. Relative to the
    source signal candle this is ``open(t + 1 + 4) / open(t + 1) - 1``.
    """

    horizon_bars: int = 4
    round_trip_cost_bps: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        if self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps must not be negative")
        if self.name is not None and not self.name.strip():
            raise ValueError("label name must not be empty")

    @property
    def label_name(self) -> str:
        if self.name is not None:
            return self.name.strip()
        base = f"execution_return_{self.horizon_bars}bars"
        if self.round_trip_cost_bps:
            return f"{base}_cost={self.round_trip_cost_bps:g}bps"
        return base


@dataclass(slots=True)
class LabelResult:
    frame: pl.DataFrame

    def __len__(self) -> int:
        return self.frame.height


@dataclass(slots=True)
class ResearchDataset:
    """Long-form factor observations joined to one or more labels."""

    frame: pl.DataFrame

    def valid(self) -> pl.DataFrame:
        return self.frame.filter(pl.col("is_valid"))

    def purged_time_split(
        self,
        split_time: datetime,
        *,
        embargo: timedelta = timedelta(0),
    ) -> tuple[ResearchDataset, ResearchDataset]:
        """Split without allowing training labels to overlap validation time."""

        if embargo < timedelta(0):
            raise ValueError("embargo must not be negative")
        training = self.frame.filter(pl.col("exit_time") < split_time)
        validation = self.frame.filter(pl.col("timestamp") >= split_time + embargo)
        return ResearchDataset(training), ResearchDataset(validation)

    def to_wide(
        self,
        *,
        label_name: str | None = None,
        factor_value: str = "value",
    ) -> pl.DataFrame:
        if factor_value not in {"value", "raw_value"}:
            raise ValueError("factor_value must be 'value' or 'raw_value'")
        labels = self.frame.get_column("label_name").unique().sort().to_list()
        if label_name is None:
            if len(labels) != 1:
                raise ValueError("label_name is required when the dataset has multiple labels")
            label_name = labels[0]
        selected = self.frame.filter(pl.col("label_name") == label_name)
        keys = ["venue", "instrument_id", "bar_type", "timestamp"]
        features = selected.pivot(
            on="factor_name",
            index=keys,
            values=factor_value,
        )
        label_columns = [
            *keys,
            "label_name",
            "label_value",
            "gross_return",
            "net_return",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "label_is_valid",
            "label_quality_flags",
        ]
        labels_frame = selected.select(*label_columns).unique(subset=keys)
        return features.join(labels_frame, on=keys, how="left").sort("instrument_id", "timestamp")


@dataclass(slots=True)
class FactorAnalysisReport:
    summary: pl.DataFrame
    overall_ic: pl.DataFrame
    ic_series: pl.DataFrame
    ic_summary: pl.DataFrame
    factor_returns: pl.DataFrame
    factor_return_summary: pl.DataFrame
    quantile_returns: pl.DataFrame
    quantile_spread: pl.DataFrame
    periodic_ic: pl.DataFrame
    decay: pl.DataFrame
    factor_correlation: pl.DataFrame
    autocorrelation: pl.DataFrame


@dataclass(slots=True)
class RedundancyAnalysisReport:
    pairwise_correlation: pl.DataFrame
    vif: pl.DataFrame
    unique_contribution: pl.DataFrame
    unique_contribution_summary: pl.DataFrame
    clusters: pl.DataFrame
