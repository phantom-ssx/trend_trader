"""Market-data-backed factor execution engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from trend_trader.data import MarketDataClient
from trend_trader.data.models import STORED_BAR_TYPES, DataType, bar_minutes
from trend_trader.factors.base import Factor
from trend_trader.factors.models import (
    FactorRequest,
    FactorResult,
    FactorSpec,
    ProcessingConfig,
    factor_request,
)
from trend_trader.factors.processing import FactorProcessor
from trend_trader.factors.registry import FactorRegistry, default_registry


class FactorClient:
    """Calculate registered factors from the unified market-data query layer."""

    def __init__(
        self,
        data: MarketDataClient | None = None,
        *,
        registry: FactorRegistry | None = None,
    ) -> None:
        self.data = data or MarketDataClient()
        self.registry = registry or default_registry
        self.processor = FactorProcessor()

    def query(self, request: FactorRequest) -> FactorResult:
        requested = [(spec, self.registry.get(spec.name), True) for spec in request.factors]
        requested_names = {spec.name for spec in request.factors}
        exposure_items = [
            (FactorSpec(name), self.registry.get(name), False)
            for name in request.processing.neutralize.exposures
            if name not in requested_names
        ]
        calculations = [*requested, *exposure_items]
        factor_names = [factor.factor_name(spec) for spec, factor, _ in requested]
        if len(factor_names) != len(set(factor_names)):
            raise ValueError("factor request contains duplicate factor specifications")
        dependency_types = {
            dependency.data_type
            for _, factor, _ in calculations
            for dependency in factor.dependencies
        }
        dependency_starts = {
            data_type: request.start
            - timedelta(
                minutes=(
                    max(
                        factor.required_history_bars(spec, request.bar_type)
                        for spec, factor, _ in calculations
                        if any(item.data_type is data_type for item in factor.dependencies)
                    )
                    + 1
                )
                * bar_minutes(request.bar_type)
            )
            for data_type in dependency_types
        }
        availability_lag = timedelta(minutes=bar_minutes(request.bar_type))
        raw_frames: list[pl.DataFrame] = []
        for instrument_id in request.instrument_ids:
            inputs = self._load_inputs(
                instrument_id,
                request,
                dependency_starts,
            )
            for spec, factor, is_requested in calculations:
                raw = factor.compute(inputs, spec, request.bar_type)
                self._validate_raw(raw, factor)
                raw_frames.append(
                    raw.with_columns((pl.col("timestamp") + availability_lag).alias("timestamp"))
                    .filter(
                        (pl.col("timestamp") >= request.start) & (pl.col("timestamp") < request.end)
                    )
                    .with_columns(
                        pl.lit(request.venue).alias("venue"),
                        pl.lit(instrument_id).alias("instrument_id"),
                        pl.lit(request.bar_type).alias("bar_type"),
                        pl.lit(factor.factor_name(spec)).alias("factor_name"),
                        pl.lit(factor.name).alias("factor_key"),
                        pl.lit(factor.version).alias("factor_version"),
                        pl.lit(is_requested).alias("_requested"),
                    )
                )
        all_raw = pl.concat(raw_frames, how="vertical_relaxed").select(
            "venue",
            "instrument_id",
            "bar_type",
            "timestamp",
            "factor_name",
            "factor_key",
            "factor_version",
            pl.col("raw_value").cast(pl.Float64),
            "_requested",
        )
        exposures = all_raw.filter(
            pl.col("factor_key").is_in(request.processing.neutralize.exposures)
        ).select("venue", "instrument_id", "timestamp", "factor_key", "raw_value")
        targets = all_raw.filter(pl.col("_requested")).drop("_requested")
        processed = self.processor.apply(
            targets,
            request.processing,
            exposures=exposures if request.processing.neutralize.exposures else None,
        )
        return FactorResult(
            processed.select(
                "venue",
                "instrument_id",
                "bar_type",
                "timestamp",
                "factor_name",
                "factor_key",
                "factor_version",
                "raw_value",
                "value",
                "is_valid",
                "quality_flags",
            ).sort("factor_name", "instrument_id", "timestamp")
        )

    def factor(
        self,
        name: str,
        instrument_ids: list[str] | tuple[str, ...],
        bar_type: str,
        start: datetime | str,
        end: datetime | str,
        *,
        venue: str = "OKX",
        processing: ProcessingConfig | None = None,
        **params: Any,
    ) -> FactorResult:
        return self.query(
            factor_request(
                [FactorSpec(name, params)],
                instrument_ids,
                start,
                end,
                bar_type,
                venue=venue,
                processing=processing,
            )
        )

    def _load_inputs(
        self,
        instrument_id: str,
        request: FactorRequest,
        dependency_starts: dict[DataType, datetime],
    ) -> dict[DataType, pl.DataFrame]:
        inputs: dict[DataType, pl.DataFrame] = {}
        dependencies = set(dependency_starts)
        if DataType.CANDLES in dependencies:
            inputs[DataType.CANDLES] = self.data.candles(
                instrument_id,
                request.bar_type,
                dependency_starts[DataType.CANDLES],
                request.end,
                venue=request.venue,
            )
        periodic_methods = {
            DataType.CONTRACT_BASIS: "contract_basis",
            DataType.OPEN_INTEREST: "open_interest",
            DataType.LONG_SHORT_RATIO: "long_short_ratio",
            DataType.TAKER_VOLUME: "taker_volume",
        }
        for data_type, method_name in periodic_methods.items():
            if data_type not in dependencies:
                continue
            stored_minutes = bar_minutes(STORED_BAR_TYPES[data_type])
            requested_minutes = bar_minutes(request.bar_type)
            if requested_minutes < stored_minutes or requested_minutes % stored_minutes:
                raise ValueError(
                    f"factor bar_type {request.bar_type} is incompatible with "
                    f"{data_type.value} base interval {STORED_BAR_TYPES[data_type]}"
                )
            method = getattr(self.data, method_name)
            inputs[data_type] = method(
                instrument_id,
                request.bar_type,
                dependency_starts[data_type],
                request.end,
                venue=request.venue,
            )
        if DataType.FUNDING_RATES in dependencies:
            inputs[DataType.FUNDING_RATES] = self.data.funding_rates(
                instrument_id,
                dependency_starts[DataType.FUNDING_RATES],
                request.end,
                venue=request.venue,
            )
        if DataType.LIQUIDATIONS in dependencies:
            inputs[DataType.LIQUIDATIONS] = self.data.liquidations(
                instrument_id,
                dependency_starts[DataType.LIQUIDATIONS],
                request.end,
                venue=request.venue,
            )
        if DataType.MARKET_CAP in dependencies:
            market_start = _day_floor(dependency_starts[DataType.MARKET_CAP]) - timedelta(days=1)
            market_end = _day_floor(request.end)
            if market_end < request.end:
                market_end += timedelta(days=1)
            if market_end <= market_start:
                market_end = market_start + timedelta(days=1)
            inputs[DataType.MARKET_CAP] = self.data.market_cap(
                _base_asset(instrument_id),
                market_start,
                market_end,
                venue="GLOBAL",
            )
        return inputs

    @staticmethod
    def _validate_raw(frame: pl.DataFrame, factor: Factor) -> None:
        missing = sorted({"timestamp", "raw_value"}.difference(frame.columns))
        if missing:
            raise ValueError(f"factor {factor.name} result is missing columns: {missing}")
        if frame.get_column("timestamp").n_unique() != frame.height:
            raise ValueError(f"factor {factor.name} returned duplicate timestamps")


def _day_floor(value: datetime) -> datetime:
    return datetime(value.year, value.month, value.day, tzinfo=UTC)


def _base_asset(instrument_id: str) -> str:
    base = instrument_id.split("-", maxsplit=1)[0].strip().upper()
    if not base:
        raise ValueError(f"cannot derive base asset from instrument_id {instrument_id!r}")
    return base
