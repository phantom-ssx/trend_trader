"""Shared mechanics for factor and strategy experiment pipelines."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl

from trend_trader.data import MarketDataClient
from trend_trader.data.models import DataType, bar_minutes
from trend_trader.experiments.versioning import data_fingerprint, factor_code_version, git_revision
from trend_trader.factors import (
    FactorRequest,
    FactorSpec,
    NeutralizeConfig,
    OutlierConfig,
    ProcessingConfig,
    StandardizeConfig,
    default_registry,
)
from trend_trader.research import ExecutionReturnSpec, FactorResearchClient, ResearchDataset


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    experiment_id: str
    artifact_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PreparedExperiment:
    git_commit: str
    git_dirty: bool
    specs: tuple[FactorSpec, ...]
    factor_versions: tuple[dict[str, str], ...]
    universe: pl.DataFrame
    instruments: tuple[str, ...]


def prepare_experiment(
    data: MarketDataClient,
    config: Any,
    workdir: Path,
) -> PreparedExperiment:
    git_commit, git_dirty = git_revision(
        workdir,
        allow_dirty=config.experiment.allow_dirty_git,
    )
    specs = tuple(FactorSpec(item.name, item.params) for item in config.factor_configs)
    versions = tuple(factor_code_version(spec) for spec in specs)
    universe = resolve_universe(data, config)
    instruments = tuple(str(value) for value in universe["instrument_id"].to_list())
    if not instruments:
        raise ValueError("experiment universe is empty")
    return PreparedExperiment(
        git_commit=git_commit,
        git_dirty=git_dirty,
        specs=specs,
        factor_versions=versions,
        universe=universe,
        instruments=instruments,
    )


def build_component_research(
    data: MarketDataClient,
    config: Any,
    prepared: PreparedExperiment,
) -> ResearchDataset:
    request = FactorRequest(
        factors=prepared.specs,
        instrument_ids=prepared.instruments,
        start=config.data.start,
        end=config.data.end,
        bar_type=config.data.timeframe,
        venue=config.data.universe.venue,
        processing=processing_config(config),
    )
    # Labels describe predictive targets. Strategy trading costs are applied later
    # to actual portfolio turnover, not embedded in every supervised target.
    labels = tuple(
        ExecutionReturnSpec(horizon_bars=horizon, round_trip_cost_bps=0)
        for horizon in config.label.horizons
    )
    dataset = FactorResearchClient(data).build(request, labels)
    if dataset.frame.is_empty():
        raise ValueError("experiment produced no factor-label observations")
    return dataset


def fingerprint_experiment_data(
    data: MarketDataClient,
    config: Any,
    prepared: PreparedExperiment,
) -> tuple[str, dict[str, Any]]:
    calculations = [(spec, default_registry.get(spec.name)) for spec in prepared.specs] + [
        (FactorSpec(name), default_registry.get(name)) for name in config.preprocess.neutralize
    ]
    history_bars = (
        max(
            factor.required_history_bars(spec, config.data.timeframe)
            for spec, factor in calculations
        )
        + 1
    )
    data_types = tuple(
        sorted(
            {
                DataType.CANDLES.value,
                *(
                    dependency.data_type.value
                    for _, factor in calculations
                    for dependency in factor.dependencies
                ),
            }
        )
    )
    step = timedelta(minutes=bar_minutes(config.data.timeframe))
    return data_fingerprint(
        data,
        instruments=prepared.instruments,
        data_types=data_types,
        venue=config.data.universe.venue,
        bar_type=config.data.timeframe,
        start=(config.data.start - step * history_bars).isoformat(),
        end=(config.data.end + step * max(config.label.horizons)).isoformat(),
    )


def resolve_universe(data: MarketDataClient, config: Any) -> pl.DataFrame:
    settings = config.data.universe
    if settings.mode == "explicit":
        return pl.DataFrame(
            {
                "instrument_id": list(settings.instruments),
                "as_of": [config.data.start] * len(settings.instruments),
                "rank": list(range(1, len(settings.instruments) + 1)),
                "selection_mode": ["explicit"] * len(settings.instruments),
            }
        )
    result = data.trading_universe(
        config.data.start,
        name=settings.name,
        venue=settings.venue,
        instrument_type=settings.instrument_type,
        settle_currency=settings.settle_currency,
        contract_type=settings.contract_type,
        min_listing_days=settings.min_listing_days,
        min_volume_usd_24h=settings.min_quote_volume_24h,
        min_open_interest_usd=settings.min_open_interest_usd,
        max_spread_bps=settings.max_spread_bps,
        top_n=settings.top_n_by_volume,
        refresh=False,
        persist=False,
    )
    if "instrument_id" not in result.columns:
        raise ValueError("universe result is missing instrument_id")
    return result


def processing_config(config: Any) -> ProcessingConfig:
    winsorize = config.preprocess.winsorize
    return ProcessingConfig(
        outlier=OutlierConfig(
            method="winsorize" if winsorize is not None else "none",
            scope=winsorize.scope if winsorize is not None else "cross_sectional",
            lower_quantile=winsorize.lower if winsorize is not None else 0.01,
            upper_quantile=winsorize.upper if winsorize is not None else 0.99,
            window=config.preprocess.rolling_window,
            min_periods=config.preprocess.rolling_min_periods,
        ),
        standardize=StandardizeConfig(
            method=config.preprocess.normalize,
            scope=config.preprocess.normalize_scope,
            window=config.preprocess.rolling_window,
            min_periods=config.preprocess.rolling_min_periods,
            min_cross_section=config.evaluation.min_cross_section,
        ),
        neutralize=NeutralizeConfig(
            exposures=config.preprocess.neutralize,
            min_observations=config.evaluation.min_cross_section,
        ),
    )


def translate_combination_params(
    params: dict[str, Any],
    references: dict[str, str],
) -> dict[str, Any]:
    translated = copy.deepcopy(params)
    weights = translated.get("weights")
    if isinstance(weights, dict):
        translated["weights"] = {
            references.get(str(name), str(name)): value for name, value in weights.items()
        }
    _translate_rule_references(translated, references)
    rules = translated.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, dict):
                _translate_rule_references(rule, references)
    return translated


def first_float(frame: pl.DataFrame, column: str) -> float | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    value = frame[column][0]
    return float(value) if value is not None else None


def _translate_rule_references(config: dict[str, Any], references: dict[str, str]) -> None:
    conditions = config.get("conditions")
    if isinstance(conditions, list):
        for condition in conditions:
            if isinstance(condition, dict) and "factor" in condition:
                factor = str(condition["factor"])
                condition["factor"] = references.get(factor, factor)
    for key in ("score_factor", "default_score_factor", "fail_score_factor"):
        if key in config:
            factor = str(config[key])
            config[key] = references.get(factor, factor)
