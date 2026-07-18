"""Base contract and shared helpers for factors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import polars as pl

from trend_trader.data.models import DataType, bar_minutes
from trend_trader.factors.models import DataDependency, FactorSpec


class Factor(ABC):
    """Stateless batch factor implementation."""

    name: str
    version = "1"
    dependencies: tuple[DataDependency, ...] = (DataDependency(DataType.CANDLES),)

    def required_history_bars(self, spec: FactorSpec, bar_type: str) -> int:
        return 0

    def factor_name(self, spec: FactorSpec) -> str:
        if not spec.params:
            return self.name
        encoded = ",".join(f"{key}={spec.params[key]}" for key in sorted(spec.params))
        return f"{self.name}[{encoded}]"

    @abstractmethod
    def compute(
        self,
        inputs: Mapping[DataType, pl.DataFrame],
        spec: FactorSpec,
        bar_type: str,
    ) -> pl.DataFrame:
        """Return ``timestamp`` and ``raw_value`` columns."""


def positive_int(params: Mapping[str, Any], name: str, default: int) -> int:
    value = int(params.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def duration_bars(value: Any, bar_type: str, *, default: str) -> int:
    """Convert an integer bar count or duration such as ``24h`` to bars."""

    value = default if value is None else value
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("lookback bars must be positive")
        return value
    text = str(value).strip().lower()
    minutes = bar_minutes(text)
    output_minutes = bar_minutes(bar_type)
    if minutes < output_minutes or minutes % output_minutes:
        raise ValueError(f"duration {text!r} must be a multiple of bar_type {bar_type!r}")
    return minutes // output_minutes


def annualization_factor(bar_type: str) -> float:
    periods_per_year = (365 * 24 * 60) / bar_minutes(bar_type)
    return periods_per_year**0.5


def anchor(inputs: Mapping[DataType, pl.DataFrame]) -> pl.DataFrame:
    return inputs[DataType.CANDLES].select("timestamp").sort("timestamp")


def bar_timedelta(bar_type: str) -> timedelta:
    return timedelta(minutes=bar_minutes(bar_type))
