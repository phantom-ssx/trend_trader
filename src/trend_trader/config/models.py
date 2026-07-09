from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    parquet_path: Path
    exchange: str = "OKX"
    inst_id: str = "BTC-USDT-SWAP"
    bar: str = "1m"


class StrategyConfig(BaseModel):
    trade_size: float = 0.001
    fast_period: int = 10
    slow_period: int = 30


class BacktestConfig(BaseModel):
    data: DataConfig
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    starting_balance: float = 10_000.0
    base_currency: str = "USDT"


class OkxRuntimeConfig(BaseModel):
    inst_id: str = "BTC-USDT-SWAP"
    instrument_family: str = "BTC-USDT"
    bar: str = "1m"
    trade_size: float = 0.001
    demo: bool = True
    margin_mode: str = "cross"


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_backtest_config(path: Path) -> BacktestConfig:
    return BacktestConfig.model_validate(load_toml(path))


def load_okx_runtime_config(path: Path) -> OkxRuntimeConfig:
    return OkxRuntimeConfig.model_validate(load_toml(path))
