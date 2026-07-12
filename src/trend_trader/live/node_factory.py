from __future__ import annotations

import os
from decimal import Decimal

from nautilus_trader.adapters.okx.config import OKXDataClientConfig, OKXExecClientConfig
from nautilus_trader.adapters.okx.factories import (
    OKXLiveDataClientFactory,
    OKXLiveExecClientFactory,
)
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.core.nautilus_pyo3 import OKXEnvironment, OKXInstrumentType, OKXMarginMode
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Currency

from trend_trader.backtest.nautilus_engine import make_bar_type
from trend_trader.config.models import OkxRuntimeConfig
from trend_trader.strategies.hourly_ma_exit import HourlyMaExitConfig, HourlyMaExitStrategy
from trend_trader.strategies.ma_spread_atr import MaSpreadAtrConfig, MaSpreadAtrStrategy

OKX_CLIENT_NAME = "OKX"


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def _environment(demo: bool) -> OKXEnvironment:
    return OKXEnvironment.DEMO if demo else OKXEnvironment.LIVE


def _margin_mode(value: str) -> OKXMarginMode:
    normalized = value.lower()
    if normalized == "cross":
        return OKXMarginMode.CROSS
    if normalized == "isolated":
        return OKXMarginMode.ISOLATED
    raise ValueError("margin_mode must be 'cross' or 'isolated'")


def build_okx_client_configs(
    config: OkxRuntimeConfig,
) -> tuple[OKXDataClientConfig, OKXExecClientConfig]:
    credentials = {
        "api_key": _required_env("OKX_API_KEY"),
        "api_secret": _required_env("OKX_API_SECRET"),
        "api_passphrase": _required_env("OKX_PASSPHRASE"),
    }
    common = {
        **credentials,
        "instrument_types": (OKXInstrumentType.SWAP,),
        "instrument_families": (config.instrument_family,),
        "environment": _environment(config.demo),
    }
    return (
        OKXDataClientConfig(**common),
        OKXExecClientConfig(**common, margin_mode=_margin_mode(config.margin_mode)),
    )


def build_trading_node(config: OkxRuntimeConfig) -> TradingNode:
    data_config, exec_config = build_okx_client_configs(config)
    node = TradingNode(
        config=TradingNodeConfig(
            trader_id="OKX-DEMO-001" if config.demo else "OKX-LIVE-001",
            data_clients={OKX_CLIENT_NAME: data_config},
            exec_clients={OKX_CLIENT_NAME: exec_config},
        )
    )
    node.add_data_client_factory(OKX_CLIENT_NAME, OKXLiveDataClientFactory)
    node.add_exec_client_factory(OKX_CLIENT_NAME, OKXLiveExecClientFactory)
    instrument_id = InstrumentId.from_str(f"{config.inst_id}.OKX")
    strategy_config = {
        "instrument_id": instrument_id,
        "bar_type": make_bar_type(instrument_id, config.bar),
        "settlement_currency": Currency.from_str(config.instrument_family.rsplit("-", 1)[-1]),
        "trade_size": Decimal(str(config.trade_size)),
        "sizing": config.sizing,
        "leverage": Decimal(str(config.leverage)),
        "fast_period": config.fast_period,
        "slow_period": config.slow_period,
        "spread_threshold": config.spread_threshold,
        "atr_period": config.atr_period,
        "atr_pct_min": config.atr_pct_min,
        "min_order_notional": Decimal(str(config.min_order_notional)),
        "warmup_bars": config.warmup_bars,
        "load_history_on_start": True,
        "bark_url": os.getenv("BARK_URL"),
        "trading_mode": "模拟盘" if config.demo else "实盘",
    }
    if config.strategy == "best-filter":
        strategy = MaSpreadAtrStrategy(MaSpreadAtrConfig(**strategy_config))
    elif config.strategy == "hourly-exit-filter":
        strategy = HourlyMaExitStrategy(
            HourlyMaExitConfig(
                **strategy_config,
                exit_threshold=config.exit_threshold,
                cooldown_bars=config.cooldown_bars,
            )
        )
    else:
        raise ValueError(
            "Runtime trading supports strategy='best-filter' or 'hourly-exit-filter'"
        )
    node.trader.add_strategy(
        strategy
    )
    return node
