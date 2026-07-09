from __future__ import annotations

import os

from nautilus_trader.adapters.okx.config import OKXDataClientConfig, OKXExecClientConfig
from nautilus_trader.adapters.okx.factories import (
    OKXLiveDataClientFactory,
    OKXLiveExecClientFactory,
)
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.core.nautilus_pyo3 import OKXEnvironment, OKXInstrumentType, OKXMarginMode
from nautilus_trader.live.node import TradingNode

from trend_trader.config.models import OkxRuntimeConfig

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
    return node
