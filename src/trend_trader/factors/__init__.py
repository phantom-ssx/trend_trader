"""Unified factor calculation API."""

from trend_trader.factors.derivatives import DERIVATIVE_FACTORS
from trend_trader.factors.engine import FactorClient
from trend_trader.factors.liquidity import LIQUIDITY_FACTORS
from trend_trader.factors.models import (
    FactorRequest,
    FactorResult,
    FactorSpec,
    NeutralizeConfig,
    OutlierConfig,
    ProcessingConfig,
    StandardizeConfig,
    factor_request,
)
from trend_trader.factors.price import PRICE_FACTORS
from trend_trader.factors.registry import FactorRegistry, default_registry
from trend_trader.factors.volatility import VOLATILITY_FACTORS

for _factor in (*PRICE_FACTORS, *VOLATILITY_FACTORS, *LIQUIDITY_FACTORS, *DERIVATIVE_FACTORS):
    if _factor.name not in default_registry.names():
        default_registry.register(_factor)

__all__ = [
    "FactorClient",
    "FactorRegistry",
    "FactorRequest",
    "FactorResult",
    "FactorSpec",
    "NeutralizeConfig",
    "OutlierConfig",
    "ProcessingConfig",
    "StandardizeConfig",
    "default_registry",
    "factor_request",
]
