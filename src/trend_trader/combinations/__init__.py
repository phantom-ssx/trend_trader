"""Point-in-time multi-factor combination layer."""

from trend_trader.combinations.base import (
    FactorCombinationRegistry,
    FactorCombinationRequest,
    FactorCombinationResult,
    FactorCombiner,
)
from trend_trader.combinations.client import (
    FactorCombinationClient,
    default_combination_registry,
)

__all__ = [
    "FactorCombinationClient",
    "FactorCombinationRegistry",
    "FactorCombinationRequest",
    "FactorCombinationResult",
    "FactorCombiner",
    "default_combination_registry",
]
