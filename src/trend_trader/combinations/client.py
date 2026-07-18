"""Public multi-factor combination client and default method registry."""

from __future__ import annotations

from trend_trader.combinations.base import (
    FactorCombinationRegistry,
    FactorCombinationRequest,
    FactorCombinationResult,
)
from trend_trader.combinations.ic import IcWeightedCombiner
from trend_trader.combinations.model import WalkForwardModelCombiner
from trend_trader.combinations.static import LinearCombiner, RankCombiner, RuleCombiner
from trend_trader.research import ResearchDataset

default_combination_registry = FactorCombinationRegistry()
for _combiner in (
    RuleCombiner(),
    LinearCombiner(),
    RankCombiner(),
    IcWeightedCombiner(),
    WalkForwardModelCombiner("machine_learning"),
    WalkForwardModelCombiner("deep_learning", forced_model="mlp"),
):
    default_combination_registry.register(_combiner)


class FactorCombinationClient:
    def __init__(self, registry: FactorCombinationRegistry | None = None) -> None:
        self.registry = registry or default_combination_registry

    def combine(
        self,
        dataset: ResearchDataset,
        request: FactorCombinationRequest,
    ) -> FactorCombinationResult:
        return self.registry.get(request.method).combine(dataset, request)
