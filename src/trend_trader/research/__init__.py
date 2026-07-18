"""Execution-aware factor research and analysis."""

from trend_trader.research.analysis import FactorAnalyzer
from trend_trader.research.dataset import FactorResearchClient
from trend_trader.research.labels import ExecutionReturnLabeler
from trend_trader.research.models import (
    ExecutionReturnSpec,
    FactorAnalysisReport,
    LabelResult,
    RedundancyAnalysisReport,
    ResearchDataset,
)
from trend_trader.research.redundancy import FactorRedundancyAnalyzer

__all__ = [
    "ExecutionReturnLabeler",
    "ExecutionReturnSpec",
    "FactorAnalysisReport",
    "FactorAnalyzer",
    "FactorResearchClient",
    "FactorRedundancyAnalyzer",
    "LabelResult",
    "RedundancyAnalysisReport",
    "ResearchDataset",
]
