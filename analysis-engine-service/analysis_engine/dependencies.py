"""
Analysis Engine Service Dependency Configuration

This module configures dependency injection for the analysis engine service,
including setting up the adapter pattern to resolve circular dependencies.
"""
from fastapi import Depends
from typing import Annotated

from analysis_engine.adapters.analysis_engine_adapter import AnalysisEngineAdapter
from analysis_engine.adapters.analysis_engine_adapter_factory import AnalysisEngineAdapterFactory
from analysis_engine.services.analysis_service import AnalysisService
from analysis_engine.services.market_regime_detector import MarketRegimeService
from analysis_engine.services.tool_effectiveness_service import ToolEffectivenessService
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository

# Single instances of services/repositories
tool_effectiveness_repository = ToolEffectivenessRepository()
analysis_service = AnalysisService(tool_effectiveness_repository=tool_effectiveness_repository)
market_regime_service = MarketRegimeService()
tool_effectiveness_service = ToolEffectivenessService(repository=tool_effectiveness_repository)
adapter_factory = AnalysisEngineAdapterFactory()

async def get_analysis_engine_adapter() -> AnalysisEngineAdapter:
    """Dependency provider for AnalysisEngineAdapter"""
    return await adapter_factory.create_adapter()

# Type hints for dependency injection
AnalysisEngineAdapterDep = Annotated[AnalysisEngineAdapter, Depends(get_analysis_engine_adapter)]

# Export commonly used dependencies
__all__ = [
    "AnalysisEngineAdapterDep",
    "get_analysis_engine_adapter"
]
