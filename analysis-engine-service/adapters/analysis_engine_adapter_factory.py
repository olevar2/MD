"""
Analysis Engine Adapter Factory

This module provides a factory for creating instances of the AnalysisEngineAdapter.
"""
import logging
from typing import Optional
from analysis_engine.services.analysis_service import AnalysisService
from analysis_engine.services.market_regime_detector import MarketRegimeService
from analysis_engine.services.tool_effectiveness_service import ToolEffectivenessService
from analysis_engine.adapters.analysis_engine_adapter import AnalysisEngineAdapter
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AnalysisEngineAdapterFactory:
    """Factory for creating instances of the AnalysisEngineAdapter"""

    def __init__(self):
        """Initialize factory dependencies"""
        self.tool_effectiveness_repository = ToolEffectivenessRepository()

    @with_resilience('create_adapter')
    @async_with_exception_handling
    async def create_adapter(self) ->AnalysisEngineAdapter:
        """Create and return a configured instance of AnalysisEngineAdapter"""
        try:
            analysis_service = AnalysisService(tool_effectiveness_repository
                =self.tool_effectiveness_repository)
            await analysis_service.initialize()
            market_regime_service = MarketRegimeService()
            tool_effectiveness_service = ToolEffectivenessService(repository
                =self.tool_effectiveness_repository)
            adapter = AnalysisEngineAdapter(analysis_service=
                analysis_service, market_regime_service=
                market_regime_service, tool_effectiveness_service=
                tool_effectiveness_service)
            logger.info('Successfully created AnalysisEngineAdapter instance')
            return adapter
        except Exception as e:
            logger.error(f'Error creating AnalysisEngineAdapter: {str(e)}')
            raise
