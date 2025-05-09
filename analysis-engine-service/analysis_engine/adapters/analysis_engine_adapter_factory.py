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

class AnalysisEngineAdapterFactory:
    """Factory for creating instances of the AnalysisEngineAdapter"""
    
    def __init__(self):
        """Initialize factory dependencies"""
        # Initialize repositories
        self.tool_effectiveness_repository = ToolEffectivenessRepository()
        
    async def create_adapter(self) -> AnalysisEngineAdapter:
        """Create and return a configured instance of AnalysisEngineAdapter"""
        try:
            # Initialize required services
            analysis_service = AnalysisService(
                tool_effectiveness_repository=self.tool_effectiveness_repository
            )
            await analysis_service.initialize()
            
            market_regime_service = MarketRegimeService()
            tool_effectiveness_service = ToolEffectivenessService(
                repository=self.tool_effectiveness_repository
            )
            
            # Create and return the adapter
            adapter = AnalysisEngineAdapter(
                analysis_service=analysis_service,
                market_regime_service=market_regime_service,
                tool_effectiveness_service=tool_effectiveness_service
            )
            
            logger.info("Successfully created AnalysisEngineAdapter instance")
            return adapter
            
        except Exception as e:
            logger.error(f"Error creating AnalysisEngineAdapter: {str(e)}")
            raise
