"""
Services package for the Analysis Engine Service.

This package contains services for various analysis tasks.
"""

from analysis_engine.services.market_regime_detector import MarketRegimeService
from analysis_engine.services.standardized_market_regime_service import StandardizedMarketRegimeService
from analysis_engine.services.tool_effectiveness import ToolEffectivenessService

# Factory functions
def get_market_regime_service(use_standardized: bool = True) -> MarketRegimeService:
    """
    Get a market regime service instance.
    
    Args:
        use_standardized: Whether to use the standardized service
        
    Returns:
        MarketRegimeService instance
    """
    if use_standardized:
        return StandardizedMarketRegimeService()
    else:
        return MarketRegimeService()
