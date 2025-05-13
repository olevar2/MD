"""
Analysis Engine Adapter

This module provides an adapter for the Analysis Engine Service that implements
the IAnalysisEngine interface, allowing other services to interact with it in
a decoupled way.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from common_lib.interfaces.analysis_engine_interface import IAnalysisEngine
from common_lib.models import MarketData
from analysis_engine.services.analysis_service import AnalysisService
from analysis_engine.services.market_regime_detector import MarketRegimeService
from analysis_engine.services.tool_effectiveness_service import ToolEffectivenessService
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AnalysisEngineAdapter(IAnalysisEngine):
    """
    Adapter implementing IAnalysisEngine interface for the Analysis Engine Service.
    This adapter decouples other services from direct dependencies on the analysis engine.
    """

    def __init__(self, analysis_service: AnalysisService,
        market_regime_service: MarketRegimeService,
        tool_effectiveness_service: ToolEffectivenessService):
    """
      init  .
    
    Args:
        analysis_service: Description of analysis_service
        market_regime_service: Description of market_regime_service
        tool_effectiveness_service: Description of tool_effectiveness_service
    
    """

        self.analysis_service = analysis_service
        self.market_regime_service = market_regime_service
        self.tool_effectiveness_service = tool_effectiveness_service
        logger.info('AnalysisEngineAdapter initialized')

    @async_with_exception_handling
    async def run_analysis(self, analyzer_name: str, market_data: Union[
        MarketData, Dict[str, MarketData]], parameters: Optional[Dict[str,
        Any]]=None) ->Dict[str, Any]:
        """
        Run a specific type of analysis on market data
        """
        try:
            if parameters:
                analyzer = await self.analysis_service.get_analyzer(
                    analyzer_name, parameters)
            else:
                analyzer = await self.analysis_service.get_analyzer(
                    analyzer_name)
            result = await self.analysis_service.run_analysis(analyzer_name,
                market_data)
            return result
        except Exception as e:
            logger.error(f'Error running analysis {analyzer_name}: {str(e)}')
            return {'error': str(e), 'analyzer': analyzer_name, 'is_valid':
                False}

    @with_resilience('get_market_regime')
    @async_with_exception_handling
    async def get_market_regime(self, symbol: str, timeframe: str,
        from_date: Optional[datetime]=None, to_date: Optional[datetime]=None
        ) ->Dict[str, Any]:
        """
        Get the market regime analysis for a symbol
        """
        try:
            regime_analysis = await self.market_regime_service.analyze_regime(
                symbol=symbol, timeframe=timeframe, from_date=from_date,
                to_date=to_date)
            return regime_analysis
        except Exception as e:
            logger.error(f'Error getting market regime for {symbol}: {str(e)}')
            return {'error': str(e), 'symbol': symbol, 'timeframe':
                timeframe, 'is_valid': False}

    @with_resilience('get_confluence_analysis')
    @async_with_exception_handling
    async def get_confluence_analysis(self, market_data: Dict[str,
        MarketData], parameters: Optional[Dict[str, Any]]=None) ->Dict[str, Any
        ]:
        """
        Get confluence analysis across multiple timeframes/indicators
        """
        try:
            confluence_result = (await self.analysis_service.
                run_confluence_analysis(market_data=market_data, parameters
                =parameters))
            return confluence_result
        except Exception as e:
            logger.error(f'Error running confluence analysis: {str(e)}')
            return {'error': str(e), 'is_valid': False}

    @with_analysis_resilience('analyze_pattern_effectiveness')
    @async_with_exception_handling
    async def analyze_pattern_effectiveness(self, pattern_name: str, symbol:
        str, timeframe: str, from_date: Optional[datetime]=None, to_date:
        Optional[datetime]=None) ->Dict[str, Any]:
        """
        Analyze the effectiveness of a specific pattern
        """
        try:
            effectiveness = (await self.tool_effectiveness_service.
                analyze_effectiveness(tool_id=pattern_name, instrument=
                symbol, timeframe=timeframe, from_date=from_date, to_date=
                to_date))
            return effectiveness
        except Exception as e:
            logger.error(
                f'Error analyzing pattern effectiveness for {pattern_name}: {str(e)}'
                )
            return {'error': str(e), 'pattern': pattern_name, 'symbol':
                symbol, 'timeframe': timeframe, 'is_valid': False}
