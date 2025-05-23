"""
Standardized Market Regime Service

This module provides a standardized interface for market regime detection
using the standardized client API.
"""
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import asyncio
from analysis_engine.services.tool_effectiveness import MarketRegime
from analysis_engine.services.market_regime_detector import MarketRegimeService


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class StandardizedMarketRegimeService(MarketRegimeService):
    """
    Standardized service for detecting market regimes using the standardized client API.

    This service extends the base MarketRegimeService to use the standardized client API
    for market regime detection when available, falling back to the legacy implementation
    when necessary.
    """

    def __init__(self):
        """Initialize the standardized market regime service"""
        super().__init__()
        self.logger = logging.getLogger(__name__)

    @with_exception_handling
    def detect_market_regime(self, symbol: str, timeframe: str, ohlc_data:
        Optional[pd.DataFrame]=None, lookback_periods: int=100) ->Dict[str, Any
        ]:
        """
        Detect the current market regime for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            ohlc_data: Optional OHLC data to use for detection
            lookback_periods: Number of periods to look back for analysis

        Returns:
            Dictionary with detected regime information
        """
        try:
            from analysis_engine.clients.standardized import get_client_factory
            client = get_client_factory().get_market_regime_client()
            if ohlc_data is not None:
                if isinstance(ohlc_data, pd.DataFrame):
                    if ('timestamp' not in ohlc_data.columns and ohlc_data.
                        index.name != 'timestamp'):
                        ohlc_data = ohlc_data.reset_index()
                    ohlc_data = ohlc_data.to_dict('records')
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(client.
                    detect_market_regime(symbol=symbol, timeframe=timeframe,
                    ohlc_data=ohlc_data))
                if result and 'regime' in result:
                    return result
        except (ImportError, Exception) as e:
            self.logger.warning(
                f'Error using standardized client to get market regime: {str(e)}'
                )
        if ohlc_data is not None:
            return self.detect_current_regime(symbol, timeframe, ohlc_data)
        else:
            return {'regime': MarketRegime.UNKNOWN, 'confidence': 0.0,
                'metrics': {}, 'error':
                'No OHLC data provided and standardized client unavailable'}

    @async_with_exception_handling
    async def detect_market_regime_async(self, symbol: str, timeframe: str,
        ohlc_data: Optional[pd.DataFrame]=None, lookback_periods: int=100
        ) ->Dict[str, Any]:
        """
        Detect the current market regime for a symbol and timeframe asynchronously.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            ohlc_data: Optional OHLC data to use for detection
            lookback_periods: Number of periods to look back for analysis

        Returns:
            Dictionary with detected regime information
        """
        try:
            from analysis_engine.clients.standardized import get_client_factory
            client = get_client_factory().get_market_regime_client()
            if ohlc_data is not None:
                if isinstance(ohlc_data, pd.DataFrame):
                    if ('timestamp' not in ohlc_data.columns and ohlc_data.
                        index.name != 'timestamp'):
                        ohlc_data = ohlc_data.reset_index()
                    ohlc_data = ohlc_data.to_dict('records')
                result = await client.detect_market_regime(symbol=symbol,
                    timeframe=timeframe, ohlc_data=ohlc_data)
                if result and 'regime' in result:
                    return result
        except (ImportError, Exception) as e:
            self.logger.warning(
                f'Error using standardized client to get market regime: {str(e)}'
                )
        if ohlc_data is not None:
            return self.detect_current_regime(symbol, timeframe, ohlc_data)
        else:
            return {'regime': MarketRegime.UNKNOWN, 'confidence': 0.0,
                'metrics': {}, 'error':
                'No OHLC data provided and standardized client unavailable'}

    @async_with_exception_handling
    async def detect_current_regime(self, symbol: str, timeframe: str,
        price_data: pd.DataFrame) ->Dict[str, Any]:
        """
        Detect the current market regime for a specific symbol and timeframe

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            price_data: OHLC price data

        Returns:
            Dictionary with detected regime and supporting metrics
        """
        try:
            from analysis_engine.clients.standardized import get_client_factory
            client = get_client_factory().get_market_regime_client()
            if isinstance(price_data, pd.DataFrame):
                if ('timestamp' not in price_data.columns and price_data.
                    index.name != 'timestamp'):
                    price_data = price_data.reset_index()
                ohlc_data = price_data.to_dict('records')
            result = await client.detect_market_regime(symbol=symbol,
                timeframe=timeframe, ohlc_data=ohlc_data)
            if result and 'regime' in result:
                return result
        except (ImportError, Exception) as e:
            self.logger.warning(
                f'Error using standardized client to get market regime: {str(e)}'
                )
        return await super().detect_current_regime(symbol, timeframe,
            price_data)
