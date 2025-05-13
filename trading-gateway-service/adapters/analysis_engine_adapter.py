"""
Analysis Engine Adapter for Trading Gateway Service.

This module provides an adapter for the Analysis Engine Service, allowing the Trading Gateway
to use standardized technical indicators and market analysis functionality.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.service_client.http_client import AsyncHTTPServiceClient
from common_lib.errors.base_exceptions import ServiceError, ValidationError
from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer
from core.exceptions_bridge_1 import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class AnalysisEngineAdapter(IAnalysisProvider, IIndicatorProvider,
    IPatternRecognizer):
    """
    Adapter for the Analysis Engine Service.

    This adapter provides access to technical indicators and market analysis
    functionality from the Analysis Engine Service.
    """

    def __init__(self, config: Optional[Dict[str, Any]]=None, logger:
        Optional[logging.Logger]=None):
        """
        Initialize the Analysis Engine Adapter.

        Args:
            config: Configuration for the adapter
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        service_url = self.config.get('analysis_engine_url',
            'http://analysis-engine-service:8000')
        client_config = ServiceClientConfig(service_name='analysis-engine',
            base_url=service_url, timeout=10.0, retry_count=3,
            retry_backoff=0.5)
        self.client = AsyncHTTPServiceClient(client_config)

    @with_market_data_resilience('analyze_market')
    @async_with_exception_handling
    async def analyze_market(self, symbol: str, timeframe: str,
        analysis_type: str, start_time: datetime, end_time: Optional[
        datetime]=None, parameters: Optional[Dict[str, Any]]=None) ->Dict[
        str, Any]:
        """
        Perform market analysis.

        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe for analysis
            analysis_type: Type of analysis to perform
            start_time: Start time for data
            end_time: End time for data
            parameters: Additional parameters for the analysis

        Returns:
            Dictionary containing analysis results

        Raises:
            ServiceError: If there's an error performing the analysis
        """
        try:
            json_data = {'symbol': symbol, 'timeframe': timeframe,
                'analysis_type': analysis_type, 'start_time': start_time.
                isoformat(), 'parameters': parameters or {}}
            if end_time:
                json_data['end_time'] = end_time.isoformat()
            response = await self.client.post('/api/v1/market-analysis',
                json=json_data)
            result = response.json()
            return result
        except Exception as e:
            self.logger.error(f'Error analyzing market for {symbol}: {str(e)}')
            raise ServiceError(
                f'Failed to analyze market for {symbol}: {str(e)}')

    @with_broker_api_resilience('get_analysis_types')
    @async_with_exception_handling
    async def get_analysis_types(self) ->List[Dict[str, Any]]:
        """
        Get available analysis types.

        Returns:
            List of available analysis types

        Raises:
            ServiceError: If there's an error getting analysis types
        """
        try:
            response = await self.client.get('/api/v1/market-analysis/types')
            result = response.json()
            return result.get('analysis_types', [])
        except Exception as e:
            self.logger.error(f'Error getting analysis types: {str(e)}')
            raise ServiceError(f'Failed to get analysis types: {str(e)}')

    @async_with_exception_handling
    async def backtest_strategy(self, strategy_id: str, symbol: str,
        timeframe: str, start_time: datetime, end_time: datetime,
        parameters: Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Backtest a trading strategy.

        Args:
            strategy_id: ID of the strategy to backtest
            symbol: Symbol to backtest on
            timeframe: Timeframe for backtest
            start_time: Start time for backtest
            end_time: End time for backtest
            parameters: Additional parameters for the backtest

        Returns:
            Dictionary containing backtest results

        Raises:
            ServiceError: If there's an error backtesting the strategy
        """
        try:
            json_data = {'strategy_id': strategy_id, 'symbol': symbol,
                'timeframe': timeframe, 'start_time': start_time.isoformat(
                ), 'end_time': end_time.isoformat(), 'parameters': 
                parameters or {}}
            response = await self.client.post('/api/v1/backtest', json=
                json_data)
            result = response.json()
            return result
        except Exception as e:
            self.logger.error(
                f'Error backtesting strategy {strategy_id}: {str(e)}')
            raise ServiceError(
                f'Failed to backtest strategy {strategy_id}: {str(e)}')

    @with_broker_api_resilience('calculate_indicator')
    @async_with_exception_handling
    async def calculate_indicator(self, indicator_name: str, data: pd.
        DataFrame, parameters: Optional[Dict[str, Any]]=None) ->pd.DataFrame:
        """
        Calculate a technical indicator.

        Args:
            indicator_name: Name of the indicator to calculate
            data: Input data for the calculation
            parameters: Parameters for the indicator

        Returns:
            DataFrame with the indicator values

        Raises:
            ServiceError: If there's an error calculating the indicator
        """
        try:
            json_data = {'indicator': indicator_name, 'parameters': 
                parameters or {}, 'data': data.to_dict(orient='records')}
            response = await self.client.post('/api/v1/indicators/calculate',
                json=json_data)
            result_data = response.json()
            if 'data' in result_data:
                result_df = pd.DataFrame(result_data['data'])
                return result_df
            else:
                raise ServiceError(
                    f'Invalid response from Analysis Engine Service: {result_data}'
                    )
        except Exception as e:
            self.logger.error(
                f'Error calculating indicator {indicator_name}: {str(e)}')
            raise ServiceError(
                f'Failed to calculate indicator {indicator_name}: {str(e)}')

    @with_broker_api_resilience('get_indicator_info')
    @async_with_exception_handling
    async def get_indicator_info(self, indicator_name: str) ->Dict[str, Any]:
        """
        Get information about an indicator.

        Args:
            indicator_name: Name of the indicator

        Returns:
            Dictionary containing indicator information

        Raises:
            ServiceError: If there's an error getting indicator information
        """
        try:
            response = await self.client.get(
                f'/api/v1/indicators/{indicator_name}/info')
            result = response.json()
            return result
        except Exception as e:
            self.logger.error(
                f'Error getting indicator info for {indicator_name}: {str(e)}')
            raise ServiceError(
                f'Failed to get indicator info for {indicator_name}: {str(e)}')

    @async_with_exception_handling
    async def list_indicators(self) ->List[str]:
        """
        List available indicators.

        Returns:
            List of available indicator names

        Raises:
            ServiceError: If there's an error listing indicators
        """
        try:
            response = await self.client.get('/api/v1/indicators')
            result = response.json()
            return result.get('indicators', [])
        except Exception as e:
            self.logger.error(f'Error listing indicators: {str(e)}')
            raise ServiceError(f'Failed to list indicators: {str(e)}')

    @async_with_exception_handling
    async def detect_market_regime(self, symbol: str, timeframe: str,
        lookback_bars: int=100) ->Dict[str, Any]:
        """
        Detect the current market regime for a symbol.

        Args:
            symbol: Symbol to detect regime for
            timeframe: Timeframe for analysis
            lookback_bars: Number of bars to use for detection

        Returns:
            Dictionary containing market regime information

        Raises:
            ServiceError: If there's an error detecting market regime
        """
        try:
            json_data = {'symbol': symbol, 'timeframe': timeframe,
                'lookback_bars': lookback_bars}
            response = await self.client.post('/api/v1/market-regime/detect',
                json=json_data)
            result = response.json()
            return result
        except Exception as e:
            self.logger.error(
                f'Error detecting market regime for {symbol}: {str(e)}')
            raise ServiceError(
                f'Failed to detect market regime for {symbol}: {str(e)}')

    @with_broker_api_resilience('get_technical_indicators')
    @async_with_exception_handling
    async def get_technical_indicators(self, symbol: str, timeframe: str,
        indicators: List[Dict[str, Any]], start_time: Optional[datetime]=
        None, end_time: Optional[datetime]=None, limit: Optional[int]=None
        ) ->Dict[str, Any]:
        """
        Get technical indicators for a symbol.

        Args:
            symbol: Symbol to get indicators for
            timeframe: Timeframe for analysis
            indicators: List of indicators to calculate
            start_time: Start time for data
            end_time: End time for data
            limit: Maximum number of data points

        Returns:
            Dictionary containing indicator values

        Raises:
            ServiceError: If there's an error getting indicators
        """
        try:
            json_data = {'symbol': symbol, 'timeframe': timeframe,
                'indicators': indicators}
            if start_time:
                json_data['start_time'] = start_time.isoformat()
            if end_time:
                json_data['end_time'] = end_time.isoformat()
            if limit:
                json_data['limit'] = limit
            response = await self.client.post('/api/v1/technical-indicators',
                json=json_data)
            result = response.json()
            return result
        except Exception as e:
            self.logger.error(
                f'Error getting technical indicators for {symbol}: {str(e)}')
            raise ServiceError(
                f'Failed to get technical indicators for {symbol}: {str(e)}')

    @async_with_exception_handling
    async def recognize_patterns(self, data: pd.DataFrame, pattern_types:
        Optional[List[str]]=None) ->Dict[str, Any]:
        """
        Recognize patterns in market data.

        Args:
            data: Input data for pattern recognition
            pattern_types: Types of patterns to recognize

        Returns:
            Dictionary containing recognized patterns

        Raises:
            ServiceError: If there's an error recognizing patterns
        """
        try:
            json_data = {'data': data.to_dict(orient='records'),
                'pattern_types': pattern_types or []}
            response = await self.client.post('/api/v1/patterns/recognize',
                json=json_data)
            result = response.json()
            return result
        except Exception as e:
            self.logger.error(f'Error recognizing patterns: {str(e)}')
            raise ServiceError(f'Failed to recognize patterns: {str(e)}')

    @with_broker_api_resilience('get_pattern_types')
    @async_with_exception_handling
    async def get_pattern_types(self) ->List[Dict[str, Any]]:
        """
        Get available pattern types.

        Returns:
            List of available pattern types

        Raises:
            ServiceError: If there's an error getting pattern types
        """
        try:
            response = await self.client.get('/api/v1/patterns/types')
            result = response.json()
            return result.get('pattern_types', [])
        except Exception as e:
            self.logger.error(f'Error getting pattern types: {str(e)}')
            raise ServiceError(f'Failed to get pattern types: {str(e)}')
