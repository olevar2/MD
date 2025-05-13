"""
Analysis Engine Adapter Module

This module implements the adapter pattern for the analysis engine service,
using the interfaces defined in common-lib.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer
from common_lib.errors.base_exceptions import BaseError, ErrorCode, ValidationError, DataError, ServiceError
from analysis_engine.services.analysis_service import AnalysisService
from analysis_engine.services.indicator_service import IndicatorService
from analysis_engine.services.pattern_service import PatternService
from analysis_engine.config.settings import get_settings
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AnalysisProviderAdapter(IAnalysisProvider):
    """
    Adapter for the AnalysisService to implement the IAnalysisProvider interface.
    
    This adapter allows the AnalysisService to be used through the standardized
    IAnalysisProvider interface, enabling better service integration and
    reducing circular dependencies.
    """

    def __init__(self, analysis_service: Optional[AnalysisService]=None):
        """
        Initialize the AnalysisProviderAdapter.
        
        Args:
            analysis_service: Optional AnalysisService instance. If not provided,
                             a new instance will be created.
        """
        self._analysis_service = analysis_service or AnalysisService()
        self._settings = get_settings()
        logger.info('AnalysisProviderAdapter initialized')

    @with_analysis_resilience('analyze_market_data')
    @async_with_exception_handling
    async def analyze_market_data(self, symbol: str, timeframe: str,
        start_time: datetime, end_time: Optional[datetime]=None, indicators:
        Optional[List[str]]=None, patterns: Optional[List[str]]=None) ->Dict[
        str, Any]:
        """
        Analyze market data for a symbol.
        
        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: The timeframe (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time for the data
            end_time: Optional end time for the data
            indicators: Optional list of indicators to include in the analysis
            patterns: Optional list of patterns to detect
            
        Returns:
            Dictionary containing the analysis results
            
        Raises:
            ValidationError: If the input parameters are invalid
            DataError: If there's an issue with the data
            ServiceError: If there's a service-related error
        """
        try:
            if not symbol:
                raise ValidationError('Symbol cannot be empty', field='symbol')
            if not timeframe:
                raise ValidationError('Timeframe cannot be empty', field=
                    'timeframe')
            if not start_time:
                raise ValidationError('Start time cannot be empty', field=
                    'start_time')
            parameters = {'indicators': indicators or [], 'patterns': 
                patterns or []}
            result = await self._analysis_service.analyze_market(symbol=
                symbol, timeframe=timeframe, analysis_type='comprehensive',
                start_time=start_time, end_time=end_time, parameters=parameters
                )
            return result
        except ValidationError:
            raise
        except Exception as e:
            if 'not found' in str(e).lower():
                raise DataError(
                    f'No data found for {symbol} with timeframe {timeframe}',
                    error_code=ErrorCode.DATA_MISSING_ERROR, data_source=
                    'market_data', data_type='historical', cause=e)
            elif 'database' in str(e).lower():
                raise ServiceError(
                    f'Database error while analyzing data for {symbol}',
                    error_code=ErrorCode.SERVICE_DEPENDENCY_ERROR,
                    service_name='analysis_service', operation=
                    'analyze_market_data', cause=e)
            else:
                raise ServiceError(
                    f'Error analyzing data for {symbol}: {str(e)}',
                    error_code=ErrorCode.SERVICE_UNAVAILABLE, service_name=
                    'analysis_service', operation='analyze_market_data',
                    cause=e)

    @with_resilience('get_technical_indicators')
    @async_with_exception_handling
    async def get_technical_indicators(self, symbol: str, timeframe: str,
        indicators: List[str], start_time: datetime, end_time: Optional[
        datetime]=None, params: Optional[Dict[str, Any]]=None) ->Dict[str,
        pd.DataFrame]:
        """
        Get technical indicators for a symbol.
        
        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: The timeframe (e.g., "1m", "5m", "1h", "1d")
            indicators: List of indicators to calculate
            start_time: Start time for the data
            end_time: Optional end time for the data
            params: Optional parameters for the indicators
            
        Returns:
            Dictionary mapping indicator names to DataFrames containing the indicator values
            
        Raises:
            ValidationError: If the input parameters are invalid
            DataError: If there's an issue with the data
            ServiceError: If there's a service-related error
        """
        try:
            if not symbol:
                raise ValidationError('Symbol cannot be empty', field='symbol')
            if not timeframe:
                raise ValidationError('Timeframe cannot be empty', field=
                    'timeframe')
            if not indicators:
                raise ValidationError('Indicators list cannot be empty',
                    field='indicators')
            if not start_time:
                raise ValidationError('Start time cannot be empty', field=
                    'start_time')
            parameters = {'indicators': indicators, 'params': params or {}}
            result = await self._analysis_service.get_indicators(symbol=
                symbol, timeframe=timeframe, start_time=start_time,
                end_time=end_time, parameters=parameters)
            return result
        except ValidationError:
            raise
        except Exception as e:
            if 'not found' in str(e).lower():
                raise DataError(
                    f'No data found for {symbol} with timeframe {timeframe}',
                    error_code=ErrorCode.DATA_MISSING_ERROR, data_source=
                    'market_data', data_type='historical', cause=e)
            else:
                raise ServiceError(
                    f'Error calculating indicators for {symbol}: {str(e)}',
                    error_code=ErrorCode.SERVICE_UNAVAILABLE, service_name=
                    'analysis_service', operation=
                    'get_technical_indicators', cause=e)

    @async_with_exception_handling
    async def detect_patterns(self, symbol: str, timeframe: str, patterns:
        List[str], start_time: datetime, end_time: Optional[datetime]=None,
        params: Optional[Dict[str, Any]]=None) ->Dict[str, List[Dict[str, Any]]
        ]:
        """
        Detect patterns in market data.
        
        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: The timeframe (e.g., "1m", "5m", "1h", "1d")
            patterns: List of patterns to detect
            start_time: Start time for the data
            end_time: Optional end time for the data
            params: Optional parameters for pattern detection
            
        Returns:
            Dictionary mapping pattern names to lists of detected pattern instances
            
        Raises:
            ValidationError: If the input parameters are invalid
            DataError: If there's an issue with the data
            ServiceError: If there's a service-related error
        """
        try:
            if not symbol:
                raise ValidationError('Symbol cannot be empty', field='symbol')
            if not timeframe:
                raise ValidationError('Timeframe cannot be empty', field=
                    'timeframe')
            if not patterns:
                raise ValidationError('Patterns list cannot be empty',
                    field='patterns')
            if not start_time:
                raise ValidationError('Start time cannot be empty', field=
                    'start_time')
            parameters = {'pattern_types': patterns, 'params': params or {}}
            result = await self._analysis_service.detect_patterns(symbol=
                symbol, timeframe=timeframe, start_time=start_time,
                end_time=end_time, parameters=parameters)
            return result
        except ValidationError:
            raise
        except Exception as e:
            if 'not found' in str(e).lower():
                raise DataError(
                    f'No data found for {symbol} with timeframe {timeframe}',
                    error_code=ErrorCode.DATA_MISSING_ERROR, data_source=
                    'market_data', data_type='historical', cause=e)
            else:
                raise ServiceError(
                    f'Error detecting patterns for {symbol}: {str(e)}',
                    error_code=ErrorCode.SERVICE_UNAVAILABLE, service_name=
                    'analysis_service', operation='detect_patterns', cause=e)

    @with_resilience('get_available_indicators')
    @async_with_exception_handling
    async def get_available_indicators(self) ->List[Dict[str, Any]]:
        """
        Get a list of available indicators.
        
        Returns:
            List of dictionaries containing information about available indicators
            
        Raises:
            ServiceError: If there's a service-related error
        """
        try:
            result = await self._analysis_service.get_available_indicators()
            return result
        except Exception as e:
            raise ServiceError(f'Error getting available indicators: {str(e)}',
                error_code=ErrorCode.SERVICE_UNAVAILABLE, service_name=
                'analysis_service', operation='get_available_indicators',
                cause=e)

    @with_resilience('get_available_patterns')
    @async_with_exception_handling
    async def get_available_patterns(self) ->List[Dict[str, Any]]:
        """
        Get a list of available patterns.
        
        Returns:
            List of dictionaries containing information about available patterns
            
        Raises:
            ServiceError: If there's a service-related error
        """
        try:
            result = await self._analysis_service.get_available_patterns()
            return result
        except Exception as e:
            raise ServiceError(f'Error getting available patterns: {str(e)}',
                error_code=ErrorCode.SERVICE_UNAVAILABLE, service_name=
                'analysis_service', operation='get_available_patterns', cause=e
                )
