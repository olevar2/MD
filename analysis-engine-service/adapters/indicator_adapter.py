"""
Indicator Provider Adapter Module

This module implements the adapter pattern for the indicator service,
using the interfaces defined in common-lib.
"""
import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from common_lib.interfaces.analysis_engine import IIndicatorProvider
from common_lib.errors.base_exceptions import BaseError, ErrorCode, ValidationError, DataError, ServiceError
from analysis_engine.services.indicator_service import IndicatorService
from analysis_engine.config.settings import get_settings
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class IndicatorProviderAdapter(IIndicatorProvider):
    """
    Adapter for the IndicatorService to implement the IIndicatorProvider interface.
    
    This adapter allows the IndicatorService to be used through the standardized
    IIndicatorProvider interface, enabling better service integration and
    reducing circular dependencies.
    """

    def __init__(self, indicator_service: Optional[IndicatorService]=None):
        """
        Initialize the IndicatorProviderAdapter.
        
        Args:
            indicator_service: Optional IndicatorService instance. If not provided,
                              a new instance will be created.
        """
        self._indicator_service = indicator_service or IndicatorService()
        self._settings = get_settings()
        logger.info('IndicatorProviderAdapter initialized')

    @with_analysis_resilience('calculate_indicator')
    @async_with_exception_handling
    async def calculate_indicator(self, indicator_name: str, data: pd.
        DataFrame, params: Optional[Dict[str, Any]]=None) ->pd.DataFrame:
        """
        Calculate a technical indicator.
        
        Args:
            indicator_name: The name of the indicator to calculate
            data: The market data to calculate the indicator on
            params: Optional parameters for the indicator
            
        Returns:
            DataFrame containing the indicator values
            
        Raises:
            ValidationError: If the input parameters are invalid
            DataError: If there's an issue with the data
            ServiceError: If there's a service-related error
        """
        try:
            if not indicator_name:
                raise ValidationError('Indicator name cannot be empty',
                    field='indicator_name')
            if data is None or data.empty:
                raise ValidationError('Data cannot be empty', field='data')
            result = await self._indicator_service.calculate_indicator(
                indicator_name=indicator_name, data=data, parameters=params or
                {})
            return result
        except ValidationError:
            raise
        except Exception as e:
            if 'not found' in str(e).lower() or 'unknown indicator' in str(e
                ).lower():
                raise DataError(f'Unknown indicator: {indicator_name}',
                    error_code=ErrorCode.DATA_MISSING_ERROR, data_source=
                    'indicator_service', data_type='indicator', cause=e)
            elif 'invalid data' in str(e).lower():
                raise DataError(f'Invalid data for indicator {indicator_name}',
                    error_code=ErrorCode.DATA_VALIDATION_ERROR, data_source
                    ='indicator_service', data_type='market_data', cause=e)
            else:
                raise ServiceError(
                    f'Error calculating indicator {indicator_name}: {str(e)}',
                    error_code=ErrorCode.SERVICE_UNAVAILABLE, service_name=
                    'indicator_service', operation='calculate_indicator',
                    cause=e)

    @with_analysis_resilience('calculate_multiple_indicators')
    @async_with_exception_handling
    async def calculate_multiple_indicators(self, indicators: List[Dict[str,
        Any]], data: pd.DataFrame) ->Dict[str, pd.DataFrame]:
        """
        Calculate multiple technical indicators.
        
        Args:
            indicators: List of dictionaries containing indicator names and parameters
            data: The market data to calculate the indicators on
            
        Returns:
            Dictionary mapping indicator names to DataFrames containing the indicator values
            
        Raises:
            ValidationError: If the input parameters are invalid
            DataError: If there's an issue with the data
            ServiceError: If there's a service-related error
        """
        try:
            if not indicators:
                raise ValidationError('Indicators list cannot be empty',
                    field='indicators')
            if data is None or data.empty:
                raise ValidationError('Data cannot be empty', field='data')
            result = {}
            for indicator_info in indicators:
                indicator_name = indicator_info.get('name')
                if not indicator_name:
                    raise ValidationError('Indicator name cannot be empty',
                        field='name')
                params = indicator_info.get('parameters', {})
                indicator_result = (await self._indicator_service.
                    calculate_indicator(indicator_name=indicator_name, data
                    =data, parameters=params))
                result[indicator_name] = indicator_result
            return result
        except ValidationError:
            raise
        except Exception as e:
            if 'not found' in str(e).lower() or 'unknown indicator' in str(e
                ).lower():
                raise DataError(f'Unknown indicator in the list',
                    error_code=ErrorCode.DATA_MISSING_ERROR, data_source=
                    'indicator_service', data_type='indicator', cause=e)
            elif 'invalid data' in str(e).lower():
                raise DataError(f'Invalid data for indicators', error_code=
                    ErrorCode.DATA_VALIDATION_ERROR, data_source=
                    'indicator_service', data_type='market_data', cause=e)
            else:
                raise ServiceError(
                    f'Error calculating multiple indicators: {str(e)}',
                    error_code=ErrorCode.SERVICE_UNAVAILABLE, service_name=
                    'indicator_service', operation=
                    'calculate_multiple_indicators', cause=e)

    @with_resilience('get_indicator_info')
    @async_with_exception_handling
    async def get_indicator_info(self, indicator_name: str) ->Dict[str, Any]:
        """
        Get information about a specific indicator.
        
        Args:
            indicator_name: The name of the indicator
            
        Returns:
            Dictionary containing information about the indicator
            
        Raises:
            ValidationError: If the input parameters are invalid
            DataError: If there's an issue with the data
            ServiceError: If there's a service-related error
        """
        try:
            if not indicator_name:
                raise ValidationError('Indicator name cannot be empty',
                    field='indicator_name')
            result = await self._indicator_service.get_indicator_info(
                indicator_name=indicator_name)
            return result
        except ValidationError:
            raise
        except Exception as e:
            if 'not found' in str(e).lower() or 'unknown indicator' in str(e
                ).lower():
                raise DataError(f'Unknown indicator: {indicator_name}',
                    error_code=ErrorCode.DATA_MISSING_ERROR, data_source=
                    'indicator_service', data_type='indicator', cause=e)
            else:
                raise ServiceError(
                    f'Error getting indicator info for {indicator_name}: {str(e)}'
                    , error_code=ErrorCode.SERVICE_UNAVAILABLE,
                    service_name='indicator_service', operation=
                    'get_indicator_info', cause=e)

    @with_resilience('get_all_indicators_info')
    @async_with_exception_handling
    async def get_all_indicators_info(self) ->List[Dict[str, Any]]:
        """
        Get information about all available indicators.
        
        Returns:
            List of dictionaries containing information about all available indicators
            
        Raises:
            ServiceError: If there's a service-related error
        """
        try:
            indicators = await self._indicator_service.list_indicators()
            result = []
            for indicator_name in indicators:
                indicator_info = (await self._indicator_service.
                    get_indicator_info(indicator_name=indicator_name))
                result.append(indicator_info)
            return result
        except Exception as e:
            raise ServiceError(f'Error getting all indicators info: {str(e)}',
                error_code=ErrorCode.SERVICE_UNAVAILABLE, service_name=
                'indicator_service', operation='get_all_indicators_info',
                cause=e)
