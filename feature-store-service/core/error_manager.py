"""
Error Management Service for Indicator Calculations.

Provides comprehensive error handling and recovery strategies for indicator calculations.
"""
from typing import Dict, Any, List, Optional, Callable
import logging
from datetime import datetime
import traceback
import pandas as pd
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class IndicatorError(Exception):
    """Base class for indicator-related errors."""

    def __init__(self, message: str, error_type: str, details: Optional[
        Dict[str, Any]]=None):
    """
      init  .
    
    Args:
        message: Description of message
        error_type: Description of error_type
        details: Description of details
        Any]]: Description of Any]]
    
    """

        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class CalculationError(IndicatorError):
    """Error raised when indicator calculation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]]=None):
        super().__init__(message, 'calculation_error', details)


class DataError(IndicatorError):
    """Error raised when there are issues with input data."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]]=None):
        super().__init__(message, 'data_error', details)


class ParameterError(IndicatorError):
    """Error raised when there are issues with indicator parameters."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]]=None):
        super().__init__(message, 'parameter_error', details)


class IndicatorErrorManager:
    """
    Manages error handling and recovery for indicator calculations.
    
    This class provides centralized error handling, logging, and recovery
    strategies for various types of errors that can occur during indicator
    calculations.
    """

    def __init__(self):
    """
      init  .
    
    """

        self.error_registry: Dict[str, List[IndicatorError]] = {}
        self.recovery_strategies: Dict[str, Callable] = {'calculation_error':
            self._handle_calculation_error, 'data_error': self.
            _handle_data_error, 'parameter_error': self._handle_parameter_error
            }

    def register_error(self, error: IndicatorError, indicator_name: str
        ) ->None:
        """
        Register an error occurrence for tracking and analysis.
        
        Args:
            error: The error that occurred
            indicator_name: Name of the indicator where the error occurred
        """
        if indicator_name not in self.error_registry:
            self.error_registry[indicator_name] = []
        self.error_registry[indicator_name].append(error)
        logger.error(f'Indicator error: {error.error_type} in {indicator_name}'
            , extra={'error_type': error.error_type, 'indicator':
            indicator_name, 'details': error.details, 'stack_trace':
            traceback.format_exc()})

    @with_exception_handling
    def handle_error(self, error: IndicatorError, indicator_name: str
        ) ->Optional[Any]:
        """
        Handle an error using appropriate recovery strategy.
        
        Args:
            error: The error to handle
            indicator_name: Name of the indicator where the error occurred
            
        Returns:
            Optional recovery result depending on the strategy
        """
        self.register_error(error, indicator_name)
        strategy = self.recovery_strategies.get(error.error_type)
        if not strategy:
            logger.warning(
                f'No recovery strategy for error type: {error.error_type}')
            return None
        try:
            return strategy(error, indicator_name)
        except Exception as e:
            logger.error(
                f'Error recovery failed for {indicator_name}: {str(e)}',
                extra={'original_error': error.error_type})
            return None

    @with_exception_handling
    def _handle_calculation_error(self, error: CalculationError,
        indicator_name: str) ->Optional[pd.DataFrame]:
        """
        Handle calculation errors by attempting recovery strategies.
        
        Args:
            error: The calculation error
            indicator_name: Name of the indicator
            
        Returns:
            Optional recovered calculation result
        """
        logger.info(f'Handling calculation error for {indicator_name}')
        params = error.details.get('parameters', {})
        data = error.details.get('input_data')
        if data is None:
            logger.error('No input data available for recovery')
            return None
        try:
            if 'period' in params:
                params['period'] = max(2, params['period'] - 1)
                logger.info(
                    f"Retrying with adjusted period: {params['period']}")
            if len(data) > 100:
                logger.info('Retrying with recent data subset')
        except Exception as e:
            logger.error(f'Recovery strategies failed: {str(e)}')
        return None

    @with_exception_handling
    def _handle_data_error(self, error: DataError, indicator_name: str
        ) ->Optional[pd.DataFrame]:
        """
        Handle data errors through data cleaning and repair strategies.
        
        Args:
            error: The data error
            indicator_name: Name of the indicator
            
        Returns:
            Optional cleaned/repaired data
        """
        logger.info(f'Handling data error for {indicator_name}')
        data = error.details.get('input_data')
        if data is None:
            logger.error('No input data available for recovery')
            return None
        try:
            if 'invalid_rows' in error.details:
                invalid_rows = error.details['invalid_rows']
                cleaned_data = data.drop(invalid_rows)
                logger.info(f'Removed {len(invalid_rows)} invalid rows')
                return cleaned_data
            if data.isna().any().any():
                cleaned_data = data.interpolate(method='linear')
                logger.info('Interpolated missing values')
                return cleaned_data
        except Exception as e:
            logger.error(f'Data recovery failed: {str(e)}')
        return None

    @with_exception_handling
    def _handle_parameter_error(self, error: ParameterError, indicator_name:
        str) ->Optional[Dict[str, Any]]:
        """
        Handle parameter errors by suggesting corrections.
        
        Args:
            error: The parameter error
            indicator_name: Name of the indicator
            
        Returns:
            Optional corrected parameters
        """
        logger.info(f'Handling parameter error for {indicator_name}')
        params = error.details.get('parameters', {})
        if not params:
            logger.error('No parameters available for recovery')
            return None
        corrected_params = params.copy()
        try:
            for param, value in params.items():
                if isinstance(value, (int, float)) and value <= 0:
                    if param.endswith('_period'):
                        corrected_params[param] = 14
                    else:
                        corrected_params[param] = abs(value)
                if isinstance(value, str):
                    if param == 'price_source' and value not in ['close',
                        'open', 'high', 'low']:
                        corrected_params[param] = 'close'
                    elif param.endswith('_period'):
                        try:
                            pd.Timedelta(value)
                        except ValueError:
                            corrected_params[param] = '1d'
            logger.info('Corrected parameter values', extra={'corrected':
                corrected_params})
            return corrected_params
        except Exception as e:
            logger.error(f'Parameter recovery failed: {str(e)}')
        return None

    def get_error_summary(self, indicator_name: Optional[str]=None) ->Dict[
        str, Any]:
        """
        Get a summary of errors for analysis.
        
        Args:
            indicator_name: Optional name to filter errors for specific indicator
            
        Returns:
            Dictionary with error statistics and details
        """
        if indicator_name:
            errors = self.error_registry.get(indicator_name, [])
            indicators = [indicator_name]
        else:
            errors = [e for errs in self.error_registry.values() for e in errs]
            indicators = list(self.error_registry.keys())
        error_counts = {}
        for error in errors:
            error_counts[error.error_type] = error_counts.get(error.
                error_type, 0) + 1
        return {'total_errors': len(errors), 'error_types': error_counts,
            'affected_indicators': indicators, 'latest_error': errors[-1] if
            errors else None, 'timestamp': datetime.utcnow()}
