"""
Advanced error recovery service for indicator calculations.

Provides sophisticated error recovery strategies and analysis for
handling indicator calculation errors.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from feature_store_service.error.error_manager import (
    IndicatorError,
    CalculationError,
    DataError,
    ParameterError
)

logger = logging.getLogger(__name__)

class RecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.success_count = 0
        self.failure_count = 0
        
    def can_handle(self, error: IndicatorError) -> bool:
        """Check if this strategy can handle the given error."""
        raise NotImplementedError
        
    def recover(self, error: IndicatorError, **kwargs) -> Any:
        """Attempt to recover from the error."""
        raise NotImplementedError
        
    def update_stats(self, success: bool) -> None:
        """Update success/failure statistics."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            
    @property
    def success_rate(self) -> float:
        """Calculate success rate of the strategy."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class DataRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for data-related errors."""
    
    def __init__(self):
        super().__init__(
            name="data_recovery",
            description="Recovers from data quality issues through cleaning and interpolation"
        )
        
    def can_handle(self, error: IndicatorError) -> bool:
        return isinstance(error, DataError)
        
    def recover(self, error: DataError, **kwargs) -> Optional[pd.DataFrame]:
        """
        Recover from data errors through various cleaning strategies.
        
        Args:
            error: The data error to recover from
            **kwargs: Additional recovery parameters
            
        Returns:
            Optional[pd.DataFrame]: Cleaned data if recovery successful, None otherwise
        """
        data = error.details.get('input_data')
        if data is None or not isinstance(data, pd.DataFrame):
            return None
            
        try:
            # Strategy 1: Handle missing values
            if data.isna().any().any():
                cleaned_data = self._handle_missing_values(data)
                if cleaned_data is not None:
                    self.update_stats(True)
                    return cleaned_data
                    
            # Strategy 2: Handle outliers
            if error.details.get('has_outliers', False):
                cleaned_data = self._handle_outliers(data)
                if cleaned_data is not None:
                    self.update_stats(True)
                    return cleaned_data
                    
            # Strategy 3: Handle inconsistent data
            if error.details.get('has_inconsistencies', False):
                cleaned_data = self._handle_inconsistencies(data)
                if cleaned_data is not None:
                    self.update_stats(True)
                    return cleaned_data
                    
        except Exception as e:
            logger.error(f"Data recovery failed: {str(e)}")
            self.update_stats(False)
            
        return None

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values through interpolation."""
        # For time series data, use time-based interpolation
        if 'timestamp' in data.columns:
            data = data.set_index('timestamp')
            data = data.interpolate(method='time')
            return data.reset_index()
            
        # For non-time series data, use default interpolation
        return data.interpolate(method='linear')

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using statistical methods."""
        result = data.copy()
        
        for column in result.select_dtypes(include=[np.number]).columns:
            # Calculate z-scores
            z_scores = np.abs((result[column] - result[column].mean()) / result[column].std())
            
            # Replace outliers with rolling median
            mask = z_scores > 3
            if mask.any():
                rolling_median = result[column].rolling(window=5, center=True).median()
                result.loc[mask, column] = rolling_median[mask]
                
        return result

    def _handle_inconsistencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle data inconsistencies."""
        result = data.copy()
        
        # Handle OHLCV inconsistencies
        if all(col in result.columns for col in ['high', 'low', 'open', 'close']):
            # Ensure high is highest
            result['high'] = result[['high', 'low', 'open', 'close']].max(axis=1)
            # Ensure low is lowest
            result['low'] = result[['high', 'low', 'open', 'close']].min(axis=1)
            
        return result


class CalculationRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for calculation-related errors."""
    
    def __init__(self):
        super().__init__(
            name="calculation_recovery",
            description="Recovers from calculation errors through parameter adjustment"
        )
        
    def can_handle(self, error: IndicatorError) -> bool:
        return isinstance(error, CalculationError)
        
    def recover(self, error: CalculationError, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Recover from calculation errors through parameter adjustment.
        
        Args:
            error: The calculation error to recover from
            **kwargs: Additional recovery parameters
            
        Returns:
            Optional[Dict[str, Any]]: Adjusted parameters if recovery successful
        """
        params = error.details.get('parameters', {})
        if not params:
            return None
            
        try:
            # Strategy 1: Adjust calculation periods
            if self._is_period_related_error(error):
                adjusted_params = self._adjust_calculation_period(params)
                if adjusted_params:
                    self.update_stats(True)
                    return adjusted_params
                    
            # Strategy 2: Adjust calculation method
            if self._is_method_related_error(error):
                adjusted_params = self._adjust_calculation_method(params)
                if adjusted_params:
                    self.update_stats(True)
                    return adjusted_params
                    
        except Exception as e:
            logger.error(f"Calculation recovery failed: {str(e)}")
            self.update_stats(False)
            
        return None

    def _is_period_related_error(self, error: CalculationError) -> bool:
        """Check if error is related to calculation period."""
        error_msg = error.args[0].lower()
        return any(term in error_msg for term in ['period', 'window', 'length'])

    def _is_method_related_error(self, error: CalculationError) -> bool:
        """Check if error is related to calculation method."""
        error_msg = error.args[0].lower()
        return any(term in error_msg for term in ['method', 'algorithm', 'calculation'])

    def _adjust_calculation_period(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adjust calculation period parameters."""
        adjusted = params.copy()
        
        # Identify period parameters
        period_params = {
            k: v for k, v in params.items() 
            if isinstance(v, (int, float)) and 'period' in k.lower()
        }
        
        for param, value in period_params.items():
            # Try reducing the period
            if value > 2:
                adjusted[param] = max(2, value - 1)
                
        return adjusted if adjusted != params else None

    def _adjust_calculation_method(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adjust calculation method parameters."""
        adjusted = params.copy()
        
        # Map of alternative calculation methods
        method_alternatives = {
            'ema': 'sma',
            'weighted': 'simple',
            'linear': 'nearest',
            'cubic': 'linear'
        }
        
        # Look for method parameters and try alternatives
        for param, value in params.items():
            if isinstance(value, str) and value.lower() in method_alternatives:
                adjusted[param] = method_alternatives[value.lower()]
                
        return adjusted if adjusted != params else None


class ParameterRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for parameter-related errors."""
    
    def __init__(self):
        super().__init__(
            name="parameter_recovery",
            description="Recovers from parameter errors through validation and correction"
        )
        
    def can_handle(self, error: IndicatorError) -> bool:
        return isinstance(error, ParameterError)
        
    def recover(self, error: ParameterError, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Recover from parameter errors through validation and correction.
        
        Args:
            error: The parameter error to recover from
            **kwargs: Additional recovery parameters
            
        Returns:
            Optional[Dict[str, Any]]: Corrected parameters if recovery successful
        """
        params = error.details.get('parameters', {})
        if not params:
            return None
            
        try:
            # Strategy 1: Fix invalid values
            corrected_params = self._fix_invalid_values(params)
            if corrected_params:
                self.update_stats(True)
                return corrected_params
                
            # Strategy 2: Apply default values
            corrected_params = self._apply_defaults(params)
            if corrected_params:
                self.update_stats(True)
                return corrected_params
                
        except Exception as e:
            logger.error(f"Parameter recovery failed: {str(e)}")
            self.update_stats(False)
            
        return None

    def _fix_invalid_values(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix invalid parameter values."""
        corrected = params.copy()
        made_changes = False
        
        for param, value in params.items():
            # Handle negative numbers
            if isinstance(value, (int, float)) and value <= 0:
                if param.endswith('_period'):
                    corrected[param] = 14  # Common default period
                else:
                    corrected[param] = abs(value)
                made_changes = True
                
            # Handle invalid strings
            elif isinstance(value, str):
                if param == 'price_source' and value not in ['close', 'open', 'high', 'low']:
                    corrected[param] = 'close'
                    made_changes = True
                elif param.endswith('_period'):
                    try:
                        pd.Timedelta(value)
                    except ValueError:
                        corrected[param] = '1d'
                        made_changes = True
                        
        return corrected if made_changes else None

    def _apply_defaults(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply default values for missing parameters."""
        defaults = {
            'price_source': 'close',
            'period': 14,
            'ma_type': 'sma',
            'alpha': 0.5,
            'adjustment': True
        }
        
        corrected = params.copy()
        made_changes = False
        
        for param, default in defaults.items():
            if param not in corrected:
                corrected[param] = default
                made_changes = True
                
        return corrected if made_changes else None


class ErrorRecoveryService:
    """
    Service for managing and executing error recovery strategies.
    
    This service coordinates various recovery strategies and maintains
    statistics about their effectiveness.
    """
    
    def __init__(self):
        self.strategies: Dict[str, RecoveryStrategy] = {
            'data': DataRecoveryStrategy(),
            'calculation': CalculationRecoveryStrategy(),
            'parameter': ParameterRecoveryStrategy()
        }
        
    def recover(self, error: IndicatorError, **kwargs) -> Optional[Any]:
        """
        Attempt to recover from an error using appropriate strategy.
        
        Args:
            error: The error to recover from
            **kwargs: Additional recovery parameters
            
        Returns:
            Optional[Any]: Recovery result if successful, None otherwise
        """
        for strategy in self.strategies.values():
            if strategy.can_handle(error):
                logger.info(f"Attempting recovery using {strategy.name}")
                result = strategy.recover(error, **kwargs)
                if result is not None:
                    logger.info(f"Recovery successful using {strategy.name}")
                    return result
                    
        logger.warning("No successful recovery strategy found")
        return None
        
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about recovery strategy effectiveness."""
        return {
            name: {
                'success_count': strategy.success_count,
                'failure_count': strategy.failure_count,
                'success_rate': strategy.success_rate
            }
            for name, strategy in self.strategies.items()
        }
