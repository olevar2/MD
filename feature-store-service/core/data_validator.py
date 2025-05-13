"""
Indicator Data Validation Service

Provides comprehensive validation for technical indicator calculations and inputs.
"""
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import pandas as pd
import numpy as np
from data_pipeline_service.validation.validation_engine import ValidationStrategy, ValidationResult, ValidationSeverity


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class IndicatorValidationType(str, Enum):
    """Types of indicator validation checks"""
    OHLCV = 'ohlcv'
    INDICATOR_INPUT = 'indicator_input'
    TIME_SERIES = 'time_series'
    CALCULATION = 'calculation'


class DataValidationService:
    """
    Service for validating indicator data and calculations.
    
    This service ensures data quality and integrity for indicator calculations
    by performing various validation checks on input data and results.
    """

    def __init__(self):
    """
      init  .
    
    """

        self.validation_rules = {'ohlcv': self._validate_ohlcv,
            'indicator_input': self._validate_indicator_input,
            'time_series': self._validate_time_series, 'calculation': self.
            _validate_calculation}
        self.error_handlers = {'missing_data': self._handle_missing_data,
            'outliers': self._handle_outliers, 'calculation_error': self.
            _handle_calculation_error}

    def _validate_ohlcv(self, data: pd.DataFrame) ->ValidationResult:
        """
        Validate OHLCV data structure and integrity.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            ValidationResult with validation outcome
        """
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data must be a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        required_columns = ['timestamp', 'open', 'high', 'low', 'close',
            'volume']
        missing_columns = [col for col in required_columns if col not in
            data.columns]
        if missing_columns:
            return ValidationResult(is_valid=False, message=
                f'Missing required columns: {missing_columns}', details={
                'missing_columns': missing_columns}, severity=
                ValidationSeverity.ERROR)
        type_checks = {'timestamp': 'datetime64[ns]', 'open': ['float64',
            'float32'], 'high': ['float64', 'float32'], 'low': ['float64',
            'float32'], 'close': ['float64', 'float32'], 'volume': [
            'float64', 'float32', 'int64', 'int32']}
        type_issues = []
        for col, expected_types in type_checks.items():
            if not isinstance(expected_types, list):
                expected_types = [expected_types]
            if str(data[col].dtype) not in expected_types:
                type_issues.append(
                    f'{col} (expected {expected_types}, got {data[col].dtype})'
                    )
        if type_issues:
            return ValidationResult(is_valid=False, message=
                'Data type validation failed', details={'type_issues':
                type_issues}, severity=ValidationSeverity.ERROR)
        price_issues = []
        invalid_hl = data[data['high'] < data['low']]
        if not invalid_hl.empty:
            price_issues.append('Found rows where High < Low')
        invalid_ho = data[data['high'] < data['open']]
        invalid_hc = data[data['high'] < data['close']]
        if not invalid_ho.empty:
            price_issues.append('Found rows where High < Open')
        if not invalid_hc.empty:
            price_issues.append('Found rows where High < Close')
        invalid_lo = data[data['low'] > data['open']]
        invalid_lc = data[data['low'] > data['close']]
        if not invalid_lo.empty:
            price_issues.append('Found rows where Low > Open')
        if not invalid_lc.empty:
            price_issues.append('Found rows where Low > Close')
        if price_issues:
            return ValidationResult(is_valid=False, message=
                'Price relationship validation failed', details={
                'price_issues': price_issues}, severity=ValidationSeverity.
                ERROR)
        return ValidationResult(is_valid=True, message=
            'OHLCV validation passed', severity=ValidationSeverity.INFO)

    @with_exception_handling
    def _validate_indicator_input(self, data: pd.DataFrame, params:
        Optional[Dict[str, Any]]=None) ->ValidationResult:
        """
        Validate indicator input parameters and data.
        
        Args:
            data: DataFrame containing input data
            params: Optional indicator parameters to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Input data must be a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        if data.empty:
            return ValidationResult(is_valid=False, message=
                'Input data is empty', severity=ValidationSeverity.ERROR)
        if params:
            param_issues = []
            for param, value in params.items():
                if isinstance(value, (int, float)):
                    if value <= 0:
                        param_issues.append(f'{param} must be positive')
                elif isinstance(value, str) and param.endswith('_period'):
                    try:
                        pd.Timedelta(value)
                    except ValueError:
                        param_issues.append(
                            f'{param} has invalid time period format')
            if param_issues:
                return ValidationResult(is_valid=False, message=
                    'Invalid indicator parameters', details={'param_issues':
                    param_issues}, severity=ValidationSeverity.ERROR)
        return ValidationResult(is_valid=True, message=
            'Indicator input validation passed', severity=
            ValidationSeverity.INFO)

    def _validate_time_series(self, data: pd.DataFrame) ->ValidationResult:
        """
        Validate time series properties of the data.
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            ValidationResult with validation outcome
        """
        if 'timestamp' not in data.columns:
            return ValidationResult(is_valid=False, message=
                'Missing timestamp column', severity=ValidationSeverity.ERROR)
        data = data.sort_values('timestamp')
        duplicates = data[data['timestamp'].duplicated()]
        if not duplicates.empty:
            return ValidationResult(is_valid=False, message=
                'Found duplicate timestamps', details={'duplicate_count':
                len(duplicates)}, severity=ValidationSeverity.ERROR)
        timestamps = pd.to_datetime(data['timestamp'])
        time_diffs = timestamps.diff()
        expected_interval = time_diffs.mode()[0]
        gaps = time_diffs[time_diffs > 2 * expected_interval]
        if not gaps.empty:
            return ValidationResult(is_valid=False, message=
                'Found gaps in time series', details={'gap_count': len(gaps
                ), 'max_gap': str(gaps.max()), 'expected_interval': str(
                expected_interval)}, severity=ValidationSeverity.WARNING)
        return ValidationResult(is_valid=True, message=
            'Time series validation passed', severity=ValidationSeverity.INFO)

    def _validate_calculation(self, result: pd.DataFrame,
        expected_properties: Optional[Dict[str, Any]]=None) ->ValidationResult:
        """
        Validate indicator calculation results.
        
        Args:
            result: DataFrame containing calculation results
            expected_properties: Optional dict of expected properties to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        if not isinstance(result, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Result must be a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        validation_issues = []
        nan_cols = result.columns[result.isna().any()].tolist()
        if nan_cols:
            validation_issues.append({'issue': 'nan_values', 'details':
                f'Found NaN values in columns: {nan_cols}'})
        inf_cols = result.columns[np.isinf(result).any()].tolist()
        if inf_cols:
            validation_issues.append({'issue': 'infinity_values', 'details':
                f'Found infinity values in columns: {inf_cols}'})
        if expected_properties:
            for prop, expected in expected_properties.items():
                if prop == 'columns' and not all(col in result.columns for
                    col in expected):
                    missing = [col for col in expected if col not in result
                        .columns]
                    validation_issues.append({'issue': 'missing_columns',
                        'details': f'Missing expected columns: {missing}'})
                elif prop == 'length' and len(result) != expected:
                    validation_issues.append({'issue': 'wrong_length',
                        'details':
                        f'Expected length {expected}, got {len(result)}'})
        if validation_issues:
            return ValidationResult(is_valid=False, message=
                'Calculation validation failed', details={'issues':
                validation_issues}, severity=ValidationSeverity.ERROR)
        return ValidationResult(is_valid=True, message=
            'Calculation validation passed', severity=ValidationSeverity.INFO)

    def _handle_missing_data(self, data: pd.DataFrame) ->pd.DataFrame:
        """Handle missing data through forward fill then backward fill."""
        return data.ffill().bfill()

    def _handle_outliers(self, data: pd.DataFrame, columns: List[str],
        threshold: float=3.0) ->pd.DataFrame:
        """Handle outliers using z-score method."""
        result = data.copy()
        for column in columns:
            if column in data.columns:
                z_scores = np.abs((data[column] - data[column].mean()) /
                    data[column].std())
                result.loc[z_scores > threshold, column] = np.nan
        return self._handle_missing_data(result)

    def _handle_calculation_error(self, error: Exception, indicator_name: str
        ) ->None:
        """Log calculation errors and provide recovery suggestions."""
        error_msg = f'Calculation error in {indicator_name}: {str(error)}'
        raise ValueError(error_msg)

    @with_exception_handling
    def validate(self, validation_type: Union[str, IndicatorValidationType],
        data: pd.DataFrame, **kwargs) ->ValidationResult:
        """
        Validate data according to specified validation type.
        
        Args:
            validation_type: Type of validation to perform
            data: Data to validate
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult with validation outcome
        """
        if isinstance(validation_type, IndicatorValidationType):
            validation_type = validation_type.value
        validator = self.validation_rules.get(validation_type)
        if not validator:
            return ValidationResult(is_valid=False, message=
                f'Unknown validation type: {validation_type}', severity=
                ValidationSeverity.ERROR)
        try:
            return validator(data, **kwargs)
        except Exception as e:
            return ValidationResult(is_valid=False, message=
                f'Validation error: {str(e)}', severity=ValidationSeverity.
                ERROR)
