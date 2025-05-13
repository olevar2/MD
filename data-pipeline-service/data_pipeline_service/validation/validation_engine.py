"""
Data Validation Engine.

Provides comprehensive data validation functionality for forex market data.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import pandas as pd
from common_lib.exceptions import DataError, DataValidationError, ForexTradingPlatformError
from core_foundations.utils.logger import get_logger
logger = get_logger('validation-engine')


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, is_valid: bool, message: str='', details: Optional[
        Dict[str, Any]]=None, severity: ValidationSeverity=
        ValidationSeverity.ERROR):
        self.is_valid = is_valid
        self.message = message
        self.details = details or {}
        self.severity = severity

    def __bool__(self) ->bool:
        """Allow using ValidationResult in boolean context."""
        return self.is_valid


class ValidationStrategy(ABC):
    """Abstract base class for all validation strategies."""

    @abstractmethod
    def validate(self, data: Any) ->ValidationResult:
        """
        Validate the provided data.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        pass


class CompositeValidator:
    """
    Composite validator that runs multiple validation strategies.
    
    This class allows combining multiple validation strategies and
    executing them sequentially on the data.
    """

    def __init__(self, name: str):
        self.name = name
        self.validators: List[ValidationStrategy] = []

    def add_validator(self, validator: ValidationStrategy) ->None:
        """Add a validator to the composite."""
        self.validators.append(validator)

    @with_exception_handling
    def validate(self, data: Any) ->Tuple[bool, List[ValidationResult]]:
        """
        Run all validators on the data.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (overall_validity, list_of_validation_results)
        """
        results = []
        is_valid = True
        for validator in self.validators:
            try:
                result = validator.validate(data)
                results.append(result)
                if (not result.is_valid and result.severity ==
                    ValidationSeverity.ERROR):
                    is_valid = False
            except DataValidationError as e:
                logger.warning(
                    f'Data validation failed using {validator.__class__.__name__}: {e}'
                    )
                results.append(e.to_dict())
                is_valid = False
            except (DataError, ForexTradingPlatformError) as e:
                logger.warning(
                    f'Platform error during validation using {validator.__class__.__name__}: {e}'
                    )
                results.append({'error_type': e.__class__.__name__,
                    'message': str(e), 'validator': validator.__class__.
                    __name__, 'details': getattr(e, 'details', {})})
                is_valid = False
            except Exception as e:
                logger.exception(
                    f'Unexpected error during validation using {validator.__class__.__name__}: {e}'
                    )
                results.append({'error_type': 'UnexpectedValidationError',
                    'message':
                    f'An unexpected error occurred in {validator.__class__.__name__}: {str(e)}'
                    , 'validator': validator.__class__.__name__, 'details':
                    {'exception_type': type(e).__name__}})
                is_valid = False
        return is_valid, results


class DataValidationEngine:
    """
    Main validation engine for data quality checks.
    
    Orchestrates the validation process across different data types
    using appropriate validators.
    """

    def __init__(self):
        self.validators: Dict[str, CompositeValidator] = {}

    def register_validator(self, data_type: str, validator: CompositeValidator
        ) ->None:
        """
        Register a validator for a specific data type.
        
        Args:
            data_type: Type of data this validator handles (e.g., 'ohlcv', 'tick')
            validator: CompositeValidator instance for this data type
        """
        self.validators[data_type] = validator

    def validate(self, data: Any, data_type: str) ->bool:
        """
        Validate data using the appropriate validator.
        
        Args:
            data: Data to validate
            data_type: Type of data to determine which validator to use
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            DataValidationError if no validator exists for the data type
        """
        if data_type not in self.validators:
            raise DataValidationError(
                f'No validator registered for data type: {data_type}')
        validator = self.validators[data_type]
        is_valid, results = validator.validate(data)
        for result in results:
            if result.is_valid:
                continue
            if result.severity == ValidationSeverity.ERROR:
                logger.error(f'Validation error: {result.message}', extra=
                    result.details)
            elif result.severity == ValidationSeverity.WARNING:
                logger.warning(f'Validation warning: {result.message}',
                    extra=result.details)
            else:
                logger.info(f'Validation info: {result.message}', extra=
                    result.details)
        return is_valid


class SchemaValidator(ValidationStrategy):
    """Validates that data conforms to the expected schema."""

    def __init__(self, required_columns: List[str], dtypes: Optional[Dict[
        str, Any]]=None):
        """
        Initialize SchemaValidator.
        
        Args:
            required_columns: List of columns that must be present
            dtypes: Optional dictionary of column name to expected dtype
        """
        self.required_columns = required_columns
        self.dtypes = dtypes or {}

    def validate(self, data: pd.DataFrame) ->ValidationResult:
        """Validate data schema."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data is not a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        missing_columns = [col for col in self.required_columns if col not in
            data.columns]
        if missing_columns:
            return ValidationResult(is_valid=False, message=
                f'Missing required columns: {missing_columns}', details={
                'missing_columns': missing_columns}, severity=
                ValidationSeverity.ERROR)
        dtype_issues = []
        for col, expected_dtype in self.dtypes.items():
            if col in data.columns:
                actual_dtype = data[col].dtype
                if not self._check_dtype_compatibility(actual_dtype,
                    expected_dtype):
                    dtype_issues.append({'column': col, 'expected': str(
                        expected_dtype), 'actual': str(actual_dtype)})
        if dtype_issues:
            return ValidationResult(is_valid=False, message=
                'Data type issues detected', details={'dtype_issues':
                dtype_issues}, severity=ValidationSeverity.ERROR)
        return ValidationResult(is_valid=True, message=
            'Schema validation passed')

    def _check_dtype_compatibility(self, actual, expected) ->bool:
        """
        Check if the actual dtype is compatible with expected dtype.
        Handles common type conversion cases.
        """
        if str(actual) == str(expected):
            return True
        if str(expected) in ('float', 'float64') and str(actual) in ('int',
            'int64'):
            return True
        if 'datetime' in str(expected) and 'datetime' in str(actual):
            return True
        return False


class NullValidator(ValidationStrategy):
    """Validates that data does not contain unexpected null values."""

    def __init__(self, nullable_columns: Optional[List[str]]=None,
        non_nullable_columns: Optional[List[str]]=None, max_null_percentage:
        Optional[Dict[str, float]]=None, severity: ValidationSeverity=
        ValidationSeverity.ERROR):
        """
        Initialize NullValidator.
        
        Args:
            nullable_columns: Columns that are allowed to contain nulls
            non_nullable_columns: Columns that should not contain any nulls
            max_null_percentage: Maximum allowed percentage of nulls per column
            severity: Severity level for null validation errors
        """
        self.nullable_columns = nullable_columns or []
        self.non_nullable_columns = non_nullable_columns or []
        self.max_null_percentage = max_null_percentage or {}
        self.severity = severity

    def validate(self, data: pd.DataFrame) ->ValidationResult:
        """Validate null values in data."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data is not a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        null_issues = []
        for col in self.non_nullable_columns:
            if col in data.columns and data[col].isnull().any():
                null_count = data[col].isnull().sum()
                null_issues.append({'column': col, 'null_count': int(
                    null_count), 'percentage': float(null_count / len(data) *
                    100)})
        for col, threshold in self.max_null_percentage.items():
            if col in data.columns:
                null_percentage = data[col].isnull().mean() * 100
                if null_percentage > threshold:
                    null_issues.append({'column': col, 'null_count': int(
                        data[col].isnull().sum()), 'percentage': float(
                        null_percentage), 'threshold': float(threshold)})
        if null_issues:
            return ValidationResult(is_valid=False, message=
                'Null value issues detected', details={'null_issues':
                null_issues}, severity=self.severity)
        return ValidationResult(is_valid=True, message='Null validation passed'
            )


class OutlierValidator(ValidationStrategy):
    """Validates that data does not contain statistical outliers."""

    def __init__(self, columns_to_check: List[str], method: str='zscore',
        threshold: float=3.0, severity: ValidationSeverity=
        ValidationSeverity.WARNING):
        """
        Initialize OutlierValidator.
        
        Args:
            columns_to_check: Columns to check for outliers
            method: Method to detect outliers ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            severity: Severity level for outlier validation issues
        """
        self.columns_to_check = columns_to_check
        self.method = method.lower()
        self.threshold = threshold
        self.severity = severity
        if self.method not in ('zscore', 'iqr'):
            raise ValueError(f'Unsupported outlier detection method: {method}')

    def validate(self, data: pd.DataFrame) ->ValidationResult:
        """Validate outliers in data."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data is not a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        outlier_issues = {}
        for col in self.columns_to_check:
            if col not in data.columns:
                continue
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            if data[col].isnull().all():
                continue
            outlier_mask = self._detect_outliers(data[col])
            outlier_indices = outlier_mask.index[outlier_mask].tolist()
            if len(outlier_indices) > 0:
                outlier_issues[col] = {'count': len(outlier_indices),
                    'percentage': len(outlier_indices) / len(data) * 100,
                    'sample_indices': outlier_indices[:10]}
        if outlier_issues:
            return ValidationResult(is_valid=True, message=
                'Outliers detected in data', details={'outlier_issues':
                outlier_issues}, severity=self.severity)
        return ValidationResult(is_valid=True, message='No outliers detected')

    def _detect_outliers(self, series: pd.Series) ->pd.Series:
        """
        Detect outliers in a series.
        
        Returns:
            Boolean series where True indicates outliers
        """
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return pd.Series(False, index=series.index)
        if self.method == 'zscore':
            mean = clean_series.mean()
            std = clean_series.std()
            if std == 0:
                return pd.Series(False, index=series.index)
            z_scores = (clean_series - mean) / std
            return z_scores.abs() > self.threshold
        elif self.method == 'iqr':
            q1 = clean_series.quantile(0.25)
            q3 = clean_series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                return pd.Series(False, index=series.index)
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            return (clean_series < lower_bound) | (clean_series > upper_bound)
        return pd.Series(False, index=series.index)


class ForexPriceValidator(ValidationStrategy):
    """Validates Forex price data for consistency and reasonable values."""

    def __init__(self, instrument_config: Dict[str, Dict[str, Any]],
        check_open_range: bool=True, check_high_low: bool=True,
        check_close_range: bool=True, severity: ValidationSeverity=
        ValidationSeverity.ERROR):
        """
        Initialize ForexPriceValidator.
        
        Args:
            instrument_config: Configuration for instruments with expected price ranges
                Format: {
                    'EUR/USD': {
                        'min_price': 1.0,
                        'max_price': 1.5,
                        'max_spread_pips': 10
                    },
                    ...
                }
            check_open_range: Whether to check if open prices are within range
            check_high_low: Whether to check if high >= low
            check_close_range: Whether to check if close prices are within range
            severity: Severity level for price validation issues
        """
        self.instrument_config = instrument_config
        self.check_open_range = check_open_range
        self.check_high_low = check_high_low
        self.check_close_range = check_close_range
        self.severity = severity

    def validate(self, data: pd.DataFrame) ->ValidationResult:
        """Validate Forex price data."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data is not a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        required_columns = []
        if 'instrument' not in data.columns:
            return ValidationResult(is_valid=False, message=
                "Missing 'instrument' column", severity=ValidationSeverity.
                ERROR)
        if self.check_open_range and 'open' not in data.columns:
            return ValidationResult(is_valid=False, message=
                "Missing 'open' column for open range validation", severity
                =ValidationSeverity.ERROR)
        if self.check_high_low:
            if 'high' not in data.columns or 'low' not in data.columns:
                return ValidationResult(is_valid=False, message=
                    "Missing 'high' or 'low' columns for high-low validation",
                    severity=ValidationSeverity.ERROR)
        if self.check_close_range and 'close' not in data.columns:
            return ValidationResult(is_valid=False, message=
                "Missing 'close' column for close range validation",
                severity=ValidationSeverity.ERROR)
        issues = []
        for instrument, group in data.groupby('instrument'):
            if instrument not in self.instrument_config:
                continue
            config = self.instrument_config[instrument]
            min_price = config_manager.get('min_price')
            max_price = config_manager.get('max_price')
            if (self.check_open_range and min_price is not None and 
                max_price is not None):
                invalid_open = group[(group['open'] < min_price) | (group[
                    'open'] > max_price)]
                if not invalid_open.empty:
                    issues.append({'instrument': instrument, 'issue':
                        'open_price_out_of_range', 'count': len(
                        invalid_open), 'min_valid': min_price, 'max_valid':
                        max_price})
            if self.check_high_low:
                invalid_hl = group[group['high'] < group['low']]
                if not invalid_hl.empty:
                    issues.append({'instrument': instrument, 'issue':
                        'high_less_than_low', 'count': len(invalid_hl)})
            if (self.check_close_range and min_price is not None and 
                max_price is not None):
                invalid_close = group[(group['close'] < min_price) | (group
                    ['close'] > max_price)]
                if not invalid_close.empty:
                    issues.append({'instrument': instrument, 'issue':
                        'close_price_out_of_range', 'count': len(
                        invalid_close), 'min_valid': min_price, 'max_valid':
                        max_price})
        if issues:
            return ValidationResult(is_valid=False, message=
                'Forex price validation issues detected', details={'issues':
                issues}, severity=self.severity)
        return ValidationResult(is_valid=True, message=
            'Forex price validation passed')


class ForexSpreadValidator(ValidationStrategy):
    """Validates Forex spread data for reasonable values."""

    def __init__(self, instrument_config: Dict[str, Dict[str, Any]],
        bid_column: str='bid', ask_column: str='ask', severity:
        ValidationSeverity=ValidationSeverity.WARNING):
        """
        Initialize ForexSpreadValidator.
        
        Args:
            instrument_config: Configuration for instruments with expected spread ranges
                Format: {
                    'EUR/USD': {
                        'decimals': 5,
                        'max_spread_pips': 10
                    },
                    ...
                }
            bid_column: Name of the bid price column
            ask_column: Name of the ask price column
            severity: Severity level for spread validation issues
        """
        self.instrument_config = instrument_config
        self.bid_column = bid_column
        self.ask_column = ask_column
        self.severity = severity

    def validate(self, data: pd.DataFrame) ->ValidationResult:
        """Validate Forex spread data."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data is not a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        if 'instrument' not in data.columns:
            return ValidationResult(is_valid=False, message=
                "Missing 'instrument' column", severity=ValidationSeverity.
                ERROR)
        if self.bid_column not in data.columns:
            return ValidationResult(is_valid=False, message=
                f"Missing bid column '{self.bid_column}'", severity=
                ValidationSeverity.ERROR)
        if self.ask_column not in data.columns:
            return ValidationResult(is_valid=False, message=
                f"Missing ask column '{self.ask_column}'", severity=
                ValidationSeverity.ERROR)
        issues = []
        for instrument, group in data.groupby('instrument'):
            if instrument not in self.instrument_config:
                continue
            config = self.instrument_config[instrument]
            decimals = config_manager.get('decimals', 5)
            max_spread_pips = config_manager.get('max_spread_pips')
            if max_spread_pips is None:
                continue
            pip_multiplier = 10 ** decimals
            spread_pips = (group[self.ask_column] - group[self.bid_column]
                ) * pip_multiplier
            high_spread = group[spread_pips > max_spread_pips]
            if not high_spread.empty:
                issues.append({'instrument': instrument, 'issue':
                    'spread_exceeded_max', 'count': len(high_spread),
                    'max_allowed_pips': max_spread_pips, 'max_found_pips':
                    float(spread_pips.max())})
            negative_spread = group[spread_pips < 0]
            if not negative_spread.empty:
                issues.append({'instrument': instrument, 'issue':
                    'negative_spread', 'count': len(negative_spread)})
        if issues:
            return ValidationResult(is_valid=False, message=
                'Forex spread validation issues detected', details={
                'issues': issues}, severity=self.severity)
        return ValidationResult(is_valid=True, message=
            'Forex spread validation passed')


class TimeSeriesContinuityValidator(ValidationStrategy):
    """Validates that a time series has expected continuity (no gaps larger than expected)."""

    def __init__(self, timestamp_column: str, expected_interval: pd.
        Timedelta, max_gap: Optional[pd.Timedelta]=None, severity:
        ValidationSeverity=ValidationSeverity.WARNING):
        """
        Initialize TimeSeriesContinuityValidator.
        
        Args:
            timestamp_column: Name of the timestamp column
            expected_interval: Expected time interval between consecutive rows
            max_gap: Maximum allowed gap (defaults to 2x expected_interval)
            severity: Severity level for continuity validation issues
        """
        self.timestamp_column = timestamp_column
        self.expected_interval = expected_interval
        self.max_gap = max_gap or expected_interval * 2
        self.severity = severity

    def validate(self, data: pd.DataFrame) ->ValidationResult:
        """Validate time series continuity."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data is not a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        if self.timestamp_column not in data.columns:
            return ValidationResult(is_valid=False, message=
                f"Missing timestamp column '{self.timestamp_column}'",
                severity=ValidationSeverity.ERROR)
        sorted_data = data.sort_values(by=self.timestamp_column).copy()
        gaps = sorted_data[self.timestamp_column].diff()
        large_gaps = sorted_data[gaps > self.max_gap].copy()
        if not large_gaps.empty:
            sample_gaps = large_gaps.head(10)
            gap_details = []
            for i, row in sample_gaps.iterrows():
                prev_ts = row[self.timestamp_column] - gaps[i]
                gap_details.append({'position': int(i), 'gap_size': str(
                    gaps[i]), 'from_timestamp': str(prev_ts),
                    'to_timestamp': str(row[self.timestamp_column])})
            return ValidationResult(is_valid=True, message=
                f'Time series continuity issues detected - {len(large_gaps)} gaps larger than {self.max_gap}'
                , details={'total_gaps': len(large_gaps), 'max_gap': str(
                gaps.max()), 'sample_gaps': gap_details}, severity=self.
                severity)
        return ValidationResult(is_valid=True, message=
            'Time series continuity validation passed')
