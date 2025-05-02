"""
Advanced Data Validation Engine

This module provides comprehensive validation for financial market data
with specialized validators for OHLCV and tick data formats, configurable
severity levels, and detailed validation reporting.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np
from common_lib.exceptions import DataValidationError  # Added import
from common_lib.schemas import OHLCVData, TickData

from .validation_engine import ValidationEngine  # Assuming relative import works


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = 0      # Informational only, no action required
    WARNING = 1   # Potential issue but not critical
    ERROR = 2     # Serious issue that requires attention
    CRITICAL = 3  # Fatal issue that prevents use of the data


class ValidationResult:
    """Results of a data validation check"""
    
    def __init__(
        self,
        is_valid: bool,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        affected_rows: Optional[List[int]] = None,
        affected_columns: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation result
        
        Args:
            is_valid: Whether validation passed
            message: Description of validation result
            severity: Severity level of any issues
            affected_rows: Indices of problematic rows
            affected_columns: Names of problematic columns
            details: Additional information about the validation
        """
        self.is_valid = is_valid
        self.message = message
        self.severity = severity
        self.affected_rows = affected_rows or []
        self.affected_columns = affected_columns or []
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'is_valid': self.is_valid,
            'message': self.message,
            'severity': self.severity.name,
            'affected_rows_count': len(self.affected_rows),
            'affected_columns': self.affected_columns,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }
        
    def __str__(self) -> str:
        """String representation"""
        return (f"ValidationResult(valid={self.is_valid}, "
                f"severity={self.severity.name}, "
                f"message='{self.message}')")


class DataValidator:
    """Base class for data validators"""
    
    def __init__(
        self,
        name: str,
        description: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        """
        Initialize data validator
        
        Args:
            name: Validator name
            description: Description of what this validator checks
            severity: Default severity level for issues
        """
        self.name = name
        self.description = description
        self.severity = severity
        self.logger = logging.getLogger(f"validator.{name}")
        
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate the provided data
        
        Args:
            data: DataFrame to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        raise NotImplementedError("Subclasses must implement validate method")


class NullValueValidator(DataValidator):
    """Validates for null/missing values in data"""
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        max_null_fraction: float = 0.0,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        """
        Initialize null value validator
        
        Args:
            columns: Columns to check, or None for all columns
            max_null_fraction: Maximum allowed fraction of nulls (0-1)
            severity: Severity level for issues
        """
        super().__init__(
            name="null_value_validator",
            description="Checks for null or missing values in data",
            severity=severity
        )
        self.columns = columns
        self.max_null_fraction = max_null_fraction
        
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Check for null values in the data"""
        # Determine which columns to check
        check_columns = self.columns or data.columns.tolist()
        
        # Filter to only existing columns
        check_columns = [col for col in check_columns if col in data.columns]
        
        if not check_columns:
            return ValidationResult(
                is_valid=False,
                message="No valid columns to check",
                severity=ValidationSeverity.ERROR
            )
            
        # Check for nulls in each column
        problem_columns = []
        affected_rows = []
        column_null_counts = {}
        
        for column in check_columns:
            null_mask = data[column].isnull()
            null_count = null_mask.sum()
            null_fraction = null_count / len(data) if len(data) > 0 else 0
            
            column_null_counts[column] = {
                'count': int(null_count),
                'fraction': float(null_fraction)
            }
            
            if null_fraction > self.max_null_fraction:
                problem_columns.append(column)
                # Add row indices with nulls
                affected_rows.extend(data[null_mask].index.tolist())
        
        # Remove duplicates from affected_rows
        affected_rows = sorted(set(affected_rows))
        
        if problem_columns:
            return ValidationResult(
                is_valid=False,
                message=f"Found {len(problem_columns)} columns with excessive null values",
                severity=self.severity,
                affected_rows=affected_rows,
                affected_columns=problem_columns,
                details={'null_counts': column_null_counts}
            )
        else:
            return ValidationResult(
                is_valid=True,
                message="No excessive null values found",
                severity=ValidationSeverity.INFO,
                details={'null_counts': column_null_counts}
            )


class OutlierValidator(DataValidator):
    """Validates for outliers in numerical data"""
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'zscore',
        threshold: float = 3.0,
        severity: ValidationSeverity = ValidationSeverity.WARNING
    ):
        """
        Initialize outlier validator
        
        Args:
            columns: Columns to check, or None for all numerical columns
            method: Detection method ('zscore', 'iqr', or 'percentile')
            threshold: Threshold value for outlier detection
            severity: Severity level for issues
        """
        super().__init__(
            name="outlier_validator", 
            description="Checks for outliers in numerical data",
            severity=severity
        )
        self.columns = columns
        self.method = method.lower()
        self.threshold = threshold
        
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Check for outliers in the data"""
        # Determine which columns to check (only numerical)
        if self.columns:
            check_columns = [col for col in self.columns if col in data.columns 
                             and pd.api.types.is_numeric_dtype(data[col])]
        else:
            check_columns = data.select_dtypes(include=['number']).columns.tolist()
            
        if not check_columns:
            return ValidationResult(
                is_valid=False,
                message="No valid numerical columns to check",
                severity=ValidationSeverity.ERROR
            )
            
        # Check for outliers in each column
        problem_columns = []
        affected_rows = []
        outlier_details = {}
        
        for column in check_columns:
            # Skip columns with all nulls
            if data[column].isnull().all():
                continue
                
            # Detect outliers using the specified method
            outlier_mask = self._detect_outliers(data[column])
            outlier_count = outlier_mask.sum()
            
            outlier_details[column] = {
                'count': int(outlier_count),
                'fraction': float(outlier_count / len(data)) if len(data) > 0 else 0,
                'method': self.method,
                'threshold': self.threshold
            }
            
            if outlier_count > 0:
                problem_columns.append(column)
                # Add row indices with outliers
                affected_rows.extend(data[outlier_mask].index.tolist())
        
        # Remove duplicates from affected_rows
        affected_rows = sorted(set(affected_rows))
        
        if problem_columns:
            return ValidationResult(
                is_valid=False,
                message=f"Found outliers in {len(problem_columns)} columns",
                severity=self.severity,
                affected_rows=affected_rows,
                affected_columns=problem_columns,
                details={'outliers': outlier_details}
            )
        else:
            return ValidationResult(
                is_valid=True,
                message="No outliers found",
                severity=ValidationSeverity.INFO,
                details={'outliers': outlier_details}
            )
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """
        Detect outliers in a series using the specified method
        
        Args:
            series: Data series to check
            
        Returns:
            Boolean series with True for outliers
        """
        # Skip nulls
        data = series.dropna()
        
        if len(data) == 0:
            return pd.Series(False, index=series.index)
            
        # Initialize mask with False for all rows
        mask = pd.Series(False, index=series.index)
        
        if self.method == 'zscore':
            # Z-score method
            mean = data.mean()
            std = data.std()
            
            if std > 0:  # Avoid division by zero
                zscores = np.abs((series - mean) / std)
                mask[zscores > self.threshold] = True
                
        elif self.method == 'iqr':
            # IQR method
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            if iqr > 0:  # Avoid division by zero
                lower_bound = q1 - (self.threshold * iqr)
                upper_bound = q3 + (self.threshold * iqr)
                
                mask[(series < lower_bound) | (series > upper_bound)] = True
                
        elif self.method == 'percentile':
            # Percentile method
            lower_bound = data.quantile(self.threshold / 100)
            upper_bound = data.quantile(1 - (self.threshold / 100))
            
            mask[(series < lower_bound) | (series > upper_bound)] = True
            
        else:
            self.logger.warning(f"Unknown outlier detection method: {self.method}")
            
        return mask


class OHLCVValidator(DataValidator):
    """Validates OHLCV (Open, High, Low, Close, Volume) data"""
    
    def __init__(
        self,
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: Optional[str] = 'volume',
        timestamp_col: str = 'timestamp',
        check_gaps: bool = True,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        """
        Initialize OHLCV validator
        
        Args:
            open_col: Column name for open prices
            high_col: Column name for high prices
            low_col: Column name for low prices
            close_col: Column name for close prices
            volume_col: Column name for volume, or None if not available
            timestamp_col: Column name for timestamp
            check_gaps: Whether to check for time gaps
            severity: Severity level for issues
        """
        super().__init__(
            name="ohlcv_validator",
            description="Validates OHLCV price data structure and integrity",
            severity=severity
        )
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.timestamp_col = timestamp_col
        self.check_gaps = check_gaps
        
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate OHLCV data structure and integrity"""
        # Check required columns exist
        required_cols = [self.open_col, self.high_col, self.low_col, 
                         self.close_col, self.timestamp_col]
        if self.volume_col:
            required_cols.append(self.volume_col)
            
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            return ValidationResult(
                is_valid=False,
                message=f"Missing required columns: {', '.join(missing_cols)}",
                severity=self.severity,
                affected_columns=missing_cols
            )
            
        # Check for price integrity: High >= Open/Close >= Low
        integrity_issues = []
        affected_rows = []
        
        # High >= Open
        high_open_mask = data[self.high_col] < data[self.open_col]
        if high_open_mask.any():
            integrity_issues.append("high < open")
            affected_rows.extend(data[high_open_mask].index.tolist())
            
        # High >= Close
        high_close_mask = data[self.high_col] < data[self.close_col]
        if high_close_mask.any():
            integrity_issues.append("high < close")
            affected_rows.extend(data[high_close_mask].index.tolist())
            
        # Open >= Low
        open_low_mask = data[self.open_col] < data[self.low_col]
        if open_low_mask.any():
            integrity_issues.append("open < low")
            affected_rows.extend(data[open_low_mask].index.tolist())
            
        # Close >= Low
        close_low_mask = data[self.close_col] < data[self.low_col]
        if close_low_mask.any():
            integrity_issues.append("close < low")
            affected_rows.extend(data[close_low_mask].index.tolist())
            
        # Check volume is non-negative if present
        if self.volume_col:
            neg_volume_mask = data[self.volume_col] < 0
            if neg_volume_mask.any():
                integrity_issues.append("negative volume")
                affected_rows.extend(data[neg_volume_mask].index.tolist())
                
        # Check for timestamp gaps if requested
        gap_details = None
        if self.check_gaps and len(data) > 1:
            try:
                # Sort by timestamp if necessary
                if not data[self.timestamp_col].is_monotonic_increasing:
                    sorted_data = data.sort_values(self.timestamp_col)
                else:
                    sorted_data = data
                
                # Convert to datetime if it's not already
                timestamps = pd.to_datetime(sorted_data[self.timestamp_col])
                
                # Calculate time differences
                time_diffs = timestamps.diff().dropna()
                
                # Find the most common time difference (should be the timeframe)
                timeframe = time_diffs.mode().iloc[0]
                
                # Identify gaps (time diff > 1.5x expected timeframe)
                gaps = time_diffs[time_diffs > timeframe * 1.5]
                
                if not gaps.empty:
                    gap_indices = gaps.index.tolist()
                    integrity_issues.append(f"timestamp gaps ({len(gaps)} found)")
                    affected_rows.extend(gap_indices)
                    
                    gap_details = {
                        'gap_count': len(gaps),
                        'timeframe': timeframe.total_seconds(),
                        'gap_positions': [int(i) for i in gap_indices],
                        'gap_sizes': [g.total_seconds() for g in gaps]
                    }
                    
            except Exception as e:
                self.logger.warning(f"Error checking timestamp gaps: {str(e)}")
        
        # Remove duplicates from affected_rows
        affected_rows = sorted(set(affected_rows))
        
        # Create validation result
        if integrity_issues:
            return ValidationResult(
                is_valid=False,
                message=f"OHLCV integrity issues found: {', '.join(integrity_issues)}",
                severity=self.severity,
                affected_rows=affected_rows,
                affected_columns=[self.open_col, self.high_col, self.low_col, 
                                  self.close_col, self.timestamp_col],
                details={'issues': integrity_issues, 'gaps': gap_details}
            )
        else:
            return ValidationResult(
                is_valid=True,
                message="OHLCV data is valid",
                severity=ValidationSeverity.INFO,
                details={'gaps': gap_details}
            )


class TickDataValidator(DataValidator):
    """Validates tick data for Forex trading"""
    
    def __init__(
        self,
        bid_col: str = 'bid',
        ask_col: str = 'ask',
        timestamp_col: str = 'timestamp',
        max_spread_ratio: float = 0.01,  # 1% of price as max spread
        check_sequence: bool = True,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        """
        Initialize tick data validator
        
        Args:
            bid_col: Column name for bid price
            ask_col: Column name for ask price
            timestamp_col: Column name for timestamp
            max_spread_ratio: Maximum allowed spread as fraction of price
            check_sequence: Whether to check timestamp sequence
            severity: Severity level for issues
        """
        super().__init__(
            name="tick_data_validator",
            description="Validates Forex tick data integrity",
            severity=severity
        )
        self.bid_col = bid_col
        self.ask_col = ask_col
        self.timestamp_col = timestamp_col
        self.max_spread_ratio = max_spread_ratio
        self.check_sequence = check_sequence
        
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate tick data structure and integrity"""
        # Check required columns exist
        required_cols = [self.bid_col, self.ask_col, self.timestamp_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            return ValidationResult(
                is_valid=False,
                message=f"Missing required columns: {', '.join(missing_cols)}",
                severity=self.severity,
                affected_columns=missing_cols
            )
            
        # Check bid-ask integrity (ask >= bid)
        integrity_issues = []
        affected_rows = []
        
        bid_ask_mask = data[self.ask_col] < data[self.bid_col]
        if bid_ask_mask.any():
            integrity_issues.append("ask < bid")
            affected_rows.extend(data[bid_ask_mask].index.tolist())
            
        # Check for excessive spreads
        if self.max_spread_ratio > 0:
            spread = data[self.ask_col] - data[self.bid_col]
            mid_price = (data[self.ask_col] + data[self.bid_col]) / 2
            spread_ratio = spread / mid_price
            
            excessive_spread_mask = spread_ratio > self.max_spread_ratio
            if excessive_spread_mask.any():
                integrity_issues.append(f"excessive spread (>{self.max_spread_ratio*100}%)")
                affected_rows.extend(data[excessive_spread_mask].index.tolist())
            
        # Check for timestamp sequence issues if requested
        if self.check_sequence and len(data) > 1:
            # Check if timestamps are in ascending order
            if not pd.to_datetime(data[self.timestamp_col]).is_monotonic_increasing:
                integrity_issues.append("timestamp sequence not monotonically increasing")
                
                # Find rows with timestamp sequence issues
                timestamps = pd.to_datetime(data[self.timestamp_col])
                prev_ts = timestamps.iloc[0]
                
                for i, ts in enumerate(timestamps.iloc[1:], 1):
                    if ts < prev_ts:
                        affected_rows.append(data.index[i])
                    prev_ts = ts
        
        # Remove duplicates from affected_rows
        affected_rows = sorted(set(affected_rows))
        
        # Create validation result
        if integrity_issues:
            return ValidationResult(
                is_valid=False,
                message=f"Tick data integrity issues found: {', '.join(integrity_issues)}",
                severity=self.severity,
                affected_rows=affected_rows,
                affected_columns=[self.bid_col, self.ask_col, self.timestamp_col],
                details={'issues': integrity_issues}
            )
        else:
            return ValidationResult(
                is_valid=True,
                message="Tick data is valid",
                severity=ValidationSeverity.INFO
            )


class ValidationStrategy(ABC):
    """Abstract base class for all validation strategies."""

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Perform validation on the given data.

        Args:
            data: The pandas DataFrame to validate.

        Returns:
            A ValidationResult object indicating the outcome.
        """
        pass


class DataValidationEngine:
    """
    Orchestrates the data validation process using multiple strategies.
    """
    def __init__(self, strategies: Optional[List[ValidationStrategy]] = None):
        self.strategies = strategies or []
        self.logger = logging.getLogger("DataValidationEngine")

    def add_strategy(self, strategy: ValidationStrategy):
        """Add a validation strategy to the engine."""
        self.strategies.append(strategy)

    def validate_data(self, data: pd.DataFrame) -> List[ValidationResult]:
        """
        Run all registered validation strategies on the data.

        Args:
            data: The pandas DataFrame to validate.

        Returns:
            A list of ValidationResult objects from each strategy.
        """
        results = []
        if not isinstance(data, pd.DataFrame):
             results.append(ValidationResult(
                 is_valid=False,
                 message="Input data is not a pandas DataFrame",
                 severity=ValidationSeverity.CRITICAL
             ))
             return results

        for strategy in self.strategies:
            try:
                result = strategy.validate(data)
                results.append(result)
            except DataValidationError as e:  # Catch specific validation errors
                self.logger.warning(f"Validation strategy {type(strategy).__name__} failed: {e}")
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Strategy {type(strategy).__name__} execution error: {e}",
                    severity=ValidationSeverity.CRITICAL
                ))
            except Exception as e:
                self.logger.error(f"Validation strategy {type(strategy).__name__} failed: {e}", exc_info=True)
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Strategy {type(strategy).__name__} execution error: {e}",
                    severity=ValidationSeverity.CRITICAL
                ))
        return results

    def overall_validity(self, results: List[ValidationResult]) -> bool:
        """
        Determine the overall validity based on results, considering severity.

        Args:
            results: A list of ValidationResult objects.

        Returns:
            True if the data is considered valid overall (no ERROR or CRITICAL issues),
            False otherwise.
        """
        return not any(r.severity >= ValidationSeverity.ERROR for r in results if not r.is_valid)
