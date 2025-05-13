"""
OHLCV specific validation components.

Provides validation strategies specifically designed for OHLCV (Open-High-Low-Close-Volume) data.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from common_lib.exceptions import DataValidationError
from common_lib.schemas import OHLCVData
from common_lib.utils.date_utils import ensure_timezone
from core.validation_engine import ValidationStrategy, ValidationResult, ValidationSeverity
from datetime import datetime, timedelta, timezone
from typing import Optional
from ..models.schemas import TimeframeEnum
from ..exceptions.validation_exceptions import ValidationError


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def validate_instrument_format(instrument: str) ->None:
    """
    Validate the format of an instrument identifier.
    
    Args:
        instrument: The instrument identifier to validate (e.g., EUR_USD)
        
    Raises:
        ValidationError: If the instrument format is invalid
    """
    valid_formats = ['^[A-Z]{3}_[A-Z]{3}$', '^[A-Z]{1,5}$',
        '^[A-Z]{3,6}\\d{2,3}$', '^[A-Z]{3,4}_[A-Z]{3}$']
    import re
    for pattern in valid_formats:
        if re.match(pattern, instrument):
            return
    raise ValidationError(
        f'Invalid instrument format: {instrument}. Expected format examples: EUR_USD, AAPL, SPX500, BTC_USD'
        )


def validate_timeframe(timeframe: TimeframeEnum) ->None:
    """
    Validate that the timeframe is supported.
    
    Args:
        timeframe: The timeframe to validate
        
    Raises:
        ValidationError: If the timeframe is not supported
    """
    valid_timeframes = [t.value for t in TimeframeEnum]
    if timeframe.value not in valid_timeframes:
        raise ValidationError(
            f"Invalid timeframe: {timeframe}. Supported timeframes: {', '.join(valid_timeframes)}"
            )


def validate_date_range(start_time: datetime, end_time: datetime,
    max_range_days: Optional[int]=365) ->None:
    """
    Validate that the date range is valid and within allowed limits.
    
    Args:
        start_time: The start time of the range
        end_time: The end time of the range
        max_range_days: Maximum allowed range in days (None for no limit)
        
    Raises:
        ValidationError: If the date range is invalid
    """
    if start_time >= end_time:
        raise ValidationError('Start time must be before end time')
    now = datetime.utcnow()
    if end_time > now:
        raise ValidationError('End time cannot be in the future')
    if max_range_days is not None:
        max_timedelta = timedelta(days=max_range_days)
        if end_time - start_time > max_timedelta:
            raise ValidationError(
                f'Date range exceeds maximum allowed ({max_range_days} days)')


def is_valid_timeframe_conversion(source_timeframe: TimeframeEnum,
    target_timeframe: TimeframeEnum) ->bool:
    """
    Check if converting from source to target timeframe is valid.
    Target timeframe must be larger than source timeframe.
    
    Args:
        source_timeframe: The source timeframe
        target_timeframe: The target timeframe
        
    Returns:
        bool: True if conversion is valid, False otherwise
    """
    timeframe_to_minutes = {TimeframeEnum.ONE_MINUTE: 1, TimeframeEnum.
        FIVE_MINUTES: 5, TimeframeEnum.FIFTEEN_MINUTES: 15, TimeframeEnum.
        THIRTY_MINUTES: 30, TimeframeEnum.ONE_HOUR: 60, TimeframeEnum.
        FOUR_HOURS: 240, TimeframeEnum.ONE_DAY: 1440, TimeframeEnum.
        ONE_WEEK: 10080}
    source_minutes = timeframe_to_minutes[source_timeframe]
    target_minutes = timeframe_to_minutes[target_timeframe]
    return (target_minutes > source_minutes and target_minutes %
        source_minutes == 0)


class CandlestickPatternValidator(ValidationStrategy):
    """
    Validates if OHLCV data forms valid candlestick patterns.
    
    This validator ensures that candlestick data follows expected patterns,
    such as high >= open, high >= close, low <= open, low <= close.
    """

    def __init__(self, severity: ValidationSeverity=ValidationSeverity.ERROR):
        """
        Initialize CandlestickPatternValidator.
        
        Args:
            severity: Severity level for candlestick validation issues
        """
        self.severity = severity

    def validate(self, data: pd.DataFrame) ->ValidationResult:
        """Validate candlestick pattern integrity."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data is not a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in
            data.columns]
        if missing_columns:
            return ValidationResult(is_valid=False, message=
                f'Missing required columns: {missing_columns}', details={
                'missing_columns': missing_columns}, severity=
                ValidationSeverity.ERROR)
        issues = []
        total_rows = len(data)
        high_open_invalid = data[data['high'] < data['open']]
        if not high_open_invalid.empty:
            issues.append({'issue': 'high_less_than_open', 'count': len(
                high_open_invalid), 'percentage': 100 * len(
                high_open_invalid) / total_rows})
        high_close_invalid = data[data['high'] < data['close']]
        if not high_close_invalid.empty:
            issues.append({'issue': 'high_less_than_close', 'count': len(
                high_close_invalid), 'percentage': 100 * len(
                high_close_invalid) / total_rows})
        low_open_invalid = data[data['low'] > data['open']]
        if not low_open_invalid.empty:
            issues.append({'issue': 'low_greater_than_open', 'count': len(
                low_open_invalid), 'percentage': 100 * len(low_open_invalid
                ) / total_rows})
        low_close_invalid = data[data['low'] > data['close']]
        if not low_close_invalid.empty:
            issues.append({'issue': 'low_greater_than_close', 'count': len(
                low_close_invalid), 'percentage': 100 * len(
                low_close_invalid) / total_rows})
        if issues:
            return ValidationResult(is_valid=False, message=
                'Candlestick pattern integrity issues detected', details={
                'issues': issues}, severity=self.severity)
        return ValidationResult(is_valid=True, message=
            'Candlestick pattern validation passed')


class VolumeChangeValidator(ValidationStrategy):
    """
    Validates if volume changes are within acceptable ranges.
    
    This validator detects suspicious volume patterns, such as large spikes or drops.
    """

    def __init__(self, max_relative_change: float=10.0, window_size: int=20,
        min_volume_for_check: float=0.0, severity: ValidationSeverity=
        ValidationSeverity.WARNING):
        """
        Initialize VolumeChangeValidator.
        
        Args:
            max_relative_change: Maximum allowed change relative to moving average (e.g., 10.0 = 1000%)
            window_size: Window size for calculating moving average
            min_volume_for_check: Only check volumes above this threshold
            severity: Severity level for volume change validation issues
        """
        self.max_relative_change = max_relative_change
        self.window_size = window_size
        self.min_volume_for_check = min_volume_for_check
        self.severity = severity

    def validate(self, data: pd.DataFrame) ->ValidationResult:
        """Validate volume changes."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data is not a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        if 'volume' not in data.columns:
            return ValidationResult(is_valid=False, message=
                "Missing 'volume' column", severity=ValidationSeverity.ERROR)
        if len(data) <= self.window_size:
            return ValidationResult(is_valid=True, message=
                'Not enough data points for volume change analysis')
        if self.min_volume_for_check > 0:
            data_filtered = data[data['volume'] >= self.min_volume_for_check
                ].copy()
            if len(data_filtered) == 0:
                return ValidationResult(is_valid=True, message=
                    'All volumes below minimum threshold for checking')
        else:
            data_filtered = data.copy()
        if 'instrument' in data_filtered.columns:
            grouped = data_filtered.groupby('instrument')
        else:
            data_filtered['_group'] = 'all'
            grouped = data_filtered.groupby('_group')
        issues = []
        for group_name, group_data in grouped:
            if 'timestamp' in group_data.columns:
                group_data = group_data.sort_values('timestamp')
            volume = group_data['volume']
            volume_ma = volume.rolling(window=self.window_size, min_periods=1
                ).mean()
            relative_change = (volume - volume_ma) / (volume_ma + 1e-10)
            excessive_change = group_data[abs(relative_change) > self.
                max_relative_change].copy()
            if not excessive_change.empty:
                top_changes = excessive_change.nlargest(10, 'volume')
                change_details = []
                for _, row in top_changes.iterrows():
                    change_details.append({'volume': float(row['volume']),
                        'relative_change': float(relative_change.loc[_]),
                        'timestamp': str(row['timestamp']) if 'timestamp' in
                        row else None})
                issues.append({'instrument': str(group_name), 'count': len(
                    excessive_change), 'max_change': float(relative_change.
                    abs().max()), 'examples': change_details})
        if issues:
            return ValidationResult(is_valid=True, message=
                'Suspicious volume changes detected', details={'issues':
                issues}, severity=self.severity)
        return ValidationResult(is_valid=True, message=
            'Volume change validation passed')


class GapDetectionValidator(ValidationStrategy):
    """
    Validates OHLCV data for unexpected time gaps.

    Checks if the time difference between consecutive candles matches the
    expected timeframe, identifying missing candles or irregular intervals.
    """

    def __init__(self, timeframe: TimeframeEnum, timestamp_column: str=
        'timestamp', allowed_gap_multiplier: float=1.5, severity:
        ValidationSeverity=ValidationSeverity.WARNING):
        """
        Initialize GapDetectionValidator.

        Args:
            timeframe: The expected timeframe of the OHLCV data.
            timestamp_column: Name of the timestamp column.
            allowed_gap_multiplier: Multiplier for the timeframe to define an acceptable gap.
            severity: Severity level for gap detection issues.
        """
        self.timeframe = timeframe
        self.timestamp_column = timestamp_column
        self.allowed_gap_multiplier = allowed_gap_multiplier
        self.severity = severity
        timeframe_map = {TimeframeEnum.ONE_MINUTE: timedelta(minutes=1),
            TimeframeEnum.FIVE_MINUTES: timedelta(minutes=5), TimeframeEnum
            .FIFTEEN_MINUTES: timedelta(minutes=15), TimeframeEnum.
            THIRTY_MINUTES: timedelta(minutes=30), TimeframeEnum.ONE_HOUR:
            timedelta(hours=1), TimeframeEnum.FOUR_HOURS: timedelta(hours=4
            ), TimeframeEnum.ONE_DAY: timedelta(days=1), TimeframeEnum.
            ONE_WEEK: timedelta(weeks=1)}
        self.expected_interval = timeframe_map.get(timeframe)
        if not self.expected_interval:
            raise ValueError(
                f'Unsupported timeframe for gap detection: {timeframe}')

    @with_exception_handling
    def validate(self, data: pd.DataFrame) ->ValidationResult:
        """Validate OHLCV data for time gaps."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data is not a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        if self.timestamp_column not in data.columns:
            return ValidationResult(is_valid=False, message=
                f'Missing timestamp column: {self.timestamp_column}',
                severity=ValidationSeverity.ERROR)
        if len(data) < 2:
            return ValidationResult(is_valid=True, message=
                'Not enough data points for gap detection')
        try:
            if not pd.api.types.is_datetime64_any_dtype(data[self.
                timestamp_column]):
                data[self.timestamp_column] = pd.to_datetime(data[self.
                    timestamp_column])
            sorted_data = data.sort_values(self.timestamp_column).copy()
        except Exception as e:
            return ValidationResult(is_valid=False, message=
                f'Failed to process timestamp column: {str(e)}', severity=
                ValidationSeverity.ERROR)
        if 'instrument' in sorted_data.columns:
            groups = sorted_data.groupby('instrument')
        else:
            sorted_data['_group'] = 'all'
            groups = sorted_data.groupby('_group')
        issues = []
        max_allowed_gap = self.expected_interval * self.allowed_gap_multiplier
        for group_name, group_data in groups:
            if len(group_data) < 2:
                continue
            time_diffs = group_data[self.timestamp_column].diff().dropna()
            gaps = group_data.iloc[time_diffs[time_diffs > max_allowed_gap]
                .index]
            if not gaps.empty:
                gap_details = []
                for i, row in gaps.head(10).iterrows():
                    prev_idx = group_data.index.get_loc(i) - 1
                    if prev_idx >= 0:
                        prev_ts = group_data.iloc[prev_idx][self.
                            timestamp_column]
                        gap_size = row[self.timestamp_column] - prev_ts
                        gap_details.append({'timestamp': str(row[self.
                            timestamp_column]), 'previous_timestamp': str(
                            prev_ts), 'gap_seconds': gap_size.total_seconds()})
                issues.append({'instrument': str(group_name), 'count': len(
                    gaps), 'percentage': 100 * len(gaps) / len(group_data),
                    'expected_interval_seconds': self.expected_interval.
                    total_seconds(), 'max_allowed_gap_seconds':
                    max_allowed_gap.total_seconds(), 'examples': gap_details})
        if issues:
            return ValidationResult(is_valid=True, message=
                'Time gaps detected in OHLCV data', details={'issues':
                issues}, severity=self.severity)
        return ValidationResult(is_valid=True, message=
            'No significant time gaps detected')


class VolumeChangeValidator(ValidationStrategy):
    """
    Validates OHLCV volume data for unusual changes or patterns.

    Checks for zero volume candles (potentially indicating market inactivity
    or data issues) and extreme spikes or drops in volume.
    """

    def __init__(self, volume_column: str='volume', allow_zero_volume: bool
        =False, max_volume_spike_factor: Optional[float]=10.0,
        min_volume_drop_factor: Optional[float]=0.1, rolling_window: int=20,
        severity: ValidationSeverity=ValidationSeverity.WARNING):
        """
        Initialize VolumeChangeValidator.

        Args:
            volume_column: Name of the volume column.
            allow_zero_volume: Whether zero volume candles are considered valid.
            max_volume_spike_factor: Factor for detecting volume spikes vs rolling average.
            min_volume_drop_factor: Factor for detecting volume drops vs rolling average.
            rolling_window: Window size for calculating rolling average volume.
            severity: Severity level for volume change issues.
        """
        self.volume_column = volume_column
        self.allow_zero_volume = allow_zero_volume
        self.max_volume_spike_factor = max_volume_spike_factor
        self.min_volume_drop_factor = min_volume_drop_factor
        self.rolling_window = rolling_window
        self.severity = severity

    def validate(self, data: pd.DataFrame) ->ValidationResult:
        """Validate OHLCV volume changes."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data is not a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        if self.volume_column not in data.columns:
            return ValidationResult(is_valid=True, message=
                f"Volume column '{self.volume_column}' not found, skipping validation."
                )
        issues = []
        total_rows = len(data)
        if not self.allow_zero_volume:
            zero_volume = data[data[self.volume_column] <= 0]
            if not zero_volume.empty:
                issues.append({'issue': 'zero_or_negative_volume', 'count':
                    len(zero_volume), 'percentage': 100 * len(zero_volume) /
                    total_rows, 'examples': zero_volume.head().to_dict(
                    'records')})
        if (self.max_volume_spike_factor is not None or self.
            min_volume_drop_factor is not None) and len(data
            ) > self.rolling_window:
            if 'instrument' in data.columns:
                groups = data.groupby('instrument')
            else:
                data['_group'] = 'all'
                groups = data.groupby('_group')
            change_issues = []
            for group_name, group_data in groups:
                if len(group_data) <= self.rolling_window:
                    continue
                rolling_avg = group_data[self.volume_column].rolling(window
                    =self.rolling_window, min_periods=5).mean()
                rolling_avg = rolling_avg.replace(0, np.nan).fillna(method=
                    'ffill').fillna(method='bfill').fillna(1e-09)
                group_issues = {}
                if self.max_volume_spike_factor is not None:
                    spikes = group_data[group_data[self.volume_column] > 
                        rolling_avg * self.max_volume_spike_factor]
                    if not spikes.empty:
                        group_issues['spikes'] = {'count': len(spikes),
                            'percentage': 100 * len(spikes) / len(
                            group_data), 'examples': spikes.head().to_dict(
                            'records')}
                if self.min_volume_drop_factor is not None:
                    drops = group_data[group_data[self.volume_column] < 
                        rolling_avg * self.min_volume_drop_factor]
                    if not drops.empty:
                        group_issues['drops'] = {'count': len(drops),
                            'percentage': 100 * len(drops) / len(group_data
                            ), 'examples': drops.head().to_dict('records')}
                if group_issues:
                    change_issues.append({'instrument': str(group_name), **
                        group_issues})
            if change_issues:
                issues.append({'issue': 'volume_changes_detected',
                    'details': change_issues})
        if issues:
            return ValidationResult(is_valid=True, message=
                'OHLCV volume change issues detected', details={'issues':
                issues}, severity=self.severity)
        return ValidationResult(is_valid=True, message=
            'Volume change validation passed')

    @with_exception_handling
    def validate(self, data: pd.DataFrame) ->ValidationResult:
        """Validate volume consistency based on price range."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=False, message=
                'Data is not a pandas DataFrame', severity=
                ValidationSeverity.ERROR)
        try:
            df = pd.DataFrame([d.dict() for d in data])
            if df.empty:
                return
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(
                'UTC')
            df = df.sort_values('timestamp').set_index('timestamp')
            volatility = df['high'] - df['low']
            median_price = (df['high'] + df['low']) / 2
            median_price[median_price == 0] = 1e-09
            relative_volatility = volatility / median_price
            volume_volatility_ratio = df['volume'] / (relative_volatility +
                1e-09)
            q1 = volume_volatility_ratio.quantile(0.25)
            q3 = volume_volatility_ratio.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(volume_volatility_ratio < lower_bound) | (
                volume_volatility_ratio > upper_bound)]
            if not outliers.empty:
                error_details = outliers[['open', 'high', 'low', 'close',
                    'volume']].to_dict('records')
                raise DataValidationError(
                    f'{len(outliers)} records found with potentially inconsistent volume relative to price range.'
                    , validation_errors={'inconsistent_volume_records':
                    error_details}, validator=self.__class__.__name__)
        except DataValidationError:
            raise
        except Exception as e:
            self.logger.exception(
                f'Unexpected error during volume consistency validation: {e}')
            raise DataValidationError(
                f'Unexpected error in {self.__class__.__name__}: {e}',
                validator=self.__class__.__name__) from e
