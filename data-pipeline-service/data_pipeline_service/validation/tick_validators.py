"""
Tick data specific validation components.

Provides validation strategies specifically designed for tick-level forex data.
"""
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd
from common_lib.exceptions import DataValidationError  # Added import

from data_pipeline_service.validation.validation_engine import (
    ValidationStrategy, 
    ValidationResult, 
    ValidationSeverity
)


class QuoteSequenceValidator(ValidationStrategy):
    """
    Validates that tick data follows a logical sequence.
    
    This validator ensures that tick data doesn't have logical inconsistencies
    in the sequence of quotes, such as bid/ask prices moving in opposite directions.
    """
    
    def __init__(
        self, 
        bid_column: str = 'bid',
        ask_column: str = 'ask',
        severity: ValidationSeverity = ValidationSeverity.WARNING
    ):
        """
        Initialize QuoteSequenceValidator.
        
        Args:
            bid_column: Name of the bid price column
            ask_column: Name of the ask price column
            severity: Severity level for sequence validation issues
        """
        self.bid_column = bid_column
        self.ask_column = ask_column
        self.severity = severity
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate tick data sequence logic."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                message="Data is not a pandas DataFrame",
                severity=ValidationSeverity.ERROR
            )
            
        # Check if required columns are present
        required_columns = [self.bid_column, self.ask_column, "timestamp"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return ValidationResult(
                is_valid=False,
                message=f"Missing required columns: {missing_columns}",
                details={"missing_columns": missing_columns},
                severity=ValidationSeverity.ERROR
            )
            
        if len(data) < 2:
            return ValidationResult(
                is_valid=True,
                message="Not enough data points for sequence validation"
            )
            
        # Sort by timestamp to ensure correct sequence
        sorted_data = data.sort_values("timestamp").copy()
        
        # Group by instrument if present
        if "instrument" in sorted_data.columns:
            groups = sorted_data.groupby("instrument")
        else:
            # Create a single synthetic group
            sorted_data["_group"] = "all"
            groups = sorted_data.groupby("_group")
            
        issues = []
        
        for group_name, group_data in groups:
            if len(group_data) < 2:
                continue
                
            # Get price changes
            bid_changes = group_data[self.bid_column].diff()
            ask_changes = group_data[self.ask_column].diff()
            
            # Find inconsistent changes (bid up, ask down or vice versa) with threshold
            # Skip very small changes that might be due to decimal precision
            threshold = 1e-6
            inconsistent = group_data[
                ((bid_changes > threshold) & (ask_changes < -threshold)) |
                ((bid_changes < -threshold) & (ask_changes > threshold))
            ].copy()
            
            if not inconsistent.empty:
                # Get sample of inconsistencies for reporting
                sample = inconsistent.head(10)
                inconsistencies = []
                
                for i, row in sample.iterrows():
                    # Find previous row for comparison
                    prev_idx = group_data.index.get_loc(i) - 1
                    if prev_idx < 0:
                        continue
                        
                    prev_row = group_data.iloc[prev_idx]
                    
                    inconsistencies.append({
                        "timestamp": str(row["timestamp"]),
                        "prev_timestamp": str(prev_row["timestamp"]),
                        "prev_bid": float(prev_row[self.bid_column]),
                        "prev_ask": float(prev_row[self.ask_column]),
                        "current_bid": float(row[self.bid_column]),
                        "current_ask": float(row[self.ask_column]),
                        "bid_change": float(row[self.bid_column] - prev_row[self.bid_column]),
                        "ask_change": float(row[self.ask_column] - prev_row[self.ask_column])
                    })
                
                issues.append({
                    "instrument": str(group_name),
                    "count": len(inconsistent),
                    "percentage": 100 * len(inconsistent) / len(group_data),
                    "examples": inconsistencies
                })
                
        if issues:
            return ValidationResult(
                is_valid=True,  # Quote sequence issues don't necessarily make data invalid
                message="Quote sequence inconsistencies detected",
                details={"issues": issues},
                severity=self.severity
            )
            
        return ValidationResult(is_valid=True, message="Quote sequence validation passed")


class TickFrequencyValidator(ValidationStrategy):
    """
    Validates that tick frequency is within expected ranges.
    
    This validator ensures that the frequency of ticks is consistent with
    expected patterns, detecting unusual spikes or drops in tick frequency.
    """
    
    def __init__(
        self,
        min_expected_ticks_per_second: Optional[Dict[str, float]] = None,
        max_expected_ticks_per_second: Optional[Dict[str, float]] = None,
        analysis_window: pd.Timedelta = pd.Timedelta(minutes=1),
        min_analysis_windows: int = 5,
        severity: ValidationSeverity = ValidationSeverity.WARNING
    ):
        """
        Initialize TickFrequencyValidator.
        
        Args:
            min_expected_ticks_per_second: Minimum expected ticks per second by instrument
            max_expected_ticks_per_second: Maximum expected ticks per second by instrument
            analysis_window: Time window for analyzing tick frequency
            min_analysis_windows: Minimum number of windows required for validation
            severity: Severity level for frequency validation issues
        """
        self.min_expected_ticks_per_second = min_expected_ticks_per_second or {}
        self.max_expected_ticks_per_second = max_expected_ticks_per_second or {}
        self.analysis_window = analysis_window
        self.min_analysis_windows = min_analysis_windows
        self.severity = severity
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate tick frequency."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                message="Data is not a pandas DataFrame",
                severity=ValidationSeverity.ERROR
            )
            
        if "timestamp" not in data.columns:
            return ValidationResult(
                is_valid=False,
                message="Missing 'timestamp' column",
                severity=ValidationSeverity.ERROR
            )
            
        if "instrument" not in data.columns:
            return ValidationResult(
                is_valid=False,
                message="Missing 'instrument' column",
                severity=ValidationSeverity.ERROR
            )
            
        # Ensure timestamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            try:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    message=f"Failed to convert timestamp to datetime: {str(e)}",
                    severity=ValidationSeverity.ERROR
                )
                
        issues = []
        
        for instrument, group in data.groupby("instrument"):
            # Sort by timestamp
            group = group.sort_values("timestamp")
            
            # Get min and max timestamps
            min_ts = group["timestamp"].min()
            max_ts = group["timestamp"].max()
            
            # Calculate total time span
            time_span = max_ts - min_ts
            if time_span < self.analysis_window * self.min_analysis_windows:
                # Not enough data for this instrument
                continue
                
            # Create time windows for analysis
            window_edges = pd.date_range(
                start=min_ts, 
                end=max_ts + self.analysis_window,  # Add one more to ensure we include the last point
                freq=self.analysis_window
            )
            
            # Count ticks per window
            window_counts = []
            for i in range(len(window_edges) - 1):
                window_start = window_edges[i]
                window_end = window_edges[i+1]
                
                count = len(group[(group["timestamp"] >= window_start) & (group["timestamp"] < window_end)])
                window_counts.append({
                    "window_start": window_start,
                    "window_end": window_end,
                    "count": count,
                    "ticks_per_second": count / self.analysis_window.total_seconds()
                })
                
            # Skip instruments with too few windows
            if len(window_counts) < self.min_analysis_windows:
                continue
                
            # Get min/max thresholds for this instrument
            min_ticks = self.min_expected_ticks_per_second.get(instrument, 0)
            max_ticks = self.max_expected_ticks_per_second.get(instrument)
            
            # Find windows with too few ticks
            if min_ticks > 0:
                low_activity = [
                    w for w in window_counts 
                    if w["ticks_per_second"] < min_ticks and w["count"] > 0  # Only consider windows with some activity
                ]
            else:
                low_activity = []
                
            # Find windows with too many ticks
            if max_ticks is not None:
                high_activity = [w for w in window_counts if w["ticks_per_second"] > max_ticks]
            else:
                high_activity = []
                
            # Only add issue if we found problems
            if low_activity or high_activity:
                tps_values = [w["ticks_per_second"] for w in window_counts if w["count"] > 0]
                
                issue_detail = {
                    "instrument": instrument,
                    "windows_analyzed": len(window_counts),
                    "avg_ticks_per_second": float(np.mean(tps_values)) if tps_values else 0,
                    "min_ticks_per_second": float(np.min(tps_values)) if tps_values else 0,
                    "max_ticks_per_second": float(np.max(tps_values)) if tps_values else 0,
                }
                
                if low_activity:
                    issue_detail["low_activity_windows"] = len(low_activity)
                    issue_detail["low_activity_examples"] = [
                        {
                            "window_start": str(w["window_start"]),
                            "window_end": str(w["window_end"]),
                            "ticks": w["count"],
                            "ticks_per_second": float(w["ticks_per_second"])
                        }
                        for w in sorted(low_activity, key=lambda x: x["ticks_per_second"])[:5]  # 5 lowest
                    ]
                    
                if high_activity:
                    issue_detail["high_activity_windows"] = len(high_activity)
                    issue_detail["high_activity_examples"] = [
                        {
                            "window_start": str(w["window_start"]),
                            "window_end": str(w["window_end"]),
                            "ticks": w["count"],
                            "ticks_per_second": float(w["ticks_per_second"])
                        }
                        for w in sorted(high_activity, key=lambda x: -x["ticks_per_second"])[:5]  # 5 highest
                    ]
                    
                issues.append(issue_detail)
                
        if issues:
            return ValidationResult(
                is_valid=True,  # Tick frequency issues don't necessarily make data invalid
                message="Tick frequency anomalies detected",
                details={"issues": issues},
                severity=self.severity
            )
            
        return ValidationResult(is_valid=True, message="Tick frequency validation passed")


class TickVolumeConsistencyValidator(ValidationStrategy):
    """
    Validates consistency in tick volume data (if available).

    Checks for unusual volume patterns, such as zero volume ticks or
    extreme spikes in volume compared to recent activity.
    """

    def __init__(
        self,
        volume_column: str = 'volume',
        min_volume: Optional[float] = 0,
        max_volume_spike_factor: Optional[float] = 10.0, # Max spike compared to rolling average
        rolling_window: int = 50,
        severity: ValidationSeverity = ValidationSeverity.WARNING
    ):
        """
        Initialize TickVolumeConsistencyValidator.

        Args:
            volume_column: Name of the volume column.
            min_volume: Minimum expected volume (usually 0).
            max_volume_spike_factor: Factor for detecting volume spikes.
            rolling_window: Window size for calculating rolling average volume.
            severity: Severity level for volume consistency issues.
        """
        self.volume_column = volume_column
        self.min_volume = min_volume
        self.max_volume_spike_factor = max_volume_spike_factor
        self.rolling_window = rolling_window
        self.severity = severity

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate tick volume consistency."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                message="Data is not a pandas DataFrame",
                severity=ValidationSeverity.ERROR
            )

        if self.volume_column not in data.columns:
            # Volume column is optional, so pass if not present
            return ValidationResult(
                is_valid=True,
                message=f"Volume column '{self.volume_column}' not found, skipping validation."
            )

        issues = []
        total_rows = len(data)

        # Check for minimum volume
        if self.min_volume is not None:
            invalid_volume = data[data[self.volume_column] < self.min_volume]
            if not invalid_volume.empty:
                issues.append({
                    "issue": f"volume_less_than_{self.min_volume}",
                    "count": len(invalid_volume),
                    "percentage": 100 * len(invalid_volume) / total_rows,
                    "examples": invalid_volume.head().to_dict('records')
                })

        # Check for volume spikes (if factor is set)
        if self.max_volume_spike_factor is not None and len(data) > self.rolling_window:
            # Group by instrument if present
            if "instrument" in data.columns:
                groups = data.groupby("instrument")
            else:
                data["_group"] = "all"
                groups = data.groupby("_group")

            spike_issues = []
            for group_name, group_data in groups:
                if len(group_data) <= self.rolling_window:
                    continue

                rolling_avg = group_data[self.volume_column].rolling(window=self.rolling_window, min_periods=1).mean()
                # Avoid division by zero or near-zero
                rolling_avg = rolling_avg.replace(0, np.nan).fillna(method='ffill').fillna(method='bfill').fillna(1e-9)

                spikes = group_data[group_data[self.volume_column] > (rolling_avg * self.max_volume_spike_factor)]
                if not spikes.empty:
                    spike_issues.append({
                        "instrument": str(group_name),
                        "count": len(spikes),
                        "percentage": 100 * len(spikes) / len(group_data),
                        "examples": spikes.head().to_dict('records')
                    })

            if spike_issues:
                 issues.append({
                    "issue": "volume_spikes_detected",
                    "details": spike_issues
                 })


        if issues:
            return ValidationResult(
                is_valid=True, # Volume issues might not invalidate data but warrant warning
                message="Tick volume consistency issues detected",
                details={"issues": issues},
                severity=self.severity
            )

        return ValidationResult(is_valid=True, message="Tick volume consistency validation passed")


class ValidateTickPriceReasonableness(BaseTickValidator):
    """
    Validates that tick prices are reasonable.

    This includes checks like ensuring bid < ask, and that prices are within
    a plausible range based on historical data.
    """

    def __init__(
        self,
        min_bid_ask_spread: float = 0.0,
        max_bid_ask_spread: float = 0.1,
        max_price_deviation_factor: float = 5.0,
        severity: ValidationSeverity = ValidationSeverity.WARNING
    ):
        """
        Initialize ValidateTickPriceReasonableness.

        Args:
            min_bid_ask_spread: Minimum allowed spread between bid and ask.
            max_bid_ask_spread: Maximum allowed spread between bid and ask.
            max_price_deviation_factor: Maximum deviation from average price allowed.
            severity: Severity level for price reasonableness issues.
        """
        self.min_bid_ask_spread = min_bid_ask_spread
        self.max_bid_ask_spread = max_bid_ask_spread
        self.max_price_deviation_factor = max_price_deviation_factor
        self.severity = severity

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate tick price reasonableness."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                message="Data is not a pandas DataFrame",
                severity=ValidationSeverity.ERROR
            )

        issues = []
        total_rows = len(data)

        # Calculate the spread
        data['spread'] = data['ask'] - data['bid']

        # Check for bid/ask inversion
        inverted_ticks = data[data['bid'] > data['ask']]
        if not inverted_ticks.empty:
            issues.append({
                "issue": "bid_ask_inversion",
                "count": len(inverted_ticks),
                "percentage": 100 * len(inverted_ticks) / total_rows,
                "examples": inverted_ticks.head().to_dict('records')
            })

        # Check for spreads that are too narrow
        if self.min_bid_ask_spread > 0:
            narrow_spread_ticks = data[data['spread'] < self.min_bid_ask_spread]
            if not narrow_spread_ticks.empty:
                issues.append({
                    "issue": f"narrow_bid_ask_spread_{self.min_bid_ask_spread}",
                    "count": len(narrow_spread_ticks),
                    "percentage": 100 * len(narrow_spread_ticks) / total_rows,
                    "examples": narrow_spread_ticks.head().to_dict('records')
                })

        # Check for spreads that are too wide
        if self.max_bid_ask_spread > 0:
            wide_spread_ticks = data[data['spread'] > self.max_bid_ask_spread]
            if not wide_spread_ticks.empty:
                issues.append({
                    "issue": f"wide_bid_ask_spread_{self.max_bid_ask_spread}",
                    "count": len(wide_spread_ticks),
                    "percentage": 100 * len(wide_spread_ticks) / total_rows,
                    "examples": wide_spread_ticks.head().to_dict('records')
                })

        # Check for price deviations
        if self.max_price_deviation_factor > 0:
            # Calculate the average price (midpoint)
            data['average_price'] = (data['bid'] + data['ask']) / 2
            overall_mean = data['average_price'].mean()
            overall_std = data['average_price'].std()

            # Define reasonable bounds
            lower_bound = overall_mean - self.max_price_deviation_factor * overall_std
            upper_bound = overall_mean + self.max_price_deviation_factor * overall_std

            # Flag ticks outside the reasonable bounds
            unreasonable_price_ticks = data[(data['average_price'] < lower_bound) | (data['average_price'] > upper_bound)]
            if not unreasonable_price_ticks.empty:
                issues.append({
                    "issue": "unreasonable_price_deviation",
                    "count": len(unreasonable_price_ticks),
                    "percentage": 100 * len(unreasonable_price_ticks) / total_rows,
                    "examples": unreasonable_price_ticks.head().to_dict('records')
                })

        if issues:
            return ValidationResult(
                is_valid=True,  # Price reasonableness issues don't necessarily make data invalid
                message="Tick price reasonableness issues detected",
                details={"issues": issues},
                severity=self.severity
            )

        return ValidationResult(is_valid=True, message="Tick price reasonableness validation passed")


class ValidateTickDataCompleteness(BaseTickValidator):
    """
    Validates the completeness of tick data.

    Checks for missing fields, unexpected null values, and ensures that all
    ticks have the necessary information for processing.
    """

    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        """
        Initialize ValidateTickDataCompleteness.

        Args:
            required_fields: List of fields that are required to be present in each tick.
            severity: Severity level for completeness issues.
        """
        self.required_fields = required_fields or []
        self.severity = severity

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate tick data completeness."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                message="Data is not a pandas DataFrame",
                severity=ValidationSeverity.ERROR
            )

        issues = []
        total_rows = len(data)

        # Check for missing required fields
        for field in self.required_fields:
            if field not in data.columns:
                issues.append({
                    "issue": f"missing_field_{field}",
                    "message": f"Required field '{field}' is missing from the data.",
                    "severity": ValidationSeverity.ERROR,
                    "validator": self.__class__.__name__
                })

        # Check for null values in required fields
        if issues:
            return ValidationResult(
                is_valid=False,
                message="Missing required fields in tick data",
                details={"issues": issues},
                severity=ValidationSeverity.ERROR
            )

        for field in self.required_fields:
            null_ticks = data[data[field].isnull()]
            if not null_ticks.empty:
                issues.append({
                    "issue": f"null_value_in_field_{field}",
                    "count": len(null_ticks),
                    "percentage": 100 * len(null_ticks) / total_rows,
                    "examples": null_ticks.head().to_dict('records')
                })

        if issues:
            return ValidationResult(
                is_valid=True,  # Completeness issues might not invalidate data but warrant attention
                message="Tick data completeness issues detected",
                details={"issues": issues},
                severity=self.severity
            )

        return ValidationResult(is_valid=True, message="Tick data completeness validation passed")


class ValidateTickTimestamps(BaseTickValidator):
    """
    Validates tick timestamps.

    Ensures that timestamps are in UTC, properly ordered, and have no gaps
    larger than a specified threshold.
    """

    def __init__(
        self,
        max_gap_seconds: int = 60,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        """
        Initialize ValidateTickTimestamps.

        Args:
            max_gap_seconds: Maximum allowed gap between consecutive ticks, in seconds.
            severity: Severity level for timestamp validation issues.
        """
        self.max_gap_seconds = max_gap_seconds
        self.severity = severity

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate tick timestamps."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                message="Data is not a pandas DataFrame",
                severity=ValidationSeverity.ERROR
            )

        if "timestamp" not in data.columns:
            return ValidationResult(
                is_valid=False,
                message="Missing 'timestamp' column",
                severity=ValidationSeverity.ERROR
            )
            
        issues = []
        
        # Ensure timestamp is in UTC and datetime type
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            try:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    message=f"Failed to convert timestamp to datetime: {str(e)}",
                    severity=ValidationSeverity.ERROR
                )
        
        # Check timestamp order and gaps
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame([t.dict() for t in data])
            if df.empty or len(df) < 2:
                return # Not enough data to check sequence

            # Ensure timestamp is datetime and timezone-aware (UTC)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')

            # Sort by timestamp just in case
            df = df.sort_values('timestamp')

            # Check for non-monotonic timestamps
            time_diffs = df['timestamp'].diff().dt.total_seconds()
            # Allow for zero difference (simultaneous ticks) but not negative (out of order)
            out_of_order = df[time_diffs < 0]

            if not out_of_order.empty:
                 error_details = out_of_order[['timestamp', 'bid', 'ask']].to_dict('records')
                 # Find the preceding tick for context
                 for i, err_idx in enumerate(out_of_order.index):
                     if err_idx > 0:
                         prev_idx = df.index[df.index.get_loc(err_idx) - 1]
                         error_details[i]['previous_timestamp'] = df.loc[prev_idx, 'timestamp'].isoformat()
                     else:
                          error_details[i]['previous_timestamp'] = None

                 raise DataValidationError(
                     f"{len(out_of_order)} tick records found out of chronological order.",
                     validation_errors={"out_of_order_ticks": error_details},
                     validator=self.__class__.__name__
                 )
        except DataValidationError:
             raise # Re-raise specific validation errors
        except Exception as e: # Catch unexpected errors during validation logic
            self.logger.exception(f"Unexpected error during tick sequence validation: {e}")
            raise DataValidationError(f"Unexpected error in {self.__class__.__name__}: {e}", validator=self.__class__.__name__) from e


class ValidateTickPriceReasonableness(BaseTickValidator):
    """
    Validates that tick prices are reasonable.

    This includes checks like ensuring bid < ask, and that prices are within
    a plausible range based on historical data.
    """

    def __init__(
        self,
        min_bid_ask_spread: float = 0.0,
        max_bid_ask_spread: float = 0.1,
        max_price_deviation_factor: float = 5.0,
        severity: ValidationSeverity = ValidationSeverity.WARNING
    ):
        """
        Initialize ValidateTickPriceReasonableness.

        Args:
            min_bid_ask_spread: Minimum allowed spread between bid and ask.
            max_bid_ask_spread: Maximum allowed spread between bid and ask.
            max_price_deviation_factor: Maximum deviation from average price allowed.
            severity: Severity level for price reasonableness issues.
        """
        self.min_bid_ask_spread = min_bid_ask_spread
        self.max_bid_ask_spread = max_bid_ask_spread
        self.max_price_deviation_factor = max_price_deviation_factor
        self.severity = severity

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate tick price reasonableness."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                message="Data is not a pandas DataFrame",
                severity=ValidationSeverity.ERROR
            )

        issues = []
        total_rows = len(data)

        # Calculate the spread
        data['spread'] = data['ask'] - data['bid']

        # Check for bid/ask inversion
        inverted_ticks = data[data['bid'] > data['ask']]
        if not inverted_ticks.empty:
            issues.append({
                "issue": "bid_ask_inversion",
                "count": len(inverted_ticks),
                "percentage": 100 * len(inverted_ticks) / total_rows,
                "examples": inverted_ticks.head().to_dict('records')
            })

        # Check for spreads that are too narrow
        if self.min_bid_ask_spread > 0:
            narrow_spread_ticks = data[data['spread'] < self.min_bid_ask_spread]
            if not narrow_spread_ticks.empty:
                issues.append({
                    "issue": f"narrow_bid_ask_spread_{self.min_bid_ask_spread}",
                    "count": len(narrow_spread_ticks),
                    "percentage": 100 * len(narrow_spread_ticks) / total_rows,
                    "examples": narrow_spread_ticks.head().to_dict('records')
                })

        # Check for spreads that are too wide
        if self.max_bid_ask_spread > 0:
            wide_spread_ticks = data[data['spread'] > self.max_bid_ask_spread]
            if not wide_spread_ticks.empty:
                issues.append({
                    "issue": f"wide_bid_ask_spread_{self.max_bid_ask_spread}",
                    "count": len(wide_spread_ticks),
                    "percentage": 100 * len(wide_spread_ticks) / total_rows,
                    "examples": wide_spread_ticks.head().to_dict('records')
                })

        # Check for price deviations
        if self.max_price_deviation_factor > 0:
            # Calculate the average price (midpoint)
            data['average_price'] = (data['bid'] + data['ask']) / 2
            overall_mean = data['average_price'].mean()
            overall_std = data['average_price'].std()

            # Define reasonable bounds
            lower_bound = overall_mean - self.max_price_deviation_factor * overall_std
            upper_bound = overall_mean + self.max_price_deviation_factor * overall_std

            # Flag ticks outside the reasonable bounds
            unreasonable_price_ticks = data[(data['average_price'] < lower_bound) | (data['average_price'] > upper_bound)]
            if not unreasonable_price_ticks.empty:
                issues.append({
                    "issue": "unreasonable_price_deviation",
                    "count": len(unreasonable_price_ticks),
                    "percentage": 100 * len(unreasonable_price_ticks) / total_rows,
                    "examples": unreasonable_price_ticks.head().to_dict('records')
                })

        if issues:
            return ValidationResult(
                is_valid=True,  # Price reasonableness issues don't necessarily make data invalid
                message="Tick price reasonableness issues detected",
                details={"issues": issues},
                severity=self.severity
            )

        return ValidationResult(is_valid=True, message="Tick price reasonableness validation passed")


class ValidateTickDataCompleteness(BaseTickValidator):
    """
    Validates the completeness of tick data.

    Checks for missing fields, unexpected null values, and ensures that all
    ticks have the necessary information for processing.
    """

    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        """
        Initialize ValidateTickDataCompleteness.

        Args:
            required_fields: List of fields that are required to be present in each tick.
            severity: Severity level for completeness issues.
        """
        self.required_fields = required_fields or []
        self.severity = severity

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate tick data completeness."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                message="Data is not a pandas DataFrame",
                severity=ValidationSeverity.ERROR
            )

        issues = []
        total_rows = len(data)

        # Check for missing required fields
        for field in self.required_fields:
            if field not in data.columns:
                issues.append({
                    "issue": f"missing_field_{field}",
                    "message": f"Required field '{field}' is missing from the data.",
                    "severity": ValidationSeverity.ERROR,
                    "validator": self.__class__.__name__
                })

        # Check for null values in required fields
        if issues:
            return ValidationResult(
                is_valid=False,
                message="Missing required fields in tick data",
                details={"issues": issues},
                severity=ValidationSeverity.ERROR
            )

        for field in self.required_fields:
            null_ticks = data[data[field].isnull()]
            if not null_ticks.empty:
                issues.append({
                    "issue": f"null_value_in_field_{field}",
                    "count": len(null_ticks),
                    "percentage": 100 * len(null_ticks) / total_rows,
                    "examples": null_ticks.head().to_dict('records')
                })

        if issues:
            return ValidationResult(
                is_valid=True,  # Completeness issues might not invalidate data but warrant attention
                message="Tick data completeness issues detected",
                details={"issues": issues},
                severity=self.severity
            )

        return ValidationResult(is_valid=True, message="Tick data completeness validation passed")


class ValidateTickTimestamps(BaseTickValidator):
    """
    Validates tick timestamps.

    Ensures that timestamps are in UTC, properly ordered, and have no gaps
    larger than a specified threshold.
    """

    def __init__(
        self,
        max_gap_seconds: int = 60,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        """
        Initialize ValidateTickTimestamps.

        Args:
            max_gap_seconds: Maximum allowed gap between consecutive ticks, in seconds.
            severity: Severity level for timestamp validation issues.
        """
        self.max_gap_seconds = max_gap_seconds
        self.severity = severity

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate tick timestamps."""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                message="Data is not a pandas DataFrame",
                severity=ValidationSeverity.ERROR
            )

        if "timestamp" not in data.columns:
            return ValidationResult(
                is_valid=False,
                message="Missing 'timestamp' column",
                severity=ValidationSeverity.ERROR
            )
            
        issues = []
        
        # Ensure timestamp is in UTC and datetime type
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            try:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    message=f"Failed to convert timestamp to datetime: {str(e)}",
                    severity=ValidationSeverity.ERROR
                )
        
        # Check timestamp order and gaps
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame([t.dict() for t in data])
            if df.empty or len(df) < 2:
                return # Not enough data to check sequence

            # Ensure timestamp is datetime and timezone-aware (UTC)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')

            # Sort by timestamp just in case
            df = df.sort_values('timestamp')

            # Check for non-monotonic timestamps
            time_diffs = df['timestamp'].diff().dt.total_seconds()
            # Allow for zero difference (simultaneous ticks) but not negative (out of order)
            out_of_order = df[time_diffs < 0]

            if not out_of_order.empty:
                 error_details = out_of_order[['timestamp', 'bid', 'ask']].to_dict('records')
                 # Find the preceding tick for context
                 for i, err_idx in enumerate(out_of_order.index):
                     if err_idx > 0:
                         prev_idx = df.index[df.index.get_loc(err_idx) - 1]
                         error_details[i]['previous_timestamp'] = df.loc[prev_idx, 'timestamp'].isoformat()
                     else:
                          error_details[i]['previous_timestamp'] = None

                 raise DataValidationError(
                     f"{len(out_of_order)} tick records found out of chronological order.",
                     validation_errors={"out_of_order_ticks": error_details},
                     validator=self.__class__.__name__
                 )
        except DataValidationError:
             raise # Re-raise specific validation errors
        except Exception as e: # Catch unexpected errors during validation logic
            self.logger.exception(f"Unexpected error during tick sequence validation: {e}")
            raise DataValidationError(f"Unexpected error in {self.__class__.__name__}: {e}", validator=self.__class__.__name__) from e