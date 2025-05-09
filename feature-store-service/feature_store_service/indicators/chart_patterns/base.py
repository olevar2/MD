"""
Base classes and utilities for chart pattern recognition.

This module provides common functionality used across different pattern recognition modules.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum

from feature_store_service.indicators.base_indicator import BaseIndicator


class BasePatternDetector(BaseIndicator):
    """
    Base class for all pattern detectors.

    This class provides common functionality for pattern detection algorithms.
    """

    category = "pattern"

    def __init__(self, **kwargs):
        """Initialize the pattern detector."""
        pass

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pattern detection for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with pattern detection values
        """
        raise NotImplementedError("Subclasses must implement calculate()")


class PatternType(Enum):
    """Enum representing different pattern types."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    CONTINUATION = "continuation"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"
    UNDEFINED = "undefined"


class PatternRecognitionBase(BaseIndicator):
    """
    Base class for all pattern recognition indicators.

    Provides common functionality for pattern detection and analysis.
    """

    category = "pattern"

    def __init__(
        self,
        lookback_period: int = 100,
        min_pattern_size: int = 10,
        max_pattern_size: int = 50,
        sensitivity: float = 0.75,
        **kwargs
    ):
        """
        Initialize Pattern Recognition Base.

        Args:
            lookback_period: Number of bars to look back for pattern recognition
            min_pattern_size: Minimum size of patterns to recognize (in bars)
            max_pattern_size: Maximum size of patterns to recognize (in bars)
            sensitivity: Sensitivity of pattern detection (0.0-1.0)
            **kwargs: Additional parameters
        """
        self.lookback_period = lookback_period
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.sensitivity = max(0.0, min(1.0, sensitivity))

    @property
    def projection_bars(self) -> int:
        """Number of bars to project pattern components into the future."""
        return 20  # Default value

    def _find_peaks_troughs(self, data: pd.DataFrame, find_peaks: bool = True) -> List[int]:
        """
        Find peaks or troughs in price data relative to the input DataFrame's index.

        Args:
            data: DataFrame with price data (e.g., a lookback slice)
            find_peaks: If True, find peaks (high points), otherwise find troughs (low points)

        Returns:
            List of relative indices (within the input data) of peaks or troughs
        """
        # Use high for peaks and low for troughs
        price_series = data['high'] if find_peaks else data['low']

        # Minimum required deviation to consider a point as a peak/trough (as % of price)
        # Use a small absolute minimum deviation as well to handle low price assets
        min_deviation_pct = 0.005 * max(0.1, self.sensitivity) # Adjusted by sensitivity
        avg_price = price_series.mean()
        min_deviation_abs = avg_price * 0.001 # Small absolute minimum
        min_deviation = max(avg_price * min_deviation_pct, min_deviation_abs)

        # Find peaks or troughs using relative indices
        result_indices = []

        # Window size for local comparison (e.g., 2 bars on each side)
        window = 2

        # We need at least 2*window + 1 data points
        if len(data) < (2 * window + 1):
            return result_indices

        for i in range(window, len(data) - window):
            is_extremum = True
            for j in range(1, window + 1):
                if find_peaks:
                    # Check if current point is strictly higher than neighbors
                    if not (price_series.iloc[i] > price_series.iloc[i-j] and price_series.iloc[i] > price_series.iloc[i+j]):
                        is_extremum = False
                        break
                else:
                    # Check if current point is strictly lower than neighbors
                    if not (price_series.iloc[i] < price_series.iloc[i-j] and price_series.iloc[i] < price_series.iloc[i+j]):
                        is_extremum = False
                        break

            if is_extremum:
                # Check significance: difference from the max/min in the window
                window_slice = price_series.iloc[i-window : i+window+1]
                if find_peaks:
                    significance_check = price_series.iloc[i] - window_slice.drop(price_series.index[i]).max() > min_deviation
                else:
                    significance_check = window_slice.drop(price_series.index[i]).min() - price_series.iloc[i] > min_deviation

                if significance_check:
                    result_indices.append(i) # Append relative index

        return result_indices

    def _calculate_trendline_loc(self, data: pd.DataFrame, locations: List[int], use_high: bool) -> Tuple[float, float]:
        """
        Calculate a trendline through the selected points using absolute locations.

        Args:
            data: The full DataFrame with price data
            locations: List of absolute integer locations (iloc) to use for trendline calculation
            use_high: If True, use 'high' values, otherwise use 'low' values

        Returns:
            Tuple of (slope, intercept) where slope is per index unit (location)
        """
        if not locations or len(locations) < 2:
            return 0, 0

        # Get x (locations) and y (prices) coordinates
        x_vals = np.array(locations)
        price_col = 'high' if use_high else 'low'
        # Ensure locations are valid before accessing .iloc
        valid_locations = [loc for loc in locations if 0 <= loc < len(data)]
        if len(valid_locations) < 2:
             return 0, 0
        y_vals = data[price_col].iloc[valid_locations].values
        x_vals = np.array(valid_locations) # Use only valid locations for fitting

        try:
            # Linear regression: y = slope * x + intercept
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
        except (np.linalg.LinAlgError, ValueError):
             # Handle cases where polyfit fails (e.g., all points collinear vertically)
             slope, intercept = 0, np.mean(y_vals) if len(y_vals) > 0 else 0

        return slope, intercept

    def _horizontal_line_fit_loc(self, data: pd.DataFrame, locations: List[int], is_resistance: bool) -> Optional[float]:
        """
        Check if points align horizontally and calculate horizontal line using absolute locations.

        Args:
            data: The full DataFrame with price data
            locations: List of absolute integer locations (iloc) to check
            is_resistance: If True, check for horizontal resistance, otherwise support

        Returns:
            Horizontal line value if points align, None otherwise
        """
        if not locations or len(locations) < 2:
            return None

        # Get price values using absolute locations
        price_col = 'high' if is_resistance else 'low'
        # Ensure locations are valid before accessing .iloc
        valid_locations = [loc for loc in locations if 0 <= loc < len(data)]
        if len(valid_locations) < 2:
             return None
        values = data[price_col].iloc[valid_locations]

        # Calculate statistics
        mean_val = values.mean()
        std_val = values.std()

        # Check if values are horizontal (small std dev relative to mean, adjusted by sensitivity)
        # Avoid division by zero
        if mean_val == 0: return None
        is_horizontal = (std_val / mean_val) < (0.01 / max(self.sensitivity, 0.1))

        if is_horizontal:
            return mean_val
        else:
            return None

    def _find_contiguous_regions_loc(self, series: pd.Series) -> List[Tuple[int, int]]:
        """
        Find contiguous regions where series values are 1, returning absolute locations.

        Args:
            series: Series with pattern markers (0 or 1) indexed like the main DataFrame

        Returns:
            List of (start_loc, end_loc) tuples for each contiguous region using iloc
        """
        regions = []
        in_region = False
        start_loc = 0

        for i in range(len(series)):
            # Use .iloc for positional access
            if series.iloc[i] == 1 and not in_region:
                # Start of a new region
                in_region = True
                start_loc = i # Store the location
            elif series.iloc[i] != 1 and in_region:
                # End of a region
                in_region = False
                regions.append((start_loc, i - 1)) # End location is inclusive

        # Handle case where pattern extends to the end
        if in_region:
            regions.append((start_loc, len(series) - 1))

        return regions

    def _calculate_pattern_strength(self, data: pd.DataFrame) -> None:
        """
        Calculate and add pattern strength metrics to the DataFrame.

        Args:
            data: DataFrame with pattern recognition columns
        """
        # Get all pattern columns (excluding neckline/support/resistance)
        pattern_cols = [col for col in data.columns if col.startswith("pattern_") and
                        col.count("_") == 1 and
                        "neckline" not in col and
                        "support" not in col and
                        "resistance" not in col and
                        "upper" not in col and
                        "lower" not in col]

        # Skip if no patterns found or strength column exists
        if not pattern_cols or "pattern_strength" in data.columns:
             # Initialize strength column if it doesn't exist
            if "pattern_strength" not in data.columns:
                data["pattern_strength"] = 0
            # return # Already calculated or no patterns to calculate for

        # Initialize strength column
        data["pattern_strength"] = 0

        # Calculate strength for each pattern type found
        for col in pattern_cols:
            # Find contiguous pattern regions using absolute locations
            pattern_regions = self._find_contiguous_regions_loc(data[col])

            for start_loc, end_loc in pattern_regions:
                if end_loc < start_loc: continue # Skip invalid regions

                # Calculate pattern strength based on various factors
                length = end_loc - start_loc + 1

                # Ensure indices are valid before slicing
                if start_loc < 0 or end_loc >= len(data): continue

                pattern_slice = data.iloc[start_loc:end_loc+1]
                if pattern_slice.empty: continue

                price_range = pattern_slice['high'].max() - pattern_slice['low'].min()
                avg_price = pattern_slice['close'].mean()

                # Calculate volume increase (handle potential missing volume or start index)
                volume_increase = 1.0
                if 'volume' in data.columns:
                    prev_volume_start = max(0, start_loc - 10)
                    if start_loc > prev_volume_start: # Ensure there's a previous period
                        prev_volume_slice = data['volume'].iloc[prev_volume_start:start_loc]
                        current_volume_slice = pattern_slice['volume']
                        if not prev_volume_slice.empty and prev_volume_slice.mean() != 0:
                            volume_increase = current_volume_slice.mean() / prev_volume_slice.mean()
                        elif current_volume_slice.mean() > 0: # Handle case where previous volume is zero
                             volume_increase = 2.0 # Assign a default high increase factor

                # Normalize factors (avoid division by zero)
                normalized_length = min(1.0, length / self.max_pattern_size) if self.max_pattern_size > 0 else 0
                normalized_range = min(1.0, price_range / (avg_price * 0.1)) if avg_price > 0 else 0
                normalized_volume = min(1.0, max(0, volume_increase - 1)) # Strength from volume *increase*

                # Calculate strength (0-100) - weighted average
                # Weights can be adjusted based on importance
                length_weight = 0.4
                range_weight = 0.4
                volume_weight = 0.2

                pattern_strength = int(
                    (normalized_length * length_weight +
                     normalized_range * range_weight +
                     normalized_volume * volume_weight) * 100
                )

                # Update strength in the DataFrame, taking the max if multiple patterns overlap
                current_strength = data["pattern_strength"].iloc[start_loc:end_loc+1]
                data.iloc[start_loc:end_loc+1, data.columns.get_loc("pattern_strength")] = np.maximum(current_strength, pattern_strength)
