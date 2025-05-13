"""
Wedge Pattern Recognition Module.

This module provides functionality to detect rising and falling wedge patterns in price data.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from core.base_1 import PatternRecognitionBase


class WedgePattern(PatternRecognitionBase):
    """
    Wedge Pattern Detector.
    
    Identifies rising and falling wedge patterns in price data.
    """
    
    def __init__(
        self, 
        lookback_period: int = 100,
        min_pattern_size: int = 10,
        max_pattern_size: int = 50,
        sensitivity: float = 0.75,
        **kwargs
    ):
        """
        Initialize Wedge Pattern Detector.
        
        Args:
            lookback_period: Number of bars to look back for pattern recognition
            min_pattern_size: Minimum size of patterns to recognize (in bars)
            max_pattern_size: Maximum size of patterns to recognize (in bars)
            sensitivity: Sensitivity of pattern detection (0.0-1.0)
            **kwargs: Additional parameters
        """
        super().__init__(
            lookback_period=lookback_period,
            min_pattern_size=min_pattern_size,
            max_pattern_size=max_pattern_size,
            sensitivity=sensitivity,
            **kwargs
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate wedge pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pattern recognition values
        """
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Initialize pattern columns with zeros
        result["pattern_wedge_rising"] = 0
        result["pattern_wedge_falling"] = 0
        
        # Find wedge patterns
        result = self._find_wedge_patterns(result)
        
        return result
    
    def _find_wedge_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Find Rising and Falling Wedge patterns.
        
        Args:
            data: DataFrame with price data
        
        Returns:
            DataFrame with wedge pattern recognition markers
        """
        result = data.copy()
        
        # Initialize pattern columns if not already done
        for pattern in ["wedge_rising", "wedge_falling"]:
            if f"pattern_{pattern}" not in result.columns:
                result[f"pattern_{pattern}"] = 0
                
        # We need at least min_pattern_size bars for a valid pattern
        if len(result) < self.min_pattern_size:
            return result
            
        # Iterate through potential pattern end points
        for end_loc in range(self.min_pattern_size, len(result)):
            start_loc = max(0, end_loc - self.lookback_period)
            lookback_data = result.iloc[start_loc:end_loc]

            if len(lookback_data) < self.min_pattern_size: continue

            # Find peaks and troughs within the lookback window
            peak_indices_rel = self._find_peaks_troughs(lookback_data, find_peaks=True)
            trough_indices_rel = self._find_peaks_troughs(lookback_data, find_peaks=False)

            # Convert relative indices to absolute locations
            peak_locs = [start_loc + idx for idx in peak_indices_rel]
            trough_locs = [start_loc + idx for idx in trough_indices_rel]

            # We need at least 2 peaks and 2 troughs to form a wedge
            if len(peak_locs) < 2 or len(trough_locs) < 2:
                continue
                
            # Calculate trend lines for highs and lows using absolute locations
            high_slope, high_intercept = self._calculate_trendline_loc(result, peak_locs, use_high=True)
            low_slope, low_intercept = self._calculate_trendline_loc(result, trough_locs, use_high=False)
            
            # Define slope thresholds based on sensitivity
            slope_threshold = 0.002 * max(self.sensitivity, 0.1)

            # Rising wedge: Both trendlines slope upward, converging (low_slope > high_slope)
            is_rising_wedge = (high_slope > slope_threshold and low_slope > slope_threshold and low_slope > high_slope)
            
            # Falling wedge: Both trendlines slope downward, converging (high_slope > low_slope)
            is_falling_wedge = (high_slope < -slope_threshold and low_slope < -slope_threshold and high_slope > low_slope)
            
            pattern_type = None
            if is_rising_wedge:
                pattern_type = "wedge_rising"
            elif is_falling_wedge:
                pattern_type = "wedge_falling"
            
            if pattern_type:
                # Find the start and end locations of the pattern
                pattern_start_loc = min(min(peak_locs), min(trough_locs))
                pattern_end_loc = max(max(peak_locs), max(trough_locs))
                
                # Check if pattern is within reasonable size constraints
                pattern_width = pattern_end_loc - pattern_start_loc
                if self.min_pattern_size <= pattern_width <= self.max_pattern_size:
                    # Mark the pattern in the result DataFrame
                    result.iloc[pattern_start_loc:pattern_end_loc + 1, result.columns.get_loc(f"pattern_{pattern_type}")] = 1
            
                    # Add upper and lower trendlines
                    upper_trendline_col = f"pattern_{pattern_type}_upper"
                    lower_trendline_col = f"pattern_{pattern_type}_lower"
                    
                    if upper_trendline_col not in result.columns:
                        result[upper_trendline_col] = np.nan
                    if lower_trendline_col not in result.columns:
                        result[lower_trendline_col] = np.nan
                    
                    # Calculate trendline values for projection
                    projection_end_loc = min(len(result), pattern_end_loc + self.projection_bars)
                    for loc in range(pattern_start_loc, projection_end_loc):
                        upper_val = (high_slope * loc) + high_intercept
                        lower_val = (low_slope * loc) + low_intercept
                        
                        result.iloc[loc, result.columns.get_loc(upper_trendline_col)] = upper_val
                        result.iloc[loc, result.columns.get_loc(lower_trendline_col)] = lower_val
        
        return result
