"""
Rectangle Pattern Recognition Module.

This module provides functionality to detect rectangle patterns in price data.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from core.base_1 import PatternRecognitionBase


class RectanglePattern(PatternRecognitionBase):
    """
    Rectangle Pattern Detector.
    
    Identifies rectangle patterns (horizontal support and resistance channels) in price data.
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
        Initialize Rectangle Pattern Detector.
        
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
        Calculate rectangle pattern recognition for the given data.
        
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
        result["pattern_rectangle"] = 0
        
        # Find rectangle pattern
        result = self._find_rectangle_pattern(result)
        
        return result
    
    def _find_rectangle_pattern(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Find Rectangle pattern (horizontal support and resistance channels).
        
        Args:
            data: DataFrame with price data
        
        Returns:
            DataFrame with rectangle pattern recognition markers
        """
        result = data.copy()
        pattern_name = "rectangle"
        
        # Initialize pattern column if not already done
        if f"pattern_{pattern_name}" not in result.columns:
            result[f"pattern_{pattern_name}"] = 0
            
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

            # We need at least 2 peaks and 2 troughs to form a rectangle
            if len(peak_locs) < 2 or len(trough_locs) < 2:
                continue
                
            # Check for horizontal support and resistance using absolute locations
            resistance_line = self._horizontal_line_fit_loc(result, peak_locs, is_resistance=True)
            support_line = self._horizontal_line_fit_loc(result, trough_locs, is_resistance=False)
            
            if resistance_line is None or support_line is None or resistance_line <= support_line:
                continue # Not a valid channel or lines crossed
                
            # Calculate channel height as percentage of average price
            pattern_start_loc = min(min(peak_locs), min(trough_locs))
            pattern_end_loc = max(max(peak_locs), max(trough_locs))
            avg_price = result['close'].iloc[pattern_start_loc:pattern_end_loc+1].mean()
            if avg_price == 0: continue
            channel_height_pct = (resistance_line - support_line) / avg_price
            
            # Rectangle should have a reasonable channel height (e.g., 1-10% of price, adjusted by sensitivity)
            min_height_pct = 0.01 * max(self.sensitivity, 0.1)
            max_height_pct = 0.10 * (1 / max(self.sensitivity, 0.1))
            reasonable_height = (min_height_pct <= channel_height_pct <= max_height_pct)
            
            # Check if pattern is within reasonable size constraints
            pattern_width = pattern_end_loc - pattern_start_loc
            reasonable_size = self.min_pattern_size <= pattern_width <= self.max_pattern_size
            
            if reasonable_height and reasonable_size:
                # Mark the pattern in the result DataFrame
                result.iloc[pattern_start_loc:pattern_end_loc + 1, result.columns.get_loc(f"pattern_{pattern_name}")] = 1
                
                # Add support and resistance lines
                support_col = f"pattern_{pattern_name}_support"
                resistance_col = f"pattern_{pattern_name}_resistance"
                
                if support_col not in result.columns:
                    result[support_col] = np.nan
                if resistance_col not in result.columns:
                    result[resistance_col] = np.nan
                
                # Set support and resistance values for projection
                projection_end_loc = min(len(result), pattern_end_loc + self.projection_bars)
                result.iloc[pattern_start_loc:projection_end_loc, result.columns.get_loc(support_col)] = support_line
                result.iloc[pattern_start_loc:projection_end_loc, result.columns.get_loc(resistance_col)] = resistance_line
        
        return result
