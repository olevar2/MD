"""
Double Top and Double Bottom Pattern Recognition Module.

This module provides functionality to detect double top and double bottom patterns in price data.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from feature_store_service.indicators.chart_patterns.base import PatternRecognitionBase


class DoubleFormationPattern(PatternRecognitionBase):
    """
    Double Formation Pattern Detector.
    
    Identifies double top and double bottom patterns in price data.
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
        Initialize Double Formation Pattern Detector.
        
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
        Calculate double formation pattern recognition for the given data.
        
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
        result["pattern_double_top"] = 0
        result["pattern_double_bottom"] = 0
        
        # Find double top pattern
        result = self._find_double_formation(result, is_top=True)
        
        # Find double bottom pattern
        result = self._find_double_formation(result, is_top=False)
        
        return result
    
    def _find_double_formation(self, data: pd.DataFrame, is_top: bool) -> pd.DataFrame:
        """
        Find Double Top or Double Bottom patterns.
        
        Args:
            data: DataFrame with price data
            is_top: If True, look for double top (bearish), otherwise double bottom (bullish)
        
        Returns:
            DataFrame with pattern recognition markers
        """
        result = data.copy()
        pattern_name = "double_top" if is_top else "double_bottom"
        
        # Use appropriate price series based on pattern type
        price_series = result['high'] if is_top else result['low']
        
        # Initialize pattern column if not already done
        if f"pattern_{pattern_name}" not in result.columns:
            result[f"pattern_{pattern_name}"] = 0
            
        # We need at least min_pattern_size bars for a valid pattern
        if len(result) < self.min_pattern_size:
            return result
            
        # Get the lookback data
        lookback_len = min(len(result), self.lookback_period)
        lookback_data = result.iloc[-lookback_len:]
        
        # Find peaks or troughs
        peaks_or_troughs_indices = self._find_peaks_troughs(lookback_data, find_peaks=is_top)
        absolute_indices = [lookback_data.index[idx] for idx in peaks_or_troughs_indices]

        # We need at least 2 peaks/troughs for a double formation
        if len(absolute_indices) < 2:
            return result
            
        # Check each consecutive pair of peaks/troughs
        for i in range(len(absolute_indices) - 1):
            first_abs_idx = absolute_indices[i]
            second_abs_idx = absolute_indices[i+1]
            
            # Extract heights (y-values) from the original result DataFrame
            first_height = price_series.loc[first_abs_idx]
            second_height = price_series.loc[second_abs_idx]
            
            # For a valid pattern:
            # 1. The two peaks/troughs should be at similar levels
            # 2. Pattern should form within a reasonable timeframe
            # 3. There should be a significant dip/rise between them
            
            # Check if heights are similar (within 5% * sensitivity)
            if first_height == 0: continue # Avoid division by zero
            height_diff_pct = abs(first_height - second_height) / first_height
            similar_heights = height_diff_pct < (0.05 / max(self.sensitivity, 0.1))
            
            # Check if pattern is within reasonable size constraints (using absolute indices)
            first_loc = result.index.get_loc(first_abs_idx)
            second_loc = result.index.get_loc(second_abs_idx)
            pattern_width = second_loc - first_loc
            reasonable_size = self.min_pattern_size <= pattern_width <= self.max_pattern_size
            
            # Find the middle point (highest or lowest between the two peaks/troughs)
            middle_slice = result.loc[first_abs_idx:second_abs_idx] # Slice using absolute indices
            if middle_slice.empty or len(middle_slice) < 3: continue # Need points between peaks/troughs

            if is_top:
                middle_idx = middle_slice['low'][1:-1].idxmin() # Exclude the peaks themselves
                middle_val = result.loc[middle_idx, 'low']
                # Calculate how deep the middle point is relative to the tops
                avg_top = (first_height + second_height) / 2
                if avg_top == 0: continue
                depth_pct = (avg_top - middle_val) / avg_top
            else:
                middle_idx = middle_slice['high'][1:-1].idxmax() # Exclude the troughs themselves
                middle_val = result.loc[middle_idx, 'high']
                # Calculate how high the middle point is relative to the bottoms
                avg_bottom = (first_height + second_height) / 2
                if avg_bottom == 0: continue
                depth_pct = (middle_val - avg_bottom) / avg_bottom
            
            # Check if the middle point is deep/high enough
            significant_middle = depth_pct > (0.10 * max(self.sensitivity, 0.1))
            
            if similar_heights and reasonable_size and significant_middle:
                # Mark the pattern in the result DataFrame using absolute locations
                result.iloc[first_loc:second_loc + 1, result.columns.get_loc(f"pattern_{pattern_name}")] = 1
                
                # Add neckline (support or resistance level)
                neckline_col = f"pattern_{pattern_name}_neckline"
                if neckline_col not in result.columns:
                    result[neckline_col] = np.nan # Use NaN for missing values
                    
                # Use the middle point to establish the neckline
                neckline_val = middle_val
                projection_end_loc = min(len(result), second_loc + self.projection_bars)
                result.iloc[first_loc:projection_end_loc, result.columns.get_loc(neckline_col)] = neckline_val
        
        return result
