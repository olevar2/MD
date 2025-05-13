"""
Triple Top and Triple Bottom Pattern Recognition Module.

This module provides functionality to detect triple top and triple bottom patterns in price data.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from core.base_1 import PatternRecognitionBase


class TripleFormationPattern(PatternRecognitionBase):
    """
    Triple Formation Pattern Detector.
    
    Identifies triple top and triple bottom patterns in price data.
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
        Initialize Triple Formation Pattern Detector.
        
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
        Calculate triple formation pattern recognition for the given data.
        
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
        result["pattern_triple_top"] = 0
        result["pattern_triple_bottom"] = 0
        
        # Find triple top pattern
        result = self._find_triple_formation(result, is_top=True)
        
        # Find triple bottom pattern
        result = self._find_triple_formation(result, is_top=False)
        
        return result
    
    def _find_triple_formation(self, data: pd.DataFrame, is_top: bool) -> pd.DataFrame:
        """
        Find Triple Top or Triple Bottom patterns.
        
        Args:
            data: DataFrame with price data
            is_top: If True, look for triple top (bearish), otherwise triple bottom (bullish)
        
        Returns:
            DataFrame with pattern recognition markers
        """
        result = data.copy()
        pattern_name = "triple_top" if is_top else "triple_bottom"
        
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

        # We need at least 3 peaks/troughs for a triple formation
        if len(absolute_indices) < 3:
            return result
            
        # Check each consecutive triplet of peaks/troughs
        for i in range(len(absolute_indices) - 2):
            first_abs_idx = absolute_indices[i]
            second_abs_idx = absolute_indices[i+1]
            third_abs_idx = absolute_indices[i+2]
            
            # Extract heights (y-values) from the original result DataFrame
            first_height = price_series.loc[first_abs_idx]
            second_height = price_series.loc[second_abs_idx]
            third_height = price_series.loc[third_abs_idx]
            
            # For a valid pattern:
            # 1. All three peaks/troughs should be at similar levels
            # 2. Pattern should form within a reasonable timeframe
            
            # Check if heights are similar (within 5% * sensitivity)
            heights = [first_height, second_height, third_height]
            max_height = max(heights)
            min_height = min(heights)
            if max_height == 0: continue # Avoid division by zero
            height_diff_pct = (max_height - min_height) / max_height
            similar_heights = height_diff_pct < (0.05 / max(self.sensitivity, 0.1))
            
            # Check if pattern is within reasonable size constraints (using absolute indices)
            first_loc = result.index.get_loc(first_abs_idx)
            third_loc = result.index.get_loc(third_abs_idx)
            pattern_width = third_loc - first_loc
            reasonable_size = self.min_pattern_size <= pattern_width <= self.max_pattern_size
            
            if similar_heights and reasonable_size:
                # Mark the pattern in the result DataFrame using absolute locations
                result.iloc[first_loc:third_loc + 1, result.columns.get_loc(f"pattern_{pattern_name}")] = 1
                
                # Add neckline (support or resistance level)
                neckline_col = f"pattern_{pattern_name}_neckline"
                if neckline_col not in result.columns:
                    result[neckline_col] = np.nan # Use NaN for missing values
                
                # Find the two lowest/highest points between the three peaks/troughs
                slice1 = result.loc[first_abs_idx:second_abs_idx]
                slice2 = result.loc[second_abs_idx:third_abs_idx]
                if slice1.empty or len(slice1) < 3 or slice2.empty or len(slice2) < 3: continue

                if is_top:
                    middle_idx = slice1['low'][1:-1].idxmin()
                    middle_val = result.loc[middle_idx, 'low']
                    # Calculate how deep the middle point is relative to the tops
                    avg_top = (first_height + second_height) / 2
                    if avg_top == 0: continue
                    depth_pct = (avg_top - middle_val) / avg_top
                else:
                    middle_idx = slice2['high'][1:-1].idxmax()
                    middle_val = result.loc[middle_idx, 'high']
                    # Calculate how high the middle point is relative to the bottoms
                    avg_bottom = (first_height + second_height) / 2
                    if avg_bottom == 0: continue
                    depth_pct = (middle_val - avg_bottom) / avg_bottom
                
                # Check if the middle point is deep/high enough
                significant_middle = depth_pct > (0.10 * max(self.sensitivity, 0.1))
                
                if similar_heights and reasonable_size and significant_middle:
                    # Mark the pattern in the result DataFrame using absolute locations
                    result.iloc[first_loc:third_loc + 1, result.columns.get_loc(f"pattern_{pattern_name}")] = 1
                    
                    # Add neckline (support or resistance level)
                    neckline_col = f"pattern_{pattern_name}_neckline"
                    if neckline_col not in result.columns:
                        result[neckline_col] = np.nan # Use NaN for missing values
                        
                    # Use the middle point to establish the neckline
                    neckline_val = middle_val
                    projection_end_loc = min(len(result), third_loc + self.projection_bars)
                    result.iloc[first_loc:projection_end_loc, result.columns.get_loc(neckline_col)] = neckline_val
        
        return result
