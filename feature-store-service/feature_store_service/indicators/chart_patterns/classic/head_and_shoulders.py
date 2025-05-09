"""
Head and Shoulders Pattern Recognition Module.

This module provides functionality to detect head and shoulders patterns in price data.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from feature_store_service.indicators.chart_patterns.base import PatternRecognitionBase


class HeadAndShouldersPattern(PatternRecognitionBase):
    """
    Head and Shoulders Pattern Detector.
    
    Identifies regular and inverse head and shoulders patterns in price data.
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
        Initialize Head and Shoulders Pattern Detector.
        
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
        Calculate head and shoulders pattern recognition for the given data.
        
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
        result["pattern_head_and_shoulders"] = 0
        result["pattern_inverse_head_and_shoulders"] = 0
        
        # Find regular head and shoulders pattern
        result = self._find_head_and_shoulders(result, inverse=False)
        
        # Find inverse head and shoulders pattern
        result = self._find_head_and_shoulders(result, inverse=True)
        
        return result
    
    def _find_head_and_shoulders(self, data: pd.DataFrame, inverse: bool = False) -> pd.DataFrame:
        """
        Find Head and Shoulders or Inverse Head and Shoulders patterns.
        
        Args:
            data: DataFrame with price data
            inverse: If True, look for inverse head and shoulders (bullish)
        
        Returns:
            DataFrame with pattern recognition markers
        """
        result = data.copy()
        pattern_name = "inverse_head_and_shoulders" if inverse else "head_and_shoulders"
        
        # Use appropriate price series based on pattern type
        price_series = result['low'] if inverse else result['high']
        
        # Initialize pattern column if not already done
        if f"pattern_{pattern_name}" not in result.columns:
            result[f"pattern_{pattern_name}"] = 0
            
        # We need at least min_pattern_size bars for a valid pattern
        if len(result) < self.min_pattern_size:
            return result
            
        # Get the lookback data
        lookback_len = min(len(result), self.lookback_period)
        lookback_data = result.iloc[-lookback_len:]
        
        # Find peaks or troughs using a more robust method if available, e.g., scipy.signal.find_peaks
        # For simplicity, using the existing _find_peaks_troughs
        peaks_or_troughs_indices = self._find_peaks_troughs(lookback_data, find_peaks=not inverse)
        
        # Convert relative indices to absolute indices of the original DataFrame
        absolute_indices = [lookback_data.index[idx] for idx in peaks_or_troughs_indices]

        # We need at least 5 peaks/troughs for a head and shoulders pattern
        if len(absolute_indices) < 5:
            return result
            
        # Check each window of 5 peaks/troughs for the pattern
        for i in range(len(absolute_indices) - 4):
            points_indices = absolute_indices[i:i+5]
            
            # Extract points indices for left shoulder, head, right shoulder
            left_shoulder_idx = points_indices[0]
            head_idx = points_indices[2]
            right_shoulder_idx = points_indices[4]
            
            # Extract heights (y-values) from the original result DataFrame
            ls_height = price_series.loc[left_shoulder_idx]
            head_height = price_series.loc[head_idx] 
            rs_height = price_series.loc[right_shoulder_idx]
            
            # For a valid pattern:
            # 1. Head should be higher (lower for inverse) than both shoulders
            # 2. Shoulders should be roughly at the same level
            # 3. Pattern should form within a reasonable timeframe
            
            head_higher = (
                (head_height > ls_height and head_height > rs_height) if not inverse 
                else (head_height < ls_height and head_height < rs_height)
            )
            
            # Check if shoulders are at similar levels (within 10% * sensitivity)
            # Avoid division by zero if ls_height is 0
            if ls_height == 0: continue
            shoulder_diff_pct = abs(ls_height - rs_height) / ls_height
            shoulders_similar = shoulder_diff_pct < (0.1 / max(self.sensitivity, 0.1)) # Ensure sensitivity is not zero
            
            # Check if pattern is within reasonable size constraints (using absolute indices)
            pattern_width = result.index.get_loc(right_shoulder_idx) - result.index.get_loc(left_shoulder_idx)
            reasonable_size = self.min_pattern_size <= pattern_width <= self.max_pattern_size
            
            if head_higher and shoulders_similar and reasonable_size:
                # Calculate neckline from the troughs/peaks between shoulders and head
                neckline_left_idx = points_indices[1]
                neckline_right_idx = points_indices[3]
                
                # Use the opposite price series for neckline points
                neckline_price_series = result['high'] if inverse else result['low']
                neckline_left_price = neckline_price_series.loc[neckline_left_idx]
                neckline_right_price = neckline_price_series.loc[neckline_right_idx]

                # Calculate neckline slope and intercept using absolute indices
                neckline_left_loc = result.index.get_loc(neckline_left_idx)
                neckline_right_loc = result.index.get_loc(neckline_right_idx)
                
                if neckline_right_loc == neckline_left_loc: continue # Avoid division by zero

                slope = (neckline_right_price - neckline_left_price) / (neckline_right_loc - neckline_left_loc)
                intercept = neckline_left_price - (slope * neckline_left_loc)

                # Mark the pattern in the result DataFrame using absolute indices
                start_loc = result.index.get_loc(left_shoulder_idx)
                end_loc = result.index.get_loc(right_shoulder_idx)
                result.iloc[start_loc:end_loc + 1, result.columns.get_loc(f"pattern_{pattern_name}")] = 1
                        
                # Add neckline information
                neckline_col = f"pattern_{pattern_name}_neckline"
                if neckline_col not in result.columns:
                    result[neckline_col] = np.nan # Use NaN for missing values
                    
                # Project the neckline
                projection_end_loc = min(len(result), end_loc + self.projection_bars)
                for loc in range(start_loc, projection_end_loc):
                    neckline_val = (slope * loc) + intercept
                    result.iloc[loc, result.columns.get_loc(neckline_col)] = neckline_val
                    
        return result
