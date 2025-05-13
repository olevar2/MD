"""
Fibonacci Retracement module.

This module provides implementation of Fibonacci retracement analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from core.base_4 import FibonacciBase, TrendDirection


class FibonacciRetracement(FibonacciBase):
    """
    Fibonacci Retracement
    
    Calculates Fibonacci retracement levels based on significant swing points.
    These levels often act as support or resistance during price corrections.
    """
    
    def __init__(
        self, 
        levels: Optional[List[float]] = None,
        swing_lookback: int = 30,
        auto_detect_swings: bool = True,
        manual_points: Optional[Dict[str, int]] = None,
        projection_bars: int = 0,
        **kwargs
    ):
        """
        Initialize Fibonacci Retracement indicator.
        
        Args:
            levels: List of Fibonacci retracement levels to calculate
            swing_lookback: Number of bars to look back for finding swing points
            auto_detect_swings: Whether to automatically detect swing points
            manual_points: Dictionary with manual point indices {start_idx, end_idx}
            projection_bars: Number of bars to project levels into the future
            **kwargs: Additional parameters
        """
        name = kwargs.pop('name', 'fib_retracement')
        super().__init__(name=name, **kwargs)
        
        self.levels = levels or [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.swing_lookback = swing_lookback
        self.auto_detect_swings = auto_detect_swings
        self.manual_points = manual_points
        self.projection_bars = projection_bars
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Retracement levels for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci Retracement values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Find the two required points (start and end of the trend)
        if self.auto_detect_swings:
            # Automatically detect swing points
            start_idx, start_price, end_idx, end_price = self._detect_swing_points(result)
        elif self.manual_points:
            # Use manually specified points
            idx_map = self.manual_points
            start_idx = idx_map.get('start_idx', 0)
            end_idx = idx_map.get('end_idx', len(result) - 1)
            
            # Get prices at these indices
            start_price = result.iloc[start_idx]['close']
            end_price = result.iloc[end_idx]['close']
        else:
            # Default to first and last points
            start_idx = 0
            end_idx = len(result) - 1
            start_price = result.iloc[start_idx]['close']
            end_price = result.iloc[end_idx]['close']
        
        # Determine trend direction
        trend = self._get_trend_direction(start_price, end_price)
        
        # Calculate retracement levels
        # For retracements, we need to handle the direction correctly
        # In an uptrend, 0.0 is at the end (high) and 1.0 is at the start (low)
        # In a downtrend, 0.0 is at the end (low) and 1.0 is at the start (high)
        price_range = end_price - start_price
        
        # Add columns for each retracement level
        for level in self.levels:
            level_price = end_price - (price_range * level)
            col_name = self._format_column_name(self.name, level)
            result[col_name] = level_price
        
        # Mark the start and end points
        result[f'{self.name}_start'] = False
        result[f'{self.name}_end'] = False
        
        start_idx_loc = result.index[start_idx]
        end_idx_loc = result.index[end_idx]
        
        result.loc[start_idx_loc, f'{self.name}_start'] = True
        result.loc[end_idx_loc, f'{self.name}_end'] = True
        
        # Add trend direction
        result[f'{self.name}_trend'] = trend.value
        
        return result
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Retracement',
            'description': 'Calculates Fibonacci retracement levels based on significant swing points',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'levels',
                    'description': 'List of Fibonacci retracement levels to calculate',
                    'type': 'list',
                    'default': [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
                },
                {
                    'name': 'swing_lookback',
                    'description': 'Number of bars to look back for finding swing points',
                    'type': 'int',
                    'default': 30
                },
                {
                    'name': 'auto_detect_swings',
                    'description': 'Whether to automatically detect swing points',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'manual_points',
                    'description': 'Dictionary with manual point indices {start_idx, end_idx}',
                    'type': 'dict',
                    'default': None
                },
                {
                    'name': 'projection_bars',
                    'description': 'Number of bars to project levels into the future',
                    'type': 'int',
                    'default': 0
                }
            ]
        }