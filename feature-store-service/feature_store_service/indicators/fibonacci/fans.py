"""
Fibonacci Fan module.

This module provides implementation of Fibonacci fan analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from feature_store_service.indicators.fibonacci.base import FibonacciBase, TrendDirection


class FibonacciFan(FibonacciBase):
    """
    Fibonacci Fan
    
    Calculates Fibonacci fan lines from a significant swing point.
    These diagonal lines use Fibonacci ratios to identify potential support 
    and resistance levels as price moves over time.
    """
    
    def __init__(
        self, 
        levels: Optional[List[float]] = None,
        swing_lookback: int = 30,
        auto_detect_swings: bool = True,
        manual_points: Optional[Dict[str, int]] = None,
        projection_bars: int = 50,
        **kwargs
    ):
        """
        Initialize Fibonacci Fan indicator.
        
        Args:
            levels: List of Fibonacci fan levels to calculate
            swing_lookback: Number of bars to look back for finding swing points
            auto_detect_swings: Whether to automatically detect swing points
            manual_points: Dictionary with manual point indices {start_idx, end_idx}
            projection_bars: Number of bars to project levels into the future
            **kwargs: Additional parameters
        """
        name = kwargs.pop('name', 'fib_fan')
        super().__init__(name=name, **kwargs)
        
        self.levels = levels or [0.236, 0.382, 0.5, 0.618, 0.786]
        self.swing_lookback = swing_lookback
        self.auto_detect_swings = auto_detect_swings
        self.manual_points = manual_points
        self.projection_bars = projection_bars
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Fan levels for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci Fan values
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
        
        # Calculate the price range
        price_range = end_price - start_price
        
        # Calculate the time range in bars
        time_range = end_idx - start_idx
        
        # Calculate fan lines for each level
        for level in self.levels:
            col_name = self._format_column_name(self.name, level)
            result[col_name] = np.nan
            
            # Calculate the fan line values
            for i in range(start_idx, min(len(result), end_idx + self.projection_bars + 1)):
                # Calculate the time distance from start in bars
                time_distance = i - start_idx
                
                # Calculate the price at this point based on the fan level
                if trend == TrendDirection.UPTREND:
                    # For uptrend, fan lines go up from start point
                    fan_price = start_price + (price_range * level * time_distance / time_range)
                else:
                    # For downtrend, fan lines go down from start point
                    fan_price = start_price - (price_range * level * time_distance / time_range)
                
                # Set the fan line value
                if i < len(result):
                    result.iloc[i][col_name] = fan_price
        
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
            'name': 'Fibonacci Fan',
            'description': 'Calculates Fibonacci fan lines from a significant swing point',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'levels',
                    'description': 'List of Fibonacci fan levels to calculate',
                    'type': 'list',
                    'default': [0.236, 0.382, 0.5, 0.618, 0.786]
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
                    'description': 'Dictionary with manual point indices',
                    'type': 'dict',
                    'default': None
                },
                {
                    'name': 'projection_bars',
                    'description': 'Number of bars to project levels into the future',
                    'type': 'int',
                    'default': 50
                }
            ]
        }