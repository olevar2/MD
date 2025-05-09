"""
Fibonacci Extension module.

This module provides implementation of Fibonacci extension analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from feature_store_service.indicators.fibonacci.base import FibonacciBase, TrendDirection


class FibonacciExtension(FibonacciBase):
    """
    Fibonacci Extension
    
    Calculates Fibonacci extension levels based on three significant points.
    These levels often act as targets for price movements beyond the initial trend.
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
        Initialize Fibonacci Extension indicator.
        
        Args:
            levels: List of Fibonacci extension levels to calculate
            swing_lookback: Number of bars to look back for finding swing points
            auto_detect_swings: Whether to automatically detect swing points
            manual_points: Dictionary with manual point indices {start_idx, end_idx, retracement_idx}
            projection_bars: Number of bars to project levels into the future
            **kwargs: Additional parameters
        """
        name = kwargs.pop('name', 'fib_extension')
        super().__init__(name=name, **kwargs)
        
        self.levels = levels or [0.0, 0.382, 0.618, 1.0, 1.382, 1.618, 2.0, 2.618]
        self.swing_lookback = swing_lookback
        self.auto_detect_swings = auto_detect_swings
        self.manual_points = manual_points
        self.projection_bars = projection_bars
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Extension levels for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci Extension values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Find the three required points
        if self.auto_detect_swings:
            # Automatically detect the three points (start, end, retracement)
            start_idx, start_price, end_idx, end_price, retr_idx, retr_price = self._detect_three_points(result)
        elif self.manual_points:
            # Use manually specified points
            idx_map = self.manual_points
            start_idx = idx_map.get('start_idx', 0)
            end_idx = idx_map.get('end_idx', len(result) // 2)
            retr_idx = idx_map.get('retracement_idx', len(result) - 1)
            
            # Get prices at these indices
            start_price = result.iloc[start_idx]['close']
            end_price = result.iloc[end_idx]['close']
            retr_price = result.iloc[retr_idx]['close']
        else:
            # Default to evenly spaced points
            start_idx = 0
            end_idx = len(result) // 2
            retr_idx = len(result) - 1
            
            start_price = result.iloc[start_idx]['close']
            end_price = result.iloc[end_idx]['close']
            retr_price = result.iloc[retr_idx]['close']
        
        # Determine trend direction
        trend = self._get_trend_direction(start_price, end_price)
        
        # Calculate the price range for the initial move
        price_range = end_price - start_price
        
        # Calculate extension levels from the retracement point
        extension_levels = {}
        for level in self.levels:
            # Extension is calculated from the retracement point
            extension_price = retr_price + (price_range * level)
            extension_levels[level] = extension_price
        
        # Add columns for each extension level
        for level, price in extension_levels.items():
            col_name = self._format_column_name(self.name, level)
            result[col_name] = price
        
        # Mark the start, end, and retracement points
        result[f'{self.name}_start'] = False
        result[f'{self.name}_end'] = False
        result[f'{self.name}_retracement'] = False
        
        start_idx_loc = result.index[start_idx]
        end_idx_loc = result.index[end_idx]
        retr_idx_loc = result.index[retr_idx]
        
        result.loc[start_idx_loc, f'{self.name}_start'] = True
        result.loc[end_idx_loc, f'{self.name}_end'] = True
        result.loc[retr_idx_loc, f'{self.name}_retracement'] = True
        
        # Add trend direction
        result[f'{self.name}_trend'] = trend.value
        
        return result
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Extension',
            'description': 'Calculates Fibonacci extension levels based on three significant points',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'levels',
                    'description': 'List of Fibonacci extension levels to calculate',
                    'type': 'list',
                    'default': [0.0, 0.382, 0.618, 1.0, 1.382, 1.618, 2.0, 2.618]
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
                    'description': 'Dictionary with manual point indices {start_idx, end_idx, retracement_idx}',
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