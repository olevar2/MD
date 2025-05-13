"""
Fibonacci Circles module.

This module provides implementation of Fibonacci circles analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import math

from core.base_4 import FibonacciBase, TrendDirection, find_swing_highs, find_swing_lows


class FibonacciCircles(FibonacciBase):
    """
    Fibonacci Circles
    
    Calculates Fibonacci circles from a significant pivot point.
    These circles use Fibonacci ratios to identify potential support and resistance
    areas at specific distances from the pivot.
    """
    
    def __init__(
        self,
        levels: Optional[List[float]] = None,
        swing_lookback: int = 50,
        auto_detect_points: bool = True,
        manual_points: Optional[Dict[str, int]] = None, # {center_idx: idx, radius_ref_idx: idx}
        projection_bars: int = 50,
        time_price_ratio: float = 1.0, # Scaling factor: how many price units equal one time unit (bar)
        **kwargs
    ):
        """
        Initialize Fibonacci Circles indicator.

        Args:
            levels: List of Fibonacci levels for circle radii (e.g., [0.382, 0.5, 0.618, 1.0])
            swing_lookback: Number of bars to look back for finding center and radius points
            auto_detect_points: Whether to automatically detect center and radius points
            manual_points: Dictionary with manual point indices {center_idx, radius_ref_idx}
            projection_bars: How far in time to calculate circle points beyond data range
            time_price_ratio: Scaling factor between time (bars) and price axes. Crucial for circle shape.
            **kwargs: Additional parameters
        """
        name = kwargs.pop('name', 'fib_circle')
        super().__init__(name=name, **kwargs)
        
        self.levels = levels or [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618]
        self.swing_lookback = swing_lookback
        self.auto_detect_points = auto_detect_points
        self.manual_points = manual_points
        self.projection_bars = projection_bars
        self.time_price_ratio = time_price_ratio
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Circles for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci Circle values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Find the center and radius reference points
        if self.auto_detect_points:
            # Automatically detect points
            # For circles, we need a center point and a radius reference point
            # Use a significant swing point as center and another as radius reference
            center_idx, radius_ref_idx = self._detect_circle_points(result)
        elif self.manual_points:
            # Use manually specified points
            idx_map = self.manual_points
            center_idx = idx_map.get('center_idx', 0)
            radius_ref_idx = idx_map.get('radius_ref_idx', len(result) // 2)
        else:
            # Default to evenly spaced points
            center_idx = 0
            radius_ref_idx = len(result) // 2
        
        # Get prices and indices
        center_price = result.iloc[center_idx]['close']
        radius_ref_price = result.iloc[radius_ref_idx]['close']
        
        # Calculate the initial radius in price units
        initial_radius_a_price = abs(radius_ref_price - center_price)
        
        # Calculate circle points for each level
        for level in self.levels:
            radius_price = initial_radius_a_price * level # Radius in price units
            if radius_price <= 0: continue

            col_name_upper = f"fib_circle_upper_{str(level).replace('.', '_')}"
            col_name_lower = f"fib_circle_lower_{str(level).replace('.', '_')}"

            # Convert price radius to time radius using the scaling factor
            # radius_time^2 * time_price_ratio^2 + price_offset^2 = radius_price^2
            # radius_time = radius_price / time_price_ratio (if circle is truly round in scaled space)
            radius_time_bars = radius_price / self.time_price_ratio
            
            # Calculate circle points
            result[col_name_upper] = np.nan
            result[col_name_lower] = np.nan
            
            # Calculate for each bar within range
            for i in range(max(0, center_idx - int(radius_time_bars) - 5), 
                          min(len(result), center_idx + int(radius_time_bars) + self.projection_bars + 5)):
                # Calculate time distance from center in bars
                time_distance = i - center_idx
                
                # Calculate the price offset at this time distance
                # Using the circle equation: x^2 + y^2 = r^2
                # Where x is time_distance * time_price_ratio and y is price_offset
                time_component = (time_distance * self.time_price_ratio) ** 2
                if time_component > radius_price ** 2:
                    # Outside the circle's time range
                    continue
                    
                price_offset = math.sqrt(radius_price ** 2 - time_component)
                
                # Set the upper and lower circle points
                result.iloc[i][col_name_upper] = center_price + price_offset
                result.iloc[i][col_name_lower] = center_price - price_offset
        
        # Mark the center and radius reference points
        result[f'{self.name}_center'] = False
        result[f'{self.name}_radius_ref'] = False
        
        center_idx_loc = result.index[center_idx]
        radius_ref_idx_loc = result.index[radius_ref_idx]
        
        result.loc[center_idx_loc, f'{self.name}_center'] = True
        result.loc[radius_ref_idx_loc, f'{self.name}_radius_ref'] = True
        
        return result
    
    def _detect_circle_points(self, data: pd.DataFrame) -> Tuple[int, int]:
        """
        Detect center and radius reference points for Fibonacci circles.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (center_idx, radius_ref_idx)
        """
        # Find swing highs and lows
        swing_highs = find_swing_highs(data, n=self.swing_lookback)
        swing_lows = find_swing_lows(data, n=self.swing_lookback)
        
        # Combine and sort by index
        all_swings = []
        for idx, price in swing_highs:
            all_swings.append((idx, price, 'high'))
        for idx, price in swing_lows:
            all_swings.append((idx, price, 'low'))
        
        all_swings.sort(key=lambda x: x[0])
        
        # Need at least 2 swing points
        if len(all_swings) < 2:
            # Use first and middle points if not enough swings
            center_idx = 0
            radius_ref_idx = len(data) // 2
            return center_idx, radius_ref_idx
        
        # Use the first swing point as center and second as radius reference
        center_idx = all_swings[0][0]
        radius_ref_idx = all_swings[1][0]
        
        return center_idx, radius_ref_idx
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Circles',
            'description': 'Calculates Fibonacci circles from a significant pivot point',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'levels',
                    'description': 'List of Fibonacci levels for circle radii',
                    'type': 'list',
                    'default': [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618]
                },
                {
                    'name': 'swing_lookback',
                    'description': 'Number of bars to look back for finding center and radius points',
                    'type': 'int',
                    'default': 50
                },
                {
                    'name': 'auto_detect_points',
                    'description': 'Whether to automatically detect center and radius points',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'manual_points',
                    'description': 'Dictionary with manual point indices {center_idx, radius_ref_idx}',
                    'type': 'dict',
                    'default': None
                },
                {
                    'name': 'projection_bars',
                    'description': 'How far in time to calculate circle points beyond data range',
                    'type': 'int',
                    'default': 50
                },
                {
                    'name': 'time_price_ratio',
                    'description': 'Scaling factor between time (bars) and price axes',
                    'type': 'float',
                    'default': 1.0
                }
            ]
        }