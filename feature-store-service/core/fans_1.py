"""
Gann Fan Module.

This module provides implementation of Gann fan.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from core.base_5 import BaseGannIndicator


class GannFan(BaseGannIndicator):
    """
    Gann Fan
    
    Gann Fan consists of a set of Gann angles drawn from a significant 
    pivot point (high or low). These lines act as potential support and 
    resistance levels.
    """
    
    def __init__(
        self, 
        pivot_type: str = "swing_low",
        fan_angles: Optional[List[str]] = None,
        lookback_period: int = 100,
        price_scaling: float = 1.0,
        projection_bars: int = 50,
        **kwargs
    ):
        """
        Initialize Gann Fan indicator.
        
        Args:
            pivot_type: Type of pivot to use ('swing_low', 'swing_high')
            fan_angles: List of Gann angles for the fan (default includes common angles)
            lookback_period: Number of bars to look back for finding pivot
            price_scaling: Price scaling factor for angle calculation
            projection_bars: Number of bars to project fan lines into the future
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.pivot_type = pivot_type
        self.lookback_period = lookback_period
        self.price_scaling = price_scaling
        self.projection_bars = projection_bars
        
        # Define available Gann angles and their ratios
        self.angle_ratios = {
            "1x8": 1/8, "1x4": 1/4, "1x3": 1/3, "1x2": 1/2, "1x1": 1,
            "2x1": 2, "3x1": 3, "4x1": 4, "8x1": 8
        }
        
        # Default fan angles
        default_angles = ["1x8", "1x4", "1x2", "1x1", "2x1", "4x1", "8x1"]
        
        if fan_angles is None:
            self.fan_angles = default_angles
        else:
            self.fan_angles = [a for a in fan_angles if a in self.angle_ratios]
            
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Fan lines for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Gann Fan values
        """
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Find pivot point
        pivot_idx, pivot_price = self._find_pivot_point(result, self.pivot_type, self.lookback_period)
        
        if pivot_idx is None:
            return result
            
        # Calculate Gann fan lines from the pivot point
        pivot_pos = result.index.get_loc(pivot_idx)
        last_pos = len(result) - 1
        
        for angle_type in self.fan_angles:
            result = self._calculate_fan_line(result, pivot_pos, pivot_price, angle_type, last_pos)
        
        # Add pivot point marker
        result['gann_fan_pivot_idx'] = False
        result.loc[pivot_idx, 'gann_fan_pivot_idx'] = True
        result['gann_fan_pivot_price'] = pivot_price
        
        return result
        
    def _calculate_fan_line(
        self, 
        data: pd.DataFrame, 
        pivot_pos: int, 
        pivot_price: float, 
        angle_type: str, 
        last_pos: int
    ) -> pd.DataFrame:
        """
        Calculate a specific Gann fan line.
        """
        ratio = self.angle_ratios[angle_type]
        slope = ratio * self.price_scaling
        
        col_name = f"gann_fan_{angle_type}"
        data[col_name] = None
        
        # Determine direction based on pivot type
        is_uptrend_fan = self.pivot_type == "swing_low"
        
        # Calculate fan line projecting forward from the pivot
        for i in range(pivot_pos, last_pos + self.projection_bars + 1):
            time_diff = i - pivot_pos
            
            if is_uptrend_fan:
                # Fan lines go up from a swing low
                fan_price = pivot_price + (slope * time_diff)
            else:
                # Fan lines go down from a swing high
                fan_price = pivot_price - (slope * time_diff)
            
            if i <= last_pos:
                data.iloc[i, data.columns.get_loc(col_name)] = fan_price
                
        return data

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Fan',
            'description': 'Calculates Gann fan lines from a pivot point',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'pivot_type',
                    'description': 'Type of pivot point to use',
                    'type': 'string',
                    'default': 'swing_low',
                    'options': ['swing_low', 'swing_high']
                },
                {
                    'name': 'fan_angles',
                    'description': 'List of Gann angles for the fan',
                    'type': 'list',
                    'default': ["1x8", "1x4", "1x2", "1x1", "2x1", "4x1", "8x1"]
                },
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for finding pivot',
                    'type': 'int',
                    'default': 100
                },
                {
                    'name': 'price_scaling',
                    'description': 'Price scaling factor for angle calculation',
                    'type': 'float',
                    'default': 1.0
                },
                {
                    'name': 'projection_bars',
                    'description': 'Number of bars to project fan lines into the future',
                    'type': 'int',
                    'default': 50
                }
            ]
        }
