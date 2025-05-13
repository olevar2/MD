"""
Gann Square of 9 Module.

This module provides implementation of Gann Square of 9.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import math

from core.base_5 import BaseGannIndicator


class GannSquare(BaseGannIndicator):
    """
    Gann Square (Square of 9, Square of 144, etc.)
    
    Gann Squares are used to identify potential support and resistance levels
    based on geometric relationships derived from price and time squares.
    The Square of 9 is the most common.
    """
    
    def __init__(
        self, 
        square_type: str = "square_of_9",
        pivot_price: Optional[float] = None,
        auto_detect_pivot: bool = True,
        lookback_period: int = 100,
        num_levels: int = 4,
        **kwargs
    ):
        """
        Initialize Gann Square indicator.
        
        Args:
            square_type: Type of square ('square_of_9', 'square_of_144', etc.)
            pivot_price: The central price point for the square (None for auto-detect)
            auto_detect_pivot: Whether to automatically detect the pivot price
            lookback_period: Number of bars to look back for finding pivot price
            num_levels: Number of square levels to calculate outwards from the pivot
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.square_type = square_type
        self.pivot_price = pivot_price
        self.auto_detect_pivot = auto_detect_pivot
        self.lookback_period = lookback_period
        self.num_levels = num_levels
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Square levels for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Gann Square levels
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Determine pivot price
        if self.pivot_price is None and self.auto_detect_pivot:
            pivot_price = self._detect_pivot_price(result)
        elif self.pivot_price is not None:
            pivot_price = self.pivot_price
        else:
            # Default to the last closing price if no pivot is provided or detected
            pivot_price = result['close'].iloc[-1]
            
        if pivot_price is None or pivot_price <= 0:
            # Cannot calculate square with invalid pivot price
            return result
            
        # Calculate Gann Square levels
        sqrt_pivot = math.sqrt(pivot_price)
        
        for i in range(1, self.num_levels + 1):
            # Calculate levels based on Square of 9 logic (adding/subtracting 0.125, 0.25, etc. to sqrt)
            # These correspond to 45, 90, 135, 180, etc. degrees on the square
            
            # Support levels (below pivot)
            level_down_45 = (sqrt_pivot - 0.125 * i)**2
            level_down_90 = (sqrt_pivot - 0.250 * i)**2
            level_down_135 = (sqrt_pivot - 0.375 * i)**2
            level_down_180 = (sqrt_pivot - 0.500 * i)**2
            
            # Resistance levels (above pivot)
            level_up_45 = (sqrt_pivot + 0.125 * i)**2
            level_up_90 = (sqrt_pivot + 0.250 * i)**2
            level_up_135 = (sqrt_pivot + 0.375 * i)**2
            level_up_180 = (sqrt_pivot + 0.500 * i)**2
            
            # Add levels to DataFrame (as horizontal lines)
            result[f'gann_sq_sup_45_{i}'] = level_down_45
            result[f'gann_sq_sup_90_{i}'] = level_down_90
            result[f'gann_sq_sup_135_{i}'] = level_down_135
            result[f'gann_sq_sup_180_{i}'] = level_down_180
            
            result[f'gann_sq_res_45_{i}'] = level_up_45
            result[f'gann_sq_res_90_{i}'] = level_up_90
            result[f'gann_sq_res_135_{i}'] = level_up_135
            result[f'gann_sq_res_180_{i}'] = level_up_180
            
        # Add pivot price marker
        result['gann_square_pivot_price'] = pivot_price
        
        return result

    def _detect_pivot_price(self, data: pd.DataFrame) -> Optional[float]:
        """
        Detect a significant pivot price (e.g., recent major high or low).
        """
        lookback_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data
        
        if lookback_data.empty:
            return None
            
        # Use the average of the highest high and lowest low in the lookback period
        high_price = lookback_data['high'].max()
        low_price = lookback_data['low'].min()
        
        if high_price is not None and low_price is not None:
            return (high_price + low_price) / 2
        
        # Fallback to last close
        return lookback_data['close'].iloc[-1]

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Square',
            'description': 'Calculates Gann Square support and resistance levels',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'square_type',
                    'description': 'Type of square (currently only Square of 9 supported)',
                    'type': 'string',
                    'default': 'square_of_9',
                    'options': ['square_of_9'] # Add more later if needed
                },
                {
                    'name': 'pivot_price',
                    'description': 'Central price point for the square',
                    'type': 'float',
                    'default': None
                },
                {
                    'name': 'auto_detect_pivot',
                    'description': 'Whether to automatically detect the pivot price',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for finding pivot price',
                    'type': 'int',
                    'default': 100
                },
                {
                    'name': 'num_levels',
                    'description': 'Number of square levels to calculate outwards',
                    'type': 'int',
                    'default': 4
                }
            ]
        }
