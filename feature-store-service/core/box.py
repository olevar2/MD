"""
Gann Box Module.

This module provides implementation of Gann box.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from core.base_5 import BaseGannIndicator


class GannBox(BaseGannIndicator):
    """
    Gann Box

    Draws a box between two points (typically significant highs and lows).
    The box is then divided by time and price lines based on Gann or Fibonacci ratios.
    These divisions can identify potential support/resistance and time turning points.
    """

    def __init__(
        self,
        start_pivot_type: str = "major_low",
        end_pivot_type: str = "major_high",
        lookback_period: int = 100,
        auto_detect_pivots: bool = True,
        manual_start_idx: Optional[int] = None,
        manual_start_price: Optional[float] = None,
        manual_end_idx: Optional[int] = None,
        manual_end_price: Optional[float] = None,
        price_divisions: List[float] = [0.25, 0.382, 0.5, 0.618, 0.75],
        time_divisions: List[float] = [0.25, 0.382, 0.5, 0.618, 0.75],
        **kwargs
    ):
        """
        Initialize Gann Box indicator.

        Args:
            start_pivot_type: Type of pivot for box start ('major_low', 'major_high')
            end_pivot_type: Type of pivot for box end ('major_low', 'major_high')
            lookback_period: Number of bars to look back for finding pivots
            auto_detect_pivots: Whether to automatically detect the pivots
            manual_start_idx: Manual start index position (if auto_detect_pivots=False)
            manual_start_price: Manual start price (if auto_detect_pivots=False)
            manual_end_idx: Manual end index position (if auto_detect_pivots=False)
            manual_end_price: Manual end price (if auto_detect_pivots=False)
            price_divisions: List of price division ratios (0.0-1.0)
            time_divisions: List of time division ratios (0.0-1.0)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.start_pivot_type = start_pivot_type
        self.end_pivot_type = end_pivot_type
        self.lookback_period = lookback_period
        self.auto_detect_pivots = auto_detect_pivots
        self.manual_start_idx = manual_start_idx
        self.manual_start_price = manual_start_price
        self.manual_end_idx = manual_end_idx
        self.manual_end_price = manual_end_price
        self.price_divisions = sorted(price_divisions)
        self.time_divisions = sorted(time_divisions)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Box for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Box information
        """
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        result = data.copy()

        # Find start and end points
        if self.auto_detect_pivots:
            start_idx, start_price = self._find_pivot_point(result, self.start_pivot_type, self.lookback_period)
            end_idx, end_price = self._find_pivot_point(result, self.end_pivot_type, self.lookback_period)
        else:
            # Use manual pivots
            if self.manual_start_idx is not None and self.manual_start_price is not None and \
               self.manual_end_idx is not None and self.manual_end_price is not None:
                if 0 <= self.manual_start_idx < len(result) and 0 <= self.manual_end_idx < len(result):
                    start_idx = result.index[self.manual_start_idx]
                    start_price = self.manual_start_price
                    end_idx = result.index[self.manual_end_idx]
                    end_price = self.manual_end_price
                else:
                    # Invalid manual indices
                    return result
            else:
                # Missing manual parameters
                return result

        if start_idx is None or end_idx is None:
            # Could not find valid pivots
            return result

        # Ensure start comes before end
        if result.index.get_loc(start_idx) > result.index.get_loc(end_idx):
            start_idx, end_idx = end_idx, start_idx
            start_price, end_price = end_price, start_price

        # Get positions
        start_pos = result.index.get_loc(start_idx)
        end_pos = result.index.get_loc(end_idx)

        # Calculate box dimensions
        time_range = end_pos - start_pos
        price_range = end_price - start_price

        # Add box boundaries
        result['gann_box_start_idx'] = False
        result['gann_box_end_idx'] = False
        result.loc[start_idx, 'gann_box_start_idx'] = True
        result.loc[end_idx, 'gann_box_end_idx'] = True
        result['gann_box_start_price'] = start_price
        result['gann_box_end_price'] = end_price

        # Calculate and add price divisions
        for ratio in self.price_divisions:
            price_level = start_price + (price_range * ratio)
            col_name = f'gann_box_price_{int(ratio * 100)}'
            result[col_name] = price_level

        # Calculate and add time divisions (as markers)
        for ratio in self.time_divisions:
            time_pos = start_pos + int(time_range * ratio)
            if 0 <= time_pos < len(result):
                time_idx = result.index[time_pos]
                col_name = f'gann_box_time_{int(ratio * 100)}'
                if col_name not in result.columns:
                    result[col_name] = False
                result.loc[time_idx, col_name] = True

        return result

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Box',
            'description': 'Creates a box between two points with price and time divisions',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'start_pivot_type',
                    'description': 'Type of pivot for box start',
                    'type': 'string',
                    'default': 'major_low',
                    'options': ['major_low', 'major_high']
                },
                {
                    'name': 'end_pivot_type',
                    'description': 'Type of pivot for box end',
                    'type': 'string',
                    'default': 'major_high',
                    'options': ['major_low', 'major_high']
                },
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for finding pivots',
                    'type': 'int',
                    'default': 100
                },
                {
                    'name': 'auto_detect_pivots',
                    'description': 'Whether to automatically detect the pivots',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'manual_start_idx',
                    'description': 'Manual start index position',
                    'type': 'int',
                    'default': None
                },
                {
                    'name': 'manual_start_price',
                    'description': 'Manual start price',
                    'type': 'float',
                    'default': None
                },
                {
                    'name': 'manual_end_idx',
                    'description': 'Manual end index position',
                    'type': 'int',
                    'default': None
                },
                {
                    'name': 'manual_end_price',
                    'description': 'Manual end price',
                    'type': 'float',
                    'default': None
                },
                {
                    'name': 'price_divisions',
                    'description': 'List of price division ratios (0.0-1.0)',
                    'type': 'list',
                    'default': [0.25, 0.382, 0.5, 0.618, 0.75]
                },
                {
                    'name': 'time_divisions',
                    'description': 'List of time division ratios (0.0-1.0)',
                    'type': 'list',
                    'default': [0.25, 0.382, 0.5, 0.618, 0.75]
                }
            ]
        }
