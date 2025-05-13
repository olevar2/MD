"""
Gann Angles Module.

This module provides implementation of Gann angles.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from core.base_5 import BaseGannIndicator


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class GannAngles(BaseGannIndicator):
    """
    Gann Angles

    Gann Angles are diagonal lines that represent different rates of
    price movement over time, based on the concept that prices tend
    to move at specific angles.

    The primary Gann angle is the 1x1 (45°) line, which represents
    one unit of price for one unit of time.
    """

    def __init__(self, pivot_type: str='swing_low', angle_types: Optional[
        List[str]]=None, lookback_period: int=100, price_scaling: float=1.0,
        projection_bars: int=50, **kwargs):
        """
        Initialize Gann Angles indicator.

        Args:
            pivot_type: Type of pivot to use ('swing_low', 'swing_high', 'recent_low', 'recent_high', 'major_low', 'major_high')
            angle_types: List of Gann angles to calculate (None = all angles)
            lookback_period: Number of bars to look back for finding pivot
            price_scaling: Price scaling factor for angle calculation
            projection_bars: Number of bars to project angles into the future
            **kwargs: Additional parameters

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(**kwargs)
        self.angle_ratios = {'1x8': 1 / 8, '1x4': 1 / 4, '1x3': 1 / 3,
            '1x2': 1 / 2, '1x1': 1, '2x1': 2, '3x1': 3, '4x1': 4, '8x1': 8}
        valid_pivot_types = ['swing_low', 'swing_high', 'recent_low',
            'recent_high', 'major_low', 'major_high']
        if pivot_type not in valid_pivot_types:
            raise ValueError(
                f'Invalid pivot_type: {pivot_type}. Must be one of {valid_pivot_types}'
                )
        self.pivot_type = pivot_type
        if not isinstance(lookback_period, int):
            raise ValueError('lookback_period must be an integer')
        if lookback_period <= 0:
            raise ValueError('lookback_period must be positive')
        self.lookback_period = lookback_period
        if not isinstance(price_scaling, (int, float)):
            raise ValueError('price_scaling must be a number')
        if price_scaling <= 0:
            raise ValueError('price_scaling must be positive')
        self.price_scaling = float(price_scaling)
        if not isinstance(projection_bars, int):
            raise ValueError('projection_bars must be an integer')
        if projection_bars < 0:
            raise ValueError('projection_bars must be non-negative')
        self.projection_bars = projection_bars
        all_angles = list(self.angle_ratios.keys())
        if angle_types is None:
            self.angle_types = all_angles
        else:
            if not isinstance(angle_types, list):
                raise ValueError('angle_types must be a list or None')
            invalid_angles = [a for a in angle_types if a not in all_angles]
            if invalid_angles:
                print(
                    f'Warning: Invalid angle types {invalid_angles} will be ignored'
                    )
            self.angle_types = [a for a in angle_types if a in all_angles]
            if not self.angle_types:
                print(
                    "Warning: No valid angle types provided, using default '1x1' angle"
                    )
                self.angle_types = ['1x1']

    @with_exception_handling
    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate Gann Angles for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Angle values

        Raises:
            ValueError: If data is invalid or required columns are missing
            TypeError: If data is not a pandas DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError('data must be a pandas DataFrame')
        if data.empty:
            raise ValueError('data DataFrame is empty')
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        result = data.copy()
        try:
            pivot_idx, pivot_price = self._find_pivot_point(result, self.
                pivot_type, self.lookback_period)
            if pivot_idx is None or pivot_price is None:
                print(
                    f'Warning: Could not find a valid pivot point for {self.pivot_type}'
                    )
                return result
            try:
                pivot_pos = result.index.get_loc(pivot_idx)
            except KeyError:
                print(
                    f'Warning: Pivot index {pivot_idx} not found in data index'
                    )
                return result
            last_pos = len(result) - 1
            for angle_type in self.angle_types:
                col_name_up = f'gann_angle_up_{angle_type}'
                col_name_down = f'gann_angle_down_{angle_type}'
                result[col_name_up] = None
                result[col_name_down] = None
            for angle_type in self.angle_types:
                try:
                    result = self._calculate_angle(result, pivot_pos,
                        pivot_price, angle_type, last_pos)
                except Exception as e:
                    print(
                        f'Warning: Error calculating {angle_type} angle: {str(e)}'
                        )
            result['gann_angle_pivot_idx'] = False
            result.loc[pivot_idx, 'gann_angle_pivot_idx'] = True
            result['gann_angle_pivot_price'] = pivot_price
        except Exception as e:
            print(f'Error calculating Gann angles: {str(e)}')
            for angle_type in self.angle_types:
                col_name_up = f'gann_angle_up_{angle_type}'
                col_name_down = f'gann_angle_down_{angle_type}'
                result[col_name_up] = None
                result[col_name_down] = None
        return result

    def _calculate_angle(self, data: pd.DataFrame, pivot_pos: int,
        pivot_price: float, angle_type: str, last_pos: int) ->pd.DataFrame:
        """
        Calculate a specific Gann angle line.
        """
        ratio = self.angle_ratios[angle_type]
        slope = ratio * self.price_scaling
        col_name_up = f'gann_angle_up_{angle_type}'
        col_name_down = f'gann_angle_down_{angle_type}'
        data[col_name_up] = None
        data[col_name_down] = None
        for i in range(last_pos + self.projection_bars + 1):
            time_diff = i - pivot_pos
            up_angle_price = pivot_price + slope * time_diff
            down_angle_price = pivot_price - slope * time_diff
            if i <= last_pos:
                data.iloc[i, data.columns.get_loc(col_name_up)
                    ] = up_angle_price
                data.iloc[i, data.columns.get_loc(col_name_down)
                    ] = down_angle_price
        return data

    @classmethod
    def get_info(cls) ->Dict[str, Any]:
        """Get indicator information."""
        return {'name': 'Gann Angles', 'description':
            'Calculates Gann angle lines from a pivot point', 'category':
            cls.category, 'parameters': [{'name': 'pivot_type',
            'description': 'Type of pivot point to use', 'type': 'string',
            'default': 'swing_low', 'options': ['swing_low', 'swing_high',
            'recent_low', 'recent_high']}, {'name': 'angle_types',
            'description': 'List of Gann angles to calculate', 'type':
            'list', 'default': None}, {'name': 'lookback_period',
            'description': 'Number of bars to look back for finding pivot',
            'type': 'int', 'default': 100}, {'name': 'price_scaling',
            'description': 'Price scaling factor for angle calculation',
            'type': 'float', 'default': 1.0}, {'name': 'projection_bars',
            'description':
            'Number of bars to project angles into the future', 'type':
            'int', 'default': 50}]}
