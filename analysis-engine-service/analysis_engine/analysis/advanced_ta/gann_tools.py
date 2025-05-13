"""
Gann Analysis Tools Module

This module provides technical analysis tools based on W.D. Gann's concepts,
including Gann angles, Gann fan, Gann square, and Gann grid.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import math
from enum import Enum
from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class GannAngleType(Enum):
    """Standard Gann angles"""
    ANGLE_1X1 = '1x1'
    ANGLE_1X2 = '1x2'
    ANGLE_1X3 = '1x3'
    ANGLE_1X4 = '1x4'
    ANGLE_1X8 = '1x8'
    ANGLE_2X1 = '2x1'
    ANGLE_3X1 = '3x1'
    ANGLE_4X1 = '4x1'
    ANGLE_8X1 = '8x1'


class GannAngles(AdvancedAnalysisBase):
    """
    Gann Angles
    
    Calculates Gann angles from significant price points which can be used to
    identify support, resistance, and potential trend changes.
    """

    def __init__(self, name: str='GannAngles', parameters: Dict[str, Any]=None
        ):
        """
        Initialize Gann Angles analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {'price_column': 'close',
            'start_from_significant_point': True, 'angles': ['1x1', '1x2',
            '1x3', '1x4', '1x8', '2x1', '3x1', '4x1', '8x1'], 'price_scale':
            1.0, 'time_scale': 1.0, 'projection_bars': 20}
        if parameters:
            default_params.update(parameters)
        super().__init__(name, default_params)

    @with_exception_handling
    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate Gann angles
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Gann angle levels
        """
        result_df = df.copy()
        if self.parameters['start_from_significant_point']:
            start_idx, start_price = self._find_significant_point(result_df)
        else:
            start_idx = 0
            start_price = result_df.iloc[0][self.parameters['price_column']]
        price_scale = self.parameters['price_scale']
        time_scale = self.parameters['time_scale']
        result_df['gann_start_point'] = np.nan
        result_df.iloc[start_idx, result_df.columns.get_loc('gann_start_point')
            ] = start_price
        for angle_name in self.parameters['angles']:
            angle_column = f'gann_angle_{angle_name}'
            result_df[angle_column] = np.nan
            try:
                price_units, time_units = angle_name.split('x')
                price_ratio = float(price_units)
                time_ratio = float(time_units)
            except ValueError:
                continue
            for i in range(start_idx, len(result_df)):
                time_diff = i - start_idx
                if time_diff == 0:
                    angle_value = start_price
                else:
                    price_change = time_diff * time_scale * price_ratio / (
                        time_ratio * price_scale)
                    if self.parameters['start_from_significant_point'
                        ] and start_price == result_df['high'].max():
                        angle_value = start_price - price_change
                    else:
                        angle_value = start_price + price_change
                result_df.iloc[i, result_df.columns.get_loc(angle_column)
                    ] = angle_value
        return result_df

    def _find_significant_point(self, df: pd.DataFrame) ->Tuple[int, float]:
        """
        Find a significant price point (high or low) to start Gann angles
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (index, price) for the significant point
        """
        high_val = df['high'].max()
        low_val = df['low'].min()
        high_idx = df['high'].idxmax()
        low_idx = df['low'].idxmin()
        high_idx_loc = df.index.get_loc(high_idx)
        low_idx_loc = df.index.get_loc(low_idx)
        if high_idx_loc > low_idx_loc:
            return high_idx_loc, high_val
        else:
            return low_idx_loc, low_val

    @classmethod
    def get_info(cls) ->Dict[str, Any]:
        """Get indicator information"""
        return {'name': 'Gann Angles', 'description':
            'Calculates Gann angles from significant price points',
            'category': 'gann', 'parameters': [{'name': 'price_column',
            'description': 'Column to use for price data', 'type': 'str',
            'default': 'close'}, {'name': 'start_from_significant_point',
            'description':
            'Start angles from significant high or low point', 'type':
            'bool', 'default': True}, {'name': 'angles', 'description':
            'List of Gann angles to calculate', 'type': 'list', 'default':
            ['1x1', '1x2', '1x3', '1x4', '1x8', '2x1', '3x1', '4x1', '8x1']
            }, {'name': 'price_scale', 'description':
            'Price scaling factor for angle calculation', 'type': 'float',
            'default': 1.0}, {'name': 'time_scale', 'description':
            'Time scaling factor for angle calculation', 'type': 'float',
            'default': 1.0}]}


class GannFan(AdvancedAnalysisBase):
    """
    Gann Fan
    
    Creates a fan of Gann angles emanating from a significant price point,
    useful for identifying potential support and resistance levels.
    """

    def __init__(self, name: str='GannFan', parameters: Dict[str, Any]=None):
        """
        Initialize Gann Fan analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {'price_column': 'close',
            'start_from_significant_point': True, 'angles': ['1x1', '1x2',
            '1x3', '1x4', '1x8', '2x1', '3x1', '4x1', '8x1'], 'price_scale':
            1.0, 'time_scale': 1.0}
        if parameters:
            default_params.update(parameters)
        super().__init__(name, default_params)

    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate Gann fan lines
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Gann fan lines
        """
        gann_angles = GannAngles(name='GannFanAngles', parameters=self.
            parameters)
        result_df = gann_angles.calculate(df)
        old_columns = [col for col in result_df.columns if col.startswith(
            'gann_angle_')]
        for old_col in old_columns:
            new_col = old_col.replace('gann_angle_', 'gann_fan_')
            result_df[new_col] = result_df[old_col]
            result_df = result_df.drop(old_col, axis=1)
        if 'gann_start_point' in result_df.columns:
            result_df = result_df.rename(columns={'gann_start_point':
                'gann_fan_start'})
        return result_df

    @classmethod
    def get_info(cls) ->Dict[str, Any]:
        """Get indicator information"""
        return {'name': 'Gann Fan', 'description':
            'Creates a fan of Gann angles from a significant price point',
            'category': 'gann', 'parameters': [{'name': 'price_column',
            'description': 'Column to use for price data', 'type': 'str',
            'default': 'close'}, {'name': 'start_from_significant_point',
            'description': 'Start fan from significant high or low point',
            'type': 'bool', 'default': True}, {'name': 'angles',
            'description': 'List of Gann angles to include in fan', 'type':
            'list', 'default': ['1x1', '1x2', '1x3', '1x4', '1x8', '2x1',
            '3x1', '4x1', '8x1']}, {'name': 'price_scale', 'description':
            'Price scaling factor for angle calculation', 'type': 'float',
            'default': 1.0}, {'name': 'time_scale', 'description':
            'Time scaling factor for angle calculation', 'type': 'float',
            'default': 1.0}]}


class GannSquare9(AdvancedAnalysisBase):
    """
    Gann Square of 9
    
    The Gann Square of 9 is a mathematical calculator used to identify key price levels
    and time cycles in financial markets. It creates a spiral of numbers starting from 
    a central value (usually a significant price point) and arranges them in a square-shaped
    spiral pattern. Numbers on the same angle (0°, 45°, 90°, 135°, 180°, etc.) from the 
    central value are considered to have harmonic relationships.
    """

    def __init__(self, name: str='GannSquare9', parameters: Dict[str, Any]=None
        ):
        """
        Initialize Gann Square of 9 analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {'price_column': 'close', 'central_value_method':
            'auto', 'custom_central_value': None, 'square_size': 9,
            'angle_intervals': [0, 45, 90, 135, 180, 225, 270, 315],
            'price_scale': 1.0, 'levels_to_calculate': 5,
            'calculate_natural_squares': True, 'cardinal_directions': True}
        if parameters:
            default_params.update(parameters)
        super().__init__(name, default_params)

    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate Gann Square of 9 support and resistance levels
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Gann Square of 9 levels
        """
        result_df = df.copy()
        central_value = self._determine_central_value(df)
        if central_value is None:
            return result_df
        levels = self._calculate_square9_levels(central_value)
        result_df['gann_sq9_central'] = central_value
        for angle_name, level_values in levels.items():
            for level_idx, level_value in enumerate(level_values):
                col_name = (
                    f"gann_sq9_{angle_name.replace(' ', '_')}_{level_idx + 1}")
                result_df[col_name] = level_value
        if self.parameters['calculate_natural_squares']:
            natural_squares = self._calculate_natural_squares(central_value)
            for i, square_val in enumerate(natural_squares):
                result_df[f'gann_sq9_square_{i + 1}'] = square_val
        if self.parameters['cardinal_directions']:
            cardinal_levels = self._calculate_cardinal_levels(central_value)
            for direction, level_value in cardinal_levels.items():
                result_df[f'gann_sq9_{direction}'] = level_value
        return result_df

    def _determine_central_value(self, df: pd.DataFrame) ->float:
        """
        Determine the central value for the Square of 9 based on parameters
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Central value for Square of 9 calculations
        """
        method = self.parameters['central_value_method'].lower()
        price_col = self.parameters['price_column']
        if method == 'custom' and self.parameters['custom_central_value'
            ] is not None:
            return self.parameters['custom_central_value']
        elif method == 'last_close' and len(df) > 0:
            return df.iloc[-1][price_col]
        elif method == 'pivot_high':
            return df['high'].max()
        elif method == 'pivot_low':
            return df['low'].min()
        elif len(df) >= 10:
            recent_data = df.iloc[-10:]
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            recent_close = recent_data.iloc[-1][price_col]
            return (recent_close * 2 + recent_high + recent_low) / 4
        elif len(df) > 0:
            return df.iloc[-1][price_col]
        return None

    def _calculate_square9_levels(self, central_value: float) ->Dict[str,
        List[float]]:
        """
        Calculate price levels based on the Gann Square of 9
        
        Args:
            central_value: The center value of the square
            
        Returns:
            Dictionary with angle-based levels
        """
        levels = {}
        price_scale = self.parameters['price_scale']
        num_levels = self.parameters['levels_to_calculate']
        for angle in self.parameters['angle_intervals']:
            angle_name = f'{angle}deg'
            levels[angle_name] = []
            for level in range(1, num_levels + 1):
                position = self._get_position_at_angle_and_level(angle, level)
                level_price = central_value + position * price_scale
                levels[angle_name].append(level_price)
        return levels

    def _get_position_at_angle_and_level(self, angle: int, level: int) ->int:
        """
        Get the position number at a specific angle and level in the Square of 9
        
        Args:
            angle: The angle in degrees (0-360)
            level: The level (spiral number)
            
        Returns:
            The position number at that angle and level
        """
        angle = angle % 360
        base = (2 * level - 1) ** 2
        if 0 <= angle < 90:
            position = base + int(angle / 90.0 * (2 * level))
        elif 90 <= angle < 180:
            position = base + 2 * level + int((angle - 90) / 90.0 * (2 * level)
                )
        elif 180 <= angle < 270:
            position = base + 4 * level - int((angle - 180) / 90.0 * (2 *
                level))
        else:
            position = base + 2 * level - int((angle - 270) / 90.0 * (2 *
                level))
        return position

    def _calculate_natural_squares(self, central_value: float) ->List[float]:
        """
        Calculate natural square values from the central value
        
        Args:
            central_value: The center value of the square
            
        Returns:
            List of natural square values
        """
        natural_squares = []
        price_scale = self.parameters['price_scale']
        num_levels = self.parameters['levels_to_calculate']
        for i in range(1, num_levels + 1):
            positive_square = central_value + i ** 2 * price_scale
            negative_square = central_value - i ** 2 * price_scale
            natural_squares.append(positive_square)
            if i > 0:
                natural_squares.append(negative_square)
        return natural_squares

    def _calculate_cardinal_levels(self, central_value: float) ->Dict[str,
        float]:
        """
        Calculate cardinal direction values (N, S, E, W) from the central value
        
        Args:
            central_value: The center value of the square
            
        Returns:
            Dictionary with cardinal direction values
        """
        cardinal_levels = {}
        price_scale = self.parameters['price_scale']
        num_levels = self.parameters['levels_to_calculate']
        directions = {'north': 90, 'east': 0, 'south': 270, 'west': 180}
        for direction, angle in directions.items():
            position = self._get_position_at_angle_and_level(angle, num_levels)
            level_price = central_value + position * price_scale
            cardinal_levels[direction] = level_price
        return cardinal_levels

    @classmethod
    def get_info(cls) ->Dict[str, Any]:
        """Get indicator information"""
        return {'name': 'Gann Square of 9', 'description':
            "Calculates support and resistance levels based on W.D. Gann's Square of 9"
            , 'category': 'gann', 'parameters': [{'name': 'price_column',
            'description': 'Column to use for price data', 'type': 'str',
            'default': 'close'}, {'name': 'central_value_method',
            'description': 'Method to determine central value', 'type':
            'str', 'default': 'auto', 'options': ['auto', 'last_close',
            'pivot_high', 'pivot_low', 'custom']}, {'name':
            'custom_central_value', 'description':
            'Custom value to use as central point', 'type': 'float',
            'default': None}, {'name': 'angle_intervals', 'description':
            'Angles to analyze in the Square of 9', 'type': 'list',
            'default': [0, 45, 90, 135, 180, 225, 270, 315]}, {'name':
            'price_scale', 'description':
            'Price scaling factor for level calculation', 'type': 'float',
            'default': 1.0}, {'name': 'levels_to_calculate', 'description':
            'Number of levels to calculate in each direction', 'type':
            'int', 'default': 5}]}
