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


class GannAngleType(Enum):
    """Standard Gann angles"""
    ANGLE_1X1 = "1x1"     # 45°
    ANGLE_1X2 = "1x2"     # 26.57°
    ANGLE_1X3 = "1x3"     # 18.43°
    ANGLE_1X4 = "1x4"     # 14.04°
    ANGLE_1X8 = "1x8"     # 7.13°
    ANGLE_2X1 = "2x1"     # 63.43°
    ANGLE_3X1 = "3x1"     # 71.57°
    ANGLE_4X1 = "4x1"     # 75.96°
    ANGLE_8X1 = "8x1"     # 82.87°


class GannAngles(AdvancedAnalysisBase):
    """
    Gann Angles
    
    Calculates Gann angles from significant price points which can be used to
    identify support, resistance, and potential trend changes.
    """
    
    def __init__(
        self,
        name: str = "GannAngles",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize Gann Angles analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {
            "price_column": "close",
            "start_from_significant_point": True,
            "angles": ["1x1", "1x2", "1x3", "1x4", "1x8", "2x1", "3x1", "4x1", "8x1"],
            "price_scale": 1.0,  # Price scaling factor for angle calculation
            "time_scale": 1.0,   # Time scaling factor for angle calculation
            "projection_bars": 20
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann angles
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Gann angle levels
        """
        result_df = df.copy()
        
        # Determine the starting point
        if self.parameters["start_from_significant_point"]:
            start_idx, start_price = self._find_significant_point(result_df)
        else:
            start_idx = 0
            start_price = result_df.iloc[0][self.parameters["price_column"]]
        
        # Get scaling factors
        price_scale = self.parameters["price_scale"]
        time_scale = self.parameters["time_scale"]
        
        # Add column for the starting point
        result_df["gann_start_point"] = np.nan
        result_df.iloc[start_idx, result_df.columns.get_loc("gann_start_point")] = start_price
        
        # Calculate Gann angles
        for angle_name in self.parameters["angles"]:
            angle_column = f"gann_angle_{angle_name}"
            result_df[angle_column] = np.nan
            
            # Parse the angle ratio (e.g., "1x2" means 1 price unit per 2 time units)
            try:
                price_units, time_units = angle_name.split('x')
                price_ratio = float(price_units)
                time_ratio = float(time_units)
            except ValueError:
                # Skip invalid angle name
                continue
            
            # Calculate the angle values for each bar from the start point
            for i in range(start_idx, len(result_df)):
                time_diff = i - start_idx
                
                if time_diff == 0:
                    angle_value = start_price
                else:
                    # Calculate price change based on Gann angle
                    price_change = (time_diff * time_scale * price_ratio) / (time_ratio * price_scale)
                    
                    # Direction depends on whether we're starting from a high or low
                    if self.parameters["start_from_significant_point"] and start_price == result_df["high"].max():
                        # Starting from a high, so angles trend down
                        angle_value = start_price - price_change
                    else:
                        # Starting from a low or default start, so angles trend up
                        angle_value = start_price + price_change
                
                result_df.iloc[i, result_df.columns.get_loc(angle_column)] = angle_value
        
        return result_df
    
    def _find_significant_point(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Find a significant price point (high or low) to start Gann angles
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (index, price) for the significant point
        """
        # Significant point could be recent high or low
        high_val = df["high"].max()
        low_val = df["low"].min()
        
        high_idx = df["high"].idxmax()
        low_idx = df["low"].idxmin()
        
        # Convert to integer index location
        high_idx_loc = df.index.get_loc(high_idx)
        low_idx_loc = df.index.get_loc(low_idx)
        
        # Use the more recent significant point
        if high_idx_loc > low_idx_loc:
            return high_idx_loc, high_val
        else:
            return low_idx_loc, low_val
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Gann Angles',
            'description': 'Calculates Gann angles from significant price points',
            'category': 'gann',
            'parameters': [
                {
                    'name': 'price_column',
                    'description': 'Column to use for price data',
                    'type': 'str',
                    'default': 'close'
                },
                {
                    'name': 'start_from_significant_point',
                    'description': 'Start angles from significant high or low point',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'angles',
                    'description': 'List of Gann angles to calculate',
                    'type': 'list',
                    'default': ["1x1", "1x2", "1x3", "1x4", "1x8", "2x1", "3x1", "4x1", "8x1"]
                },
                {
                    'name': 'price_scale',
                    'description': 'Price scaling factor for angle calculation',
                    'type': 'float',
                    'default': 1.0
                },
                {
                    'name': 'time_scale',
                    'description': 'Time scaling factor for angle calculation',
                    'type': 'float',
                    'default': 1.0
                }
            ]
        }


class GannFan(AdvancedAnalysisBase):
    """
    Gann Fan
    
    Creates a fan of Gann angles emanating from a significant price point,
    useful for identifying potential support and resistance levels.
    """
    
    def __init__(
        self,
        name: str = "GannFan",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize Gann Fan analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {
            "price_column": "close",
            "start_from_significant_point": True,
            "angles": ["1x1", "1x2", "1x3", "1x4", "1x8", "2x1", "3x1", "4x1", "8x1"],
            "price_scale": 1.0,
            "time_scale": 1.0
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann fan lines
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Gann fan lines
        """
        # Gann Fan is essentially a collection of Gann angles from a specific point
        # We leverage the GannAngles implementation
        gann_angles = GannAngles(
            name="GannFanAngles",
            parameters=self.parameters
        )
        
        result_df = gann_angles.calculate(df)
        
        # Rename columns to indicate this is a fan
        old_columns = [col for col in result_df.columns if col.startswith("gann_angle_")]
        for old_col in old_columns:
            new_col = old_col.replace("gann_angle_", "gann_fan_")
            result_df[new_col] = result_df[old_col]
            result_df = result_df.drop(old_col, axis=1)
            
        if "gann_start_point" in result_df.columns:
            result_df = result_df.rename(columns={"gann_start_point": "gann_fan_start"})
        
        return result_df
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Gann Fan',
            'description': 'Creates a fan of Gann angles from a significant price point',
            'category': 'gann',
            'parameters': [
                {
                    'name': 'price_column',
                    'description': 'Column to use for price data',
                    'type': 'str',
                    'default': 'close'
                },
                {
                    'name': 'start_from_significant_point',
                    'description': 'Start fan from significant high or low point',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'angles',
                    'description': 'List of Gann angles to include in fan',
                    'type': 'list',
                    'default': ["1x1", "1x2", "1x3", "1x4", "1x8", "2x1", "3x1", "4x1", "8x1"]
                },
                {
                    'name': 'price_scale',
                    'description': 'Price scaling factor for angle calculation',
                    'type': 'float',
                    'default': 1.0
                },
                {
                    'name': 'time_scale',
                    'description': 'Time scaling factor for angle calculation',
                    'type': 'float',
                    'default': 1.0
                }
            ]
        }


class GannSquare9(AdvancedAnalysisBase):
    """
    Gann Square of 9
    
    The Gann Square of 9 is a mathematical calculator used to identify key price levels
    and time cycles in financial markets. It creates a spiral of numbers starting from 
    a central value (usually a significant price point) and arranges them in a square-shaped
    spiral pattern. Numbers on the same angle (0°, 45°, 90°, 135°, 180°, etc.) from the 
    central value are considered to have harmonic relationships.
    """
    
    def __init__(
        self,
        name: str = "GannSquare9",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize Gann Square of 9 analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {
            "price_column": "close",
            "central_value_method": "auto",  # "auto", "last_close", "pivot_high", "pivot_low", "custom"
            "custom_central_value": None,
            "square_size": 9,  # Size of the Gann square (usually 9)
            "angle_intervals": [0, 45, 90, 135, 180, 225, 270, 315],  # Angles to analyze
            "price_scale": 1.0,  # Scale factor for prices
            "levels_to_calculate": 5,  # Number of levels to calculate in each direction
            "calculate_natural_squares": True,  # Include natural square values (1, 4, 9, 16, etc.)
            "cardinal_directions": True  # Include cardinal directions (N, S, E, W)
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Square of 9 support and resistance levels
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Gann Square of 9 levels
        """
        result_df = df.copy()
        
        # Determine central value (starting point for the square of 9)
        central_value = self._determine_central_value(df)
        
        if central_value is None:
            # Can't calculate without a central value
            return result_df
        
        # Calculate support and resistance levels based on the Square of 9
        levels = self._calculate_square9_levels(central_value)
        
        # Add central value and levels to result dataframe
        result_df["gann_sq9_central"] = central_value
        
        # Add each level to the dataframe
        for angle_name, level_values in levels.items():
            for level_idx, level_value in enumerate(level_values):
                col_name = f"gann_sq9_{angle_name.replace(' ', '_')}_{level_idx+1}"
                result_df[col_name] = level_value
                
        # Add natural squares if requested
        if self.parameters["calculate_natural_squares"]:
            natural_squares = self._calculate_natural_squares(central_value)
            
            for i, square_val in enumerate(natural_squares):
                result_df[f"gann_sq9_square_{i+1}"] = square_val
                
        # Add cardinal directions if requested
        if self.parameters["cardinal_directions"]:
            cardinal_levels = self._calculate_cardinal_levels(central_value)
            
            for direction, level_value in cardinal_levels.items():
                result_df[f"gann_sq9_{direction}"] = level_value
        
        return result_df
    
    def _determine_central_value(self, df: pd.DataFrame) -> float:
        """
        Determine the central value for the Square of 9 based on parameters
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Central value for Square of 9 calculations
        """
        method = self.parameters["central_value_method"].lower()
        price_col = self.parameters["price_column"]
        
        if method == "custom" and self.parameters["custom_central_value"] is not None:
            # Use custom user-provided value
            return self.parameters["custom_central_value"]
        
        elif method == "last_close" and len(df) > 0:
            # Use the last close price
            return df.iloc[-1][price_col]
        
        elif method == "pivot_high":
            # Use the highest high in the dataset
            return df["high"].max()
        
        elif method == "pivot_low":
            # Use the lowest low in the dataset
            return df["low"].min()
        
        else:  # "auto" or any other value
            # Auto-detect based on recent market action
            if len(df) >= 10:
                # Use the average of recent price action
                recent_data = df.iloc[-10:]
                recent_high = recent_data["high"].max()
                recent_low = recent_data["low"].min()
                recent_close = recent_data.iloc[-1][price_col]
                
                # Use weighted average: close has more weight than high/low
                return (recent_close * 2 + recent_high + recent_low) / 4
            
            elif len(df) > 0:
                # Not enough data, use last close
                return df.iloc[-1][price_col]
        
        # Default fallback if no suitable value found
        return None
    
    def _calculate_square9_levels(self, central_value: float) -> Dict[str, List[float]]:
        """
        Calculate price levels based on the Gann Square of 9
        
        Args:
            central_value: The center value of the square
            
        Returns:
            Dictionary with angle-based levels
        """
        levels = {}
        price_scale = self.parameters["price_scale"]
        num_levels = self.parameters["levels_to_calculate"]
        
        # For each angle interval, calculate the corresponding levels
        for angle in self.parameters["angle_intervals"]:
            angle_name = f"{angle}deg"
            levels[angle_name] = []
            
            for level in range(1, num_levels + 1):
                # Calculate the square of 9 number at this angle and level
                # Each level corresponds to one complete spiral around the square
                position = self._get_position_at_angle_and_level(angle, level)
                
                # Calculate the actual price value
                # Square of 9 formula: central_value + position * price_scale
                level_price = central_value + (position * price_scale)
                levels[angle_name].append(level_price)
        
        return levels
    
    def _get_position_at_angle_and_level(self, angle: int, level: int) -> int:
        """
        Get the position number at a specific angle and level in the Square of 9
        
        Args:
            angle: The angle in degrees (0-360)
            level: The level (spiral number)
            
        Returns:
            The position number at that angle and level
        """
        # Normalize angle to 0-360 range
        angle = angle % 360
        
        # Calculate the position based on mathematical formulas from Gann's Square of 9
        # This is an approximation that works well for most practical purposes
        
        # Each level completes a full rotation around the square
        # The formula gives an approximation of the value at that position
        base = (2 * level - 1) ** 2  # The first number in the level
        
        # Calculate position within the level based on angle
        if 0 <= angle < 90:
            # Right side of square (East to North)
            position = base + int((angle / 90.0) * (2 * level))
        elif 90 <= angle < 180:
            # Top side of square (North to West)
            position = base + (2 * level) + int(((angle - 90) / 90.0) * (2 * level))
        elif 180 <= angle < 270:
            # Left side of square (West to South)
            position = base + (4 * level) - int(((angle - 180) / 90.0) * (2 * level))
        else:  # 270 <= angle < 360
            # Bottom side of square (South to East)
            position = base + (2 * level) - int(((angle - 270) / 90.0) * (2 * level))
        
        return position
    
    def _calculate_natural_squares(self, central_value: float) -> List[float]:
        """
        Calculate natural square values from the central value
        
        Args:
            central_value: The center value of the square
            
        Returns:
            List of natural square values
        """
        natural_squares = []
        price_scale = self.parameters["price_scale"]
        num_levels = self.parameters["levels_to_calculate"]
        
        for i in range(1, num_levels + 1):
            # Add both positive and negative square values
            positive_square = central_value + (i ** 2) * price_scale
            negative_square = central_value - (i ** 2) * price_scale
            
            natural_squares.append(positive_square)
            if i > 0:  # Don't add negative for the first level (would be duplicate of central value)
                natural_squares.append(negative_square)
        
        return natural_squares
    
    def _calculate_cardinal_levels(self, central_value: float) -> Dict[str, float]:
        """
        Calculate cardinal direction values (N, S, E, W) from the central value
        
        Args:
            central_value: The center value of the square
            
        Returns:
            Dictionary with cardinal direction values
        """
        cardinal_levels = {}
        price_scale = self.parameters["price_scale"]
        num_levels = self.parameters["levels_to_calculate"]
        
        # Cardinal directions correspond to specific angles
        # North = 90°, East = 0°, South = 270°, West = 180°
        directions = {
            "north": 90,
            "east": 0,
            "south": 270,
            "west": 180
        }
        
        for direction, angle in directions.items():
            # Use the highest level for cardinal directions
            position = self._get_position_at_angle_and_level(angle, num_levels)
            level_price = central_value + (position * price_scale)
            cardinal_levels[direction] = level_price
        
        return cardinal_levels
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Gann Square of 9',
            'description': 'Calculates support and resistance levels based on W.D. Gann\'s Square of 9',
            'category': 'gann',
            'parameters': [
                {
                    'name': 'price_column',
                    'description': 'Column to use for price data',
                    'type': 'str',
                    'default': 'close'
                },
                {
                    'name': 'central_value_method',
                    'description': 'Method to determine central value',
                    'type': 'str',
                    'default': 'auto',
                    'options': ['auto', 'last_close', 'pivot_high', 'pivot_low', 'custom']
                },
                {
                    'name': 'custom_central_value',
                    'description': 'Custom value to use as central point',
                    'type': 'float',
                    'default': None
                },
                {
                    'name': 'angle_intervals',
                    'description': 'Angles to analyze in the Square of 9',
                    'type': 'list',
                    'default': [0, 45, 90, 135, 180, 225, 270, 315]
                },
                {
                    'name': 'price_scale',
                    'description': 'Price scaling factor for level calculation',
                    'type': 'float',
                    'default': 1.0
                },
                {
                    'name': 'levels_to_calculate',
                    'description': 'Number of levels to calculate in each direction',
                    'type': 'int',
                    'default': 5
                }
            ]
        }
