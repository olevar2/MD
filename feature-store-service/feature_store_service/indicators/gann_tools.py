"""
Gann Analysis Tools Module.

This module provides implementations of W.D. Gann's analytical methods
including Gann angles, squares, fans, and other geometric tools.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import math

from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.utils.swing_points import find_swing_highs, find_swing_lows # Assuming utility exists


class GannAngles(BaseIndicator):
    """
    Gann Angles
    
    Gann Angles are diagonal lines that represent different rates of 
    price movement over time, based on the concept that prices tend
    to move at specific angles.
    
    The primary Gann angle is the 1x1 (45°) line, which represents 
    one unit of price for one unit of time.
    """
    
    category = "gann"
    
    def __init__(
        self, 
        pivot_type: str = "swing_low",
        angle_types: Optional[List[str]] = None,
        lookback_period: int = 100,
        price_scaling: float = 1.0,
        projection_bars: int = 50,
        **kwargs
    ):
        """
        Initialize Gann Angles indicator.
        
        Args:
            pivot_type: Type of pivot to use ('swing_low', 'swing_high', 'recent_low', 'recent_high')
            angle_types: List of Gann angles to calculate (None = all angles)
            lookback_period: Number of bars to look back for finding pivot
            price_scaling: Price scaling factor for angle calculation
            projection_bars: Number of bars to project angles into the future
            **kwargs: Additional parameters
        """
        self.pivot_type = pivot_type
        self.lookback_period = lookback_period
        self.price_scaling = price_scaling
        self.projection_bars = projection_bars
        
        # Define available Gann angles and their ratios (price units per time unit)
        self.angle_ratios = {
            "1x8": 1/8, "1x4": 1/4, "1x3": 1/3, "1x2": 1/2, "1x1": 1,
            "2x1": 2, "3x1": 3, "4x1": 4, "8x1": 8
        }
        
        all_angles = list(self.angle_ratios.keys())
        
        if angle_types is None:
            self.angle_types = all_angles
        else:
            self.angle_types = [a for a in angle_types if a in all_angles]
            
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Angles for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Gann Angle values
        """
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Find pivot point based on specified pivot type
        pivot_idx, pivot_price = self._find_pivot_point(result)
        
        if pivot_idx is None:
            # Could not find a valid pivot point
            return result
            
        # Calculate Gann angles from the pivot point
        pivot_pos = result.index.get_loc(pivot_idx)
        last_pos = len(result) - 1
        
        for angle_type in self.angle_types:
            result = self._calculate_angle(result, pivot_pos, pivot_price, angle_type, last_pos)
        
        # Add pivot point marker
        result['gann_angle_pivot_idx'] = False
        result.loc[pivot_idx, 'gann_angle_pivot_idx'] = True
        result['gann_angle_pivot_price'] = pivot_price
        
        return result
        
    def _find_pivot_point(self, data: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        """
        Find pivot point based on the specified pivot type.
        
        Returns:
            Tuple of (pivot_index, pivot_price)
        """
        # Use only the lookback period for finding the pivot
        lookback_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data
        
        if lookback_data.empty:
            return None, None
            
        pivot_idx = None
        pivot_price = None
        
        if self.pivot_type == "swing_low":
            # Find the lowest low in the lookback period
            pivot_idx = lookback_data['low'].idxmin()
            pivot_price = lookback_data.loc[pivot_idx, 'low']
        elif self.pivot_type == "swing_high":
            # Find the highest high in the lookback period
            pivot_idx = lookback_data['high'].idxmax()
            pivot_price = lookback_data.loc[pivot_idx, 'high']
        elif self.pivot_type == "recent_low":
            # Use the most recent low
            pivot_idx = lookback_data.index[-1]
            pivot_price = lookback_data.loc[pivot_idx, 'low']
        elif self.pivot_type == "recent_high":
            # Use the most recent high
            pivot_idx = lookback_data.index[-1]
            pivot_price = lookback_data.loc[pivot_idx, 'high']
        else:
            # Default to lowest low if type is invalid
            pivot_idx = lookback_data['low'].idxmin()
            pivot_price = lookback_data.loc[pivot_idx, 'low']
            
        return pivot_idx, pivot_price

    def _calculate_angle(
        self, 
        data: pd.DataFrame, 
        pivot_pos: int, 
        pivot_price: float, 
        angle_type: str, 
        last_pos: int
    ) -> pd.DataFrame:
        """
        Calculate a specific Gann angle line.
        """
        ratio = self.angle_ratios[angle_type]
        slope = ratio * self.price_scaling
        
        col_name_up = f"gann_angle_up_{angle_type}"
        col_name_down = f"gann_angle_down_{angle_type}"
        
        data[col_name_up] = None
        data[col_name_down] = None
        
        # Calculate angle lines projecting forward and backward from the pivot
        for i in range(last_pos + self.projection_bars + 1):
            time_diff = i - pivot_pos
            
            # Calculate upward angle
            up_angle_price = pivot_price + (slope * time_diff)
            
            # Calculate downward angle
            down_angle_price = pivot_price - (slope * time_diff)
            
            if i <= last_pos:
                data.iloc[i, data.columns.get_loc(col_name_up)] = up_angle_price
                data.iloc[i, data.columns.get_loc(col_name_down)] = down_angle_price
                
        return data

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Angles',
            'description': 'Calculates Gann angle lines from a pivot point',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'pivot_type',
                    'description': 'Type of pivot point to use',
                    'type': 'string',
                    'default': 'swing_low',
                    'options': ['swing_low', 'swing_high', 'recent_low', 'recent_high']
                },
                {
                    'name': 'angle_types',
                    'description': 'List of Gann angles to calculate',
                    'type': 'list',
                    'default': None
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
                    'description': 'Number of bars to project angles into the future',
                    'type': 'int',
                    'default': 50
                }
            ]
        }


class GannFan(BaseIndicator):
    """
    Gann Fan
    
    Gann Fan consists of a set of Gann angles drawn from a significant 
    pivot point (high or low). These lines act as potential support and 
    resistance levels.
    """
    
    category = "gann"
    
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
        pivot_idx, pivot_price = self._find_pivot_point(result)
        
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
        
    def _find_pivot_point(self, data: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        """
        Find pivot point based on the specified pivot type.
        """
        lookback_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data
        
        if lookback_data.empty:
            return None, None
            
        pivot_idx = None
        pivot_price = None
        
        if self.pivot_type == "swing_low":
            pivot_idx = lookback_data['low'].idxmin()
            pivot_price = lookback_data.loc[pivot_idx, 'low']
        elif self.pivot_type == "swing_high":
            pivot_idx = lookback_data['high'].idxmax()
            pivot_price = lookback_data.loc[pivot_idx, 'high']
        else:
            # Default to lowest low if type is invalid
            pivot_idx = lookback_data['low'].idxmin()
            pivot_price = lookback_data.loc[pivot_idx, 'low']
            
        return pivot_idx, pivot_price

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


class GannSquare(BaseIndicator):
    """
    Gann Square (Square of 9, Square of 144, etc.)
    
    Gann Squares are used to identify potential support and resistance levels
    based on geometric relationships derived from price and time squares.
    The Square of 9 is the most common.
    """
    
    category = "gann"
    
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


class GannTimeCycles(BaseIndicator):
    """
    Gann Time Cycles
    
    Identifies potential future turning points in time based on Gann's cycle theories,
    often using significant past highs or lows as starting points and projecting
    forward using specific time intervals (e.g., 90, 180, 360 days/bars).
    """
    
    category = "gann"
    
    def __init__(
        self, 
        cycle_lengths: Optional[List[int]] = None,
        starting_point_type: str = "major_low",
        lookback_period: int = 200,
        auto_detect_start: bool = True,
        max_cycles: int = 5,
        **kwargs
    ):
        """
        Initialize Gann Time Cycles indicator.
        
        Args:
            cycle_lengths: List of cycle lengths in bars (e.g., [90, 180, 360])
            starting_point_type: Type of pivot to use as start ('major_low', 'major_high')
            lookback_period: Number of bars to look back for finding starting point
            auto_detect_start: Whether to automatically detect the starting point
            max_cycles: Maximum number of cycle projections for each length
            **kwargs: Additional parameters
        """
        # Default cycle lengths based on common Gann cycles (can be adapted for timeframe)
        self.cycle_lengths = cycle_lengths or [30, 60, 90, 120, 180, 270, 360]
        self.starting_point_type = starting_point_type
        self.lookback_period = lookback_period
        self.auto_detect_start = auto_detect_start
        self.max_cycles = max_cycles
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Time Cycle points for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Gann Time Cycle markers
        """
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Find starting point
        if self.auto_detect_start:
            start_pos = self._detect_starting_point(result)
        else:
            # Manual start point needs to be provided via kwargs or other means
            # For now, default to a point within the lookback period
            start_pos = max(0, len(result) - self.lookback_period)
            
        if start_pos is None:
            return result
            
        start_idx = data.index[start_pos]
        
        # Create columns for time cycle markers
        result['gann_time_cycle_point'] = 0
        result['gann_time_cycle_start'] = False
        result.loc[start_idx, 'gann_time_cycle_start'] = True
        
        # Project time cycles forward
        cycle_counter = 1
        for length in self.cycle_lengths:
            for i in range(1, self.max_cycles + 1):
                cycle_pos = start_pos + (length * i)
                if cycle_pos < len(result):
                    cycle_idx = result.index[cycle_pos]
                    # Mark the cycle point, potentially overwriting if multiple cycles align
                    result.loc[cycle_idx, 'gann_time_cycle_point'] = cycle_counter
                    
                    # Add specific cycle marker column
                    col_name = f'gann_time_cycle_{length}_{i}'
                    if col_name not in result.columns:
                        result[col_name] = False
                    result.loc[cycle_idx, col_name] = True
                    
            cycle_counter += 1 # Increment counter for next cycle length group
            
        return result

    def _detect_starting_point(self, data: pd.DataFrame) -> Optional[int]:
        """
        Detect a significant starting point (major high or low).
        
        Returns:
            Position (index) of the starting point in the data
        """
        lookback_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data
        
        if lookback_data.empty:
            return None
            
        start_pos = None
        if self.starting_point_type == "major_low":
            low_idx = lookback_data['low'].idxmin()
            start_pos = data.index.get_loc(low_idx)
        elif self.starting_point_type == "major_high":
            high_idx = lookback_data['high'].idxmax()
            start_pos = data.index.get_loc(high_idx)
        else:
            # Default to major low
            low_idx = lookback_data['low'].idxmin()
            start_pos = data.index.get_loc(low_idx)
            
        return start_pos

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Time Cycles',
            'description': 'Identifies potential future turning points based on Gann cycle lengths',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'cycle_lengths',
                    'description': 'List of cycle lengths in bars',
                    'type': 'list',
                    'default': [30, 60, 90, 120, 180, 270, 360]
                },
                {
                    'name': 'starting_point_type',
                    'description': 'Type of pivot to use as start',
                    'type': 'string',
                    'default': 'major_low',
                    'options': ['major_low', 'major_high']
                },
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for finding starting point',
                    'type': 'int',
                    'default': 200
                },
                {
                    'name': 'auto_detect_start',
                    'description': 'Whether to automatically detect the starting point',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'max_cycles',
                    'description': 'Maximum number of cycle projections for each length',
                    'type': 'int',
                    'default': 5
                }
            ]
        }


# --- Start of New Advanced Gann Tools ---

class GannGrid(BaseIndicator):
    """
    Gann Grid

    Overlays a grid on the chart based on a significant pivot point.
    The grid lines are drawn at specific price and time intervals derived
    from the pivot, often using Gann angles or fixed increments.
    Helps identify potential support/resistance and time turning points.
    """

    category = "gann"

    def __init__(
        self,
        pivot_type: str = "swing_low",
        lookback_period: int = 100,
        price_interval: Optional[float] = None, # e.g., price units per grid line
        time_interval: Optional[int] = None,   # e.g., bars per grid line
        auto_interval: bool = True, # Automatically determine intervals based on volatility/range
        num_price_lines: int = 5, # Number of lines above/below pivot
        num_time_lines: int = 10, # Number of lines forward/backward from pivot
        price_scaling: float = 1.0, # Optional scaling for price intervals if auto-calculating
        **kwargs
    ):
        """
        Initialize Gann Grid indicator.

        Args:
            pivot_type: Type of pivot to use ('swing_low', 'swing_high')
            lookback_period: Number of bars to look back for finding pivot
            price_interval: Fixed price distance between horizontal grid lines
            time_interval: Fixed time distance (bars) between vertical grid lines
            auto_interval: If True, calculate intervals based on recent price range/volatility
            num_price_lines: Number of horizontal lines above and below the pivot
            num_time_lines: Number of vertical lines forward and backward from the pivot
            price_scaling: Scaling factor for price intervals if auto-calculating
            **kwargs: Additional parameters
        """
        self.pivot_type = pivot_type
        self.lookback_period = lookback_period
        self.price_interval = price_interval
        self.time_interval = time_interval
        self.auto_interval = auto_interval
        self.num_price_lines = num_price_lines
        self.num_time_lines = num_time_lines
        self.price_scaling = price_scaling

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Grid lines for the given data.
        Adds columns for horizontal and vertical grid line levels/positions.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Grid information
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        result = data.copy()

        # Find pivot point
        pivot_idx, pivot_price = self._find_pivot_point(result)

        if pivot_idx is None or pivot_price is None:
            print("Warning: Could not find pivot for Gann Grid.")
            return result # Not enough data or pivot not found

        pivot_pos = result.index.get_loc(pivot_idx)

        # Determine price and time intervals
        p_interval, t_interval = self._determine_intervals(result, pivot_price)

        if p_interval <= 0 or t_interval <= 0:
            print("Warning: Invalid intervals calculated for Gann Grid.")
            return result

        # Calculate and store horizontal price grid lines
        for i in range(-self.num_price_lines, self.num_price_lines + 1):
            if i == 0: continue # Skip the pivot line itself
            price_level = pivot_price + (i * p_interval)
            col_name = f"gann_grid_price_{i:+}".replace('+', 'p').replace('-', 'm')
            result[col_name] = price_level

        # Calculate and store vertical time grid lines (as markers)
        result['gann_grid_time_marker'] = 0
        for i in range(-self.num_time_lines, self.num_time_lines + 1):
            if i == 0: continue
            time_pos = pivot_pos + (i * t_interval)
            if 0 <= time_pos < len(result):
                time_idx = result.index[time_pos]
                result.loc[time_idx, 'gann_grid_time_marker'] = i # Mark with relative interval number
                # Add specific marker column
                col_name = f'gann_grid_time_{i:+}'.replace('+', 'p').replace('-', 'm')
                if col_name not in result.columns:
                    result[col_name] = False
                result.loc[time_idx, col_name] = True

        # Add pivot marker
        result['gann_grid_pivot_idx'] = False
        result.loc[pivot_idx, 'gann_grid_pivot_idx'] = True
        result['gann_grid_pivot_price'] = pivot_price
        result['gann_grid_price_interval'] = p_interval
        result['gann_grid_time_interval'] = t_interval

        return result

    def _find_pivot_point(self, data: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        """Find pivot point based on the specified pivot type."""
        lookback_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data
        if len(lookback_data) < 5: return None, None

        pivot_idx, pivot_price = None, None
        try:
            if self.pivot_type == "swing_low":
                swing_lows = find_swing_lows(lookback_data['low'], n=3)
                if not swing_lows.empty:
                    pivot_idx = lookback_data.loc[swing_lows].index[-1] # Most recent swing low
                    pivot_price = lookback_data.loc[pivot_idx, 'low']
            elif self.pivot_type == "swing_high":
                swing_highs = find_swing_highs(lookback_data['high'], n=3)
                if not swing_highs.empty:
                    pivot_idx = lookback_data.loc[swing_highs].index[-1] # Most recent swing high
                    pivot_price = lookback_data.loc[pivot_idx, 'high']
        except NameError:
            print("Warning: Swing point detection utility not found. Using simple min/max for Gann Grid pivot.")
            if self.pivot_type == "swing_low":
                pivot_idx = lookback_data['low'].idxmin()
                pivot_price = lookback_data.loc[pivot_idx, 'low']
            elif self.pivot_type == "swing_high":
                pivot_idx = lookback_data['high'].idxmax()
                pivot_price = lookback_data.loc[pivot_idx, 'high']

        # Fallback if swing not found or type invalid
        if pivot_idx is None:
            pivot_idx = lookback_data['low'].idxmin() if self.pivot_type == "swing_low" else lookback_data['high'].idxmax()
            pivot_price = lookback_data.loc[pivot_idx, 'low'] if self.pivot_type == "swing_low" else lookback_data.loc[pivot_idx, 'high']

        return pivot_idx, pivot_price

    def _determine_intervals(self, data: pd.DataFrame, pivot_price: float) -> Tuple[float, int]:
        """Determine price and time intervals for the grid."""
        p_interval = self.price_interval
        t_interval = self.time_interval

        if self.auto_interval:
            lookback_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data
            if len(lookback_data) > 1:
                price_range = lookback_data['high'].max() - lookback_data['low'].min()
                avg_daily_range = (lookback_data['high'] - lookback_data['low']).mean()

                # Heuristic for price interval (e.g., fraction of range or ATR)
                if p_interval is None:
                    # Use a fraction of the average range, scaled
                    p_interval = (avg_daily_range / 4) * self.price_scaling if avg_daily_range > 0 else (pivot_price * 0.01) * self.price_scaling
                    if p_interval <= 0: p_interval = pivot_price * 0.01 # Fallback

                # Heuristic for time interval (e.g., related to price interval via 1x1 concept or fixed)
                if t_interval is None:
                    # Try to relate to price interval via scaling (Gann 1x1 idea)
                    # If price_scaling represents price units per bar for 1x1 angle
                    # Then time_interval * price_scaling = price_interval
                    if self.price_scaling > 0:
                        t_interval_calc = p_interval / self.price_scaling
                        t_interval = max(1, int(round(t_interval_calc))) # Ensure at least 1 bar
                    else:
                        t_interval = 10 # Default fixed interval if scaling is zero
            else:
                # Fallback if not enough data for auto calculation
                if p_interval is None: p_interval = pivot_price * 0.01
                if t_interval is None: t_interval = 10
        else:
            # Use fixed intervals if provided, with defaults if missing
            if p_interval is None: p_interval = pivot_price * 0.01
            if t_interval is None: t_interval = 10

        # Ensure intervals are positive
        p_interval = max(p_interval, 1e-9) # Avoid zero or negative price interval
        t_interval = max(t_interval, 1)   # Ensure time interval is at least 1

        return p_interval, t_interval

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Grid',
            'description': 'Overlays a price/time grid based on a pivot point',
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
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for finding pivot',
                    'type': 'int',
                    'default': 100
                },
                {
                    'name': 'price_interval',
                    'description': 'Fixed price distance between horizontal lines (optional)',
                    'type': 'float',
                    'default': None
                },
                {
                    'name': 'time_interval',
                    'description': 'Fixed time distance (bars) between vertical lines (optional)',
                    'type': 'int',
                    'default': None
                },
                {
                    'name': 'auto_interval',
                    'description': 'Automatically determine intervals based on volatility/range',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'num_price_lines',
                    'description': 'Number of horizontal lines above/below pivot',
                    'type': 'int',
                    'default': 5
                },
                {
                    'name': 'num_time_lines',
                    'description': 'Number of vertical lines forward/backward from pivot',
                    'type': 'int',
                    'default': 10
                },
                {
                    'name': 'price_scaling',
                    'description': 'Scaling factor for auto price interval calculation',
                    'type': 'float',
                    'default': 1.0
                }
            ]
        }


class GannBox(BaseIndicator):
    """
    Gann Box

    Draws a box between two points (typically significant highs and lows).
    The box is then divided by time and price lines based on Gann or Fibonacci ratios.
    Used to identify potential support, resistance, and time turning points within the box.
    """

    category = "gann"

    def __init__(
        self,
        point1_type: str = "swing_low",
        point2_type: str = "swing_high",
        lookback_period: int = 100,
        auto_detect_points: bool = True,
        manual_points: Optional[Dict[str, int]] = None, # {point1_idx: idx, point2_idx: idx}
        price_levels: Optional[List[float]] = None, # e.g., [0.25, 0.382, 0.5, 0.618, 0.75]
        time_levels: Optional[List[float]] = None, # e.g., [0.25, 0.382, 0.5, 0.618, 0.75]
        projection_multiplier: float = 1.0, # Extend box levels beyond point2
        **kwargs
    ):
        """
        Initialize Gann Box indicator.

        Args:
            point1_type: Type of the first point ('swing_low', 'swing_high')
            point2_type: Type of the second point ('swing_low', 'swing_high')
            lookback_period: Number of bars to look back for finding points
            auto_detect_points: Whether to automatically detect the box points
            manual_points: Dictionary with manual point indices {point1_idx, point2_idx}
            price_levels: List of fractional levels for horizontal lines (0 to 1 range)
            time_levels: List of fractional levels for vertical lines (0 to 1 range)
            projection_multiplier: Factor to extend levels beyond the box (e.g., 1.0 = no extension, 2.0 = double)
            **kwargs: Additional parameters
        """
        self.point1_type = point1_type
        self.point2_type = point2_type
        self.lookback_period = lookback_period
        self.auto_detect_points = auto_detect_points
        self.manual_points = manual_points
        self.price_levels = price_levels or [0.236, 0.382, 0.5, 0.618, 0.786]
        self.time_levels = time_levels or [0.236, 0.382, 0.5, 0.618, 0.786]
        self.projection_multiplier = max(1.0, projection_multiplier) # Ensure at least 1.0

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Box lines for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Box information
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        result = data.copy()

        # Find the two defining points of the box
        if self.auto_detect_points:
            point1_idx, point1_price, point2_idx, point2_price = self._detect_box_points(result)
        elif self.manual_points:
            idx_map = self.manual_points
            if 'point1_idx' in idx_map and 'point2_idx' in idx_map:
                p1_pos = idx_map['point1_idx']
                p2_pos = idx_map['point2_idx']
                if 0 <= p1_pos < len(data) and 0 <= p2_pos < len(data):
                    point1_idx = data.index[p1_pos]
                    point2_idx = data.index[p2_pos]
                    # Determine price based on high/low at the points
                    point1_price = data.loc[point1_idx, 'low'] if data.loc[point1_idx, 'close'] < data.loc[point2_idx, 'close'] else data.loc[point1_idx, 'high']
                    point2_price = data.loc[point2_idx, 'high'] if data.loc[point1_idx, 'close'] < data.loc[point2_idx, 'close'] else data.loc[point2_idx, 'low']
                else:
                    point1_idx, point1_price, point2_idx, point2_price = None, None, None, None
            else:
                 raise ValueError("Manual points must include 'point1_idx' and 'point2_idx' keys")
        else:
            print("Warning: No method specified for finding Gann Box points.")
            return result

        if point1_idx is None or point2_idx is None or point1_idx == point2_idx:
            print("Warning: Could not determine valid points for Gann Box.")
            return result

        # Ensure point1 is earlier than point2
        if data.index.get_loc(point1_idx) > data.index.get_loc(point2_idx):
            point1_idx, point2_idx = point2_idx, point1_idx
            point1_price, point2_price = point2_price, point1_price

        start_pos = result.index.get_loc(point1_idx)
        end_pos = result.index.get_loc(point2_idx)
        time_range = end_pos - start_pos
        price_range = abs(point2_price - point1_price)
        min_price = min(point1_price, point2_price)

        if time_range <= 0 or price_range <= 0:
            print("Warning: Invalid range for Gann Box (time or price is zero/negative).")
            return result

        # Calculate and store horizontal price levels
        for level in self.price_levels:
            price_level = min_price + (price_range * level)
            col_name = f"gann_box_price_{level:.3f}".replace('.', '_')
            # Apply level within the box duration and potentially projected
            proj_end_pos = start_pos + int(time_range * self.projection_multiplier)
            for i in range(start_pos, min(proj_end_pos + 1, len(result))):
                 result.loc[result.index[i], col_name] = price_level
            # Extend if projection goes beyond current data
            # (Simplified: just store the level, plotting handles horizontal line)
            if proj_end_pos >= len(result):
                 result[col_name] = result[col_name].ffill() # Fill forward to end

        # Calculate and store vertical time levels (as markers)
        result['gann_box_time_marker'] = 0
        for level in self.time_levels:
            time_pos = start_pos + int(round(time_range * level))
            if start_pos <= time_pos < len(result):
                time_idx = result.index[time_pos]
                marker_val = level # Mark with the level itself
                result.loc[time_idx, 'gann_box_time_marker'] = marker_val
                # Add specific marker column
                col_name = f'gann_box_time_{level:.3f}'.replace('.', '_')
                if col_name not in result.columns:
                    result[col_name] = False
                result.loc[time_idx, col_name] = True

        # Add box boundary markers
        result['gann_box_point1_idx'] = False
        result.loc[point1_idx, 'gann_box_point1_idx'] = True
        result['gann_box_point2_idx'] = False
        result.loc[point2_idx, 'gann_box_point2_idx'] = True
        result['gann_box_point1_price'] = point1_price
        result['gann_box_point2_price'] = point2_price
        result['gann_box_time_range'] = time_range
        result['gann_box_price_range'] = price_range

        return result

    def _detect_box_points(self, data: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[pd.Timestamp], Optional[float]]:
        """Detect two significant points (e.g., recent major low and high) for the box."""
        lookback_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data
        if len(lookback_data) < 5: return None, None, None, None

        point1_idx, point1_price, point2_idx, point2_price = None, None, None, None
        try:
            swing_lows = find_swing_lows(lookback_data['low'], n=3)
            swing_highs = find_swing_highs(lookback_data['high'], n=3)
            low_points = lookback_data.loc[swing_lows]
            high_points = lookback_data.loc[swing_highs]

            if low_points.empty or high_points.empty:
                return None, None, None, None

            # Use the most recent low and high from the specified types
            p1_idx_ts = low_points.index[-1] if self.point1_type == "swing_low" else high_points.index[-1]
            p1_price_val = low_points['low'].iloc[-1] if self.point1_type == "swing_low" else high_points['high'].iloc[-1]

            p2_idx_ts = high_points.index[-1] if self.point2_type == "swing_high" else low_points.index[-1]
            p2_price_val = high_points['high'].iloc[-1] if self.point2_type == "swing_high" else low_points['low'].iloc[-1]

            # Ensure points are distinct
            if p1_idx_ts != p2_idx_ts:
                point1_idx, point1_price = p1_idx_ts, p1_price_val
                point2_idx, point2_price = p2_idx_ts, p2_price_val
            else: # If types are the same or points coincide, try second most recent
                 if self.point1_type == self.point2_type:
                     if self.point1_type == "swing_low" and len(low_points) > 1:
                         point1_idx, point1_price = low_points.index[-2], low_points['low'].iloc[-2]
                         point2_idx, point2_price = low_points.index[-1], low_points['low'].iloc[-1]
                     elif self.point1_type == "swing_high" and len(high_points) > 1:
                         point1_idx, point1_price = high_points.index[-2], high_points['high'].iloc[-2]
                         point2_idx, point2_price = high_points.index[-1], high_points['high'].iloc[-1]

        except NameError:
            print("Warning: Swing point detection utility not found. Using simple min/max for Gann Box points.")
            low_idx = lookback_data['low'].idxmin()
            low_price = lookback_data.loc[low_idx, 'low']
            high_idx = lookback_data['high'].idxmax()
            high_price = lookback_data.loc[high_idx, 'high']
            point1_idx, point1_price = low_idx, low_price
            point2_idx, point2_price = high_idx, high_price

        # Fallback if detection failed
        if point1_idx is None or point2_idx is None:
             low_idx = lookback_data['low'].idxmin()
             low_price = lookback_data.loc[low_idx, 'low']
             high_idx = lookback_data['high'].idxmax()
             high_price = lookback_data.loc[high_idx, 'high']
             point1_idx, point1_price = low_idx, low_price
             point2_idx, point2_price = high_idx, high_price

        return point1_idx, point1_price, point2_idx, point2_price

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Box',
            'description': 'Draws a box with time/price levels between two points',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'point1_type',
                    'description': 'Type of the first point',
                    'type': 'string',
                    'default': 'swing_low',
                    'options': ['swing_low', 'swing_high']
                },
                {
                    'name': 'point2_type',
                    'description': 'Type of the second point',
                    'type': 'string',
                    'default': 'swing_high',
                    'options': ['swing_low', 'swing_high']
                },
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for finding points',
                    'type': 'int',
                    'default': 100
                },
                {
                    'name': 'auto_detect_points',
                    'description': 'Whether to automatically detect the box points',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'manual_points',
                    'description': 'Dictionary with manual point indices {point1_idx, point2_idx}',
                    'type': 'dict',
                    'default': None
                },
                {
                    'name': 'price_levels',
                    'description': 'List of fractional levels for horizontal lines (0 to 1)',
                    'type': 'list',
                    'default': [0.236, 0.382, 0.5, 0.618, 0.786]
                },
                {
                    'name': 'time_levels',
                    'description': 'List of fractional levels for vertical lines (0 to 1)',
                    'type': 'list',
                    'default': [0.236, 0.382, 0.5, 0.618, 0.786]
                },
                {
                    'name': 'projection_multiplier',
                    'description': 'Factor to extend levels beyond the box time',
                    'type': 'float',
                    'default': 1.0
                }
            ]
        }


class AdvancedGannSquare(BaseIndicator):
    """
    Advanced Gann Square (Square of 9, etc. with Time Squaring)

    Extends the Gann Square concept to include time projections based on the
    square's geometry. Calculates both price support/resistance levels and
    potential future time turning points derived from a pivot point.
    """

    category = "gann"

    def __init__(
        self,
        pivot_type: str = "major_low", # 'major_low', 'major_high', 'manual'
        manual_pivot_price: Optional[float] = None,
        manual_pivot_time_idx: Optional[int] = None, # Position index
        lookback_period: int = 200,
        auto_detect_pivot: bool = True,
        num_price_levels: int = 4,
        num_time_levels: int = 4,
        time_price_ratio: float = 1.0, # Scaling: Price units per Time unit (bar)
        square_increment: float = 0.125, # Increment for sqrt (0.125 for 45 deg in Sq9)
        **kwargs
    ):
        """
        Initialize Advanced Gann Square indicator.

        Args:
            pivot_type: Type of pivot ('major_low', 'major_high', 'manual')
            manual_pivot_price: Manual pivot price (if pivot_type='manual')
            manual_pivot_time_idx: Manual pivot time position index (if pivot_type='manual')
            lookback_period: Bars to look back for auto-detecting pivot
            auto_detect_pivot: Whether to automatically detect pivot point
            num_price_levels: Number of price square levels outwards
            num_time_levels: Number of time square levels outwards
            time_price_ratio: Scaling factor between time and price axes. Crucial for time squaring.
            square_increment: The base increment added/subtracted from sqrt(price) for levels (0.125=45deg Sq9)
            **kwargs: Additional parameters
        """
        self.pivot_type = pivot_type
        self.manual_pivot_price = manual_pivot_price
        self.manual_pivot_time_idx = manual_pivot_time_idx
        self.lookback_period = lookback_period
        self.auto_detect_pivot = auto_detect_pivot
        self.num_price_levels = num_price_levels
        self.num_time_levels = num_time_levels
        self.time_price_ratio = time_price_ratio # Critical calibration parameter
        self.square_increment = square_increment # e.g., 0.125 for Sq9 45deg, 0.25 for 90deg

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Advanced Gann Square price levels and time turning points.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Square price levels and time markers
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        result = data.copy()

        # Determine pivot point (price and time position)
        pivot_pos, pivot_price = self._determine_pivot(result)

        if pivot_pos is None or pivot_price is None or pivot_price <= 0:
            print("Warning: Invalid pivot point for Advanced Gann Square.")
            return result

        pivot_idx_ts = result.index[pivot_pos]
        sqrt_pivot_price = math.sqrt(pivot_price)

        # --- Calculate Price Levels --- (Similar to basic GannSquare)
        for i in range(1, self.num_price_levels + 1):
            increment = self.square_increment * i

            # Support levels
            level_down = (sqrt_pivot_price - increment)**2
            # Resistance levels
            level_up = (sqrt_pivot_price + increment)**2

            # Add levels to DataFrame (as horizontal lines)
            # Use a generic name indicating the increment level
            result[f'adv_gann_sq_sup_{i}'] = level_down
            result[f'adv_gann_sq_res_{i}'] = level_up

        # --- Calculate Time Turning Points --- (Squaring time)
        # Concept: Project the price levels onto the time axis using the time_price_ratio
        # sqrt(price) +/- n*increment -> price_level
        # sqrt(time_offset * time_price_ratio) = sqrt(price_level) ??? -> This isn't quite right.
        # Alternative: Use the sqrt(pivot) increments directly scaled to time.
        # Time offset = ( (sqrt(pivot) +/- n*increment)^2 ) / time_price_ratio ??? No.
        # Let's use the Gann method: Time = Price (scaled). So sqrt(Time) = sqrt(Price / scale)
        # Or more directly: Time offset relates to (n * increment) on the sqrt scale.
        # If 1 unit price = 1 unit time * time_price_ratio
        # Then sqrt(price) relates to sqrt(time * time_price_ratio)
        # Time offset = (n * increment * sqrt(scaling_factor?)) ^ 2

        # Let's try the direct Square of 9 time projection method:
        # Time pivots occur at (sqrt(pivot_time_equivalent) +/- N)^2 where N is integer.
        # We need a pivot_time_equivalent. Let's use the pivot_pos itself as a starting point.
        # Or, map pivot_price to time: pivot_time_equiv = pivot_price / time_price_ratio
        # sqrt_pivot_time_equiv = math.sqrt(pivot_price / self.time_price_ratio) if self.time_price_ratio > 0 else sqrt_pivot_price

        # Simpler approach: Project increments directly onto time axis using scaling
        # Time offset = (increment / time_price_ratio) ^ 2 ???
        # Let's use the idea that angles relate price and time. 1x1 angle: price_change = time_change * time_price_ratio
        # Consider the square root increments as changes on a scaled axis.
        # Time offset corresponding to price increment `inc`: time_offset = (inc / time_price_ratio)^2 ???

        # Let's stick to the standard Square of 9 time projection method:
        # Project forward/backward from pivot_pos using squares of (sqrt(reference) +/- N)
        # What is the reference? Often the pivot_pos itself, or 0 if pivot is start of data.
        # Let's use pivot_pos as the time reference point T0.
        # Projected times T = T0 +/- (N * time_increment)^2 ??? No.
        # Projected times T = T0 +/- CycleLength * N
        # Cycle lengths derived from square: e.g., 90, 180, 360 bars/days

        # Let's implement time squaring based on the price pivot:
        # Time points = pivot_pos +/- ( (sqrt(pivot_price) +/- n*increment)^2 ) / time_price_ratio ??? Seems convoluted.

        # Try Gann's method: Time turning points at squares of numbers related to sqrt(pivot_price)
        result['adv_gann_sq_time_marker'] = 0
        for i in range(1, self.num_time_levels + 1):
            increment = self.square_increment * i

            # Calculate the equivalent "time units" for the price increment
            # If price_level = (sqrt_pivot +/- increment)^2
            # Then time_offset * time_price_ratio = price_level ??? No.
            # Let's assume the increment itself represents a scaled unit.
            # Time offset = (increment / time_price_ratio)^2 ???

            # Using the Square of 9 cardinal/diagonal time projection concept:
            # Time offsets are related to (sqrt(pivot_price) +/- N*0.5)^2 or similar, scaled.
            # Let's calculate time offsets based on the price levels found, scaled by time_price_ratio.
            # Time offset = Price Level / time_price_ratio

            if self.time_price_ratio > 0:
                # Time points derived from resistance levels
                price_level_up = (sqrt_pivot_price + increment)**2
                time_offset_up = int(round(price_level_up / self.time_price_ratio))
                time_pos_up_fwd = pivot_pos + time_offset_up
                time_pos_up_bwd = pivot_pos - time_offset_up

                # Time points derived from support levels
                price_level_down = (sqrt_pivot_price - increment)**2
                if price_level_down > 0:
                    time_offset_down = int(round(price_level_down / self.time_price_ratio))
                    time_pos_down_fwd = pivot_pos + time_offset_down
                    time_pos_down_bwd = pivot_pos - time_offset_down
                else:
                    time_pos_down_fwd, time_pos_down_bwd = None, None

                # Mark time points
                for time_pos, marker_val in [
                    (time_pos_up_fwd, i), (time_pos_up_bwd, -i),
                    (time_pos_down_fwd, i), (time_pos_down_bwd, -i)
                ]:
                    if time_pos is not None and 0 <= time_pos < len(result):
                        time_idx = result.index[time_pos]
                        # Mark with relative interval number
                        result.loc[time_idx, 'adv_gann_sq_time_marker'] = marker_val
                        
                        # Add specific marker column
                        col_name = f'adv_gann_sq_time_{abs(marker_val)}'
                        if col_name not in result.columns:
                            result[col_name] = False
                        result.loc[time_idx, col_name] = True

        # Add pivot marker
        result['adv_gann_sq_pivot_idx'] = False
        result.loc[pivot_idx_ts, 'adv_gann_sq_pivot_idx'] = True
        result['adv_gann_sq_pivot_price'] = pivot_price
        result['adv_gann_sq_pivot_pos'] = pivot_pos

        return result

    def _determine_pivot(self, data: pd.DataFrame) -> Tuple[Optional[int], Optional[float]]:
        """Determine pivot point (position and price)."""
        if self.pivot_type == 'manual' and self.manual_pivot_price is not None and self.manual_pivot_time_idx is not None:
            if 0 <= self.manual_pivot_time_idx < len(data):
                return self.manual_pivot_time_idx, self.manual_pivot_price
            else:
                return None, None # Manual index out of bounds

        if not self.auto_detect_pivot:
             # Use last bar if not auto-detecting and not manual
             if not data.empty:
                 last_pos = len(data) - 1
                 last_price = data['close'].iloc[-1]
                 return last_pos, last_price
             else:
                 return None, None

        # Auto-detect major high or low
        lookback_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data
        if len(lookback_data) < 5: return None, None

        pivot_pos, pivot_price = None, None
        try:
            if self.pivot_type == "major_low":
                swing_lows = find_swing_lows(lookback_data['low'], n=5) # Wider lookback for major swings
                if not swing_lows.empty:
                    pivot_idx_ts = lookback_data.loc[swing_lows].index[-1] # Most recent major low
                    pivot_price = lookback_data.loc[pivot_idx_ts, 'low']
                    pivot_pos = data.index.get_loc(pivot_idx_ts)
            elif self.pivot_type == "major_high":
                swing_highs = find_swing_highs(lookback_data['high'], n=5)
                if not swing_highs.empty:
                    pivot_idx_ts = lookback_data.loc[swing_highs].index[-1] # Most recent major high
                    pivot_price = lookback_data.loc[pivot_idx_ts, 'high']
                    pivot_pos = data.index.get_loc(pivot_idx_ts)
        except NameError:
            print("Warning: Swing point detection utility not found. Using simple min/max for Adv Gann Square pivot.")
            if self.pivot_type == "major_low":
                pivot_idx_ts = lookback_data['low'].idxmin()
                pivot_price = lookback_data.loc[pivot_idx_ts, 'low']
                pivot_pos = data.index.get_loc(pivot_idx_ts)
            elif self.pivot_type == "major_high":
                pivot_idx_ts = lookback_data['high'].idxmax()
                pivot_price = lookback_data.loc[pivot_idx_ts, 'high']
                pivot_pos = data.index.get_loc(pivot_idx_ts)

        # Fallback if detection failed
        if pivot_pos is None:
            if not data.empty:
                 last_pos = len(data) - 1
                 last_price = data['close'].iloc[-1]
                 # Use last bar as pivot if detection fails
                 pivot_pos, pivot_price = last_pos, last_price
                 print("Warning: Pivot auto-detection failed, using last bar as pivot.")
            else:
                 return None, None

        return pivot_pos, pivot_price

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Advanced Gann Square',
            'description': 'Gann Square of 9 (etc.) with price levels and time squaring',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'pivot_type',
                    'description': 'Type of pivot point to use',
                    'type': 'string',
                    'default': 'major_low',
                    'options': ['major_low', 'major_high', 'manual']
                },
                {
                    'name': 'manual_pivot_price',
                    'description': 'Manual pivot price (if pivot_type=\'manual\')',
                    'type': 'float',
                    'default': None
                },
                {
                    'name': 'manual_pivot_time_idx',
                    'description': 'Manual pivot time position index (if pivot_type=\'manual\')',
                    'type': 'int',
                    'default': None
                },
                {
                    'name': 'lookback_period',
                    'description': 'Bars to look back for auto-detecting pivot',
                    'type': 'int',
                    'default': 200
                },
                {
                    'name': 'auto_detect_pivot',
                    'description': 'Whether to automatically detect pivot point',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'num_price_levels',
                    'description': 'Number of price square levels outwards',
                    'type': 'int',
                    'default': 4
                },
                {
                    'name': 'num_time_levels',
                    'description': 'Number of time square levels outwards',
                    'type': 'int',
                    'default': 4
                },
                {
                    'name': 'time_price_ratio',
                    'description': 'Scaling factor: Price units per Time unit (bar)',
                    'type': 'float',
                    'default': 1.0 # Needs calibration
                },
                {
                    'name': 'square_increment',
                    'description': 'Sqrt increment for levels (0.125=45deg Sq9)',
                    'type': 'float',
                    'default': 0.125
                }
            ]
        }


class GannWheel(BaseIndicator):
    """
    Gann Wheel (Square of 52 / Circle)

    A tool based on the Square of 9, often adapted to a circular format or specific
    time cycles like the 52 weeks in a year. It identifies potential support/resistance
    levels and time turning points based on angular relationships (degrees) from a pivot.
    Commonly used degrees are 45, 90, 120, 180, 240, 270, 315, 360.
    """

    category = "gann"

    def __init__(
        self,
        pivot_type: str = "major_low", # 'major_low', 'major_high', 'manual'
        manual_pivot_price: Optional[float] = None,
        manual_pivot_time_idx: Optional[int] = None, # Position index
        lookback_period: int = 200,
        auto_detect_pivot: bool = True,
        degrees: Optional[List[int]] = None, # e.g., [45, 90, 180, 240, 270, 360]
        num_cycles: int = 3, # Number of wheel cycles/rotations
        time_price_ratio: float = 1.0, # Scaling: Price units per Time unit (bar)
        units_per_cycle: float = 360.0, # Units (degrees) in one full cycle/rotation
        **kwargs
    ):
        """
        Initialize Gann Wheel indicator.

        Args:
            pivot_type: Type of pivot ('major_low', 'major_high', 'manual')
            manual_pivot_price: Manual pivot price (if pivot_type='manual')
            manual_pivot_time_idx: Manual pivot time position index (if pivot_type='manual')
            lookback_period: Bars to look back for auto-detecting pivot
            auto_detect_pivot: Whether to automatically detect pivot point
            degrees: List of key angles (degrees) on the wheel to calculate levels for.
            num_cycles: Number of full cycles (360 degrees) to calculate.
            time_price_ratio: Scaling factor: Price units per Time unit (bar). Critical.
            units_per_cycle: The number of units (degrees) representing one full cycle (usually 360).
            **kwargs: Additional parameters
        """
        self.pivot_type = pivot_type
        self.manual_pivot_price = manual_pivot_price
        self.manual_pivot_time_idx = manual_pivot_time_idx
        self.lookback_period = lookback_period
        self.auto_detect_pivot = auto_detect_pivot
        self.degrees = degrees or [45, 90, 120, 135, 180, 225, 240, 270, 315, 360]
        self.num_cycles = num_cycles
        self.time_price_ratio = time_price_ratio # Needs calibration
        self.units_per_cycle = units_per_cycle # Usually 360 degrees

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Wheel price levels and time turning points.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Wheel price levels and time markers
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        result = data.copy()

        # Determine pivot point (price and time position)
        pivot_pos, pivot_price = self._determine_pivot(result)

        if pivot_pos is None or pivot_price is None or pivot_price <= 0:
            print("Warning: Invalid pivot point for Gann Wheel.")
            return result

        pivot_idx_ts = result.index[pivot_pos]
        sqrt_pivot_price = math.sqrt(pivot_price)

        # Calculate the base increment for one full cycle (360 degrees or units_per_cycle)
        # In Square of 9, one cycle (360 deg) corresponds to adding 2 to the square root.
        sqrt_increment_per_unit = 2.0 / self.units_per_cycle

        # --- Calculate Price Levels --- based on degrees
        for cycle in range(self.num_cycles + 1): # Include 0th cycle (pivot itself)
            for degree in self.degrees:
                total_units = cycle * self.units_per_cycle + degree
                sqrt_increment = total_units * sqrt_increment_per_unit

                # Calculate price levels (Resistance = adding increment, Support = subtracting)
                price_res = (sqrt_pivot_price + sqrt_increment)**2
                price_sup = (sqrt_pivot_price - sqrt_increment)**2

                # Add levels to DataFrame
                col_res = f'gann_wheel_res_{cycle}_{degree}deg'
                col_sup = f'gann_wheel_sup_{cycle}_{degree}deg'
                result[col_res] = price_res
                if price_sup > 0:
                    result[col_sup] = price_sup
                else:
                     result[col_sup] = np.nan # Avoid negative prices

        # --- Calculate Time Turning Points --- based on degrees
        result['gann_wheel_time_marker'] = 0
        if self.time_price_ratio > 0:
            for cycle in range(self.num_cycles + 1):
                for degree in self.degrees:
                    total_units = cycle * self.units_per_cycle + degree
                    sqrt_increment = total_units * sqrt_increment_per_unit

                    # Calculate equivalent price levels first
                    price_res = (sqrt_pivot_price + sqrt_increment)**2
                    price_sup = (sqrt_pivot_price - sqrt_increment)**2

                    # Convert price levels to time offsets using time_price_ratio
                    time_offset_res = int(round(price_res / self.time_price_ratio))
                    time_offset_sup = int(round(price_sup / self.time_price_ratio)) if price_sup > 0 else 0

                    # Calculate time positions relative to pivot_pos
                    time_pos_res_fwd = pivot_pos + time_offset_res
                    time_pos_res_bwd = pivot_pos - time_offset_res
                    time_pos_sup_fwd = pivot_pos + time_offset_sup if time_offset_sup > 0 else None
                    time_pos_sup_bwd = pivot_pos - time_offset_sup if time_offset_sup > 0 else None

                    # Mark time points
                    for time_pos, marker_val, level_type in [
                        (time_pos_res_fwd, degree + cycle*360, 'res'),
                        (time_pos_res_bwd, -(degree + cycle*360), 'res'),
                        (time_pos_sup_fwd, degree + cycle*360, 'sup'),
                        (time_pos_sup_bwd, -(degree + cycle*360), 'sup')
                    ]:
                        if time_pos is not None and 0 <= time_pos < len(result):
                            time_idx = result.index[time_pos]
                            # Use degree as marker value (can be large)
                            result.loc[time_idx, 'gann_wheel_time_marker'] = marker_val
                            # Add specific marker column
                            direction = 'fwd' if marker_val > 0 else 'bwd'
                            col_name = f'gann_wheel_time_{level_type}_{cycle}_{degree}deg_{direction}'
                            if col_name not in result.columns:
                                result[col_name] = False
                            result.loc[time_idx, col_name] = True

        # Add pivot marker
        result['gann_wheel_pivot_idx'] = False
        result.loc[pivot_idx_ts, 'gann_wheel_pivot_idx'] = True
        result['gann_wheel_pivot_price'] = pivot_price
        result['gann_wheel_pivot_pos'] = pivot_pos

        return result    def _determine_pivot(self, data: pd.DataFrame) -> Tuple[Optional[int], Optional[float]]:
        """Determine pivot point (position and price). (Same as AdvancedGannSquare)"""
        if self.pivot_type == 'manual' and self.manual_pivot_price is not None and self.manual_pivot_time_idx is not None:
            if 0 <= self.manual_pivot_time_idx < len(data):
                return self.manual_pivot_time_idx, self.manual_pivot_price
            else:
                return None, None # Manual index out of bounds

        if not self.auto_detect_pivot:
             if not data.empty:
                 last_pos = len(data) - 1
                 last_price = data['close'].iloc[-1]
                 return last_pos, last_price
             else:
                 return None, None

        # Auto-detect major high or low
        lookback_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data
        if len(lookback_data) < 5: return None, None

        pivot_pos, pivot_price = None, None
        try:
            if self.pivot_type == "major_low":
                swing_lows = find_swing_lows(lookback_data['low'], n=5)
                if not swing_lows.empty:
                    pivot_idx_ts = lookback_data.loc[swing_lows].index[-1]
                    pivot_price = lookback_data.loc[pivot_idx_ts, 'low']
                    pivot_pos = data.index.get_loc(pivot_idx_ts)
            elif self.pivot_type == "major_high":
                swing_highs = find_swing_highs(lookback_data['high'], n=5)
                if not swing_highs.empty:
                    pivot_idx_ts = lookback_data.loc[swing_highs].index[-1]
                    pivot_price = lookback_data.loc[pivot_idx_ts, 'high']
                    pivot_pos = data.index.get_loc(pivot_idx_ts)
        except NameError:
            print("Warning: Swing point detection utility not found. Using simple min/max for Gann Wheel pivot.")
            if self.pivot_type == "major_low":
                pivot_idx_ts = lookback_data['low'].idxmin()
                pivot_price = lookback_data.loc[pivot_idx_ts, 'low']
                pivot_pos = data.index.get_loc(pivot_idx_ts)
            elif self.pivot_type == "major_high":
                pivot_idx_ts = lookback_data['high'].idxmax()
                pivot_price = lookback_data.loc[pivot_idx_ts, 'high']
                pivot_pos = data.index.get_loc(pivot_idx_ts)

        # Fallback if detection failed
        if pivot_pos is None:
            if not data.empty:
                 last_pos = len(data) - 1
                 last_price = data['close'].iloc[-1]
                 pivot_pos, pivot_price = last_pos, last_price
                 print("Warning: Pivot auto-detection failed, using last bar as pivot.")
            else:
                 return None, None

        return pivot_pos, pivot_price

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Wheel',
            'description': 'Calculates price/time levels based on angular degrees from a pivot (Sq9/Circle)',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'pivot_type',
                    'description': 'Type of pivot point to use',
                    'type': 'string',
                    'default': 'major_low',
                    'options': ['major_low', 'major_high', 'manual']
                },
                {
                    'name': 'manual_pivot_price',
                    'description': 'Manual pivot price (if pivot_type=\'manual\')',
                    'type': 'float',
                    'default': None
                },
                {
                    'name': 'manual_pivot_time_idx',
                    'description': 'Manual pivot time position index (if pivot_type=\'manual\')',
                    'type': 'int',
                    'default': None
                },
                {
                    'name': 'lookback_period',
                    'description': 'Bars to look back for auto-detecting pivot',
                    'type': 'int',
                    'default': 200
                },
                {
                    'name': 'auto_detect_pivot',
                    'description': 'Whether to automatically detect pivot point',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'degrees',
                    'description': 'List of key angles (degrees) on the wheel',
                    'type': 'list',
                    'default': [45, 90, 120, 135, 180, 225, 240, 270, 315, 360]
                },
                {
                    'name': 'num_cycles',
                    'description': 'Number of full wheel cycles/rotations',
                    'type': 'int',
                    'default': 3
                },
                {
                    'name': 'time_price_ratio',
                    'description': 'Scaling factor: Price units per Time unit (bar)',
                    'type': 'float',
                    'default': 1.0 # Needs calibration
                },
                {
                    'name': 'units_per_cycle',
                    'description': 'Units (e.g., degrees) in one full cycle',
                    'type': 'float',
                    'default': 360.0
                }
            ]
        }


# --- End of New Advanced Gann Tools ---


# --- Start of Gann Hexagon and Square of 144 ---

class GannHexagon(BaseIndicator):
    \"\"\"
    Gann Hexagon

    Identifies potential turning points based on hexagonal geometry projected
    from a pivot point. Uses 60-degree increments (and multiples) on the
    Square of 9 concept to find price and time levels.
    \"\"\"

    category = \"gann\"

    def __init__(
        self,
        pivot_type: str = \"major_low\", # \'major_low\', \'major_high\', \'manual\'
        manual_pivot_price: Optional[float] = None,
        manual_pivot_time_idx: Optional[int] = None, # Position index
        lookback_period: int = 200,
        auto_detect_pivot: bool = True,
        degrees: Optional[List[int]] = None, # e.g., [60, 120, 180, 240, 300, 360]
        num_cycles: int = 3, # Number of hexagon cycles/rotations
        time_price_ratio: float = 1.0, # Scaling: Price units per Time unit (bar)
        units_per_cycle: float = 360.0, # Units (degrees) in one full cycle
        **kwargs
    ):
        \"\"\"
        Initialize Gann Hexagon indicator.

        Args:
            pivot_type: Type of pivot (\'major_low\', \'major_high\', \'manual\')
            manual_pivot_price: Manual pivot price (if pivot_type=\'manual\')
            manual_pivot_time_idx: Manual pivot time position index (if pivot_type=\'manual\')
            lookback_period: Bars to look back for auto-detecting pivot
            auto_detect_pivot: Whether to automatically detect pivot point
            degrees: List of key angles (degrees, typically multiples of 60) on the hexagon.
            num_cycles: Number of full cycles (360 degrees) to calculate.
            time_price_ratio: Scaling factor: Price units per Time unit (bar). Critical.
            units_per_cycle: The number of units (degrees) representing one full cycle (usually 360).
            **kwargs: Additional parameters
        \"\"\"
        self.pivot_type = pivot_type
        self.manual_pivot_price = manual_pivot_price
        self.manual_pivot_time_idx = manual_pivot_time_idx
        self.lookback_period = lookback_period
        self.auto_detect_pivot = auto_detect_pivot
        # Default to hexagonal degrees
        self.degrees = degrees or [60, 120, 180, 240, 300, 360]
        self.num_cycles = num_cycles
        self.time_price_ratio = time_price_ratio # Needs calibration
        self.units_per_cycle = units_per_cycle # Usually 360 degrees

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Calculate Gann Hexagon price levels and time turning points.
        Uses the same Square of 9 root calculation as Gann Wheel, but typically
        focuses on 60-degree increments.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Hexagon price levels and time markers
        \"\"\"
        required_cols = [\'high\', \'low\', \'close\']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f\"Data must contain \'{col}\' column\")

        result = data.copy()

        # Determine pivot point (price and time position)
        pivot_pos, pivot_price = self._determine_pivot(result)

        if pivot_pos is None or pivot_price is None or pivot_price <= 0:
            print(\"Warning: Invalid pivot point for Gann Hexagon.\")
            return result

        pivot_idx_ts = result.index[pivot_pos]
        sqrt_pivot_price = math.sqrt(pivot_price)

        # Calculate the base increment for one full cycle (360 degrees)
        # In Square of 9, one cycle (360 deg) corresponds to adding 2 to the square root.
        sqrt_increment_per_unit = 2.0 / self.units_per_cycle

        # --- Calculate Price Levels --- based on degrees
        for cycle in range(self.num_cycles + 1): # Include 0th cycle (pivot itself)
            for degree in self.degrees:
                total_units = cycle * self.units_per_cycle + degree
                sqrt_increment = total_units * sqrt_increment_per_unit

                # Calculate price levels
                price_res = (sqrt_pivot_price + sqrt_increment)**2
                price_sup = (sqrt_pivot_price - sqrt_increment)**2

                # Add levels to DataFrame
                col_res = f\'gann_hex_res_{cycle}_{degree}deg\'
                col_sup = f\'gann_hex_sup_{cycle}_{degree}deg\'
                result[col_res] = price_res
                if price_sup > 0:
                    result[col_sup] = price_sup
                else:
                     result[col_sup] = np.nan # Avoid negative prices

        # --- Calculate Time Turning Points --- based on degrees
        result[\'gann_hex_time_marker\'] = 0
        if self.time_price_ratio > 0:
            for cycle in range(self.num_cycles + 1):
                for degree in self.degrees:
                    total_units = cycle * self.units_per_cycle + degree
                    sqrt_increment = total_units * sqrt_increment_per_unit

                    # Calculate equivalent price levels first
                    price_res = (sqrt_pivot_price + sqrt_increment)**2
                    price_sup = (sqrt_pivot_price - sqrt_increment)**2

                    # Convert price levels to time offsets using time_price_ratio
                    time_offset_res = int(round(price_res / self.time_price_ratio))
                    time_offset_sup = int(round(price_sup / self.time_price_ratio)) if price_sup > 0 else 0

                    # Calculate time positions relative to pivot_pos
                    time_pos_res_fwd = pivot_pos + time_offset_res
                    time_pos_res_bwd = pivot_pos - time_offset_res
                    time_pos_sup_fwd = pivot_pos + time_offset_sup if time_offset_sup > 0 else None
                    time_pos_sup_bwd = pivot_pos - time_offset_sup if time_offset_sup > 0 else None

                    # Mark time points
                    for time_pos, marker_val, level_type in [\
                        (time_pos_res_fwd, degree + cycle*360, \'res\'),
                        (time_pos_res_bwd, -(degree + cycle*360), \'res\'),
                        (time_pos_sup_fwd, degree + cycle*360, \'sup\'),
                        (time_pos_sup_bwd, -(degree + cycle*360), \'sup\')
                    ]:
                        if time_pos is not None and 0 <= time_pos < len(result):\
                            time_idx = result.index[time_pos]
                            result.loc[time_idx, \'gann_hex_time_marker\'] = marker_val
                            # Add specific marker column
                            direction = \'fwd\' if marker_val > 0 else \'bwd\'
                            col_name = f\'gann_hex_time_{level_type}_{cycle}_{degree}deg_{direction}\'
                            if col_name not in result.columns:
                                result[col_name] = False
                            result.loc[time_idx, col_name] = True

        # Add pivot marker
        result[\'gann_hex_pivot_idx\'] = False
        result.loc[pivot_idx_ts, \'gann_hex_pivot_idx\'] = True
        result[\'gann_hex_pivot_price\'] = pivot_price
        result[\'gann_hex_pivot_pos\'] = pivot_pos

        return result

    # Re-use the pivot detection logic from AdvancedGannSquare/GannWheel
    _determine_pivot = AdvancedGannSquare._determine_pivot

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        \"\"\"Get indicator information.\"\"\"
        return {
            \'name\': \'Gann Hexagon\',
            \'description\': \'Calculates price/time levels based on hexagonal degrees (60 deg increments) from a pivot\',
            \'category\': cls.category,
            \'parameters\': [
                {\
                    \'name\': \'pivot_type\',\
                    \'description\': \'Type of pivot point to use\',\
                    \'type\': \'string\',\
                    \'default\': \'major_low\',\
                    \'options\': [\'major_low\', \'major_high\', \'manual\']
                },
                {\
                    \'name\': \'manual_pivot_price\',\
                    \'description\': \'Manual pivot price (if pivot_type=\\\'manual\\\')\',\
                    \'type\': \'float\',\
                    \'default\': None
                },
                {\
                    \'name\': \'manual_pivot_time_idx\',\
                    \'description\': \'Manual pivot time position index (if pivot_type=\\\'manual\\\')\',\
                    \'type\': \'int\',\
                    \'default\': None
                },
                {\
                    \'name\': \'lookback_period\',\
                    \'description\': \'Bars to look back for auto-detecting pivot\',\
                    \'type\': \'int\',\
                    \'default\': 200
                },
                {\
                    \'name\': \'auto_detect_pivot\',\
                    \'description\': \'Whether to automatically detect pivot point\',\
                    \'type\': \'bool\',\
                    \'default\': True
                },
                {\
                    \'name\': \'degrees\',\
                    \'description\': \'List of key angles (degrees) on the hexagon\',\
                    \'type\': \'list\',\
                    \'default\': [60, 120, 180, 240, 300, 360]\
                },
                {\
                    \'name\': \'num_cycles\',\
                    \'description\': \'Number of full hexagon cycles/rotations\',\
                    \'type\': \'int\',\
                    \'default\': 3
                },
                {\
                    \'name\': \'time_price_ratio\',\
                    \'description\': \'Scaling factor: Price units per Time unit (bar)\',\
                    \'type\': \'float\',\
                    \'default\': 1.0 # Needs calibration\
                },
                {\
                    \'name\': \'units_per_cycle\',\
                    \'description\': \'Units (e.g., degrees) in one full cycle\',\
                    \'type\': \'float\',\
                    \'default\': 360.0\
                }\
            ]\
        }


class GannSquare144(BaseIndicator):
    \"\"\"
    Gann Square of 144

    Calculates potential support and resistance levels based on the Square of 144 concept.
    This often relates to a 144-day or 144-bar cycle, or uses increments derived
    from the square root of 144 (which is 12). This implementation uses the
    standard square root method but allows adjusting the base increment.
    \"\"\"

    category = \"gann\"

    def __init__(
        self,
        pivot_type: str = \"major_low\", # \'major_low\', \'major_high\', \'manual\'
        manual_pivot_price: Optional[float] = None,
        manual_pivot_time_idx: Optional[int] = None, # Position index
        lookback_period: int = 200, # Longer lookback might be relevant for 144 cycle
        auto_detect_pivot: bool = True,
        num_levels: int = 4,
        base_increment: float = 0.125, # Standard Sq9 increment, adjust if 144 implies different scaling
        use_time_squaring: bool = False, # Optionally add time squaring like AdvancedGannSquare
        time_price_ratio: float = 1.0, # Required if use_time_squaring is True
        **kwargs
    ):
        \"\"\"
        Initialize Gann Square of 144 indicator.

        Args:
            pivot_type: Type of pivot (\'major_low\', \'major_high\', \'manual\')
            manual_pivot_price: Manual pivot price (if pivot_type=\'manual\')
            manual_pivot_time_idx: Manual pivot time position index (if pivot_type=\'manual\')
            lookback_period: Bars for auto-detecting pivot (consider >= 144)
            auto_detect_pivot: Whether to automatically detect pivot point
            num_levels: Number of square levels to calculate outwards
            base_increment: The fundamental increment added/subtracted from sqrt(price).
                            Default is 0.125 (Sq9 45deg). Adjust based on specific Sq144 interpretation.
            use_time_squaring: If True, calculate time turning points as well.
            time_price_ratio: Scaling factor for time squaring (Price units per Time unit).
            **kwargs: Additional parameters
        \"\"\"
        self.pivot_type = pivot_type
        self.manual_pivot_price = manual_pivot_price
        self.manual_pivot_time_idx = manual_pivot_time_idx
        self.lookback_period = lookback_period
        self.auto_detect_pivot = auto_detect_pivot
        self.num_levels = num_levels
        self.base_increment = base_increment # Key parameter for Sq144 interpretation
        self.use_time_squaring = use_time_squaring
        self.time_price_ratio = time_price_ratio

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Calculate Gann Square of 144 price levels (and optionally time points).

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Square 144 levels and optional time markers
        \"\"\"
        required_cols = [\'high\', \'low\', \'close\']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f\"Data must contain \'{col}\' column\")

        result = data.copy()

        # Determine pivot point (price and time position)
        pivot_pos, pivot_price = self._determine_pivot(result)

        if pivot_pos is None or pivot_price is None or pivot_price <= 0:
            print(\"Warning: Invalid pivot point for Gann Square 144.\")
            return result

        pivot_idx_ts = result.index[pivot_pos]
        sqrt_pivot_price = math.sqrt(pivot_price)

        # --- Calculate Price Levels ---
        for i in range(1, self.num_levels + 1):
            increment = self.base_increment * i

            # Support levels
            level_down = (sqrt_pivot_price - increment)**2
            # Resistance levels
            level_up = (sqrt_pivot_price + increment)**2

            # Add levels to DataFrame
            result[f\'gann_sq144_sup_{i}\'] = level_down if level_down > 0 else np.nan
            result[f\'gann_sq144_res_{i}\'] = level_up

        # --- Optionally Calculate Time Turning Points --- (Similar to AdvancedGannSquare)
        if self.use_time_squaring:
            result[\'gann_sq144_time_marker\'] = 0
            if self.time_price_ratio > 0:
                for i in range(1, self.num_levels + 1): # Use num_levels for time too
                    increment = self.base_increment * i

                    # Time points derived from resistance levels
                    price_level_up = (sqrt_pivot_price + increment)**2
                    time_offset_up = int(round(price_level_up / self.time_price_ratio))
                    time_pos_up_fwd = pivot_pos + time_offset_up
                    time_pos_up_bwd = pivot_pos - time_offset_up

                    # Time points derived from support levels
                    price_level_down = (sqrt_pivot_price - increment)**2
                    if price_level_down > 0:
                        time_offset_down = int(round(price_level_down / self.time_price_ratio))
                        time_pos_down_fwd = pivot_pos + time_offset_down
                        time_pos_down_bwd = pivot_pos - time_offset_down
                    else:
                        time_pos_down_fwd, time_pos_down_bwd = None, None

                    # Mark time points
                    for time_pos, marker_val in [\
                        (time_pos_up_fwd, i), (time_pos_up_bwd, -i),\
                        (time_pos_down_fwd, i), (time_pos_down_bwd, -i)\
                    ]:
                        if time_pos is not None and 0 <= time_pos < len(result):\
                            time_idx = result.index[time_pos]
                            result.loc[time_idx, \'gann_sq144_time_marker\'] = marker_val
                            # Add specific marker column
                            direction = \'fwd\' if marker_val > 0 else \'bwd\'
                            level_type = \'res\' if time_pos in [time_pos_up_fwd, time_pos_up_bwd] else \'sup\'
                            col_name = f\'gann_sq144_time_{level_type}_{abs(marker_val)}_{direction}\'
                            if col_name not in result.columns:
                                result[col_name] = False
                            result.loc[time_idx, col_name] = True
            else:
                print(\"Warning: time_price_ratio must be positive for time squaring.\")


        # Add pivot marker
        result[\'gann_sq144_pivot_idx\'] = False
        result.loc[pivot_idx_ts, \'gann_sq144_pivot_idx\'] = True
        result[\'gann_sq144_pivot_price\'] = pivot_price
        result[\'gann_sq144_pivot_pos\'] = pivot_pos

        return result

    # Re-use the pivot detection logic
    _determine_pivot = AdvancedGannSquare._determine_pivot

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        \"\"\"Get indicator information.\"\"\"
        return {
            \'name\': \'Gann Square 144\',
            \'description\': \'Calculates Gann Square levels, potentially related to 144 cycle/scaling\',
            \'category\': cls.category,
            \'parameters\': [
                {\
                    \'name\': \'pivot_type\',\
                    \'description\': \'Type of pivot point to use\',\
                    \'type\': \'string\',\
                    \'default\': \'major_low\',\
                    \'options\': [\'major_low\', \'major_high\', \'manual\']
                },
                {\
                    \'name\': \'manual_pivot_price\',\
                    \'description\': \'Manual pivot price (if pivot_type=\\\'manual\\\')\',\
                    \'type\': \'float\',\
                    \'default\': None
                },
                {\
                    \'name\': \'manual_pivot_time_idx\',\
                    \'description\': \'Manual pivot time position index (if pivot_type=\\\'manual\\\')\',\
                    \'type\': \'int\',\
                    \'default\': None
                },
                {\
                    \'name\': \'lookback_period\',\
                    \'description\': \'Bars for auto-detecting pivot (e.g., >= 144)\',\
                    \'type\': \'int\',\
                    \'default\': 200
                },
                {\
                    \'name\': \'auto_detect_pivot\',\
                    \'description\': \'Whether to automatically detect pivot point\',\
                    \'type\': \'bool\',\
                    \'default\': True
                },
                {\
                    \'name\': \'num_levels\',\
                    \'description\': \'Number of square levels outwards\',\
                    \'type\': \'int\',\
                    \'default\': 4
                },
                {\
                    \'name\': \'base_increment\',\
                    \'description\': \'Sqrt increment for levels (adjust for Sq144 interpretation)\',\
                    \'type\': \'float\',\
                    \'default\': 0.125
                },
                {\
                    \'name\': \'use_time_squaring\',\
                    \'description\': \'Calculate time turning points based on square levels\',\
                    \'type\': \'bool\',\
                    \'default\': False
                },
                {\
                    \'name\': \'time_price_ratio\',\
                    \'description\': \'Scaling factor for time squaring (Price units per Time unit)\',\
                    \'type\': \'float\',\
                    \'default\': 1.0 # Needs calibration if use_time_squaring=True\
                }
            ]\
        }

# --- End of Gann Hexagon and Square of 144 ---
