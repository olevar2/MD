"""
Fibonacci Analysis Module.

This module provides implementations of Fibonacci-based technical analysis tools
including retracement levels, extension levels, fans, and time zones.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum
import math

from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.utils.swing_points import find_swing_highs, find_swing_lows # Assuming utility exists


class TrendDirection(Enum):
    """Direction of price trend for Fibonacci analysis"""
    UPTREND = "uptrend"  # Lower swing point to higher swing point
    DOWNTREND = "downtrend"  # Higher swing point to lower swing point


class FibonacciRetracement(BaseIndicator):
    """
    Fibonacci Retracement
    
    Calculates Fibonacci retracement levels based on a significant price move.
    These levels often act as support or resistance as price retraces from a trend.
    
    Standard Fibonacci retracement levels are 23.6%, 38.2%, 50%, 61.8%, and 78.6%.
    """
    
    category = "fibonacci"
    
    def __init__(
        self, 
        levels: Optional[List[float]] = None,
        swing_lookback: int = 30,
        auto_detect_swings: bool = True,
        trend_direction: Optional[str] = None,
        projection_bars: int = 50,
        **kwargs
    ):
        """
        Initialize Fibonacci Retracement indicator.
        
        Args:
            levels: List of Fibonacci levels to calculate (default is standard levels)
            swing_lookback: Number of bars to look back for finding swing points
            auto_detect_swings: Whether to automatically detect swing points
            trend_direction: Manual specification of trend direction ('uptrend' or 'downtrend')
            projection_bars: Number of bars to project levels into the future
            **kwargs: Additional parameters
        """
        # Define default Fibonacci retracement levels
        self.levels = levels or [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.swing_lookback = swing_lookback
        self.auto_detect_swings = auto_detect_swings
        self.trend_direction = trend_direction
        self.projection_bars = projection_bars
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Retracement levels for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci Retracement values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Find swing points
        if self.auto_detect_swings:
            swing_high_idx, swing_high_price, swing_low_idx, swing_low_price = self._detect_swings(result)
        else:
            # Manual specification requires external swing points
            # In this case, we'll use the highest high and lowest low in the lookback period
            lookback_data = result.iloc[-self.swing_lookback:] if len(result) > self.swing_lookback else result
            swing_high_idx = lookback_data['high'].idxmax()
            swing_high_price = lookback_data.loc[swing_high_idx, 'high']
            swing_low_idx = lookback_data['low'].idxmin()
            swing_low_price = lookback_data.loc[swing_low_idx, 'low']
        
        # Determine trend direction
        if self.trend_direction:
            direction = TrendDirection.UPTREND if self.trend_direction.lower() == 'uptrend' else TrendDirection.DOWNTREND
        else:
            # Auto-determine trend direction based on swing point order
            if swing_low_idx < swing_high_idx:
                direction = TrendDirection.UPTREND
            else:
                direction = TrendDirection.DOWNTREND
        
        # Calculate retracement levels
        if direction == TrendDirection.UPTREND:
            start_price = swing_low_price
            end_price = swing_high_price
            start_idx = swing_low_idx
            end_idx = swing_high_idx
        else:
            start_price = swing_high_price
            end_price = swing_low_price
            start_idx = swing_high_idx
            end_idx = swing_low_idx
            
        price_range = end_price - start_price
        
        # Add retracement levels to result DataFrame
        for level in self.levels:
            level_price = end_price - (price_range * level)
            col_name = f"fib_retracement_{level:.3f}".replace('.', '_')
            
            # Project levels into future bars
            result[col_name] = None
            last_idx = len(result) - 1
            
            # Fill from end_idx to last index and project forward if needed
            end_loc = result.index.get_loc(end_idx)
            for i in range(end_loc, last_idx + self.projection_bars + 1):
                if i <= last_idx:
                    result.loc[result.index[i], col_name] = level_price
                    
        # Add swing points to the result
        result['fib_swing_high_idx'] = False
        result['fib_swing_low_idx'] = False
        
        result.loc[swing_high_idx, 'fib_swing_high_idx'] = True
        result.loc[swing_low_idx, 'fib_swing_low_idx'] = True
        
        # Add trend direction and metadata
        result['fib_trend_direction'] = direction.value
        
        # Add metadata as additional columns
        result['fib_start_price'] = start_price
        result['fib_end_price'] = end_price
        result['fib_price_range'] = price_range
        
        return result
        
    def _detect_swings(self, data: pd.DataFrame) -> Tuple[pd.Timestamp, float, pd.Timestamp, float]:
        """
        Detect significant swing points in the price data.
        
        Returns:
            Tuple of (swing_high_idx, swing_high_price, swing_low_idx, swing_low_price)
        """
        # Use only the lookback period for finding swings
        lookback_data = data.iloc[-self.swing_lookback:] if len(data) > self.swing_lookback else data
        
        # Find local peaks and troughs
        # A point is a peak if its value is greater than its neighbors
        # A point is a trough if its value is less than its neighbors
        peaks = []
        troughs = []
        
        # Minimum required deviation to consider a point as a swing point (as % of price)
        min_deviation_pct = 0.005  # 0.5%
        avg_price = lookback_data['close'].mean()
        min_deviation = avg_price * min_deviation_pct
        
        # Find peaks and troughs
        for i in range(2, len(lookback_data) - 2):
            # Current point
            curr_idx = lookback_data.index[i]
            curr_high = lookback_data.iloc[i]['high']
            curr_low = lookback_data.iloc[i]['low']
            
            # Compare with surrounding points
            if (curr_high > lookback_data.iloc[i-2]['high'] and 
                curr_high > lookback_data.iloc[i-1]['high'] and
                curr_high > lookback_data.iloc[i+1]['high'] and
                curr_high > lookback_data.iloc[i+2]['high']):
                # Found a peak
                peaks.append((curr_idx, curr_high))
                
            if (curr_low < lookback_data.iloc[i-2]['low'] and
                curr_low < lookback_data.iloc[i-1]['low'] and
                curr_low < lookback_data.iloc[i+1]['low'] and
                curr_low < lookback_data.iloc[i+2]['low']):
                # Found a trough
                troughs.append((curr_idx, curr_low))
        
        # If no swing points found, use the extremes in the lookback period
        if not peaks:
            high_idx = lookback_data['high'].idxmax()
            high_price = lookback_data.loc[high_idx]['high']
            peaks = [(high_idx, high_price)]
            
        if not troughs:
            low_idx = lookback_data['low'].idxmin()
            low_price = lookback_data.loc[low_idx]['low']
            troughs = [(low_idx, low_price)]
            
        # Get most significant peak and trough
        # For peak, choose the highest high
        swing_high_idx, swing_high_price = max(peaks, key=lambda x: x[1])
        
        # For trough, choose the lowest low
        swing_low_idx, swing_low_price = min(troughs, key=lambda x: x[1])
        
        return swing_high_idx, swing_high_price, swing_low_idx, swing_low_price
        
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Retracement',
            'description': 'Calculates Fibonacci retracement levels based on a significant price move',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'levels',
                    'description': 'List of Fibonacci levels to calculate',
                    'type': 'list',
                    'default': [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
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
                    'name': 'trend_direction',
                    'description': 'Manual specification of trend direction',
                    'type': 'string',
                    'default': None,
                    'options': ['uptrend', 'downtrend']
                },
                {
                    'name': 'projection_bars',
                    'description': 'Number of bars to project levels into the future',
                    'type': 'int',
                    'default': 50
                }
            ]
        }


class FibonacciExtension(BaseIndicator):
    """
    Fibonacci Extension
    
    Calculates Fibonacci extension levels to project potential profit targets
    beyond the trend. Uses three points: start of a move, end of a move, 
    and the retracement point.
    
    Standard Fibonacci extension levels are 127.2%, 161.8%, 261.8%, and 423.6%.
    """
    
    category = "fibonacci"
    
    def __init__(
        self, 
        levels: Optional[List[float]] = None,
        swing_lookback: int = 50,
        auto_detect_swings: bool = True,
        manual_points: Optional[Dict[str, int]] = None,
        projection_bars: int = 50,
        **kwargs
    ):
        """
        Initialize Fibonacci Extension indicator.
        
        Args:
            levels: List of Fibonacci extension levels to calculate (default is standard levels)
            swing_lookback: Number of bars to look back for finding swing points
            auto_detect_swings: Whether to automatically detect the three required swing points
            manual_points: Dictionary with manual point indices {'start': idx1, 'end': idx2, 'retracement': idx3}
            projection_bars: Number of bars to project levels into the future
            **kwargs: Additional parameters
        """
        # Define default Fibonacci extension levels
        self.levels = levels or [0, 0.618, 1.0, 1.272, 1.618, 2.618, 4.236]
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
            
            if 'start' in idx_map and 'end' in idx_map and 'retracement' in idx_map:
                # Get the price values for the specified indices
                start_idx = data.index[idx_map['start']] if idx_map['start'] < len(data) else data.index[-1]
                end_idx = data.index[idx_map['end']] if idx_map['end'] < len(data) else data.index[-1]
                retr_idx = data.index[idx_map['retracement']] if idx_map['retracement'] < len(data) else data.index[-1]
                
                # Get corresponding prices
                start_price = data.loc[start_idx, 'close']
                end_price = data.loc[end_idx, 'close']
                retr_price = data.loc[retr_idx, 'close']
            else:
                # Missing required points
                raise ValueError("Manual points must include 'start', 'end', and 'retracement' keys")
        else:
            # Default to using recent significant swing points
            lookback_data = result.iloc[-self.swing_lookback:] if len(result) > self.swing_lookback else result
            
            # Find the highest high and lowest low in the lookback period
            high_idx = lookback_data['high'].idxmax()
            high_price = lookback_data.loc[high_idx, 'high']
            
            low_idx = lookback_data['low'].idxmin()
            low_price = lookback_data.loc[low_idx, 'low']
            
            # Determine if we're in an uptrend or downtrend based on order of swing points
            if lookback_data.index.get_loc(low_idx) < lookback_data.index.get_loc(high_idx):
                # Uptrend: low -> high -> retracement
                start_idx, start_price = low_idx, low_price
                end_idx, end_price = high_idx, high_price
                
                # Find retracement point after the high
                post_high_data = lookback_data.loc[end_idx:].iloc[1:]
                if not post_high_data.empty:
                    retr_idx = post_high_data['low'].idxmin()
                    retr_price = post_high_data.loc[retr_idx, 'low']
                else:
                    # No data after high, use the last price
                    retr_idx = lookback_data.index[-1]
                    retr_price = lookback_data.loc[retr_idx, 'close']
                    
            else:
                # Downtrend: high -> low -> retracement
                start_idx, start_price = high_idx, high_price
                end_idx, end_price = low_idx, low_price
                
                # Find retracement point after the low
                post_low_data = lookback_data.loc[end_idx:].iloc[1:]
                if not post_low_data.empty:
                    retr_idx = post_low_data['high'].idxmax()
                    retr_price = post_low_data.loc[retr_idx, 'high']
                else:
                    # No data after low, use the last price
                    retr_idx = lookback_data.index[-1]
                    retr_price = lookback_data.loc[retr_idx, 'close']
        
        # Determine trend direction
        uptrend = start_price < end_price
        
        # Calculate the price range for the initial move
        wave1_range = end_price - start_price
        
        # Calculate the price range for the retracement
        wave2_range = end_price - retr_price if uptrend else retr_price - end_price
        
        # Calculate extension levels
        for level in self.levels:
            # Calculate the extension price based on the retracement
            if uptrend:
                # Uptrend: extensions go up from the retracement
                level_price = retr_price + (wave1_range * level)
            else:
                # Downtrend: extensions go down from the retracement
                level_price = retr_price - (wave1_range * level)
            
            # Create column for this extension level
            col_name = f"fib_extension_{level:.3f}".replace('.', '_')
            
            # Project levels into future bars
            result[col_name] = None
            last_idx = len(result) - 1
            
            # Fill from retracement point to last index and project forward if needed
            retr_loc = result.index.get_loc(retr_idx)
            for i in range(retr_loc, last_idx + self.projection_bars + 1):
                if i <= last_idx:
                    result.loc[result.index[i], col_name] = level_price
        
        # Add wave points to the result
        result['fib_ext_point1_idx'] = False
        result['fib_ext_point2_idx'] = False
        result['fib_ext_point3_idx'] = False
        
        result.loc[start_idx, 'fib_ext_point1_idx'] = True
        result.loc[end_idx, 'fib_ext_point2_idx'] = True
        result.loc[retr_idx, 'fib_ext_point3_idx'] = True
        
        # Add trend direction and metadata
        result['fib_ext_direction'] = 'uptrend' if uptrend else 'downtrend'
        result['fib_ext_wave1_range'] = wave1_range
        result['fib_ext_wave2_range'] = wave2_range
        result['fib_ext_point1_price'] = start_price
        result['fib_ext_point2_price'] = end_price
        result['fib_ext_point3_price'] = retr_price
        
        return result
        
    def _detect_three_points(self, data: pd.DataFrame) -> Tuple[pd.Timestamp, float, pd.Timestamp, float, pd.Timestamp, float]:
        """
        Detect three significant points for Fibonacci extensions.
        
        Returns:
            Tuple of (start_idx, start_price, end_idx, end_price, retracement_idx, retracement_price)
        """
        # Use only the lookback period for finding swing points
        lookback_data = data.iloc[-self.swing_lookback:] if len(data) > self.swing_lookback else data
        
        # Find significant swing points
        swing_points = []
        
        # Minimum required deviation to consider a point as a swing point (as % of price)
        min_deviation_pct = 0.005  # 0.5%
        avg_price = lookback_data['close'].mean()
        min_deviation = avg_price * min_deviation_pct
        
        # Find peaks
        for i in range(2, len(lookback_data) - 2):
            # Check for a peak
            curr_idx = lookback_data.index[i]
            curr_high = lookback_data.iloc[i]['high']
            
            is_peak = (curr_high > lookback_data.iloc[i-2]['high'] and 
                       curr_high > lookback_data.iloc[i-1]['high'] and
                       curr_high > lookback_data.iloc[i+1]['high'] and
                       curr_high > lookback_data.iloc[i+2]['high'])
                       
            # Check for a trough
            curr_low = lookback_data.iloc[i]['low']
            
            is_trough = (curr_low < lookback_data.iloc[i-2]['low'] and
                         curr_low < lookback_data.iloc[i-1]['low'] and
                         curr_low < lookback_data.iloc[i+1]['low'] and
                         curr_low < lookback_data.iloc[i+2]['low'])
                   
            if is_peak:
                swing_points.append((curr_idx, curr_high, 'high'))
                
            if is_trough:
                swing_points.append((curr_idx, curr_low, 'low'))
        
        # Sort swing points by time
        swing_points.sort(key=lambda x: lookback_data.index.get_loc(x[0]))
        
        # If less than 3 swing points found, use the extremes and the last point
        if len(swing_points) < 3:
            high_idx = lookback_data['high'].idxmax()
            high_price = lookback_data.loc[high_idx]['high']
            
            low_idx = lookback_data['low'].idxmin()
            low_price = lookback_data.loc[low_idx]['low']
            
            last_idx = lookback_data.index[-1]
            last_price = lookback_data.loc[last_idx]['close']
            
            # Determine if we're in an uptrend or downtrend based on order
            if lookback_data.index.get_loc(low_idx) < lookback_data.index.get_loc(high_idx):
                # Uptrend: low -> high -> last
                return low_idx, low_price, high_idx, high_price, last_idx, last_price
            else:
                # Downtrend: high -> low -> last
                return high_idx, high_price, low_idx, low_price, last_idx, last_price
                
        # If we have 3+ swing points, pick the 3 most significant ones
        # Find the swing with the biggest price movement for point 1 to point 2
        max_move = 0
        start_point = None
        end_point = None
        
        for i in range(len(swing_points) - 1):
            p1 = swing_points[i]
            p2 = swing_points[i+1]
            
            # Skip if both are the same type (need high->low or low->high)
            if p1[2] == p2[2]:
                continue
                
            move = abs(p2[1] - p1[1])
            if move > max_move:
                max_move = move
                start_point = p1
                end_point = p2
                
        # If no valid pairs found, fall back to the extremes
        if start_point is None or end_point is None:
            high_idx = lookback_data['high'].idxmax()
            high_price = lookback_data.loc[high_idx]['high']
            
            low_idx = lookback_data['low'].idxmin()
            low_price = lookback_data.loc[low_idx]['low']
            
            # Determine order based on time
            if lookback_data.index.get_loc(low_idx) < lookback_data.index.get_loc(high_idx):
                start_point = (low_idx, low_price, 'low')
                end_point = (high_idx, high_price, 'high')
            else:
                start_point = (high_idx, high_price, 'high')
                end_point = (low_idx, low_price, 'low')
        
        # Find the retracement point after end_point
        end_point_loc = lookback_data.index.get_loc(end_point[0])
        retracement = None
        
        # Look for a swing in the opposite direction after end_point
        for p in swing_points:
            p_loc = lookback_data.index.get_loc(p[0])
            if p_loc > end_point_loc and p[2] != end_point[2]:
                retracement = p
                break
                
        # If no retracement found, use the last point
        if retracement is None:
            last_idx = lookback_data.index[-1]
            last_price = lookback_data.loc[last_idx]['close']
            
            # Determine type based on end_point
            last_type = 'low' if end_point[2] == 'high' else 'high'
            retracement = (last_idx, last_price, last_type)
            
        # Return the three points
        return start_point[0], start_point[1], end_point[0], end_point[1], retracement[0], retracement[1]
        
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Extension',
            'description': 'Calculates Fibonacci extension levels to project potential profit targets',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'levels',
                    'description': 'List of Fibonacci extension levels to calculate',
                    'type': 'list',
                    'default': [0, 0.618, 1.0, 1.272, 1.618, 2.618, 4.236]
                },
                {
                    'name': 'swing_lookback',
                    'description': 'Number of bars to look back for finding swing points',
                    'type': 'int',
                    'default': 50
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


class FibonacciFan(BaseIndicator):
    """
    Fibonacci Fan
    
    Calculates Fibonacci fan lines from a significant swing point.
    These diagonal lines use Fibonacci ratios to identify potential support 
    and resistance levels as price moves over time.
    """
    
    category = "fibonacci"
    
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
            levels: List of Fibonacci fan levels to calculate (default is standard levels)
            swing_lookback: Number of bars to look back for finding swing points
            auto_detect_swings: Whether to automatically detect significant swing points
            manual_points: Dictionary with manual point indices {'start': idx1, 'end': idx2}
            projection_bars: Number of bars to project levels into the future
            **kwargs: Additional parameters
        """
        # Define default Fibonacci fan levels
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
            
            if 'start' in idx_map and 'end' in idx_map:
                # Get the price values for the specified indices
                start_idx = data.index[idx_map['start']] if idx_map['start'] < len(data) else data.index[-1]
                end_idx = data.index[idx_map['end']] if idx_map['end'] < len(data) else data.index[-1]
                
                # Get corresponding prices
                start_price = data.loc[start_idx, 'close']
                end_price = data.loc[end_idx, 'close']
            else:
                # Missing required points
                raise ValueError("Manual points must include 'start' and 'end' keys")
        else:
            # Default to using the highest high and lowest low in the lookback period
            lookback_data = result.iloc[-self.swing_lookback:] if len(result) > self.swing_lookback else result
            
            high_idx = lookback_data['high'].idxmax()
            high_price = lookback_data.loc[high_idx, 'high']
            
            low_idx = lookback_data['low'].idxmin()
            low_price = lookback_data.loc[low_idx, 'low']
            
            # Use the earlier point as start and the later point as end
            if lookback_data.index.get_loc(low_idx) < lookback_data.index.get_loc(high_idx):
                start_idx, start_price = low_idx, low_price
                end_idx, end_price = high_idx, high_price
            else:
                start_idx, start_price = high_idx, high_price
                end_idx, end_price = low_idx, low_price
        
        # Calculate the price range and time range
        price_range = end_price - start_price
        
        # Get the position in the data frame for each index
        start_pos = result.index.get_loc(start_idx)
        end_pos = result.index.get_loc(end_idx)
        time_range = end_pos - start_pos
        
        # Prevent division by zero
        if time_range == 0:
            time_range = 1
            
        # Calculate fan slopes
        for level in self.levels:
            # Calculate the fan line parameters
            # The slope is determined by the ratio of the vertical distance to the horizontal distance
            price_component = price_range * level
            fan_price = start_price + price_component
            
            # This gives the slope (price change per bar)
            slope = price_component / time_range
            
            # Create column for this fan level
            col_name = f"fib_fan_{level:.3f}".replace('.', '_')
            result[col_name] = None
            
            # Project fan lines forward
            last_pos = len(result) - 1
            
            # Calculate the fan line values from start point forward
            for i in range(start_pos, last_pos + self.projection_bars + 1):
                if i <= last_pos:
                    time_diff = i - start_pos
                    result.iloc[i, result.columns.get_loc(col_name)] = start_price + (slope * time_diff)
        
        # Add pivot points to the result
        result['fib_fan_start_idx'] = False
        result['fib_fan_end_idx'] = False
        
        result.loc[start_idx, 'fib_fan_start_idx'] = True
        result.loc[end_idx, 'fib_fan_end_idx'] = True
        
        # Add metadata
        result['fib_fan_start_price'] = start_price
        result['fib_fan_end_price'] = end_price
        result['fib_fan_price_range'] = price_range
        result['fib_fan_direction'] = 'uptrend' if price_range > 0 else 'downtrend'
        
        return result
        
    def _detect_swing_points(self, data: pd.DataFrame) -> Tuple[pd.Timestamp, float, pd.Timestamp, float]:
        """
        Detect significant swing points for Fibonacci fans.
        
        Returns:
            Tuple of (start_idx, start_price, end_idx, end_price)
        """
        # Use only the lookback period for finding swings
        lookback_data = data.iloc[-self.swing_lookback:] if len(data) > self.swing_lookback else data
        
        # Find local peaks and troughs
        peaks = []
        troughs = []
        
        # Find peaks and troughs
        for i in range(2, len(lookback_data) - 2):
            # Current point
            curr_idx = lookback_data.index[i]
            curr_high = lookback_data.iloc[i]['high']
            curr_low = lookback_data.iloc[i]['low']
            
            # Check for peak
            is_peak = (curr_high > lookback_data.iloc[i-2]['high'] and 
                       curr_high > lookback_data.iloc[i-1]['high'] and
                       curr_high > lookback_data.iloc[i+1]['high'] and
                       curr_high > lookback_data.iloc[i+2]['high'])
                       
            # Check for trough
            is_trough = (curr_low < lookback_data.iloc[i-2]['low'] and
                         curr_low < lookback_data.iloc[i-1]['low'] and
                         curr_low < lookback_data.iloc[i+1]['low'] and
                         curr_low < lookback_data.iloc[i+2]['low'])
                   
            if is_peak:
                peaks.append((curr_idx, curr_high))
                
            if is_trough:
                troughs.append((curr_idx, curr_low))
        
        # If no swing points found, use the extremes
        if not peaks:
            high_idx = lookback_data['high'].idxmax()
            high_price = lookback_data.loc[high_idx]['high']
            peaks = [(high_idx, high_price)]
            
        if not troughs:
            low_idx = lookback_data['low'].idxmin()
            low_price = lookback_data.loc[low_idx]['low']
            troughs = [(low_idx, low_price)]
            
        # Find the most significant high and low
        highest_peak = max(peaks, key=lambda x: x[1])
        lowest_trough = min(troughs, key=lambda x: x[1])
        
        # Determine order based on time
        peak_idx, peak_price = highest_peak
        trough_idx, trough_price = lowest_trough
        
        if lookback_data.index.get_loc(trough_idx) < lookback_data.index.get_loc(peak_idx):
            # Uptrend: trough -> peak
            return trough_idx, trough_price, peak_idx, peak_price
        else:
            # Downtrend: peak -> trough
            return peak_idx, peak_price, trough_idx, trough_price
        
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


class FibonacciTimeZones(BaseIndicator):
    """
    Fibonacci Time Zones
    
    Projects potential future significant time points based on Fibonacci numbers.
    This tool helps identify time periods where significant market reversals 
    or continuations might occur.
    """
    
    category = "fibonacci"
    
    def __init__(
        self, 
        fib_sequence: Optional[List[int]] = None,
        starting_point: Optional[int] = None,
        auto_detect_start: bool = True,
        max_zones: int = 8,
        **kwargs
    ):
        """
        Initialize Fibonacci Time Zones indicator.
        
        Args:
            fib_sequence: List of Fibonacci numbers to use for time projections
            starting_point: Manual index for the starting point (None for auto-detect)
            auto_detect_start: Whether to automatically detect the starting point
            max_zones: Maximum number of time zones to project
            **kwargs: Additional parameters
        """
        # Define default Fibonacci sequence numbers
        self.fib_sequence = fib_sequence or [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        self.starting_point = starting_point
        self.auto_detect_start = auto_detect_start
        self.max_zones = max(1, min(max_zones, 10))  # Limit between 1 and 10
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Time Zones for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci Time Zone markers
        """
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Determine starting point
        if self.starting_point is not None and self.starting_point < len(data):
            start_pos = self.starting_point
        elif self.auto_detect_start:
            # Find a significant swing point to use as starting point
            start_pos = self._detect_starting_point(data)
        else:
            # Default to a recent significant point
            start_pos = max(0, len(data) - 50)  # 50 bars from the end
            
        start_idx = data.index[start_pos]
        
        # Create column for Fibonacci time zones
        result['fib_time_zone'] = 0
        
        # Add time zones based on Fibonacci numbers
        for i, fib in enumerate(self.fib_sequence[:self.max_zones]):
            zone_pos = start_pos + fib
            if zone_pos < len(result):
                zone_idx = result.index[zone_pos]
                result.loc[zone_idx, 'fib_time_zone'] = i + 1  # Zone number
                
        # Mark the starting point
        result['fib_time_zone_start'] = False
        result.loc[start_idx, 'fib_time_zone_start'] = True
        
        # Add information about each zone
        for i, fib in enumerate(self.fib_sequence[:self.max_zones]):
            col_name = f"fib_time_zone_{i+1}"
            result[col_name] = False
            
            zone_pos = start_pos + fib
            if zone_pos < len(result):
                zone_idx = result.index[zone_pos]
                result.loc[zone_idx, col_name] = True
        
        return result
        
    def _detect_starting_point(self, data: pd.DataFrame) -> int:
        """
        Detect a significant starting point for Fibonacci time zones.
        
        Returns:
            Position (index) of the starting point in the data
        """
        # Look for a significant price reversal or volatility event
        # We'll use a simple approach: find a point with a significant price change
        
        # Calculate daily changes
        changes = data['close'].pct_change().abs()
        
        # Use the point with maximum change in the first half of the data
        half_point = len(data) // 2
        first_half = changes.iloc[:half_point]
        
        if not first_half.empty:
            # Find the position of the maximum change
            max_change_pos = first_half.idxmax()
            return data.index.get_loc(max_change_pos)
        
        # Default to the first point if no significant change found
        return 0
        
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Time Zones',
            'description': 'Projects potential future significant time points based on Fibonacci numbers',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'fib_sequence',
                    'description': 'List of Fibonacci numbers to use for time projections',
                    'type': 'list',
                    'default': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
                },
                {
                    'name': 'starting_point',
                    'description': 'Manual index for the starting point',
                    'type': 'int',
                    'default': None
                },
                {
                    'name': 'auto_detect_start',
                    'description': 'Whether to automatically detect the starting point',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'max_zones',
                    'description': 'Maximum number of time zones to project',
                    'type': 'int',
                    'default': 8
                }
            ]
        }


# --- Start of New Advanced Fibonacci Tools ---

class FibonacciChannels(BaseIndicator):
    """
    Fibonacci Channels

    Draws parallel trend lines based on Fibonacci ratios from a base trend line.
    Helps identify potential support and resistance zones within a trend.
    Requires three points: start of trend, end of trend, and a point defining channel width.
    """

    category = "fibonacci"

    def __init__(
        self,
        levels: Optional[List[float]] = None,
        swing_lookback: int = 50,
        auto_detect_swings: bool = True,
        manual_points: Optional[Dict[str, int]] = None, # {start_idx: idx, end_idx: idx, width_ref_idx: idx}
        projection_bars: int = 50,
        **kwargs
    ):
        """
        Initialize Fibonacci Channels indicator.

        Args:
            levels: List of Fibonacci levels for channel width (e.g., [0.0, 0.618, 1.0, 1.618])
            swing_lookback: Number of bars to look back for finding trend points
            auto_detect_swings: Whether to automatically detect trend points
            manual_points: Dictionary with manual point indices {start, end, width_ref}
            projection_bars: Number of bars to project channels into the future
            **kwargs: Additional parameters
        """
        self.levels = levels or [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
        self.swing_lookback = swing_lookback
        self.auto_detect_swings = auto_detect_swings
        self.manual_points = manual_points
        self.projection_bars = projection_bars

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Channel lines for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Fibonacci Channel lines
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        result = data.copy()

        # Find start, end, and width reference points
        if self.auto_detect_swings:
            start_idx, start_price, end_idx, end_price, width_ref_idx, width_ref_price = self._detect_channel_points(result)
        elif self.manual_points:
            idx_map = self.manual_points
            if 'start_idx' in idx_map and 'end_idx' in idx_map and 'width_ref_idx' in idx_map:
                start_idx = data.index[idx_map['start_idx']] if idx_map['start_idx'] < len(data) else data.index[-1]
                end_idx = data.index[idx_map['end_idx']] if idx_map['end_idx'] < len(data) else data.index[-1]
                width_ref_idx = data.index[idx_map['width_ref_idx']] if idx_map['width_ref_idx'] < len(data) else data.index[-1]

                # Determine price based on trend direction for start/end
                temp_start_price = data.loc[start_idx, 'close']
                temp_end_price = data.loc[end_idx, 'close']
                start_price = data.loc[start_idx, 'low'] if temp_end_price > temp_start_price else data.loc[start_idx, 'high']
                end_price = data.loc[end_idx, 'high'] if temp_end_price > temp_start_price else data.loc[end_idx, 'low']
                width_ref_price = data.loc[width_ref_idx, 'close'] # Price at the width reference point
            else:
                raise ValueError("Manual points must include 'start_idx', 'end_idx', and 'width_ref_idx' keys")
        else:
            print("Warning: No method specified for finding channel points.")
            return result # Or raise error

        if start_idx is None or end_idx is None or width_ref_idx is None:
             print("Warning: Could not determine channel points.")
             return result

        start_pos = result.index.get_loc(start_idx)
        end_pos = result.index.get_loc(end_idx)
        time_range = end_pos - start_pos
        price_range = end_price - start_price

        if time_range == 0: time_range = 1 # Avoid division by zero
        slope = price_range / time_range

        # Calculate the base line's price at the width reference point's time
        width_ref_pos = result.index.get_loc(width_ref_idx)
        base_line_price_at_ref = start_price + slope * (width_ref_pos - start_pos)
        channel_width = abs(width_ref_price - base_line_price_at_ref)

        last_pos = len(result) - 1
        for level in self.levels:
            col_name = f"fib_channel_{level:.3f}".replace('.', '_')
            result[col_name] = np.nan # Initialize with NaN

            vertical_offset = channel_width * level
            # Adjust offset direction based on whether width ref point is above/below base trendline
            if width_ref_price < base_line_price_at_ref:
                 vertical_offset *= -1

            # Calculate channel line points from start_pos onwards, including projection
            for i in range(start_pos, last_pos + self.projection_bars + 1):
                current_idx = result.index[i] if i <= last_pos else result.index[-1] + pd.Timedelta(days=i-last_pos) # Crude projection index
                time_diff = i - start_pos
                base_line_price = start_price + (slope * time_diff)
                channel_line_price = base_line_price + vertical_offset

                if i <= last_pos:
                    result.loc[current_idx, col_name] = channel_line_price
                else:
                    # Extend DataFrame for projection if needed (simple approach)
                    if current_idx not in result.index:
                        result.loc[current_idx] = np.nan
                    result.loc[current_idx, col_name] = channel_line_price


        # Add metadata
        result['fib_channel_start_idx'] = False
        result.loc[start_idx, 'fib_channel_start_idx'] = True
        result['fib_channel_end_idx'] = False
        result.loc[end_idx, 'fib_channel_end_idx'] = True
        result['fib_channel_width_ref_idx'] = False
        result.loc[width_ref_idx, 'fib_channel_width_ref_idx'] = True
        result['fib_channel_base_slope'] = slope
        result['fib_channel_base_width'] = channel_width

        return result

    def _detect_channel_points(self, data: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[pd.Timestamp], Optional[float], Optional[pd.Timestamp], Optional[float]]:
        """
        Detect three points for defining the channel: start, end, and width reference.
        This is a simplified detection using recent swing highs/lows.
        """
        lookback_data = data.iloc[-self.swing_lookback:] if len(data) > self.swing_lookback else data
        if len(lookback_data) < 5: return None, None, None, None, None, None

        # Use utility functions if available, otherwise simple min/max
        try:
            swing_lows = find_swing_lows(lookback_data['low'], n=2) # Example: n=2 (look left/right 2 bars)
            swing_highs = find_swing_highs(lookback_data['high'], n=2)
            low_points = lookback_data.loc[swing_lows]
            high_points = lookback_data.loc[swing_highs]

            if low_points.empty or high_points.empty:
                return None, None, None, None

            # Find the most recent significant low and high
            last_low_idx = low_points.index[-1]
            last_high_idx = high_points.index[-1]

            # Determine trend direction based on the order of the last high/low
            if lookback_data.index.get_loc(last_low_idx) < lookback_data.index.get_loc(last_high_idx):
                # Uptrend: Start=Low, End=High
                start_idx = last_low_idx
                start_price = lookback_data.loc[start_idx, 'low']
                end_idx = last_high_idx
                end_price = lookback_data.loc[end_idx, 'high']
                trend_segment = lookback_data.loc[start_idx:end_idx]
                # Width ref: Highest high within the trend segment (excluding end point)
                segment_highs = trend_segment['high'].iloc[:-1]
                if not segment_highs.empty:
                    width_ref_idx = segment_highs.idxmax()
                    width_ref_price = segment_highs.max()
                else: # Fallback if segment is too short
                    width_ref_idx = end_idx
                    width_ref_price = end_price

            else:
                # Downtrend: Start=High, End=Low
                start_idx = last_high_idx
                start_price = lookback_data.loc[start_idx, 'high']
                end_idx = last_low_idx
                end_price = lookback_data.loc[end_idx, 'low']
                trend_segment = lookback_data.loc[start_idx:end_idx]
                 # Width ref: Lowest low within the trend segment (excluding end point)
                segment_lows = trend_segment['low'].iloc[:-1]
                if not segment_lows.empty:
                    width_ref_idx = segment_lows.idxmin()
                    width_ref_price = segment_lows.min()
                else: # Fallback
                    width_ref_idx = end_idx
                    width_ref_price = end_price

        except NameError: # Fallback if find_swing_highs/lows not available
            print("Warning: Swing point detection utility not found. Using simple min/max.")
            low_idx = lookback_data['low'].idxmin()
            low_price = lookback_data.loc[low_idx, 'low']
            high_idx = lookback_data['high'].idxmax()
            high_price = lookback_data.loc[high_idx, 'high']

            if lookback_data.index.get_loc(low_idx) < lookback_data.index.get_loc(high_idx):
                # Uptrend: low -> high
                start_idx, start_price = low_idx, low_price
                end_idx, end_price = high_idx, high_price
                trend_segment = lookback_data.loc[start_idx:end_idx]
                width_ref_idx = trend_segment['high'].idxmax() # Simplistic width ref
                width_ref_price = trend_segment.loc[width_ref_idx, 'high']
            else:
                # Downtrend: high -> low
                start_idx, start_price = high_idx, high_price
                end_idx, end_price = low_idx, low_price
                trend_segment = lookback_data.loc[start_idx:end_idx]
                width_ref_idx = trend_segment['low'].idxmin() # Simplistic width ref
                width_ref_price = trend_segment.loc[width_ref_idx, 'low']

        # Refine width point: Find point furthest from the base trend line
        start_pos = lookback_data.index.get_loc(start_idx)
        end_pos = lookback_data.index.get_loc(end_idx)
        time_range_refine = end_pos - start_pos
        price_range_refine = end_price - start_price
        if time_range_refine == 0: time_range_refine = 1
        slope_refine = price_range_refine / time_range_refine
        max_dist = 0
        refined_width_ref_idx = width_ref_idx # Default to initial guess
        refined_width_ref_price = width_ref_price

        for i in range(start_pos + 1, end_pos):
            current_idx = lookback_data.index[i]
            # Check high in uptrend, low in downtrend
            current_price = lookback_data.iloc[i]['high'] if start_price < end_price else lookback_data.iloc[i]['low']
            base_line_price = start_price + slope_refine * (i - start_pos)
            dist = abs(current_price - base_line_price)
            if dist > max_dist:
                max_dist = dist
                refined_width_ref_idx = current_idx
                refined_width_ref_price = current_price

        return start_idx, start_price, end_idx, end_price, refined_width_ref_idx, refined_width_ref_price

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Channels',
            'description': 'Draws parallel trend lines based on Fibonacci ratios',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'levels',
                    'description': 'List of Fibonacci levels for channel width',
                    'type': 'list',
                    'default': [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
                },
                {
                    'name': 'swing_lookback',
                    'description': 'Number of bars to look back for finding trend points',
                    'type': 'int',
                    'default': 50
                },
                {
                    'name': 'auto_detect_swings',
                    'description': 'Whether to automatically detect trend points',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'manual_points',
                    'description': 'Dictionary with manual point indices {start, end, width_ref}',
                    'type': 'dict',
                    'default': None
                },
                {
                    'name': 'projection_bars',
                    'description': 'Number of bars to project channels into the future',
                    'type': 'int',
                    'default': 50
                }
            ]
        }


class FibonacciCircles(BaseIndicator):
    """
    Fibonacci Circles

    Draws circles centered on a significant price point (e.g., a major high or low).
    The radii of the circles are determined by Fibonacci ratios multiplied by the price range
    of a preceding trend. They indicate potential time and price support/resistance.
    Requires a center point and a radius reference point.
    """

    category = "fibonacci"

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
        self.levels = levels or [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618]
        self.swing_lookback = swing_lookback
        self.auto_detect_points = auto_detect_points
        self.manual_points = manual_points
        self.projection_bars = projection_bars
        self.time_price_ratio = time_price_ratio # This is critical and often needs calibration

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Circle points for the given data.
        Note: This calculates the price levels of the circles at each time step.
              Visualizing actual circles requires a charting library capable of handling aspect ratio.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Fibonacci Circle price levels
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        result = data.copy()

        # Find center point (position index)
        if self.manual_center_idx is not None and 0 <= self.manual_center_idx < len(data):
            center_pos = self.manual_center_idx
        elif self.auto_detect_center:
            center_pos = self._detect_center_point_pos(result)
        else:
            print("Warning: No method specified for finding circle center.")
            return result # Or raise error

        if center_pos is None:
            print("Warning: Could not determine circle center point.")
            return result # Points not found

        center_idx_ts = result.index[center_pos] # Timestamp index
        center_price = result.loc[center_idx_ts, 'close'] # Use close price at center
        last_pos = len(result) - 1

        # Prepare columns for results
        for level in self.levels:
            col_name_upper = f"fib_circle_upper_{level:.3f}".replace('.', '_')
            col_name_lower = f"fib_circle_lower_{level:.3f}".replace('.', '_')
            result[col_name_upper] = np.nan
            result[col_name_lower] = np.nan

        # Estimate initial radius 'a' based on price at center or recent volatility
        # Simple heuristic: a fraction of the center price
        initial_radius_a_price = center_price * self.initial_radius_factor
        if initial_radius_a_price <= 0: initial_radius_a_price = data['close'].iloc[-10:].std() # Fallback
        if initial_radius_a_price <= 0: initial_radius_a_price = 1 # Final fallback

        # Calculate circle points for each level
        for level in self.levels:
            radius_price = initial_radius_a_price * level # Radius in price units
            if radius_price <= 0: continue

            col_name_upper = f"fib_circle_upper_{level:.3f}".replace('.', '_')
            col_name_lower = f"fib_circle_lower_{level:.3f}".replace('.', '_')

            # Convert price radius to time radius using the scaling factor
            # radius_time^2 * time_price_ratio^2 + price_offset^2 = radius_price^2
            # radius_time = radius_price / time_price_ratio (if circle is truly round in scaled space)
            radius_time_bars = radius_price / self.time_price_ratio

            # Calculate circle points based on time distance from center
            # Iterate through relevant time range (past and future projection)
            start_calc_pos = max(0, center_pos - int(np.ceil(radius_time_bars)) - self.projection_bars)
            end_calc_pos = min(last_pos + self.projection_bars, center_pos + int(np.ceil(radius_time_bars)) + self.projection_bars)

            for i in range(start_calc_pos, end_calc_pos + 1):
                time_diff_bars = i - center_pos
                time_diff_sq = time_diff_bars**2
                radius_time_sq = radius_time_bars**2

                if time_diff_sq <= radius_time_sq: # Check if time is within the circle's horizontal extent
                    # Calculate price offset using circle equation in scaled space
                    # price_offset^2 = radius_price^2 - (time_diff_bars * time_price_ratio)^2
                    price_offset_sq = radius_price**2 - (time_diff_bars * self.time_price_ratio)**2

                    if price_offset_sq >= 0: # Ensure real result
                        price_offset = math.sqrt(price_offset_sq)
                        upper_price = center_price + price_offset
                        lower_price = center_price - price_offset

                        # Assign to DataFrame, extending if necessary for projection
                        if i <= last_pos:
                            current_idx = result.index[i]
                            result.loc[current_idx, col_name_upper] = upper_price
                            result.loc[current_idx, col_name_lower] = lower_price
                        else:
                            # Extend index for projection
                            proj_idx = result.index[-1] + pd.Timedelta(days=i-last_pos) # Crude projection index
                            if proj_idx not in result.index:
                                result.loc[proj_idx] = np.nan
                            result.loc[proj_idx, col_name_upper] = upper_price
                            result.loc[proj_idx, col_name_lower] = lower_price


        # Interpolate missing values for smoother line (optional)
        result[col_name_spiral] = result[col_name_spiral].interpolate(method='linear').bfill().ffill()

        # Add metadata
        result['fib_circle_center_idx'] = False
        result.loc[center_idx_ts, 'fib_circle_center_idx'] = True
        result['fib_circle_radius_ref_idx'] = False
        result.loc[radius_ref_idx, 'fib_circle_radius_ref_idx'] = True
        result['fib_circle_center_price'] = center_price
        result['fib_circle_base_radius'] = base_radius_price

        return result

    def _detect_center_point_pos(self, data: pd.DataFrame) -> Optional[int]:
        """Detect a significant turning point (position index) to use as the spiral center."""
        lookback_data = data.iloc[-self.swing_lookback:] if len(data) > self.swing_lookback else data
        if len(lookback_data) < 5: return None

        center_idx_ts = None
        try:
            # Find the most recent significant high or low using utility
            swing_lows = find_swing_lows(lookback_data['low'], n=3)
            swing_highs = find_swing_highs(lookback_data['high'], n=3)
            low_points = lookback_data.loc[swing_lows]
            high_points = lookback_data.loc[swing_highs]

            if not low_points.empty and not high_points.empty:
                 last_low_idx = low_points.index[-1]
                 last_high_idx = high_points.index[-1]
                 # Choose the most recent one
                 center_idx_ts = last_low_idx if data.index.get_loc(last_low_idx) > data.index.get_loc(last_high_idx) else last_high_idx
            elif not low_points.empty:
                 center_idx_ts = low_points.index[-1]
            elif not high_points.empty:
                 center_idx_ts = high_points.index[-1]

        except NameError:
             # Fallback: Find the most prominent high or low in the lookback period
            low_idx_ts = lookback_data['low'].idxmin()
            high_idx_ts = lookback_data['high'].idxmax()
            low_price = lookback_data.loc[low_idx_ts, 'low']
            high_price = lookback_data.loc[high_idx_ts, 'high']

            # Choose the one with larger deviation from the mean as more significant
            mean_price = lookback_data['close'].mean()
            if abs(high_price - mean_price) > abs(low_price - mean_price):
                center_idx_ts = high_idx_ts
            else:
                center_idx_ts = low_idx_ts

        if center_idx_ts is not None:
            return data.index.get_loc(center_idx_ts)
        else:
            return None


    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Spirals',
            'description': 'Draws logarithmic spirals to forecast turning points',
            'category': cls.category,
            'parameters': [
                 {
                    'name': 'growth_factor',
                    'description': 'Growth factor related to the spiral expansion (often related to phi).',
                    'type': 'float',
                    'default': 2.618 # Example: phi^2
                },
                 {
                    'name': 'start_angle',
                    'description': 'Starting angle (radians)',
                    'type': 'float',
                    'default': 0.0
                },
                 {
                    'name': 'num_rotations',
                    'description': 'Number of full rotations',
                    'type': 'int',
                    'default': 3
                },
                 {
                    'name': 'points_per_rotation',
                    'description': 'Calculation points per rotation',
                    'type': 'int',
                    'default': 90
                },
                 {
                    'name': 'swing_lookback',
                    'description': 'Lookback period for finding the center point',
                    'type': 'int',
                    'default': 50
                },
                 {
                    'name': 'auto_detect_center',
                    'description': 'Whether to automatically detect the center point',
                    'type': 'bool',
                    'default': True
                },
                 {
                    'name': 'manual_center_idx',
                    'description': 'Manual position index for the spiral center',
                    'type': 'int',
                    'default': None
                },
                 {
                    'name': 'projection_bars',
                    'description': 'How far in time to project the spiral',
                    'type': 'int',
                    'default': 100
                },
                 {
                    'name': 'time_price_ratio',
                    'description': 'Scaling factor: Price units per Time unit (bar)',
                    'type': 'float',
                    'default': 1.0 # Needs calibration per market/timeframe
                },
                 {
                    'name': 'initial_radius_factor',
                    'description': 'Multiplier for center price for initial radius estimate',
                    'type': 'float',
                    'default': 0.01 # Needs calibration
                }
            ]
        }


class FibonacciSpeedResistanceLines(BaseIndicator):
    """
    Fibonacci Speed/Resistance Lines

    Combines elements of trendlines and Fibonacci retracements.
    Lines are drawn from a significant low/start point to intersect vertical lines
    drawn at Fibonacci levels (typically 1/3 and 2/3, or 38.2%/61.8%) of the price range
    at the time of the significant high/end point.
    """

    category = "fibonacci"

    def __init__(
        self,
        levels: Optional[List[float]] = None, # e.g., [1/3, 2/3] or [0.382, 0.618]
        swing_lookback: int = 50,
        auto_detect_swings: bool = True,
        manual_points: Optional[Dict[str, int]] = None, # {start_idx: idx, end_idx: idx}
        projection_bars: int = 50,
        **kwargs
    ):
        """
        Initialize Fibonacci Speed/Resistance Lines indicator.

        Args:
            levels: List of Fibonacci levels (fractions) to determine line placement (e.g., [0.382, 0.618])
            swing_lookback: Number of bars to look back for finding swing points
            auto_detect_swings: Whether to automatically detect swing points
            manual_points: Dictionary with manual point indices {start_idx, end_idx}
            projection_bars: Number of bars to project lines into the future
            **kwargs: Additional parameters
        """
        self.levels = levels or [0.382, 0.618] # Common levels, can also use 1/3, 2/3
        self.swing_lookback = swing_lookback
        self.auto_detect_swings = auto_detect_swings
        self.manual_points = manual_points
        self.projection_bars = projection_bars

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Speed/Resistance Lines for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Speed/Resistance Line values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        result = data.copy()

        # Find start and end points of the major trend
        if self.auto_detect_swings:
            start_idx, start_price, end_idx, end_price = self._detect_trend_points(result)
        elif self.manual_points:
            idx_map = self.manual_points
            if 'start_idx' in idx_map and 'end_idx' in idx_map:
                start_idx = data.index[idx_map['start_idx']] if idx_map['start_idx'] < len(data) else data.index[-1]
                end_idx = data.index[idx_map['end_idx']] if idx_map['end_idx'] < len(data) else data.index[-1]
                # Determine start/end price based on actual high/low of the points
                temp_start_price = data.loc[start_idx, 'close']
                temp_end_price = data.loc[end_idx, 'close']
                start_price = data.loc[start_idx, 'low'] if temp_end_price > temp_start_price else data.loc[start_idx, 'high']
                end_price = data.loc[end_idx, 'high'] if temp_end_price > temp_start_price else data.loc[end_idx, 'low']
            else:
                raise ValueError("Manual points must include 'start_idx' and 'end_idx' keys")
        else:
            print("Warning: No method specified for finding speed line points.")
            return result # Or raise error

        if start_idx is None or end_idx is None or start_idx == end_idx:
            print("Warning: Could not determine valid start/end points for speed lines.")
            return result # Points not found or identical

        # Determine trend direction and price range
        is_uptrend = end_price > start_price
        price_range = abs(end_price - start_price)
        start_pos = result.index.get_loc(start_idx)
        end_pos = result.index.get_loc(end_idx)
        last_pos = len(result) - 1
        time_range = end_pos - start_pos
        if time_range == 0: time_range = 1 # Avoid division by zero

        # Calculate speed lines
        for level in self.levels:
            col_name = f"fib_speed_line_{level:.3f}".replace('.', '_')
            result[col_name] = np.nan # Initialize with NaN

            # Calculate the target price level at the end_idx time
            # This is the price level the speed line aims for at the end point's time
            if is_uptrend:
                level_price_at_end = start_price + (price_range * (1 - level)) # Level applied from start
            else:
                level_price_at_end = start_price - (price_range * (1 - level)) # Level applied from start

            # Calculate the slope of the speed line (from start_price at start_pos to level_price_at_end at end_pos)
            speed_line_price_delta = level_price_at_end - start_price
            slope = speed_line_price_delta / time_range

            # Project the line forward from the start point
            for i in range(start_pos, last_pos + self.projection_bars + 1):
                time_diff = i - start_pos
                line_price = start_price + (slope * time_diff)

                # Assign to DataFrame, extending if necessary
                if i <= last_pos:
                    current_idx = result.index[i]
                    result.loc[current_idx, col_name] = line_price
                else:
                    # Extend DataFrame for projection (simple approach)
                    proj_idx = result.index[-1] + pd.Timedelta(days=i-last_pos) # Crude projection index
                    if proj_idx not in result.index:
                        result.loc[proj_idx] = np.nan
                    result.loc[proj_idx, col_name] = line_price


        # Add metadata
        result['fib_speed_start_idx'] = False
        result.loc[start_idx, 'fib_speed_start_idx'] = True
        result['fib_speed_end_idx'] = False
        result.loc[end_idx, 'fib_speed_end_idx'] = True
        result['fib_speed_start_price'] = start_price
        result['fib_speed_end_price'] = end_price
        result['fib_speed_is_uptrend'] = is_uptrend

        return result

    def _detect_trend_points(self, data: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[pd.Timestamp], Optional[float]]:
        """Detect start and end points of a significant trend using swing highs/lows."""
        lookback_data = data.iloc[-self.swing_lookback:] if len(data) > self.swing_lookback else data
        if len(lookback_data) < 5: return None, None, None, None

        start_idx, start_price, end_idx, end_price = None, None, None, None
        try:
            # Find major low and high in the lookback period using utility
            swing_lows = find_swing_lows(lookback_data['low'], n=3)
            swing_highs = find_swing_highs(lookback_data['high'], n=3)
            low_points = lookback_data.loc[swing_lows]
            high_points = lookback_data.loc[swing_highs]

            if low_points.empty or high_points.empty:
                 return None, None, None, None

            # Use the earliest low/high and the latest high/low in the period
            first_low_idx = low_points.index[0]
            first_high_idx = high_points.index[0]
            last_low_idx = low_points.index[-1]
            last_high_idx = high_points.index[-1]

            # Determine overall trend in the lookback window
            if data.index.get_loc(first_low_idx) < data.index.get_loc(last_high_idx) and \
               data.index.get_loc(first_high_idx) < data.index.get_loc(last_high_idx): # Crude check for uptrend
                # Uptrend: Start=Earliest Low, End=Latest High
                start_idx = first_low_idx
                start_price = lookback_data.loc[start_idx, 'low']
                end_idx = last_high_idx
                end_price = lookback_data.loc[end_idx, 'high']
            elif data.index.get_loc(first_high_idx) < data.index.get_loc(last_low_idx) and \
                 data.index.get_loc(first_low_idx) < data.index.get_loc(last_low_idx): # Crude check for downtrend
                # Downtrend: Start=Earliest High, End=Latest Low
                start_idx = first_high_idx
                start_price = lookback_data.loc[start_idx, 'high']
                end_idx = last_low_idx
                end_price = lookback_data.loc[end_idx, 'low']
            else: # Default to most recent swing if trend unclear
                 if data.index.get_loc(last_low_idx) < data.index.get_loc(last_high_idx):
                     start_idx, start_price = last_low_idx, lookback_data.loc[last_low_idx, 'low']
                     end_idx, end_price = last_high_idx, lookback_data.loc[last_high_idx, 'high']
                 else:
                     start_idx, start_price = last_high_idx, lookback_data.loc[last_high_idx, 'high']
                     end_idx, end_price = last_low_idx, lookback_data.loc[last_low_idx, 'low']


        except NameError:
            print("Warning: Swing point detection utility not found. Using simple min/max.")
            # Fallback: Find major low and high in the lookback period
            low_idx = lookback_data['low'].idxmin()
            low_price = lookback_data.loc[low_idx, 'low']
            high_idx = lookback_data['high'].idxmax()
            high_price = lookback_data.loc[high_idx, 'high']

            # Assign start/end based on time order
            if lookback_data.index.get_loc(low_idx) < lookback_data.index.get_loc(high_idx):
                # Uptrend
                start_idx, start_price = low_idx, low_price
                end_idx, end_price = high_idx, high_price
            else:
                # Downtrend
                start_idx, start_price = high_idx, high_price
                end_idx, end_price = low_idx, low_price

        return start_idx, start_price, end_idx, end_price

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Speed/Resistance Lines',
            'description': 'Combines trendlines and Fibonacci ratios for S/R analysis',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'levels',
                    'description': 'List of Fibonacci levels (fractions) for line placement',
                    'type': 'list',
                    'default': [0.382, 0.618] # Or [0.333, 0.667]
                },
                {
                    'name': 'swing_lookback',
                    'description': 'Number of bars to look back for finding swing points',
                    'type': 'int',
                    'default': 50
                },
                {
                    'name': 'auto_detect_swings',
                    'description': 'Whether to automatically detect swing points',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'manual_points',
                    'description': 'Dictionary with manual point indices {start_idx, end_idx}',
                    'type': 'dict',
                    'default': None
                },
                {
                    'name': 'projection_bars',
                    'description': 'Number of bars to project lines into the future',
                    'type': 'int',
                    'default': 50
                }
            ]
        }


class FibonacciClusters(BaseIndicator):
    """
    Fibonacci Clusters

    Identifies price zones where multiple Fibonacci levels (from different tools
    like retracements, extensions, projections, fans, etc.) converge.
    These cluster zones are considered stronger potential support/resistance areas.
    Requires results from other Fibonacci indicators as input columns in the DataFrame.
    """

    category = "fibonacci"

    def __init__(
        self,
        cluster_tolerance_pct: float = 0.005, # 0.5% tolerance for levels to be considered clustered
        min_cluster_strength: int = 3, # Minimum number of levels to form a cluster
        # fib_indicator_results: Optional[List[pd.DataFrame]] = None, # Alt: Pass results if not in main df
        **kwargs
    ):
        """
        Initialize Fibonacci Clusters indicator.

        Args:
            cluster_tolerance_pct: Price proximity (as percentage of current price) to group levels.
            min_cluster_strength: Minimum number of converging levels to define a cluster.
            # fib_indicator_results: Optional list of pre-calculated Fibonacci indicator DataFrames.
            **kwargs: Additional parameters
        """
        self.cluster_tolerance_pct = cluster_tolerance_pct
        self.min_cluster_strength = min_cluster_strength
        # self.fib_indicator_results = fib_indicator_results
        # TODO: Add ability to run specified Fib indicators internally if results not provided

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify Fibonacci cluster zones based on existing Fibonacci level columns in the input data.

        Args:
            data: DataFrame with OHLCV data and columns containing calculated Fibonacci levels
                  (e.g., 'fib_retracement_0_618', 'fib_extension_1_618', 'fib_fan_0_382', etc.).

        Returns:
            DataFrame with cluster information (cluster center, strength, upper/lower bounds).
        """
        result = data.copy()
        result['fib_cluster_center'] = np.nan
        result['fib_cluster_strength'] = 0
        result['fib_cluster_upper'] = np.nan
        result['fib_cluster_lower'] = np.nan

        # Dynamically identify Fibonacci level columns in the input DataFrame
        fib_level_columns = []
        exclude_keywords = ['idx', 'price', 'range', 'direction', 'slope', 'width',
                            'strength', 'center', 'upper', 'lower', 'trend', 'base',
                            'radius', 'spiral', 'cluster'] # Keywords for metadata columns

        for col in data.columns:
            # Basic check: starts with 'fib_' and contains numbers (likely a level)
            if col.startswith('fib_') and any(char.isdigit() for char in col):
                # More robust check: exclude known metadata columns
                is_metadata = False
                for keyword in exclude_keywords:
                    if keyword in col:
                        is_metadata = True
                        break
                if not is_metadata:
                    fib_level_columns.append(col)

        if not fib_level_columns:
            print("Warning: No Fibonacci level columns found in the input DataFrame for cluster analysis.")
            print(f"Looked for columns starting with 'fib_' and containing numbers, excluding keywords: {exclude_keywords}")
            return result
        else:
            print(f"Found potential Fibonacci level columns for clustering: {fib_level_columns}")


        # Iterate through each time step (row)
        for i in range(len(result)):
            # Use 'close' price for tolerance calculation, fallback to previous if NaN
            current_price = result.iloc[i]['close']
            if pd.isna(current_price) and i > 0:
                current_price = result.iloc[i-1]['close']
            if pd.isna(current_price): continue # Skip if no valid price

            tolerance = current_price * self.cluster_tolerance_pct

            # Get all non-null Fibonacci levels at this time step
            levels_at_i = []
            for col in fib_level_columns:
                level_val = result.iloc[i][col]
                if pd.notna(level_val):
                    levels_at_i.append(level_val)

            if len(levels_at_i) < self.min_cluster_strength:
                continue # Not enough levels to potentially form a cluster

            # Sort levels for efficient clustering
            levels_at_i.sort()

            # Find clusters using a sliding window approach based on tolerance
            clusters = []
            if levels_at_i:
                start_cluster_idx = 0
                for j in range(1, len(levels_at_i)):
                    # If the current level is outside the tolerance range of the *start* level of the current cluster
                    if abs(levels_at_i[j] - levels_at_i[start_cluster_idx]) > tolerance:
                        # Check if the completed cluster is strong enough
                        if (j - start_cluster_idx) >= self.min_cluster_strength:
                            clusters.append(levels_at_i[start_cluster_idx:j])
                        # Start a new potential cluster
                        start_cluster_idx = j
                # Check the last potential cluster
                if (len(levels_at_i) - start_cluster_idx) >= self.min_cluster_strength:
                    clusters.append(levels_at_i[start_cluster_idx:])


            # Find the strongest cluster (most levels)
            # Optionally: prioritize cluster closest to current price if strengths are equal
            best_cluster = None
            max_strength = 0
            # min_dist_to_price = float('inf') # Uncomment to prioritize proximity for ties

            for cluster in clusters:
                strength = len(cluster)
                # cluster_center = np.mean(cluster) # Uncomment if needed
                # dist_to_price = abs(cluster_center - current_price) # Uncomment if needed

                if strength > max_strength:
                    max_strength = strength
                    best_cluster = cluster
                    # min_dist_to_price = dist_to_price # Uncomment if needed
                # Uncomment block for proximity tie-breaking
                # elif strength == max_strength and dist_to_price < min_dist_to_price:
                #     best_cluster = cluster
                #     min_dist_to_price = dist_to_price


            # Store the best cluster info for this time step
            if best_cluster:
                cluster_center = np.mean(best_cluster)
                cluster_strength = len(best_cluster)
                cluster_lower = min(best_cluster)
                cluster_upper = max(best_cluster)

                result.iloc[i, result.columns.get_loc('fib_cluster_center')] = cluster_center
                result.iloc[i, result.columns.get_loc('fib_cluster_strength')] = cluster_strength
                result.iloc[i, result.columns.get_loc('fib_cluster_upper')] = cluster_upper
                result.iloc[i, result.columns.get_loc('fib_cluster_lower')] = cluster_lower

        return result

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Clusters',
            'description': 'Identifies price zones where multiple Fibonacci levels converge',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'cluster_tolerance_pct',
                    'description': 'Price proximity (percentage of current price) to group levels',
                    'type': 'float',
                    'default': 0.005
                },
                {
                    'name': 'min_cluster_strength',
                    'description': 'Minimum number of levels to form a cluster',
                    'type': 'int',
                    'default': 3
                }
                # { # Optional parameter if allowing separate results input
                #     'name': 'fib_indicator_results',
                #     'description': 'Optional list of pre-calculated Fibonacci indicator DataFrames',
                #     'type': 'list',
                #     'default': None
                # }
            ]
        }

# --- End of New Advanced Fibonacci Tools ---
