"""
Base module for Fibonacci analysis tools.

This module provides base classes, enums, and common functionality for Fibonacci analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum
import math

from core.base_indicator import BaseIndicator

# Import swing point detection functions
# Define swing point detection functions
def find_swing_highs(data, n=5, **kwargs):
    """
    Find swing high points in the data.
    
    Args:
        data: DataFrame or Series with price data
        n: Number of bars to look back and forward
        **kwargs: Additional parameters (for compatibility)
        
    Returns:
        List of (index, price) tuples for swing high points
    """
    lookback = kwargs.get('lookback', n)
    
    if isinstance(data, pd.DataFrame):
        if 'high' in data.columns:
            values = data['high'].values
        else:
            values = data.iloc[:, 0].values
    else:
        values = data.values if hasattr(data, 'values') else data
        
    swing_highs = []
    for i in range(lookback, len(values) - lookback):
        if all(values[i] > values[i-j] for j in range(1, lookback+1)) and \
           all(values[i] > values[i+j] for j in range(1, lookback+1)):
            if isinstance(data, pd.DataFrame):
                swing_highs.append((data.index[i], values[i]))
            else:
                swing_highs.append((i, values[i]))
    return swing_highs

def find_swing_lows(data, n=5, **kwargs):
    """
    Find swing low points in the data.
    
    Args:
        data: DataFrame or Series with price data
        n: Number of bars to look back and forward
        **kwargs: Additional parameters (for compatibility)
        
    Returns:
        List of (index, price) tuples for swing low points
    """
    lookback = kwargs.get('lookback', n)
    
    if isinstance(data, pd.DataFrame):
        if 'low' in data.columns:
            values = data['low'].values
        else:
            values = data.iloc[:, 0].values
    else:
        values = data.values if hasattr(data, 'values') else data
        
    swing_lows = []
    for i in range(lookback, len(values) - lookback):
        if all(values[i] < values[i-j] for j in range(1, lookback+1)) and \
           all(values[i] < values[i+j] for j in range(1, lookback+1)):
            if isinstance(data, pd.DataFrame):
                swing_lows.append((data.index[i], values[i]))
            else:
                swing_lows.append((i, values[i]))
    return swing_lows


class TrendDirection(Enum):
    """Direction of price trend for Fibonacci analysis"""
    UPTREND = "uptrend"  # Lower swing point to higher swing point
    DOWNTREND = "downtrend"  # Higher swing point to lower swing point


class FibonacciBase(BaseIndicator):
    """
    Base class for all Fibonacci indicators.
    
    Provides common functionality for Fibonacci-based analysis tools.
    """
    
    category = "fibonacci"
    
    def __init__(self, **kwargs):
        """
        Initialize Fibonacci base indicator.
        
        Args:
            **kwargs: Additional parameters
        """
        # BaseIndicator doesn't have an __init__ method, so we don't call super().__init__
        self.name = kwargs.pop('name', self.__class__.__name__)
    
    def _detect_swing_points(self, data: pd.DataFrame) -> Tuple[int, float, int, float]:
        """
        Detect significant swing points for Fibonacci analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (start_idx, start_price, end_idx, end_price)
        """
        # Use a default lookback if not specified
        lookback = getattr(self, 'swing_lookback', 5)
        
        # Find swing highs and lows
        swing_highs = find_swing_highs(data, n=lookback)
        swing_lows = find_swing_lows(data, n=lookback)
        
        # Combine and sort by index
        all_swings = []
        for idx, price in swing_highs:
            all_swings.append((idx, price, 'high'))
        for idx, price in swing_lows:
            all_swings.append((idx, price, 'low'))
        
        all_swings.sort(key=lambda x: x[0])
        
        # Need at least 2 swing points
        if len(all_swings) < 2:
            # Use first and last points if not enough swings
            start_idx = 0
            end_idx = len(data) - 1
            start_price = data.iloc[start_idx]['close']
            end_price = data.iloc[end_idx]['close']
            return start_idx, start_price, end_idx, end_price
        
        # Find the most significant swing points
        # For simplicity, use the first and last swing points
        start_idx, start_price, start_type = all_swings[0]
        end_idx, end_price, end_type = all_swings[-1]
        
        return start_idx, start_price, end_idx, end_price
    
    def _detect_three_points(self, data: pd.DataFrame) -> Tuple[int, float, int, float, int, float]:
        """
        Detect three significant points for Fibonacci extension analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (start_idx, start_price, end_idx, end_price, retr_idx, retr_price)
        """
        # First detect the initial swing points
        start_idx, start_price, end_idx, end_price = self._detect_swing_points(data)
        
        # Determine trend direction
        trend = TrendDirection.UPTREND if end_price > start_price else TrendDirection.DOWNTREND
        
        # Find a retracement point after the end point
        retr_idx = None
        retr_price = None
        
        # Look for a retracement in the opposite direction
        if trend == TrendDirection.UPTREND:
            # For uptrend, look for a lower point after the high
            for i in range(end_idx + 1, len(data)):
                if data.iloc[i]['low'] < end_price:
                    retr_idx = i
                    retr_price = data.iloc[i]['low']
                    break
        else:
            # For downtrend, look for a higher point after the low
            for i in range(end_idx + 1, len(data)):
                if data.iloc[i]['high'] > end_price:
                    retr_idx = i
                    retr_price = data.iloc[i]['high']
                    break
        
        # If no retracement found, use the last point
        if retr_idx is None:
            retr_idx = len(data) - 1
            retr_price = data.iloc[retr_idx]['close']
        
        return start_idx, start_price, end_idx, end_price, retr_idx, retr_price
    
    def _calculate_fibonacci_levels(
        self, 
        start_price: float, 
        end_price: float, 
        levels: List[float]
    ) -> Dict[float, float]:
        """
        Calculate Fibonacci levels based on start and end prices.
        
        Args:
            start_price: Starting price point
            end_price: Ending price point
            levels: List of Fibonacci ratios to calculate
            
        Returns:
            Dictionary mapping Fibonacci ratios to price levels
        """
        price_range = end_price - start_price
        result = {}
        
        for level in levels:
            level_price = start_price + (price_range * level)
            result[level] = level_price
        
        return result
    
    def _format_column_name(self, prefix: str, level: float) -> str:
        """
        Format a column name for a Fibonacci level.
        
        Args:
            prefix: Column name prefix
            level: Fibonacci level
            
        Returns:
            Formatted column name
        """
        level_str = str(level).replace('.', '_')
        return f"{prefix}_{level_str}"
    
    def _get_trend_direction(self, start_price: float, end_price: float) -> TrendDirection:
        """
        Determine trend direction based on start and end prices.
        
        Args:
            start_price: Starting price point
            end_price: Ending price point
            
        Returns:
            TrendDirection enum value
        """
        return TrendDirection.UPTREND if end_price > start_price else TrendDirection.DOWNTREND