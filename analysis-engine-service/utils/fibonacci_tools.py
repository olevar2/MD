"""
Fibonacci Tools Module

This module provides technical analysis tools based on Fibonacci retracements,
extensions, and related price projections.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from enum import Enum

from analysis_engine.analysis.advanced_ta.base import (
    AdvancedAnalysisBase,
    PatternRecognitionBase,
    detect_swings
)


class FibonacciLevel(Enum):
    """Standard Fibonacci levels"""
    LEVEL_0 = 0.0
    LEVEL_236 = 0.236
    LEVEL_382 = 0.382
    LEVEL_5 = 0.5
    LEVEL_618 = 0.618
    LEVEL_786 = 0.786
    LEVEL_1 = 1.0
    LEVEL_1272 = 1.272
    LEVEL_1618 = 1.618
    LEVEL_2618 = 2.618
    LEVEL_4236 = 4.236


class FibonacciRetracement(AdvancedAnalysisBase):
    """
    Fibonacci Retracement Analysis
    
    Calculates Fibonacci retracement levels based on significant swing high/low points,
    which can identify potential support and resistance areas.
    """
    
    def __init__(
        self,
        name: str = "FibonacciRetracement",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize Fibonacci Retracement analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {
            "price_column": "close",
            "high_column": "high",
            "low_column": "low",
            "auto_detect_swings": True,
            "lookback_period": 100,
            "levels": [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci retracement levels
        """
        result_df = df.copy()
        
        # Auto-detect swing points if enabled
        if self.parameters["auto_detect_swings"]:
            swing_points = self._auto_detect_swing_points(result_df)
        else:
            # Otherwise, use the first and last point in the dataframe
            first_idx = 0
            last_idx = len(result_df) - 1
            swing_points = [(first_idx, result_df.iloc[first_idx][self.parameters["price_column"]]),
                           (last_idx, result_df.iloc[last_idx][self.parameters["price_column"]])]
        
        # Calculate retracement levels for each swing point pair
        for i in range(len(swing_points) - 1):
            start_idx, start_price = swing_points[i]
            end_idx, end_price = swing_points[i+1]
            
            # Skip if the distance between points is too small
            price_range = abs(end_price - start_price)
            avg_price = (start_price + end_price) / 2
            if price_range / avg_price < 0.01:  # Less than 1% movement
                continue
                
            # Calculate retracement levels
            is_uptrend = end_price > start_price
            
            # Create columns for this swing
            swing_id = f"fib_swing_{i}"
            
            # Add swing start/end columns
            start_column = f"{swing_id}_start"
            end_column = f"{swing_id}_end"
            result_df[start_column] = np.nan
            result_df[end_column] = np.nan
            
            # Mark the swing start and end points
            result_df.iloc[start_idx, result_df.columns.get_loc(start_column)] = start_price
            result_df.iloc[end_idx, result_df.columns.get_loc(end_column)] = end_price
            
            # Add retracement level columns
            for level in self.parameters["levels"]:
                level_column = f"{swing_id}_level_{str(level).replace('.', '_')}"
                result_df[level_column] = np.nan
                
                # Calculate retracement price
                if is_uptrend:
                    retracement_price = end_price - (price_range * level)
                else:
                    retracement_price = end_price + (price_range * level)
                
                # Mark the level in the dataframe from the end point onwards
                result_df.iloc[end_idx:, result_df.columns.get_loc(level_column)] = retracement_price
        
        return result_df
    
    def _auto_detect_swing_points(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Automatically detect significant swing points
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of (index, price) tuples for swing points
        """
        # Use swing detection from base class if available
        if hasattr(self, 'detect_swings'):
            swing_df = detect_swings(df, lookback=5, price_col=self.parameters["price_column"])
            swing_points = []
            
            # Extract significant swings
            for i, row in swing_df.iterrows():
                if row['is_swing_high'] or row['is_swing_low']:
                    idx = df.index.get_loc(i)
                    price = row[self.parameters["price_column"]]
                    swing_points.append((idx, price))
            
            return swing_points
        
        # Simplified swing detection if detect_swings not available
        lookback = min(self.parameters["lookback_period"], len(df) - 2)
        swing_points = []
        
        high_col = self.parameters.get("high_column", self.parameters["price_column"])
        low_col = self.parameters.get("low_column", self.parameters["price_column"])
        
        # Find local maxima and minima
        for i in range(1, len(df) - 1):
            # Check if this is a significant high
            if (df[high_col].iloc[i] > df[high_col].iloc[i-1] and 
                df[high_col].iloc[i] > df[high_col].iloc[i+1]):
                swing_points.append((i, df[high_col].iloc[i]))
                
            # Check if this is a significant low
            elif (df[low_col].iloc[i] < df[low_col].iloc[i-1] and 
                  df[low_col].iloc[i] < df[low_col].iloc[i+1]):
                swing_points.append((i, df[low_col].iloc[i]))
        
        # Ensure we have at least two swing points
        if len(swing_points) < 2:
            swing_points = [(0, df[self.parameters["price_column"]].iloc[0]),
                           (len(df) - 1, df[self.parameters["price_column"]].iloc[-1])]
        
        return swing_points
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Fibonacci Retracement',
            'description': 'Calculates Fibonacci retracement levels based on swing points',
            'category': 'fibonacci',
            'parameters': [
                {
                    'name': 'price_column',
                    'description': 'Column to use for price data',
                    'type': 'str',
                    'default': 'close'
                },
                {
                    'name': 'auto_detect_swings',
                    'description': 'Automatically detect significant swing points',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for swing detection',
                    'type': 'int',
                    'default': 100
                },
                {
                    'name': 'levels',
                    'description': 'Fibonacci levels to calculate',
                    'type': 'list',
                    'default': [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
                }
            ]
        }


class FibonacciExtension(AdvancedAnalysisBase):
    """
    Fibonacci Extension Analysis
    
    Calculates Fibonacci extension levels based on three swing points (A, B, C),
    projecting potential price targets beyond point C.
    """
    
    def __init__(
        self,
        name: str = "FibonacciExtension",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize Fibonacci Extension analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {
            "price_column": "close",
            "high_column": "high",
            "low_column": "low",
            "auto_detect_swings": True,
            "lookback_period": 100,
            "levels": [0, 0.618, 1.0, 1.272, 1.618, 2.0, 2.618, 3.618, 4.236]
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci extension levels
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci extension levels
        """
        result_df = df.copy()
        
        # Auto-detect swing points if enabled
        if self.parameters["auto_detect_swings"]:
            swing_points = self._auto_detect_swing_points(result_df)
        else:
            # If not auto-detecting, take first, middle, and last point in dataframe
            first_idx = 0
            middle_idx = len(result_df) // 2
            last_idx = len(result_df) - 1
            
            swing_points = [
                (first_idx, result_df.iloc[first_idx][self.parameters["price_column"]]),
                (middle_idx, result_df.iloc[middle_idx][self.parameters["price_column"]]),
                (last_idx, result_df.iloc[last_idx][self.parameters["price_column"]])
            ]
        
        # We need at least 3 swing points for a valid extension calculation
        # (points A, B, and C to project D)
        if len(swing_points) < 3:
            return result_df
            
        # Calculate extension levels for consecutive swing point triplets
        for i in range(len(swing_points) - 2):
            # Get points A, B, C
            a_idx, a_price = swing_points[i]
            b_idx, b_price = swing_points[i+1]
            c_idx, c_price = swing_points[i+2]
            
            # Calculate the AB move and BC move
            ab_move = b_price - a_price
            bc_move = c_price - b_price
            
            # Skip if the movement is too small
            if abs(ab_move) / a_price < 0.01 or abs(bc_move) / b_price < 0.01:
                continue
                
            # Create columns for this extension
            ext_id = f"fib_ext_{i}"
            
            # Add points columns
            a_column = f"{ext_id}_point_a"
            b_column = f"{ext_id}_point_b"
            c_column = f"{ext_id}_point_c"
            
            result_df[a_column] = np.nan
            result_df[b_column] = np.nan
            result_df[c_column] = np.nan
            
            # Mark the swing points
            result_df.iloc[a_idx, result_df.columns.get_loc(a_column)] = a_price
            result_df.iloc[b_idx, result_df.columns.get_loc(b_column)] = b_price
            result_df.iloc[c_idx, result_df.columns.get_loc(c_column)] = c_price
            
            # Add extension level columns
            for level in self.parameters["levels"]:
                level_column = f"{ext_id}_level_{str(level).replace('.', '_')}"
                result_df[level_column] = np.nan
                
                # Calculate extension price (project from C based on AB move)
                extension_price = c_price + (ab_move * level)
                
                # Mark the level in the dataframe from point C onwards
                result_df.iloc[c_idx:, result_df.columns.get_loc(level_column)] = extension_price
        
        return result_df
    
    def _auto_detect_swing_points(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Automatically detect significant swing points
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of (index, price) tuples for swing points
        """
        # Similar to FibonacciRetracement._auto_detect_swing_points
        # Use swing detection from base class if available
        if hasattr(self, 'detect_swings'):
            swing_df = detect_swings(df, lookback=5, price_col=self.parameters["price_column"])
            swing_points = []
            
            # Extract significant swings
            for i, row in swing_df.iterrows():
                if row['is_swing_high'] or row['is_swing_low']:
                    idx = df.index.get_loc(i)
                    price = row[self.parameters["price_column"]]
                    swing_points.append((idx, price))
            
            return swing_points
            
        # Simplified swing detection
        swing_points = []
        high_col = self.parameters.get("high_column", self.parameters["price_column"])
        low_col = self.parameters.get("low_column", self.parameters["price_column"])
        
        # Find significant swing highs and lows
        for i in range(1, len(df) - 1):
            # Check for swing high
            if (df[high_col].iloc[i] > df[high_col].iloc[i-1] and 
                df[high_col].iloc[i] > df[high_col].iloc[i+1]):
                swing_points.append((i, df[high_col].iloc[i]))
                
            # Check for swing low
            elif (df[low_col].iloc[i] < df[low_col].iloc[i-1] and 
                  df[low_col].iloc[i] < df[low_col].iloc[i+1]):
                swing_points.append((i, df[low_col].iloc[i]))
        
        # Ensure we have at least three swing points
        if len(swing_points) < 3:
            swing_points = [
                (0, df[self.parameters["price_column"]].iloc[0]),
                (len(df) // 2, df[self.parameters["price_column"]].iloc[len(df) // 2]),
                (len(df) - 1, df[self.parameters["price_column"]].iloc[-1])
            ]
        
        return swing_points
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Fibonacci Extension',
            'description': 'Calculates Fibonacci extension levels for price projections',
            'category': 'fibonacci',
            'parameters': [
                {
                    'name': 'price_column',
                    'description': 'Column to use for price data',
                    'type': 'str',
                    'default': 'close'
                },
                {
                    'name': 'auto_detect_swings',
                    'description': 'Automatically detect significant swing points',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for swing detection',
                    'type': 'int',
                    'default': 100
                },
                {
                    'name': 'levels',
                    'description': 'Fibonacci extension levels to calculate',
                    'type': 'list',
                    'default': [0, 0.618, 1.0, 1.272, 1.618, 2.0, 2.618, 3.618, 4.236]
                }
            ]
        }


class FibonacciTimeZones(AdvancedAnalysisBase):
    """
    Fibonacci Time Zones
    
    Projects potential reversal or significant price action points based on
    Fibonacci sequence in the time domain.
    """
    
    def __init__(
        self,
        name: str = "FibonacciTimeZones",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize Fibonacci Time Zones analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {
            "start_from_significant_point": True,
            "max_levels": 8,
            "base_interval": 1  # Base interval multiplier
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci time zones
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci time zones
        """
        result_df = df.copy()
        
        # Determine the starting point
        if self.parameters["start_from_significant_point"]:
            start_idx = self._find_significant_point(result_df)
        else:
            start_idx = 0
        
        # Add a column to mark the starting point
        result_df["fib_time_start"] = 0
        result_df.iloc[start_idx, result_df.columns.get_loc("fib_time_start")] = 1
        
        # Calculate Fibonacci sequence for time zones
        fib_sequence = self._calculate_fibonacci_sequence(self.parameters["max_levels"])
        
        # Create time zone columns
        for i, fib_value in enumerate(fib_sequence):
            zone_idx = start_idx + fib_value * self.parameters["base_interval"]
            zone_column = f"fib_time_zone_{i+1}"
            
            # Add the column
            result_df[zone_column] = 0
            
            # Mark the time zone if it's within the dataframe
            if zone_idx < len(result_df):
                result_df.iloc[zone_idx, result_df.columns.get_loc(zone_column)] = 1
        
        return result_df
    
    def _find_significant_point(self, df: pd.DataFrame) -> int:
        """
        Find a significant price point to start Fibonacci time zones
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Index of the significant point
        """
        # Look for a significant high or low
        # This could be improved with more sophisticated swing detection
        high = df["high"].max()
        low = df["low"].min()
        
        high_idx = df["high"].idxmax()
        low_idx = df["low"].idxmin()
        
        # Use the earlier of the significant high or low
        if high_idx < low_idx:
            return df.index.get_loc(high_idx)
        else:
            return df.index.get_loc(low_idx)
    
    def _calculate_fibonacci_sequence(self, n: int) -> List[int]:
        """
        Calculate the Fibonacci sequence
        
        Args:
            n: Number of Fibonacci numbers to generate
            
        Returns:
            List of Fibonacci numbers
        """
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        return fib[:n]
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Fibonacci Time Zones',
            'description': 'Projects potential reversal points based on Fibonacci sequence in time',
            'category': 'fibonacci',
            'parameters': [
                {
                    'name': 'start_from_significant_point',
                    'description': 'Start sequence from a significant high or low',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'max_levels',
                    'description': 'Maximum number of Fibonacci levels to calculate',
                    'type': 'int',
                    'default': 8
                },
                {
                    'name': 'base_interval',
                    'description': 'Base interval multiplier for Fibonacci sequence',
                    'type': 'int',
                    'default': 1
                }
            ]
        }
