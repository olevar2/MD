"""
Andrews Pitchfork Module

This module provides implementation of Andrews' Pitchfork, a technical analysis
tool consisting of three parallel trendlines based on three points selected on a chart.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import math

from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase


class AndrewsPitchfork(AdvancedAnalysisBase):
    """
    Andrews Pitchfork
    
    Calculates the three trendlines that make up Andrews' Pitchfork, which can be used
    to identify potential support, resistance, and the median line for price channels.
    """
    
    def __init__(
        self,
        name: str = "AndrewsPitchfork",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize Andrews Pitchfork analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {
            "price_column": "close",
            "high_column": "high",
            "low_column": "low",
            "auto_detect_points": True,
            "use_fractals": True,  # Use fractal analysis for point detection
            "lookback_period": 50,  # Period to look back for point detection
            "additional_lines": True  # Add additional lines (e.g., mid-channel lines)
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Andrews Pitchfork trendlines
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Andrews Pitchfork trendlines
        """
        result_df = df.copy()
        
        # Automatically detect the three points for the pitchfork
        if self.parameters["auto_detect_points"]:
            pivot_points = self._detect_pivot_points(result_df)
        else:
            # Use first, middle and last point as default
            length = len(result_df)
            pivot_points = [
                (0, result_df.iloc[0][self.parameters["price_column"]]),
                (length // 2, result_df.iloc[length // 2][self.parameters["price_column"]]),
                (length - 1, result_df.iloc[-1][self.parameters["price_column"]])
            ]
        
        # We need at least 3 pivot points to draw a pitchfork
        if len(pivot_points) < 3:
            return result_df
            
        # Get the three points for the pitchfork
        p1_idx, p1_price = pivot_points[0]  # First pivot (typically a significant high or low)
        p2_idx, p2_price = pivot_points[1]  # Second pivot (a reaction high/low)
        p3_idx, p3_price = pivot_points[2]  # Third pivot (another reaction high/low)
        
        # Add columns to mark the pivot points
        result_df['pitchfork_point1'] = np.nan
        result_df['pitchfork_point2'] = np.nan
        result_df['pitchfork_point3'] = np.nan
        
        result_df.iloc[p1_idx, result_df.columns.get_loc('pitchfork_point1')] = p1_price
        result_df.iloc[p2_idx, result_df.columns.get_loc('pitchfork_point2')] = p2_price
        result_df.iloc[p3_idx, result_df.columns.get_loc('pitchfork_point3')] = p3_price
        
        # Calculate the midpoint of p2 and p3
        mid_idx = (p2_idx + p3_idx) // 2
        mid_price = (p2_price + p3_price) / 2
        
        # Calculate slope of the median line (from p1 to the midpoint)
        if mid_idx == p1_idx:
            # Avoid division by zero
            median_slope = 0
        else:
            median_slope = (mid_price - p1_price) / (mid_idx - p1_idx)
        
        # Calculate y-intercept of the median line
        median_intercept = p1_price - median_slope * p1_idx
        
        # Calculate the pitchfork lines for each bar
        result_df['pitchfork_median_line'] = np.nan
        result_df['pitchfork_upper_line'] = np.nan
        result_df['pitchfork_lower_line'] = np.nan
        
        for i in range(p1_idx, len(result_df)):
            # Median line (handle)
            median_value = median_slope * i + median_intercept
            result_df.iloc[i, result_df.columns.get_loc('pitchfork_median_line')] = median_value
            
            # Upper line (upper tine)
            upper_value = median_value + (p2_price - mid_price)
            result_df.iloc[i, result_df.columns.get_loc('pitchfork_upper_line')] = upper_value
            
            # Lower line (lower tine)
            lower_value = median_value + (p3_price - mid_price)
            result_df.iloc[i, result_df.columns.get_loc('pitchfork_lower_line')] = lower_value
        
        # Add additional lines if requested (midlines between median and upper/lower)
        if self.parameters["additional_lines"]:
            result_df['pitchfork_upper_midline'] = np.nan
            result_df['pitchfork_lower_midline'] = np.nan
            
            for i in range(p1_idx, len(result_df)):
                median_value = result_df.iloc[i]['pitchfork_median_line']
                upper_value = result_df.iloc[i]['pitchfork_upper_line']
                lower_value = result_df.iloc[i]['pitchfork_lower_line']
                
                # Midlines between median and upper/lower lines
                upper_midline = (median_value + upper_value) / 2
                lower_midline = (median_value + lower_value) / 2
                
                result_df.iloc[i, result_df.columns.get_loc('pitchfork_upper_midline')] = upper_midline
                result_df.iloc[i, result_df.columns.get_loc('pitchfork_lower_midline')] = lower_midline
        
        return result_df
    
    def _detect_pivot_points(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Automatically detect pivot points for Andrews Pitchfork
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of (index, price) tuples for pivot points
        """
        # Limit data to lookback period
        lookback = min(self.parameters["lookback_period"], len(df))
        lookback_df = df.iloc[-lookback:]
        
        pivot_points = []
        price_col = self.parameters["price_column"]
        high_col = self.parameters.get("high_column", price_col)
        low_col = self.parameters.get("low_column", price_col)
        
        # If using fractal analysis for better pivot point detection
        if self.parameters["use_fractals"]:
            # Process highs - look for fractal tops (a high with 2 lower highs on each side)
            for i in range(2, len(lookback_df) - 2):
                if (lookback_df[high_col].iloc[i] > lookback_df[high_col].iloc[i-1] and
                    lookback_df[high_col].iloc[i] > lookback_df[high_col].iloc[i-2] and
                    lookback_df[high_col].iloc[i] > lookback_df[high_col].iloc[i+1] and
                    lookback_df[high_col].iloc[i] > lookback_df[high_col].iloc[i+2]):
                    # Fractal high found
                    orig_idx = len(df) - lookback + i
                    pivot_points.append((orig_idx, lookback_df[high_col].iloc[i]))
            
            # Process lows - look for fractal bottoms (a low with 2 higher lows on each side)
            for i in range(2, len(lookback_df) - 2):
                if (lookback_df[low_col].iloc[i] < lookback_df[low_col].iloc[i-1] and
                    lookback_df[low_col].iloc[i] < lookback_df[low_col].iloc[i-2] and
                    lookback_df[low_col].iloc[i] < lookback_df[low_col].iloc[i+1] and
                    lookback_df[low_col].iloc[i] < lookback_df[low_col].iloc[i+2]):
                    # Fractal low found
                    orig_idx = len(df) - lookback + i
                    pivot_points.append((orig_idx, lookback_df[low_col].iloc[i]))
        else:
            # Simpler method - just detect local highs and lows
            for i in range(1, len(lookback_df) - 1):
                # Check for local high
                if lookback_df[high_col].iloc[i] > lookback_df[high_col].iloc[i-1] and lookback_df[high_col].iloc[i] > lookback_df[high_col].iloc[i+1]:
                    orig_idx = len(df) - lookback + i
                    pivot_points.append((orig_idx, lookback_df[high_col].iloc[i]))
                
                # Check for local low
                if lookback_df[low_col].iloc[i] < lookback_df[low_col].iloc[i-1] and lookback_df[low_col].iloc[i] < lookback_df[low_col].iloc[i+1]:
                    orig_idx = len(df) - lookback + i
                    pivot_points.append((orig_idx, lookback_df[low_col].iloc[i]))
        
        # Sort pivot points by index
        pivot_points.sort(key=lambda x: x[0])
        
        # We need exactly 3 pivot points for the Andrews Pitchfork
        # If we have more than 3, pick most significant ones
        if len(pivot_points) > 3:
            # Find the most significant high and low
            highs = sorted([p for p in pivot_points], key=lambda x: x[1], reverse=True)
            lows = sorted([p for p in pivot_points], key=lambda x: x[1])
            
            # Choose 3 points: highest high, lowest low, and the most recent high/low
            highest = highs[0]
            lowest = lows[0]
            
            # For the third point, use most recent significant pivot that's not already used
            remaining = [p for p in pivot_points if p != highest and p != lowest]
            if remaining:
                most_recent = max(remaining, key=lambda x: x[0])
                selected_points = [highest, lowest, most_recent]
            else:
                selected_points = [highest, lowest]
                
            # Sort the points by index
            selected_points.sort(key=lambda x: x[0])
            return selected_points
        
        # If we have less than 3 points, use what we have
        return pivot_points
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Andrews Pitchfork',
            'description': 'Calculates Andrews Pitchfork trendlines for price channels',
            'category': 'trendline',
            'parameters': [
                {
                    'name': 'price_column',
                    'description': 'Primary price column for calculations',
                    'type': 'str',
                    'default': 'close'
                },
                {
                    'name': 'auto_detect_points',
                    'description': 'Automatically detect pivot points for the pitchfork',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'use_fractals',
                    'description': 'Use fractal analysis for better pivot point detection',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'lookback_period',
                    'description': 'Period to look back for point detection',
                    'type': 'int',
                    'default': 50
                },
                {
                    'name': 'additional_lines',
                    'description': 'Draw additional channel lines (midlines)',
                    'type': 'bool',
                    'default': True
                }
            ]
        }


class SchiffPitchfork(AndrewsPitchfork):
    """
    Schiff Pitchfork
    
    A modified version of Andrews' Pitchfork where the median line starts from
    the midpoint between points 1 and 2, creating a different channel.
    """
    
    def __init__(
        self,
        name: str = "SchiffPitchfork",
        parameters: Dict[str, Any] = None
    ):
        """Initialize Schiff Pitchfork analyzer"""
        super().__init__(name, parameters)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Schiff Pitchfork trendlines
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Schiff Pitchfork trendlines
        """
        result_df = df.copy()
        
        # Detect pivot points (same as Andrews' Pitchfork)
        if self.parameters["auto_detect_points"]:
            pivot_points = self._detect_pivot_points(result_df)
        else:
            length = len(result_df)
            pivot_points = [
                (0, result_df.iloc[0][self.parameters["price_column"]]),
                (length // 2, result_df.iloc[length // 2][self.parameters["price_column"]]),
                (length - 1, result_df.iloc[-1][self.parameters["price_column"]])
            ]
        
        if len(pivot_points) < 3:
            return result_df
            
        # Get the three points for the pitchfork
        p1_idx, p1_price = pivot_points[0]
        p2_idx, p2_price = pivot_points[1]
        p3_idx, p3_price = pivot_points[2]
        
        # Add columns to mark the pivot points
        result_df['schiff_point1'] = np.nan
        result_df['schiff_point2'] = np.nan
        result_df['schiff_point3'] = np.nan
        
        result_df.iloc[p1_idx, result_df.columns.get_loc('schiff_point1')] = p1_price
        result_df.iloc[p2_idx, result_df.columns.get_loc('schiff_point2')] = p2_price
        result_df.iloc[p3_idx, result_df.columns.get_loc('schiff_point3')] = p3_price
        
        # Schiff modification: use the midpoint of p1 and p2 as the starting point of the median line
        mid_p1_p2_idx = (p1_idx + p2_idx) // 2
        mid_p1_p2_price = (p1_price + p2_price) / 2
        
        # Calculate the midpoint of p2 and p3
        mid_p2_p3_idx = (p2_idx + p3_idx) // 2
        mid_p2_p3_price = (p2_price + p3_price) / 2
        
        # Calculate slope of the median line (from midpoint of p1-p2 to midpoint of p2-p3)
        if mid_p2_p3_idx == mid_p1_p2_idx:
            # Avoid division by zero
            median_slope = 0
        else:
            median_slope = (mid_p2_p3_price - mid_p1_p2_price) / (mid_p2_p3_idx - mid_p1_p2_idx)
        
        # Calculate y-intercept of the median line
        median_intercept = mid_p1_p2_price - median_slope * mid_p1_p2_idx
        
        # Calculate the pitchfork lines for each bar
        result_df['schiff_median_line'] = np.nan
        result_df['schiff_upper_line'] = np.nan
        result_df['schiff_lower_line'] = np.nan
        
        for i in range(mid_p1_p2_idx, len(result_df)):
            # Median line
            median_value = median_slope * i + median_intercept
            result_df.iloc[i, result_df.columns.get_loc('schiff_median_line')] = median_value
            
            # Upper and lower lines parallel to median line and passing through p2 and p3
            p2_deviation = p2_price - (median_slope * p2_idx + median_intercept)
            p3_deviation = p3_price - (median_slope * p3_idx + median_intercept)
            
            upper_value = median_value + p2_deviation
            lower_value = median_value + p3_deviation
            
            result_df.iloc[i, result_df.columns.get_loc('schiff_upper_line')] = upper_value
            result_df.iloc[i, result_df.columns.get_loc('schiff_lower_line')] = lower_value
        
        # Add additional lines if requested
        if self.parameters["additional_lines"]:
            result_df['schiff_upper_midline'] = np.nan
            result_df['schiff_lower_midline'] = np.nan
            
            for i in range(mid_p1_p2_idx, len(result_df)):
                median_value = result_df.iloc[i]['schiff_median_line']
                upper_value = result_df.iloc[i]['schiff_upper_line']
                lower_value = result_df.iloc[i]['schiff_lower_line']
                
                upper_midline = (median_value + upper_value) / 2
                lower_midline = (median_value + lower_value) / 2
                
                result_df.iloc[i, result_df.columns.get_loc('schiff_upper_midline')] = upper_midline
                result_df.iloc[i, result_df.columns.get_loc('schiff_lower_midline')] = lower_midline
        
        return result_df
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Schiff Pitchfork',
            'description': 'Modified Andrews Pitchfork with median line starting from midpoint of points 1 and 2',
            'category': 'trendline',
            'parameters': [
                {
                    'name': 'price_column',
                    'description': 'Primary price column for calculations',
                    'type': 'str',
                    'default': 'close'
                },
                {
                    'name': 'auto_detect_points',
                    'description': 'Automatically detect pivot points for the pitchfork',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'use_fractals',
                    'description': 'Use fractal analysis for better pivot point detection',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'lookback_period',
                    'description': 'Period to look back for point detection',
                    'type': 'int',
                    'default': 50
                },
                {
                    'name': 'additional_lines',
                    'description': 'Draw additional channel lines (midlines)',
                    'type': 'bool',
                    'default': True
                }
            ]
        }
