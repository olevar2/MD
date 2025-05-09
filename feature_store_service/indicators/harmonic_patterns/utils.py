"""
Harmonic Pattern Utilities Module.

This module provides utility functions for harmonic pattern analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def identify_pivots(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Identify significant pivot points in the price data.
    
    Args:
        data: DataFrame with OHLCV data
        window: Number of bars to look before/after for pivot detection
        
    Returns:
        DataFrame with pivot points identified
    """
    # Make a copy to avoid modifying the input data
    result = data.copy()
    
    # Identify pivot highs and lows
    result['pivot_high'] = 0
    result['pivot_low'] = 0
    
    for i in range(window, len(result) - window):
        # Check if this bar's high is higher than all bars in the window before and after
        if all(result.loc[i, 'high'] > result.loc[i-window:i-1, 'high']) and \
           all(result.loc[i, 'high'] > result.loc[i+1:i+window, 'high']):
            result.loc[i, 'pivot_high'] = 1
            
        # Check if this bar's low is lower than all bars in the window before and after
        if all(result.loc[i, 'low'] < result.loc[i-window:i-1, 'low']) and \
           all(result.loc[i, 'low'] < result.loc[i+1:i+window, 'low']):
            result.loc[i, 'pivot_low'] = 1
            
    # Store pivot point values
    result['pivot_high_value'] = result['high'] * result['pivot_high']
    result['pivot_low_value'] = result['low'] * result['pivot_low']
    
    # Replace zeros with NaN for easier manipulation
    result['pivot_high_value'].replace(0, np.nan, inplace=True)
    result['pivot_low_value'].replace(0, np.nan, inplace=True)
    
    return result


def calculate_ratio(value1: float, value2: float) -> float:
    """
    Calculate ratio between two values (avoiding division by zero).
    
    Args:
        value1: First value
        value2: Second value
        
    Returns:
        Ratio between values
    """
    if value2 == 0:
        return float('inf')
    return abs(value1 / value2)


def ratio_matches(actual_ratio: float, target_ratio: float, tolerance: float) -> bool:
    """
    Check if an actual ratio matches a target ratio within tolerance.
    
    Args:
        actual_ratio: Actual calculated ratio
        target_ratio: Target ratio to match
        tolerance: Acceptable tolerance range
        
    Returns:
        True if ratio matches within tolerance, False otherwise
    """
    return abs(actual_ratio - target_ratio) <= tolerance * target_ratio