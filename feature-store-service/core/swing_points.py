"""
Swing Points Utility Module.

This module provides functions for detecting swing points in price data.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Tuple


def find_swing_highs(
    data: Union[pd.Series, np.ndarray],
    n: int = 2,
    strict: bool = True
) -> Union[pd.Series, np.ndarray]:
    """
    Find swing high points in a price series.
    
    A swing high is a local maximum where the n bars before and after are lower.
    
    Args:
        data: Price series (pandas Series or numpy array)
        n: Number of bars to look before and after for comparison
        strict: If True, all n bars must be lower; if False, any lower bar is sufficient
        
    Returns:
        Boolean series or array with True at swing high positions
    """
    if isinstance(data, pd.Series):
        swing_highs = pd.Series(False, index=data.index)
        values = data.values
    else:
        swing_highs = np.zeros(len(data), dtype=bool)
        values = data
    
    for i in range(n, len(values) - n):
        if strict:
            # All n bars before and after must be lower
            if all(values[i] > values[i-j] for j in range(1, n+1)) and \
               all(values[i] > values[i+j] for j in range(1, n+1)):
                if isinstance(swing_highs, pd.Series):
                    swing_highs.iloc[i] = True
                else:
                    swing_highs[i] = True
        else:
            # At least one bar before and after must be lower
            if any(values[i] > values[i-j] for j in range(1, n+1)) and \
               any(values[i] > values[i+j] for j in range(1, n+1)):
                if isinstance(swing_highs, pd.Series):
                    swing_highs.iloc[i] = True
                else:
                    swing_highs[i] = True
    
    return swing_highs


def find_swing_lows(
    data: Union[pd.Series, np.ndarray],
    n: int = 2,
    strict: bool = True
) -> Union[pd.Series, np.ndarray]:
    """
    Find swing low points in a price series.
    
    A swing low is a local minimum where the n bars before and after are higher.
    
    Args:
        data: Price series (pandas Series or numpy array)
        n: Number of bars to look before and after for comparison
        strict: If True, all n bars must be higher; if False, any higher bar is sufficient
        
    Returns:
        Boolean series or array with True at swing low positions
    """
    if isinstance(data, pd.Series):
        swing_lows = pd.Series(False, index=data.index)
        values = data.values
    else:
        swing_lows = np.zeros(len(data), dtype=bool)
        values = data
    
    for i in range(n, len(values) - n):
        if strict:
            # All n bars before and after must be higher
            if all(values[i] < values[i-j] for j in range(1, n+1)) and \
               all(values[i] < values[i+j] for j in range(1, n+1)):
                if isinstance(swing_lows, pd.Series):
                    swing_lows.iloc[i] = True
                else:
                    swing_lows[i] = True
        else:
            # At least one bar before and after must be higher
            if any(values[i] < values[i-j] for j in range(1, n+1)) and \
               any(values[i] < values[i+j] for j in range(1, n+1)):
                if isinstance(swing_lows, pd.Series):
                    swing_lows.iloc[i] = True
                else:
                    swing_lows[i] = True
    
    return swing_lows


def find_swing_points(
    data: Union[pd.Series, np.ndarray],
    n: int = 2,
    strict: bool = True
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Find both swing high and swing low points in a price series.
    
    Args:
        data: Price series (pandas Series or numpy array)
        n: Number of bars to look before and after for comparison
        strict: If True, all n bars must be higher/lower; if False, any higher/lower bar is sufficient
        
    Returns:
        Tuple of (swing_highs, swing_lows) boolean series or arrays
    """
    swing_highs = find_swing_highs(data, n, strict)
    swing_lows = find_swing_lows(data, n, strict)
    
    return swing_highs, swing_lows
