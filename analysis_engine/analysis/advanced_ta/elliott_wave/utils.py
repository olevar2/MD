"""
Elliott Wave Utilities Module.

This module provides utility functions for Elliott Wave analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def detect_zigzag_points(
    df: pd.DataFrame, 
    price_column: str, 
    high_column: str, 
    low_column: str, 
    zigzag_threshold: float
) -> List[Tuple[pd.Timestamp, float, bool]]:
    """
    Detect significant turning points using zigzag algorithm
    
    Args:
        df: DataFrame with OHLCV data
        price_column: Column to use for price data
        high_column: Column to use for high data
        low_column: Column to use for low data
        zigzag_threshold: Minimum percentage move for zigzag
        
    Returns:
        List of (timestamp, price, is_high) tuples for zigzag points
    """
    # Extract price columns
    if high_column in df.columns and low_column in df.columns:
        price = df[price_column].values
        highs = df[high_column].values
        lows = df[low_column].values
    else:
        # Use price column for both high and low
        price = df[price_column].values
        highs = price
        lows = price
    
    timestamps = df.index.to_pydatetime()
    
    zigzag_points = []
    in_uptrend = None
    last_price = None
    swing_high = None
    swing_low = None
    swing_high_price = -float('inf')
    swing_low_price = float('inf')
    
    for i in range(len(price)):
        current_price = price[i]
        current_high = highs[i]
        current_low = lows[i]
        
        # Initial point
        if last_price is None:
            last_price = current_price
            continue
        
        # Determine initial trend direction
        if in_uptrend is None:
            in_uptrend = current_price > last_price
            if in_uptrend:
                swing_low = (timestamps[i-1], last_price, False)
                zigzag_points.append(swing_low)
            else:
                swing_high = (timestamps[i-1], last_price, True)
                zigzag_points.append(swing_high)
            
            swing_high_price = current_high
            swing_low_price = current_low
            continue
        
        # Update swing points
        if in_uptrend:
            # Looking for higher highs
            if current_high > swing_high_price:
                swing_high_price = current_high
            
            # Check for reversal
            if current_price < (swing_high_price * (1 - zigzag_threshold)):
                # Reversal from uptrend to downtrend
                swing_high = (timestamps[i-1], swing_high_price, True)
                zigzag_points.append(swing_high)
                in_uptrend = False
                swing_low_price = current_low
        else:
            # Looking for lower lows
            if current_low < swing_low_price:
                swing_low_price = current_low
            
            # Check for reversal
            if current_price > (swing_low_price * (1 + zigzag_threshold)):
                # Reversal from downtrend to uptrend
                swing_low = (timestamps[i-1], swing_low_price, False)
                zigzag_points.append(swing_low)
                in_uptrend = True
                swing_high_price = current_high
        
        last_price = current_price
    
    # Add the most recent swing point if it hasn't been added
    if in_uptrend and swing_high_price > -float('inf'):
        swing_high = (timestamps[-1], swing_high_price, True)
        zigzag_points.append(swing_high)
    elif not in_uptrend and swing_low_price < float('inf'):
        swing_low = (timestamps[-1], swing_low_price, False)
        zigzag_points.append(swing_low)
    
    return zigzag_points


def detect_swing_points(prices: pd.Series, window: int = 5) -> List[Tuple[int, float, str]]:
    """
    Detect swing highs and lows using local extrema
    
    Args:
        prices: Series of price data
        window: Window size for local extrema detection
        
    Returns:
        List of (index, price, "high"/"low") tuples
    """
    swing_points = []
    
    for i in range(window, len(prices) - window):
        window_left = prices.iloc[i - window:i]
        window_right = prices.iloc[i + 1:i + window + 1]
        current_price = prices.iloc[i]
        
        # Check for swing high
        if current_price > max(window_left.max(), window_right.max()):
            swing_points.append((i, current_price, "high"))
            
        # Check for swing low
        if current_price < min(window_left.min(), window_right.min()):
            swing_points.append((i, current_price, "low"))
    
    return sorted(swing_points, key=lambda x: x[0])


def calculate_wave_sharpness(df: pd.DataFrame, start_idx: int, end_idx: int, price_column: str) -> float:
    """
    Calculate the sharpness of a wave by measuring its direct route vs actual route
    
    Args:
        df: DataFrame with price data
        start_idx: Starting index of the wave
        end_idx: Ending index of the wave
        price_column: Column to use for price data
        
    Returns:
        Sharpness value (higher = sharper wave)
    """
    if start_idx >= end_idx:
        return 0.0
        
    # Extract price data for the wave
    wave_slice = df[price_column].iloc[start_idx:end_idx+1]
    
    # Calculate direct distance (start point to end point)
    direct_distance = abs(wave_slice.iloc[-1] - wave_slice.iloc[0])
    
    # Calculate actual traveled distance (sum of all moves)
    traveled_distance = np.sum(np.abs(wave_slice.diff().dropna()))
    
    # Sharpness = direct / traveled (higher value means more direct/sharp move)
    if traveled_distance > 0:
        return direct_distance / traveled_distance
    else:
        return 0.0