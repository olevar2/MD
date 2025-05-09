"""
Volatility Utilities Module.

This module provides utility functions for volatility indicators.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def calculate_true_range(high: pd.Series, low: pd.Series, close_prev: pd.Series) -> pd.Series:
    """
    Calculate True Range for volatility indicators.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close_prev: Series of previous close prices
        
    Returns:
        Series with True Range values
    """
    high_low = high - low
    high_close_prev = abs(high - close_prev)
    low_close_prev = abs(low - close_prev)
    
    return pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)


def calculate_volatility_ratio(
    short_vol: pd.Series, 
    long_vol: pd.Series, 
    smoothing_period: int = 5
) -> pd.Series:
    """
    Calculate ratio between short-term and long-term volatility.
    
    Args:
        short_vol: Series of short-term volatility values
        long_vol: Series of long-term volatility values
        smoothing_period: Period for smoothing the ratio
        
    Returns:
        Series with volatility ratio values
    """
    ratio = short_vol / long_vol
    return ratio.rolling(window=smoothing_period).mean()


def calculate_volatility_percentile(
    volatility: pd.Series, 
    lookback_period: int = 100
) -> pd.Series:
    """
    Calculate percentile rank of current volatility relative to history.
    
    Args:
        volatility: Series of volatility values
        lookback_period: Historical period to use for percentile calculation
        
    Returns:
        Series with volatility percentile values (0-100)
    """
    return volatility.rolling(window=lookback_period).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x) * 100
    )


def calculate_volatility_breakout(
    volatility: pd.Series, 
    lookback_period: int = 20, 
    threshold: float = 2.0
) -> pd.Series:
    """
    Detect volatility breakouts (spikes above historical average).
    
    Args:
        volatility: Series of volatility values
        lookback_period: Period for calculating average volatility
        threshold: Threshold multiplier for breakout detection
        
    Returns:
        Series with volatility breakout signals (0 or 1)
    """
    avg_vol = volatility.rolling(window=lookback_period).mean()
    std_vol = volatility.rolling(window=lookback_period).std()
    
    breakout = pd.Series(0, index=volatility.index)
    breakout[volatility > avg_vol + threshold * std_vol] = 1
    
    return breakout


def calculate_volatility_regime(
    volatility: pd.Series, 
    lookback_period: int = 100, 
    high_threshold: float = 0.75, 
    low_threshold: float = 0.25
) -> pd.Series:
    """
    Classify volatility regime (high, normal, low).
    
    Args:
        volatility: Series of volatility values
        lookback_period: Period for calculating volatility percentiles
        high_threshold: Percentile threshold for high volatility regime
        low_threshold: Percentile threshold for low volatility regime
        
    Returns:
        Series with volatility regime values (1=high, 0=normal, -1=low)
    """
    vol_percentile = calculate_volatility_percentile(volatility, lookback_period)
    
    regime = pd.Series(0, index=volatility.index)  # Default: normal regime
    regime[vol_percentile > high_threshold * 100] = 1  # High volatility
    regime[vol_percentile < low_threshold * 100] = -1  # Low volatility
    
    return regime