"""
Trend Indicators Module.

This module provides implementations of various trend-based indicators.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from core.base_indicator import BaseIndicator


class AverageDirectionalIndex(BaseIndicator):
    """
    Average Directional Index (ADX) indicator.
    
    This trend indicator measures the strength of a trend regardless of 
    direction, with additional directional indicators (DI+ and DI-).
    """
    
    category = "trend"
    
    def __init__(self, window: int = 14, smooth_window: int = 14, **kwargs):
        """
        Initialize Average Directional Index indicator.
        
        Args:
            window: Lookback period for initial calculations
            smooth_window: Smoothing period for final ADX calculation
            **kwargs: Additional parameters
        """
        self.window = window
        self.smooth_window = smooth_window
        self.name_adx = f"adx_{window}"
        self.name_di_plus = f"di_plus_{window}"
        self.name_di_minus = f"di_minus_{window}"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ADX values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate True Range
        high_low = result['high'] - result['low']
        high_close_prev = abs(result['high'] - result['close'].shift(1))
        low_close_prev = abs(result['low'] - result['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = result['high'] - result['high'].shift(1)
        down_move = result['low'].shift(1) - result['low']
        
        # Calculate +DM and -DM
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Create Series for calculations
        tr_series = pd.Series(tr)
        plus_dm_series = pd.Series(plus_dm)
        minus_dm_series = pd.Series(minus_dm)
        
        # Wilder's smoothing technique
        tr_smoothed = tr_series.rolling(window=self.window).sum()
        plus_dm_smoothed = plus_dm_series.rolling(window=self.window).sum()
        minus_dm_smoothed = minus_dm_series.rolling(window=self.window).sum()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smoothed / tr_smoothed)
        minus_di = 100 * (minus_dm_smoothed / tr_smoothed)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX (smoothed DX)
        adx = dx.rolling(window=self.smooth_window).mean()
        
        # Add indicators to result
        result[self.name_adx] = adx
        result[self.name_di_plus] = plus_di
        result[self.name_di_minus] = minus_di
        
        return result


class Supertrend(BaseIndicator):
    """
    Supertrend indicator.
    
    This trend indicator identifies trend direction and potential reversal points
    using ATR to set the band width around the price.
    """
    
    category = "trend"
    
    def __init__(self, atr_period: int = 10, multiplier: float = 3.0, **kwargs):
        """
        Initialize Supertrend indicator.
        
        Args:
            atr_period: Period for ATR calculation
            multiplier: Multiplier for the ATR value
            **kwargs: Additional parameters
        """
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.name = f"supertrend_{atr_period}_{multiplier}"
        self.name_direction = f"supertrend_dir_{atr_period}_{multiplier}"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Supertrend values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate ATR
        high_low = result['high'] - result['low']
        high_close_prev = abs(result['high'] - result['close'].shift(1))
        low_close_prev = abs(result['low'] - result['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()
        
        # Calculate the basic upper and lower bands
        hl2 = (result['high'] + result['low']) / 2
        basic_upper = hl2 + (self.multiplier * atr)
        basic_lower = hl2 - (self.multiplier * atr)
        
        # Initialize Supertrend columns
        supertrend = pd.Series(0.0, index=result.index)
        direction = pd.Series(1, index=result.index)  # 1 for uptrend, -1 for downtrend
        
        # Calculate Supertrend
        for i in range(1, len(result)):
            # Calculate current upper and lower bands
            if basic_upper.iloc[i] < supertrend.iloc[i-1] or result['close'].iloc[i-1] > supertrend.iloc[i-1]:
                upper = basic_upper.iloc[i]
            else:
                upper = supertrend.iloc[i-1]
                
            if basic_lower.iloc[i] > supertrend.iloc[i-1] or result['close'].iloc[i-1] < supertrend.iloc[i-1]:
                lower = basic_lower.iloc[i]
            else:
                lower = supertrend.iloc[i-1]
                
            # Determine trend direction
            if supertrend.iloc[i-1] == upper.iloc[i-1]:
                # Previous trend was bearish
                if result['close'].iloc[i] > upper:
                    # Trend reversal: bearish to bullish
                    supertrend.iloc[i] = lower
                    direction.iloc[i] = 1
                else:
                    # Continue bearish
                    supertrend.iloc[i] = upper
                    direction.iloc[i] = -1
            else:
                # Previous trend was bullish
                if result['close'].iloc[i] < lower:
                    # Trend reversal: bullish to bearish
                    supertrend.iloc[i] = upper
                    direction.iloc[i] = -1
                else:
                    # Continue bullish
                    supertrend.iloc[i] = lower
                    direction.iloc[i] = 1
        
        # Add to result DataFrame
        result[self.name] = supertrend
        result[self.name_direction] = direction
        
        return result


class ParabolicSAR(BaseIndicator):
    """
    Parabolic Stop and Reverse (SAR) indicator.
    
    This trend-following indicator shows potential reversal points in price direction,
    accelerating with the trend to catch possible price reversal.
    """
    
    category = "trend"
    
    def __init__(self, acceleration: float = 0.02, max_acceleration: float = 0.2, **kwargs):
        """
        Initialize Parabolic SAR indicator.
        
        Args:
            acceleration: Starting acceleration factor
            max_acceleration: Maximum acceleration factor
            **kwargs: Additional parameters
        """
        self.acceleration = acceleration
        self.max_acceleration = max_acceleration
        self.name = "parabolic_sar"
        self.name_direction = "parabolic_sar_direction"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Parabolic SAR for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Parabolic SAR values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Initialize series
        sar = pd.Series(0.0, index=result.index)
        direction = pd.Series(1, index=result.index)  # 1 for uptrend, -1 for downtrend
        ep = pd.Series(0.0, index=result.index)  # Extreme Point
        af = pd.Series(self.acceleration, index=result.index)  # Acceleration Factor
        
        # Set initial values
        if len(result) > 1:
            # Start with a downtrend (SAR above price) if close[0] > close[1]
            if result['close'].iloc[0] > result['close'].iloc[1]:
                direction.iloc[0] = -1
                sar.iloc[0] = result['high'].iloc[0]
                ep.iloc[0] = result['low'].iloc[0]
            else:
                # Start with an uptrend (SAR below price)
                direction.iloc[0] = 1
                sar.iloc[0] = result['low'].iloc[0]
                ep.iloc[0] = result['high'].iloc[0]
        
        # Calculate SAR for each period
        for i in range(1, len(result)):
            # Previous period's values
            prev_sar = sar.iloc[i-1]
            prev_ep = ep.iloc[i-1]
            prev_af = af.iloc[i-1]
            prev_direction = direction.iloc[i-1]
            
            # Current SAR value (before adjustment)
            sar.iloc[i] = prev_sar + prev_af * (prev_ep - prev_sar)
            
            # Initialize current values
            curr_direction = prev_direction
            curr_af = prev_af
            curr_ep = prev_ep
            
            # Check for trend reversal
            if prev_direction == 1:
                # Previous trend was up (bullish)
                
                # Check if SAR crosses below current low (must reverse)
                if sar.iloc[i] > result['low'].iloc[i]:
                    curr_direction = -1  # Change to downtrend
                    sar.iloc[i] = max(result['high'].iloc[i], prev_ep)  # New SAR is the highest high
                    curr_ep = result['low'].iloc[i]  # New extreme point is current low
                    curr_af = self.acceleration  # Reset acceleration factor
                else:
                    # Continue uptrend
                    # Ensure SAR doesn't go above previous lows
                    sar.iloc[i] = min(sar.iloc[i], result['low'].iloc[i-1], result['low'].iloc[i-2] if i > 1 else float('inf'))
                    
                    # Update extreme point and acceleration factor if new high
                    if result['high'].iloc[i] > prev_ep:
                        curr_ep = result['high'].iloc[i]
                        curr_af = min(prev_af + self.acceleration, self.max_acceleration)
                    
            else:
                # Previous trend was down (bearish)
                
                # Check if SAR crosses above current high (must reverse)
                if sar.iloc[i] < result['high'].iloc[i]:
                    curr_direction = 1  # Change to uptrend
                    sar.iloc[i] = min(result['low'].iloc[i], prev_ep)  # New SAR is the lowest low
                    curr_ep = result['high'].iloc[i]  # New extreme point is current high
                    curr_af = self.acceleration  # Reset acceleration factor
                else:
                    # Continue downtrend
                    # Ensure SAR doesn't go below previous highs
                    sar.iloc[i] = max(sar.iloc[i], result['high'].iloc[i-1], result['high'].iloc[i-2] if i > 1 else 0)
                    
                    # Update extreme point and acceleration factor if new low
                    if result['low'].iloc[i] < prev_ep:
                        curr_ep = result['low'].iloc[i]
                        curr_af = min(prev_af + self.acceleration, self.max_acceleration)
            
            # Update values for current period
            direction.iloc[i] = curr_direction
            ep.iloc[i] = curr_ep
            af.iloc[i] = curr_af
        
        # Add results to DataFrame
        result[self.name] = sar
        result[self.name_direction] = direction
        
        return result
