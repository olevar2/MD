"""
Volatility Range Module.

This module provides implementations of range-based volatility indicators.
"""

import pandas as pd
from typing import Dict, Any

from feature_store_service.indicators.base_indicator import BaseIndicator


class AverageTrueRange(BaseIndicator):
    """
    Average True Range (ATR) indicator.
    
    This volatility indicator measures market volatility by decomposing the
    entire range of an asset price for a period.
    """
    
    category = "volatility"
    
    def __init__(self, window: int = 14, **kwargs):
        """
        Initialize Average True Range indicator.
        
        Args:
            window: Lookback period for the ATR calculation
            **kwargs: Additional parameters
        """
        self.window = window
        self.name = f"atr_{window}"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ATR values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate true range
        high_low = result['high'] - result['low']
        high_close_prev = abs(result['high'] - result['close'].shift(1))
        low_close_prev = abs(result['low'] - result['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR using simple moving average
        # Note: For a more accurate Wilder's smoothing method, you would use a different approach
        result[self.name] = true_range.rolling(window=self.window).mean()
        
        return result