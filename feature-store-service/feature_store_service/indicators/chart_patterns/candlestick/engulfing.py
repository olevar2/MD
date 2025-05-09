"""
Engulfing Candlestick Pattern Module.

This module provides implementation of the Engulfing candlestick pattern recognition.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from feature_store_service.indicators.chart_patterns.candlestick.base import BaseCandlestickPattern


class EngulfingPattern(BaseCandlestickPattern):
    """
    Engulfing Pattern Detector.
    
    The Engulfing pattern is a two-candle reversal pattern where the second candle's body
    completely engulfs the body of the first candle.
    
    - Bullish Engulfing: Forms in a downtrend where a bearish candle is followed by a larger bullish candle
    - Bearish Engulfing: Forms in an uptrend where a bullish candle is followed by a larger bearish candle
    """
    
    def __init__(
        self, 
        trend_lookback: int = 5,
        **kwargs
    ):
        """
        Initialize Engulfing Pattern Detector.
        
        Args:
            trend_lookback: Number of candles to look back for trend determination
            **kwargs: Additional parameters
        """
        super().__init__(pattern_name="engulfing", has_direction=True, **kwargs)
        self.trend_lookback = max(1, trend_lookback)
    
    def _detect_patterns(self, data: pd.DataFrame) -> None:
        """
        Detect Engulfing patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data (will be modified in-place)
        """
        # Need at least 2 candles to detect an engulfing pattern
        if len(data) < 2:
            return
        
        for i in range(1, len(data)):
            # Get current and previous candle bodies
            curr_open = data['open'].iloc[i]
            curr_close = data['close'].iloc[i]
            prev_open = data['open'].iloc[i-1]
            prev_close = data['close'].iloc[i-1]
            
            # Check for bullish engulfing
            if (self._is_bearish_candle(data, i-1) and 
                self._is_bullish_candle(data, i) and 
                curr_open <= prev_close and 
                curr_close >= prev_open):
                
                # Stronger signal if in a downtrend
                if self._is_in_downtrend(data, i, self.trend_lookback):
                    data.loc[data.index[i], f"candle_{self.pattern_name}_bullish"] = 1
                    
                    # Calculate strength based on size difference
                    curr_body = self._get_body_size(data, i)
                    prev_body = self._get_body_size(data, i-1)
                    
                    if curr_body > prev_body * 2:
                        data.loc[data.index[i], f"candle_{self.pattern_name}_strength"] = 0.9
                    else:
                        data.loc[data.index[i], f"candle_{self.pattern_name}_strength"] = 0.7
            
            # Check for bearish engulfing
            elif (self._is_bullish_candle(data, i-1) and 
                  self._is_bearish_candle(data, i) and 
                  curr_open >= prev_close and 
                  curr_close <= prev_open):
                
                # Stronger signal if in an uptrend
                if self._is_in_uptrend(data, i, self.trend_lookback):
                    data.loc[data.index[i], f"candle_{self.pattern_name}_bearish"] = 1
                    
                    # Calculate strength based on size difference
                    curr_body = self._get_body_size(data, i)
                    prev_body = self._get_body_size(data, i-1)
                    
                    if curr_body > prev_body * 2:
                        data.loc[data.index[i], f"candle_{self.pattern_name}_strength"] = 0.9
                    else:
                        data.loc[data.index[i], f"candle_{self.pattern_name}_strength"] = 0.7
