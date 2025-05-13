"""
Hammer and Hanging Man Candlestick Pattern Module.

This module provides implementation of the Hammer and Hanging Man candlestick pattern recognition.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from core.base_2 import BaseCandlestickPattern


class HammerPattern(BaseCandlestickPattern):
    """
    Hammer and Hanging Man Pattern Detector.
    
    The Hammer and Hanging Man are candlestick patterns with a small body and a long lower shadow.
    The upper shadow should be very small or non-existent.
    
    - Hammer: Forms in a downtrend and is a potential bullish reversal signal
    - Hanging Man: Forms in an uptrend and is a potential bearish reversal signal
    
    Both patterns have the same appearance but different implications based on the trend.
    """
    
    def __init__(
        self, 
        body_threshold: float = 0.3,
        lower_shadow_threshold: float = 0.6,
        upper_shadow_threshold: float = 0.1,
        trend_lookback: int = 5,
        **kwargs
    ):
        """
        Initialize Hammer Pattern Detector.
        
        Args:
            body_threshold: Maximum body size as a percentage of the candle range (0.0-1.0)
            lower_shadow_threshold: Minimum lower shadow size as a percentage of the candle range (0.0-1.0)
            upper_shadow_threshold: Maximum upper shadow size as a percentage of the candle range (0.0-1.0)
            trend_lookback: Number of candles to look back for trend determination
            **kwargs: Additional parameters
        """
        super().__init__(pattern_name="hammer", has_direction=True, **kwargs)
        self.body_threshold = max(0.0, min(0.5, body_threshold))
        self.lower_shadow_threshold = max(0.0, min(1.0, lower_shadow_threshold))
        self.upper_shadow_threshold = max(0.0, min(0.5, upper_shadow_threshold))
        self.trend_lookback = max(1, trend_lookback)
    
    def _detect_patterns(self, data: pd.DataFrame) -> None:
        """
        Detect Hammer and Hanging Man patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data (will be modified in-place)
        """
        for i in range(len(data)):
            # Calculate body and shadow ratios
            body_ratio = self._get_body_to_range_ratio(data, i)
            lower_shadow_ratio = self._get_lower_shadow_to_range_ratio(data, i)
            upper_shadow_ratio = self._get_upper_shadow_to_range_ratio(data, i)
            
            # Check if the candle has the right shape for a Hammer/Hanging Man
            if (body_ratio <= self.body_threshold and 
                lower_shadow_ratio >= self.lower_shadow_threshold and 
                upper_shadow_ratio <= self.upper_shadow_threshold):
                
                # Check if it's a Hammer (bullish) or Hanging Man (bearish)
                if self._is_in_downtrend(data, i, self.trend_lookback):
                    # Hammer (bullish reversal in downtrend)
                    data.loc[data.index[i], f"candle_{self.pattern_name}_bullish"] = 1
                    
                    # Stronger signal if the body is bullish (close > open)
                    if self._is_bullish_candle(data, i):
                        data.loc[data.index[i], f"candle_{self.pattern_name}_strength"] = 0.8
                    else:
                        data.loc[data.index[i], f"candle_{self.pattern_name}_strength"] = 0.6
                        
                elif self._is_in_uptrend(data, i, self.trend_lookback):
                    # Hanging Man (bearish reversal in uptrend)
                    data.loc[data.index[i], f"candle_{self.pattern_name}_bearish"] = 1
                    
                    # Stronger signal if the body is bearish (close < open)
                    if self._is_bearish_candle(data, i):
                        data.loc[data.index[i], f"candle_{self.pattern_name}_strength"] = 0.8
                    else:
                        data.loc[data.index[i], f"candle_{self.pattern_name}_strength"] = 0.6
