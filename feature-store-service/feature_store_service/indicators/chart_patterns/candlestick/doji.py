"""
Doji Candlestick Pattern Module.

This module provides implementation of the Doji candlestick pattern recognition.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from feature_store_service.indicators.chart_patterns.candlestick.base import BaseCandlestickPattern


class DojiPattern(BaseCandlestickPattern):
    """
    Doji Pattern Detector.
    
    A Doji candlestick forms when the open and close prices are virtually equal.
    The length of the upper and lower shadows can vary, forming different types of Doji.
    
    Types of Doji:
    - Standard Doji: Open and close are equal with upper and lower shadows
    - Long-Legged Doji: Open and close are equal with long upper and lower shadows
    - Dragonfly Doji: Open and close are equal at the high with a long lower shadow
    - Gravestone Doji: Open and close are equal at the low with a long upper shadow
    """
    
    def __init__(
        self, 
        body_threshold: float = 0.05,
        **kwargs
    ):
        """
        Initialize Doji Pattern Detector.
        
        Args:
            body_threshold: Maximum body size as a percentage of the candle range (0.0-1.0)
            **kwargs: Additional parameters
        """
        super().__init__(pattern_name="doji", has_direction=False, **kwargs)
        self.body_threshold = max(0.0, min(0.2, body_threshold))
    
    def _detect_patterns(self, data: pd.DataFrame) -> None:
        """
        Detect Doji patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data (will be modified in-place)
        """
        for i in range(len(data)):
            # Calculate body to range ratio
            body_ratio = self._get_body_to_range_ratio(data, i)
            
            # Check if the body is small enough to be a Doji
            if body_ratio <= self.body_threshold:
                data.loc[data.index[i], f"candle_{self.pattern_name}"] = 1
                
                # Determine Doji type
                upper_shadow_ratio = self._get_upper_shadow_to_range_ratio(data, i)
                lower_shadow_ratio = self._get_lower_shadow_to_range_ratio(data, i)
                
                # Long-Legged Doji
                if upper_shadow_ratio > 0.3 and lower_shadow_ratio > 0.3:
                    data.loc[data.index[i], f"candle_{self.pattern_name}_type"] = "long_legged"
                
                # Dragonfly Doji
                elif lower_shadow_ratio > 0.7 and upper_shadow_ratio < 0.1:
                    data.loc[data.index[i], f"candle_{self.pattern_name}_type"] = "dragonfly"
                
                # Gravestone Doji
                elif upper_shadow_ratio > 0.7 and lower_shadow_ratio < 0.1:
                    data.loc[data.index[i], f"candle_{self.pattern_name}_type"] = "gravestone"
                
                # Standard Doji
                else:
                    data.loc[data.index[i], f"candle_{self.pattern_name}_type"] = "standard"
