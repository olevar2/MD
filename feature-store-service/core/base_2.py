"""
Base Candlestick Pattern Module.

This module provides base classes and utilities for candlestick pattern recognition.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from core.base_1 import BasePatternDetector, PatternType


class BaseCandlestickPattern(BasePatternDetector):
    """
    Base class for all candlestick pattern detectors.
    
    This class provides common functionality for candlestick pattern detection.
    """
    
    def __init__(
        self, 
        pattern_name: str,
        has_direction: bool = True,
        **kwargs
    ):
        """
        Initialize Base Candlestick Pattern Detector.
        
        Args:
            pattern_name: Name of the candlestick pattern
            has_direction: Whether the pattern has bullish/bearish direction
            **kwargs: Additional parameters
        """
        self.name = f"{pattern_name}_pattern"
        self.pattern_name = pattern_name
        self.has_direction = has_direction
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate candlestick pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with candlestick pattern values
        """
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Initialize pattern columns with zeros
        if self.has_direction:
            result[f"candle_{self.pattern_name}_bullish"] = 0
            result[f"candle_{self.pattern_name}_bearish"] = 0
        else:
            result[f"candle_{self.pattern_name}"] = 0
        
        # Detect patterns
        self._detect_patterns(result)
        
        return result
    
    def _detect_patterns(self, data: pd.DataFrame) -> None:
        """
        Detect candlestick patterns in the given data.
        
        This method should be implemented by subclasses.
        
        Args:
            data: DataFrame with OHLCV data (will be modified in-place)
        """
        raise NotImplementedError("Subclasses must implement _detect_patterns()")
    
    def _is_bullish_candle(self, data: pd.DataFrame, idx: int) -> bool:
        """
        Check if the candle at the given index is bullish (close > open).
        
        Args:
            data: DataFrame with OHLCV data
            idx: Index of the candle to check
            
        Returns:
            True if the candle is bullish, False otherwise
        """
        return data['close'].iloc[idx] > data['open'].iloc[idx]
    
    def _is_bearish_candle(self, data: pd.DataFrame, idx: int) -> bool:
        """
        Check if the candle at the given index is bearish (close < open).
        
        Args:
            data: DataFrame with OHLCV data
            idx: Index of the candle to check
            
        Returns:
            True if the candle is bearish, False otherwise
        """
        return data['close'].iloc[idx] < data['open'].iloc[idx]
    
    def _get_body_size(self, data: pd.DataFrame, idx: int) -> float:
        """
        Get the body size of the candle at the given index.
        
        Args:
            data: DataFrame with OHLCV data
            idx: Index of the candle to check
            
        Returns:
            Body size of the candle (absolute difference between open and close)
        """
        return abs(data['close'].iloc[idx] - data['open'].iloc[idx])
    
    def _get_upper_shadow(self, data: pd.DataFrame, idx: int) -> float:
        """
        Get the upper shadow size of the candle at the given index.
        
        Args:
            data: DataFrame with OHLCV data
            idx: Index of the candle to check
            
        Returns:
            Upper shadow size of the candle
        """
        return data['high'].iloc[idx] - max(data['open'].iloc[idx], data['close'].iloc[idx])
    
    def _get_lower_shadow(self, data: pd.DataFrame, idx: int) -> float:
        """
        Get the lower shadow size of the candle at the given index.
        
        Args:
            data: DataFrame with OHLCV data
            idx: Index of the candle to check
            
        Returns:
            Lower shadow size of the candle
        """
        return min(data['open'].iloc[idx], data['close'].iloc[idx]) - data['low'].iloc[idx]
    
    def _get_candle_range(self, data: pd.DataFrame, idx: int) -> float:
        """
        Get the total range of the candle at the given index.
        
        Args:
            data: DataFrame with OHLCV data
            idx: Index of the candle to check
            
        Returns:
            Total range of the candle (high - low)
        """
        return data['high'].iloc[idx] - data['low'].iloc[idx]
    
    def _get_body_to_range_ratio(self, data: pd.DataFrame, idx: int) -> float:
        """
        Get the ratio of body size to total range of the candle at the given index.
        
        Args:
            data: DataFrame with OHLCV data
            idx: Index of the candle to check
            
        Returns:
            Ratio of body size to total range (0.0-1.0)
        """
        body_size = self._get_body_size(data, idx)
        candle_range = self._get_candle_range(data, idx)
        
        if candle_range == 0:
            return 0.0
        
        return body_size / candle_range
    
    def _get_upper_shadow_to_range_ratio(self, data: pd.DataFrame, idx: int) -> float:
        """
        Get the ratio of upper shadow to total range of the candle at the given index.
        
        Args:
            data: DataFrame with OHLCV data
            idx: Index of the candle to check
            
        Returns:
            Ratio of upper shadow to total range (0.0-1.0)
        """
        upper_shadow = self._get_upper_shadow(data, idx)
        candle_range = self._get_candle_range(data, idx)
        
        if candle_range == 0:
            return 0.0
        
        return upper_shadow / candle_range
    
    def _get_lower_shadow_to_range_ratio(self, data: pd.DataFrame, idx: int) -> float:
        """
        Get the ratio of lower shadow to total range of the candle at the given index.
        
        Args:
            data: DataFrame with OHLCV data
            idx: Index of the candle to check
            
        Returns:
            Ratio of lower shadow to total range (0.0-1.0)
        """
        lower_shadow = self._get_lower_shadow(data, idx)
        candle_range = self._get_candle_range(data, idx)
        
        if candle_range == 0:
            return 0.0
        
        return lower_shadow / candle_range
    
    def _is_in_uptrend(self, data: pd.DataFrame, idx: int, lookback: int = 5) -> bool:
        """
        Check if the market is in an uptrend at the given index.
        
        Args:
            data: DataFrame with OHLCV data
            idx: Index of the candle to check
            lookback: Number of candles to look back for trend determination
            
        Returns:
            True if the market is in an uptrend, False otherwise
        """
        if idx < lookback:
            return False
        
        # Simple uptrend check: current close > close 'lookback' periods ago
        return data['close'].iloc[idx] > data['close'].iloc[idx - lookback]
    
    def _is_in_downtrend(self, data: pd.DataFrame, idx: int, lookback: int = 5) -> bool:
        """
        Check if the market is in a downtrend at the given index.
        
        Args:
            data: DataFrame with OHLCV data
            idx: Index of the candle to check
            lookback: Number of candles to look back for trend determination
            
        Returns:
            True if the market is in a downtrend, False otherwise
        """
        if idx < lookback:
            return False
        
        # Simple downtrend check: current close < close 'lookback' periods ago
        return data['close'].iloc[idx] < data['close'].iloc[idx - lookback]
