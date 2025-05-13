"""
Heikin-Ashi Pattern Recognizer Module.

This module provides pattern recognition capabilities for Heikin-Ashi candlesticks.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from core.base import AdvancedPatternRecognizer
from models.models import (
    HeikinAshiPatternType,
    HeikinAshiTrendType,
    HeikinAshiCandle,
    HeikinAshiPattern
)
from utils.utils import (
    calculate_heikin_ashi,
    extract_heikin_ashi_candles,
    detect_heikin_ashi_reversal,
    detect_heikin_ashi_continuation,
    detect_heikin_ashi_strong_trend,
    detect_heikin_ashi_weak_trend
)


class HeikinAshiPatternRecognizer(AdvancedPatternRecognizer):
    """
    Recognizes patterns in Heikin-Ashi candlesticks.
    
    This class identifies common Heikin-Ashi patterns like reversals,
    continuations, and trend strength patterns.
    """
    
    category = "pattern"
    
    def __init__(
        self,
        pattern_types: Optional[List[str]] = None,
        lookback_period: int = 100,
        min_trend_length: int = 5,
        sensitivity: float = 0.75,
        **kwargs
    ):
        """
        Initialize the Heikin-Ashi pattern recognizer.
        
        Args:
            pattern_types: List of pattern types to look for (None = all patterns)
            lookback_period: Number of bars to look back for pattern recognition
            min_trend_length: Minimum number of candles in a trend
            sensitivity: Sensitivity of pattern detection (0.0-1.0)
            **kwargs: Additional parameters
        """
        super().__init__(
            lookback_period=lookback_period,
            sensitivity=sensitivity,
            pattern_types=pattern_types,
            **kwargs
        )
        
        self.min_trend_length = min_trend_length
        
        # Set pattern types to recognize
        all_patterns = [pt.value for pt in HeikinAshiPatternType]
        
        if pattern_types is None:
            self.pattern_types = all_patterns
        else:
            self.pattern_types = [pt for pt in pattern_types if pt in all_patterns]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin-Ashi pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Heikin-Ashi pattern recognition values
        """
        # Validate input data
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate Heikin-Ashi candles
        result = calculate_heikin_ashi(result)
        
        # Initialize pattern columns
        for pattern_type in HeikinAshiPatternType:
            result[f"pattern_{pattern_type.value}"] = 0
        
        # Add direction and strength columns
        result["pattern_heikin_ashi_direction"] = ""
        result["pattern_heikin_ashi_strength"] = 0.0
        result["pattern_heikin_ashi_target"] = np.nan
        result["pattern_heikin_ashi_stop"] = np.nan
        
        # Extract Heikin-Ashi candles
        candles = extract_heikin_ashi_candles(result)
        
        # Find patterns
        patterns = self._find_patterns_in_candles(candles)
        
        # Map patterns to DataFrame
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            
            if pattern_type in self.pattern_types:
                # Find the rows corresponding to this pattern
                pattern_indices = list(range(pattern.start_index, pattern.end_index + 1))
                
                if pattern_indices:
                    # Set pattern values
                    result.iloc[pattern_indices, result.columns.get_loc(f"pattern_{pattern_type}")] = 1
                    result.iloc[pattern_indices, result.columns.get_loc("pattern_heikin_ashi_direction")] = pattern.trend_type.value
                    result.iloc[pattern_indices, result.columns.get_loc("pattern_heikin_ashi_strength")] = pattern.strength
                    
                    if pattern.target_price is not None:
                        result.iloc[pattern_indices, result.columns.get_loc("pattern_heikin_ashi_target")] = pattern.target_price
                    
                    if pattern.stop_price is not None:
                        result.iloc[pattern_indices, result.columns.get_loc("pattern_heikin_ashi_stop")] = pattern.stop_price
        
        return result
    
    def find_patterns(
        self,
        data: pd.DataFrame,
        pattern_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find Heikin-Ashi patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data
            pattern_types: List of pattern types to look for (None = all patterns)
            
        Returns:
            Dictionary of pattern types and their occurrences
        """
        # Determine which patterns to look for
        if pattern_types is None:
            patterns_to_find = self.pattern_types
        else:
            patterns_to_find = [pt for pt in pattern_types if pt in self.pattern_types]
        
        # Calculate Heikin-Ashi candles
        ha_data = calculate_heikin_ashi(data)
        
        # Extract Heikin-Ashi candles
        candles = extract_heikin_ashi_candles(ha_data)
        
        # Find patterns
        patterns = self._find_patterns_in_candles(candles)
        
        # Initialize the patterns dictionary
        patterns_dict = {pattern_type: [] for pattern_type in patterns_to_find}
        
        # Convert patterns to dictionary format
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            
            if pattern_type in patterns_to_find:
                # Convert pattern to dictionary
                pattern_dict = pattern.to_dict()
                
                # Add to patterns dictionary
                patterns_dict[pattern_type].append(pattern_dict)
        
        return patterns_dict
    
    def _find_patterns_in_candles(self, candles: List[HeikinAshiCandle]) -> List[HeikinAshiPattern]:
        """
        Find patterns in Heikin-Ashi candles.
        
        Args:
            candles: List of Heikin-Ashi candles
            
        Returns:
            List of HeikinAshiPattern objects
        """
        patterns = []
        
        # Detect reversal patterns
        if HeikinAshiPatternType.REVERSAL.value in self.pattern_types:
            reversal_patterns = detect_heikin_ashi_reversal(
                candles,
                min_trend_length=self.min_trend_length,
                lookback=self.lookback_period,
                sensitivity=self.sensitivity
            )
            patterns.extend(reversal_patterns)
        
        # Detect continuation patterns
        if HeikinAshiPatternType.CONTINUATION.value in self.pattern_types:
            continuation_patterns = detect_heikin_ashi_continuation(
                candles,
                min_trend_length=self.min_trend_length,
                lookback=self.lookback_period,
                sensitivity=self.sensitivity
            )
            patterns.extend(continuation_patterns)
        
        # Detect strong trend patterns
        if HeikinAshiPatternType.STRONG_TREND.value in self.pattern_types:
            strong_trend_patterns = detect_heikin_ashi_strong_trend(
                candles,
                min_trend_length=self.min_trend_length,
                lookback=self.lookback_period,
                sensitivity=self.sensitivity
            )
            patterns.extend(strong_trend_patterns)
        
        # Detect weak trend patterns
        if HeikinAshiPatternType.WEAK_TREND.value in self.pattern_types:
            weak_trend_patterns = detect_heikin_ashi_weak_trend(
                candles,
                min_trend_length=self.min_trend_length,
                lookback=self.lookback_period,
                sensitivity=self.sensitivity
            )
            patterns.extend(weak_trend_patterns)
        
        return patterns