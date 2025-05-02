"""
Fractal Geometry Analysis Module

This module provides fractal analysis tools for forex price data including:
- Fractal pattern identification 
- Fractal dimension calculation
- Self-similarity detection
- Multi-timeframe fractal alignment

Implementation supports both standard calculation and incremental updates.
"""
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

from analysis_engine.analysis.advanced_ta.base import (
    AdvancedAnalysisBase,
    PatternRecognitionBase,
    PatternResult,
    ConfidenceLevel,
    MarketDirection,
    AnalysisTimeframe
)


class FractalType(Enum):
    """Fractal pattern types"""
    BULLISH = "bullish"  # Up fractal (high point)
    BEARISH = "bearish"  # Down fractal (low point)


class FractalPattern(PatternResult):
    """Represents a detected fractal pattern."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fractal_type = kwargs.get("fractal_type", None) # FractalType Enum
        self.pattern_name = "Fractal"


class FractalIndicator(PatternRecognitionBase):
    """
    Identifies Bill Williams' Fractal patterns.

    - Bullish Fractal (Up Fractal): A high point with two lower highs on each side.
    - Bearish Fractal (Down Fractal): A low point with two higher lows on each side.

    The pattern is confirmed two bars after the fractal point itself.
    """

    def __init__(self, lookback_period: int = 5):
        """
        Initialize the Fractal Indicator.

        Args:
            lookback_period: The number of bars required to form a fractal (typically 5).
                             A 5-bar fractal looks at the middle bar [n-2, n-1, n, n+1, n+2].
        """
        if lookback_period % 2 == 0 or lookback_period < 5:
            raise ValueError("lookback_period must be an odd number >= 5")
        parameters = {"lookback_period": lookback_period}
        super().__init__("Fractal Indicator", parameters)
        self.center_offset = lookback_period // 2

    def find_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        Find Fractal patterns in the price data.

        Args:
            df: DataFrame with OHLCV data (requires 'high' and 'low' columns).

        Returns:
            List of FractalPattern objects.
        """
        patterns = []
        n = self.parameters["lookback_period"]
        offset = self.center_offset

        if len(df) < n:
            return patterns

        highs = df['high']
        lows = df['low']

        for i in range(offset, len(df) - offset):
            center_time = df.index[i]
            center_high = highs.iloc[i]
            center_low = lows.iloc[i]

            # Check for Bullish Fractal (Up Fractal)
            is_bullish = True
            for j in range(n):
                if j == offset: continue # Skip the center bar
                if highs.iloc[i - offset + j] >= center_high:
                    is_bullish = False
                    break

            if is_bullish:
                pattern = FractalPattern(
                    pattern_name="Bullish Fractal",
                    timeframe=AnalysisTimeframe.UNKNOWN, # Should be derived
                    direction=MarketDirection.BULLISH, # Indicates potential reversal or breakout point
                    confidence=ConfidenceLevel.MEDIUM, # Fractals are common, confidence depends on context
                    start_time=df.index[i - offset], # Start of the 5-bar pattern
                    end_time=df.index[i + offset],   # End of the 5-bar pattern (confirmation bar)
                    signal_time=center_time,         # The time of the fractal high itself
                    start_price=lows.iloc[i - offset], # Price range of the pattern
                    end_price=highs.iloc[i + offset],
                    signal_price=center_high,        # The price of the fractal high
                    fractal_type=FractalType.BULLISH
                )
                patterns.append(pattern)

            # Check for Bearish Fractal (Down Fractal)
            is_bearish = True
            for j in range(n):
                if j == offset: continue # Skip the center bar
                if lows.iloc[i - offset + j] <= center_low:
                    is_bearish = False
                    break

            if is_bearish:
                pattern = FractalPattern(
                    pattern_name="Bearish Fractal",
                    timeframe=AnalysisTimeframe.UNKNOWN,
                    direction=MarketDirection.BEARISH,
                    confidence=ConfidenceLevel.MEDIUM,
                    start_time=df.index[i - offset],
                    end_time=df.index[i + offset],
                    signal_time=center_time,
                    start_price=lows.iloc[i - offset],
                    end_price=highs.iloc[i + offset],
                    signal_price=center_low,
                    fractal_type=FractalType.BEARISH
                )
                patterns.append(pattern)

        return patterns

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fractal indicators and add columns to the DataFrame.

        Adds columns:
        - fractal_bullish: Price level of the bullish fractal high (NaN if none).
        - fractal_bearish: Price level of the bearish fractal low (NaN if none).
        The signal appears on the confirmation bar (end_time of the pattern).

        Args:
            df: DataFrame containing OHLCV data.

        Returns:
            DataFrame with added fractal indicator columns.
        """
        result_df = df.copy()
        n = self.parameters["lookback_period"]
        offset = self.center_offset

        result_df['fractal_bullish'] = np.nan
        result_df['fractal_bearish'] = np.nan

        if len(df) < n:
            return result_df

        highs = result_df['high']
        lows = result_df['low']

        for i in range(offset, len(result_df) - offset):
            center_high = highs.iloc[i]
            center_low = lows.iloc[i]
            confirmation_bar_index = i + offset # Signal appears on the bar closing the pattern

            # Check for Bullish Fractal
            is_bullish = all(highs.iloc[i - offset + j] < center_high for j in range(n) if j != offset)
            if is_bullish:
                # Place the signal (fractal high price) on the confirmation bar's row
                result_df.iloc[confirmation_bar_index, result_df.columns.get_loc('fractal_bullish')] = center_high

            # Check for Bearish Fractal
            is_bearish = all(lows.iloc[i - offset + j] > center_low for j in range(n) if j != offset)
            if is_bearish:
                # Place the signal (fractal low price) on the confirmation bar's row
                result_df.iloc[confirmation_bar_index, result_df.columns.get_loc('fractal_bearish')] = center_low

        return result_df

# Example Usage
if __name__ == '__main__':
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    data = {
        'high': [10, 11, 12, 11, 10, 11, 13, 14, 13, 12, 11, 10, 9, 10, 11, 12, 13, 12, 11, 10] * 2 + [10]*10,
        'low':  [ 8,  9, 10,  9,  8,  7,  8,  9,  8,  7,  6,  7, 8,  7,  6,  5,  6,  7,  8,  9] * 2 + [ 8]*10
    }
    sample_df = pd.DataFrame(data, index=dates)
    sample_df['open'] = sample_df['low'] + (sample_df['high'] - sample_df['low']) * 0.3
    sample_df['close'] = sample_df['low'] + (sample_df['high'] - sample_df['low']) * 0.7
    sample_df['volume'] = 1000

    print("Sample Data:")
    print(sample_df.head(10))

    # Initialize indicator
    fractal_indicator = FractalIndicator(lookback_period=5)

    # Find patterns
    found_patterns = fractal_indicator.find_patterns(sample_df)
    print(f"\nFound {len(found_patterns)} Fractal patterns.")
    for i, p in enumerate(found_patterns):
         print(f"--- Pattern {i+1} ---")
         print(f"  Name: {p.pattern_name}")
         print(f"  Type: {p.fractal_type.value}")
         print(f"  Direction: {p.direction.value}")
         print(f"  Signal Time: {p.signal_time}") # Time of the fractal point
         print(f"  Confirmation Time: {p.end_time}") # Time pattern is confirmed
         print(f"  Signal Price: {p.signal_price:.2f}")

    # Calculate indicator columns
    result_with_indicator = fractal_indicator.calculate(sample_df)
    print("\nDataFrame with Fractal Indicator Columns:")
    # Display rows where fractals are identified
    print(result_with_indicator[result_with_indicator['fractal_bullish'].notna() | result_with_indicator['fractal_bearish'].notna()])
