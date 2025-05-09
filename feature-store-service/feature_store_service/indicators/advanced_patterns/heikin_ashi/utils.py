"""
Heikin-Ashi Pattern Utilities Module.

This module provides utility functions for Heikin-Ashi pattern recognition.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from feature_store_service.indicators.advanced_patterns.heikin_ashi.models import (
    HeikinAshiPatternType,
    HeikinAshiTrendType,
    HeikinAshiCandle,
    HeikinAshiPattern
)


def calculate_heikin_ashi(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Heikin-Ashi candles from OHLC data.
    
    Args:
        data: DataFrame with OHLC data
        
    Returns:
        DataFrame with Heikin-Ashi candles
    """
    # Make a copy to avoid modifying the input data
    ha_data = data.copy()
    
    # Initialize Heikin-Ashi columns
    ha_data['ha_open'] = np.nan
    ha_data['ha_high'] = np.nan
    ha_data['ha_low'] = np.nan
    ha_data['ha_close'] = np.nan
    
    # Calculate first Heikin-Ashi candle
    ha_data.loc[ha_data.index[0], 'ha_open'] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2
    ha_data.loc[ha_data.index[0], 'ha_close'] = (data['open'].iloc[0] + data['high'].iloc[0] + 
                                                data['low'].iloc[0] + data['close'].iloc[0]) / 4
    ha_data.loc[ha_data.index[0], 'ha_high'] = data['high'].iloc[0]
    ha_data.loc[ha_data.index[0], 'ha_low'] = data['low'].iloc[0]
    
    # Calculate remaining Heikin-Ashi candles
    for i in range(1, len(data)):
        ha_data.loc[ha_data.index[i], 'ha_close'] = (data['open'].iloc[i] + data['high'].iloc[i] + 
                                                    data['low'].iloc[i] + data['close'].iloc[i]) / 4
        ha_data.loc[ha_data.index[i], 'ha_open'] = (ha_data['ha_open'].iloc[i-1] + ha_data['ha_close'].iloc[i-1]) / 2
        ha_data.loc[ha_data.index[i], 'ha_high'] = max(data['high'].iloc[i], ha_data['ha_open'].iloc[i], ha_data['ha_close'].iloc[i])
        ha_data.loc[ha_data.index[i], 'ha_low'] = min(data['low'].iloc[i], ha_data['ha_open'].iloc[i], ha_data['ha_close'].iloc[i])
    
    # Add trend direction
    ha_data['ha_trend'] = np.where(ha_data['ha_close'] > ha_data['ha_open'], 
                                  HeikinAshiTrendType.BULLISH.value, 
                                  HeikinAshiTrendType.BEARISH.value)
    
    # Handle neutral candles (close = open)
    ha_data.loc[ha_data['ha_close'] == ha_data['ha_open'], 'ha_trend'] = HeikinAshiTrendType.NEUTRAL.value
    
    # Add body size and shadow size
    ha_data['ha_body_size'] = abs(ha_data['ha_close'] - ha_data['ha_open'])
    ha_data['ha_upper_shadow'] = ha_data['ha_high'] - np.maximum(ha_data['ha_open'], ha_data['ha_close'])
    ha_data['ha_lower_shadow'] = np.minimum(ha_data['ha_open'], ha_data['ha_close']) - ha_data['ha_low']
    
    return ha_data


def extract_heikin_ashi_candles(data: pd.DataFrame) -> List[HeikinAshiCandle]:
    """
    Extract Heikin-Ashi candles from DataFrame.
    
    Args:
        data: DataFrame with Heikin-Ashi data
        
    Returns:
        List of HeikinAshiCandle objects
    """
    candles = []
    
    for i, (idx, row) in enumerate(data.iterrows()):
        candle = HeikinAshiCandle(
            open=row['ha_open'],
            high=row['ha_high'],
            low=row['ha_low'],
            close=row['ha_close'],
            timestamp=idx,
            index=i
        )
        candles.append(candle)
    
    return candles


def detect_heikin_ashi_reversal(
    candles: List[HeikinAshiCandle],
    min_trend_length: int = 5,
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[HeikinAshiPattern]:
    """
    Detect Heikin-Ashi reversal patterns.
    
    Args:
        candles: List of Heikin-Ashi candles
        min_trend_length: Minimum number of candles in the trend before reversal
        lookback: Number of candles to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of HeikinAshiPattern objects representing reversal patterns
    """
    patterns = []
    
    # Ensure we have enough candles
    if len(candles) < lookback + min_trend_length:
        return patterns
    
    # Limit to lookback period
    recent_candles = candles[-lookback:]
    
    # Detect trend changes
    for i in range(min_trend_length, len(recent_candles) - 1):
        # Get current and previous candles
        current_candle = recent_candles[i]
        prev_candles = recent_candles[i-min_trend_length:i]
        next_candle = recent_candles[i+1]
        
        # Check for bullish reversal
        if all(c.close < c.open for c in prev_candles) and current_candle.close > current_candle.open and next_candle.close > next_candle.open:
            # We have a bullish reversal
            
            # Calculate pattern strength
            # Strength is based on the size of the reversal candle and the strength of the previous trend
            prev_trend_strength = sum(abs(c.open - c.close) for c in prev_candles) / len(prev_candles)
            reversal_strength = abs(current_candle.close - current_candle.open)
            
            # Normalize strength (0.0-1.0)
            avg_price = sum(c.close for c in prev_candles) / len(prev_candles)
            normalized_strength = min(1.0, (reversal_strength / avg_price) * 10 * sensitivity)
            
            # Calculate target price (measured move)
            trend_low = min(c.low for c in prev_candles + [current_candle])
            trend_high = max(c.high for c in prev_candles + [current_candle])
            trend_range = trend_high - trend_low
            target_price = current_candle.close + trend_range
            
            # Calculate stop price (below reversal low)
            stop_price = current_candle.low * 0.99  # 1% below reversal low
            
            # Create the pattern
            pattern = HeikinAshiPattern(
                pattern_type=HeikinAshiPatternType.REVERSAL,
                start_index=current_candle.index - min_trend_length,
                end_index=current_candle.index + 1,
                candles=prev_candles + [current_candle, next_candle],
                trend_type=HeikinAshiTrendType.BULLISH,
                strength=normalized_strength,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
        
        # Check for bearish reversal
        elif all(c.close > c.open for c in prev_candles) and current_candle.close < current_candle.open and next_candle.close < next_candle.open:
            # We have a bearish reversal
            
            # Calculate pattern strength
            # Strength is based on the size of the reversal candle and the strength of the previous trend
            prev_trend_strength = sum(abs(c.open - c.close) for c in prev_candles) / len(prev_candles)
            reversal_strength = abs(current_candle.close - current_candle.open)
            
            # Normalize strength (0.0-1.0)
            avg_price = sum(c.close for c in prev_candles) / len(prev_candles)
            normalized_strength = min(1.0, (reversal_strength / avg_price) * 10 * sensitivity)
            
            # Calculate target price (measured move)
            trend_low = min(c.low for c in prev_candles + [current_candle])
            trend_high = max(c.high for c in prev_candles + [current_candle])
            trend_range = trend_high - trend_low
            target_price = current_candle.close - trend_range
            
            # Calculate stop price (above reversal high)
            stop_price = current_candle.high * 1.01  # 1% above reversal high
            
            # Create the pattern
            pattern = HeikinAshiPattern(
                pattern_type=HeikinAshiPatternType.REVERSAL,
                start_index=current_candle.index - min_trend_length,
                end_index=current_candle.index + 1,
                candles=prev_candles + [current_candle, next_candle],
                trend_type=HeikinAshiTrendType.BEARISH,
                strength=normalized_strength,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
    
    return patterns


def detect_heikin_ashi_continuation(
    candles: List[HeikinAshiCandle],
    min_trend_length: int = 5,
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[HeikinAshiPattern]:
    """
    Detect Heikin-Ashi continuation patterns.
    
    Args:
        candles: List of Heikin-Ashi candles
        min_trend_length: Minimum number of candles in the trend
        lookback: Number of candles to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of HeikinAshiPattern objects representing continuation patterns
    """
    patterns = []
    
    # Ensure we have enough candles
    if len(candles) < lookback + min_trend_length:
        return patterns
    
    # Limit to lookback period
    recent_candles = candles[-lookback:]
    
    # Detect continuation patterns
    for i in range(min_trend_length, len(recent_candles) - 3):
        # Get current and previous candles
        current_candle = recent_candles[i]
        prev_candles = recent_candles[i-min_trend_length:i]
        next_candles = recent_candles[i+1:i+4]  # Look ahead 3 candles
        
        # Check for bullish continuation
        if all(c.close > c.open for c in prev_candles) and \
           current_candle.close < current_candle.open and \
           all(c.close > c.open for c in next_candles):
            # We have a bullish continuation pattern
            
            # Calculate pattern strength
            # Strength is based on the consistency of the trend and the pullback depth
            trend_strength = sum(abs(c.close - c.open) for c in prev_candles) / len(prev_candles)
            pullback_depth = (prev_candles[-1].close - current_candle.low) / prev_candles[-1].close
            continuation_strength = sum(abs(c.close - c.open) for c in next_candles) / len(next_candles)
            
            # Normalize strength (0.0-1.0)
            normalized_strength = min(1.0, (trend_strength + continuation_strength) / 2 * sensitivity)
            
            # Calculate target price (measured move)
            trend_start = min(c.low for c in prev_candles)
            trend_current = next_candles[-1].close
            trend_range = trend_current - trend_start
            target_price = trend_current + trend_range * 0.618  # Fibonacci extension
            
            # Calculate stop price (below pullback low)
            stop_price = current_candle.low * 0.99  # 1% below pullback low
            
            # Create the pattern
            pattern = HeikinAshiPattern(
                pattern_type=HeikinAshiPatternType.CONTINUATION,
                start_index=current_candle.index - min_trend_length,
                end_index=current_candle.index + 3,
                candles=prev_candles + [current_candle] + next_candles,
                trend_type=HeikinAshiTrendType.BULLISH,
                strength=normalized_strength,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
        
        # Check for bearish continuation
        elif all(c.close < c.open for c in prev_candles) and \
             current_candle.close > current_candle.open and \
             all(c.close < c.open for c in next_candles):
            # We have a bearish continuation pattern
            
            # Calculate pattern strength
            # Strength is based on the consistency of the trend and the pullback depth
            trend_strength = sum(abs(c.close - c.open) for c in prev_candles) / len(prev_candles)
            pullback_depth = (current_candle.high - prev_candles[-1].close) / prev_candles[-1].close
            continuation_strength = sum(abs(c.close - c.open) for c in next_candles) / len(next_candles)
            
            # Normalize strength (0.0-1.0)
            normalized_strength = min(1.0, (trend_strength + continuation_strength) / 2 * sensitivity)
            
            # Calculate target price (measured move)
            trend_start = max(c.high for c in prev_candles)
            trend_current = next_candles[-1].close
            trend_range = trend_start - trend_current
            target_price = trend_current - trend_range * 0.618  # Fibonacci extension
            
            # Calculate stop price (above pullback high)
            stop_price = current_candle.high * 1.01  # 1% above pullback high
            
            # Create the pattern
            pattern = HeikinAshiPattern(
                pattern_type=HeikinAshiPatternType.CONTINUATION,
                start_index=current_candle.index - min_trend_length,
                end_index=current_candle.index + 3,
                candles=prev_candles + [current_candle] + next_candles,
                trend_type=HeikinAshiTrendType.BEARISH,
                strength=normalized_strength,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
    
    return patterns


def detect_heikin_ashi_strong_trend(
    candles: List[HeikinAshiCandle],
    min_trend_length: int = 5,
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[HeikinAshiPattern]:
    """
    Detect strong trends in Heikin-Ashi candles.
    
    Args:
        candles: List of Heikin-Ashi candles
        min_trend_length: Minimum number of candles in the trend
        lookback: Number of candles to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of HeikinAshiPattern objects representing strong trend patterns
    """
    patterns = []
    
    # Ensure we have enough candles
    if len(candles) < lookback:
        return patterns
    
    # Limit to lookback period
    recent_candles = candles[-lookback:]
    
    # Detect strong trends
    for i in range(min_trend_length, len(recent_candles)):
        # Get current trend candles
        trend_candles = recent_candles[i-min_trend_length:i+1]
        
        # Check for strong bullish trend
        if all(c.close > c.open for c in trend_candles) and \
           all(c.low >= c.open for c in trend_candles[1:]):  # No lower shadows
            # We have a strong bullish trend
            
            # Calculate pattern strength
            # Strength is based on the consistency of the trend and the absence of shadows
            trend_strength = sum(abs(c.close - c.open) for c in trend_candles) / len(trend_candles)
            shadow_ratio = sum(c.low == c.open for c in trend_candles) / len(trend_candles)
            
            # Normalize strength (0.0-1.0)
            avg_price = sum(c.close for c in trend_candles) / len(trend_candles)
            normalized_strength = min(1.0, (trend_strength / avg_price) * 10 * shadow_ratio * sensitivity)
            
            # Calculate target price (measured move)
            trend_start = trend_candles[0].open
            trend_current = trend_candles[-1].close
            trend_range = trend_current - trend_start
            target_price = trend_current + trend_range * 0.618  # Fibonacci extension
            
            # Calculate stop price (below lowest low)
            stop_price = min(c.low for c in trend_candles) * 0.99  # 1% below lowest low
            
            # Create the pattern
            pattern = HeikinAshiPattern(
                pattern_type=HeikinAshiPatternType.STRONG_TREND,
                start_index=trend_candles[0].index,
                end_index=trend_candles[-1].index,
                candles=trend_candles,
                trend_type=HeikinAshiTrendType.BULLISH,
                strength=normalized_strength,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
        
        # Check for strong bearish trend
        elif all(c.close < c.open for c in trend_candles) and \
             all(c.high <= c.open for c in trend_candles[1:]):  # No upper shadows
            # We have a strong bearish trend
            
            # Calculate pattern strength
            # Strength is based on the consistency of the trend and the absence of shadows
            trend_strength = sum(abs(c.close - c.open) for c in trend_candles) / len(trend_candles)
            shadow_ratio = sum(c.high == c.open for c in trend_candles) / len(trend_candles)
            
            # Normalize strength (0.0-1.0)
            avg_price = sum(c.close for c in trend_candles) / len(trend_candles)
            normalized_strength = min(1.0, (trend_strength / avg_price) * 10 * shadow_ratio * sensitivity)
            
            # Calculate target price (measured move)
            trend_start = trend_candles[0].open
            trend_current = trend_candles[-1].close
            trend_range = trend_start - trend_current
            target_price = trend_current - trend_range * 0.618  # Fibonacci extension
            
            # Calculate stop price (above highest high)
            stop_price = max(c.high for c in trend_candles) * 1.01  # 1% above highest high
            
            # Create the pattern
            pattern = HeikinAshiPattern(
                pattern_type=HeikinAshiPatternType.STRONG_TREND,
                start_index=trend_candles[0].index,
                end_index=trend_candles[-1].index,
                candles=trend_candles,
                trend_type=HeikinAshiTrendType.BEARISH,
                strength=normalized_strength,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
    
    return patterns


def detect_heikin_ashi_weak_trend(
    candles: List[HeikinAshiCandle],
    min_trend_length: int = 5,
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[HeikinAshiPattern]:
    """
    Detect weak trends in Heikin-Ashi candles.
    
    Args:
        candles: List of Heikin-Ashi candles
        min_trend_length: Minimum number of candles in the trend
        lookback: Number of candles to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of HeikinAshiPattern objects representing weak trend patterns
    """
    patterns = []
    
    # Ensure we have enough candles
    if len(candles) < lookback:
        return patterns
    
    # Limit to lookback period
    recent_candles = candles[-lookback:]
    
    # Detect weak trends
    for i in range(min_trend_length, len(recent_candles)):
        # Get current trend candles
        trend_candles = recent_candles[i-min_trend_length:i+1]
        
        # Check for weak bullish trend
        if all(c.close > c.open for c in trend_candles) and \
           any(c.low < c.open for c in trend_candles) and \
           sum(c.high - c.close for c in trend_candles) / len(trend_candles) > \
           sum(c.close - c.open for c in trend_candles) / len(trend_candles):
            # We have a weak bullish trend (long upper shadows)
            
            # Calculate pattern strength
            # Strength is based on the consistency of the trend and the presence of shadows
            trend_strength = sum(abs(c.close - c.open) for c in trend_candles) / len(trend_candles)
            shadow_ratio = sum(c.high - c.close for c in trend_candles) / sum(c.close - c.open for c in trend_candles)
            
            # Normalize strength (0.0-1.0)
            normalized_strength = min(1.0, (1.0 / shadow_ratio) * sensitivity)
            
            # Calculate target price (measured move)
            trend_start = trend_candles[0].open
            trend_current = trend_candles[-1].close
            trend_range = trend_current - trend_start
            target_price = trend_current + trend_range * 0.382  # Smaller Fibonacci extension due to weakness
            
            # Calculate stop price (below lowest low)
            stop_price = min(c.low for c in trend_candles) * 0.99  # 1% below lowest low
            
            # Create the pattern
            pattern = HeikinAshiPattern(
                pattern_type=HeikinAshiPatternType.WEAK_TREND,
                start_index=trend_candles[0].index,
                end_index=trend_candles[-1].index,
                candles=trend_candles,
                trend_type=HeikinAshiTrendType.BULLISH,
                strength=normalized_strength,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
        
        # Check for weak bearish trend
        elif all(c.close < c.open for c in trend_candles) and \
             any(c.high > c.open for c in trend_candles) and \
             sum(c.open - c.low for c in trend_candles) / len(trend_candles) > \
             sum(c.open - c.close for c in trend_candles) / len(trend_candles):
            # We have a weak bearish trend (long lower shadows)
            
            # Calculate pattern strength
            # Strength is based on the consistency of the trend and the presence of shadows
            trend_strength = sum(abs(c.close - c.open) for c in trend_candles) / len(trend_candles)
            shadow_ratio = sum(c.open - c.low for c in trend_candles) / sum(c.open - c.close for c in trend_candles)
            
            # Normalize strength (0.0-1.0)
            normalized_strength = min(1.0, (1.0 / shadow_ratio) * sensitivity)
            
            # Calculate target price (measured move)
            trend_start = trend_candles[0].open
            trend_current = trend_candles[-1].close
            trend_range = trend_start - trend_current
            target_price = trend_current - trend_range * 0.382  # Smaller Fibonacci extension due to weakness
            
            # Calculate stop price (above highest high)
            stop_price = max(c.high for c in trend_candles) * 1.01  # 1% above highest high
            
            # Create the pattern
            pattern = HeikinAshiPattern(
                pattern_type=HeikinAshiPatternType.WEAK_TREND,
                start_index=trend_candles[0].index,
                end_index=trend_candles[-1].index,
                candles=trend_candles,
                trend_type=HeikinAshiTrendType.BEARISH,
                strength=normalized_strength,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
    
    return patterns