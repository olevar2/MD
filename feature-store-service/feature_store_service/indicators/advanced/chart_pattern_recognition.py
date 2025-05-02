"""
Chart Pattern Recognition.

This module implements algorithms to recognize classic chart patterns
with price target estimates and pattern quality rating system.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum

from feature_store_service.indicators.base_indicator import BaseIndicator


class PatternQuality(Enum):
    """Enum for pattern quality ratings."""
    POOR = 1
    FAIR = 2
    GOOD = 3
    EXCELLENT = 4
    IDEAL = 5


class ChartPatternRecognition(BaseIndicator):
    """
    Chart Pattern Recognition indicator.
    
    This indicator implements algorithms to recognize classic chart patterns,
    adds price target estimates, and develops a quality rating system.
    """
    
    category = "pattern_recognition"
    
    def __init__(
        self, 
        min_pattern_bars: int = 5,
        max_pattern_bars: int = 100, 
        noise_threshold: float = 0.03,
        **kwargs
    ):
        """
        Initialize Chart Pattern Recognition indicator.
        
        Args:
            min_pattern_bars: Minimum number of bars required for pattern detection
            max_pattern_bars: Maximum number of bars to look back for pattern detection
            noise_threshold: Threshold for filtering out market noise (as decimal)
            **kwargs: Additional parameters
        """
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.noise_threshold = noise_threshold
        self.name = "chart_patterns"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Recognize chart patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with recognized patterns, price targets, and quality ratings
        """
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Smooth the price data to reduce noise
        result['smooth_high'] = result['high'].rolling(window=3).mean()
        result['smooth_low'] = result['low'].rolling(window=3).mean()
        result['smooth_close'] = result['close'].rolling(window=3).mean()
        
        # Identify significant pivots
        self._identify_pivots(result)
        
        # Detect Head and Shoulders patterns
        self._detect_head_and_shoulders(result)
        self._detect_inverse_head_and_shoulders(result)
        
        # Detect Double Top/Bottom patterns
        self._detect_double_top(result)
        self._detect_double_bottom(result)
        
        # Detect Triangle patterns
        self._detect_ascending_triangle(result)
        self._detect_descending_triangle(result)
        self._detect_symmetrical_triangle(result)
        
        # Detect Flags and Pennants
        self._detect_bull_flag(result)
        self._detect_bear_flag(result)
        self._detect_pennant(result)
        
        # Detect Rectangle patterns
        self._detect_rectangle(result)
        
        # Calculate pattern rating and price targets
        self._calculate_pattern_ratings(result)
        self._calculate_price_targets(result)
        
        return result
    
    def _identify_pivots(self, data: pd.DataFrame, window: int = 5) -> None:
        """
        Identify significant pivot points in the price data.
        
        Args:
            data: DataFrame with OHLCV data
            window: Number of bars to look before/after for pivot detection
        """
        # Identify pivot highs and lows
        data['pivot_high'] = 0
        data['pivot_low'] = 0
        
        for i in range(window, len(data) - window):
            # Check if this bar's high is higher than all bars in the window before and after
            if all(data.loc[i, 'high'] > data.loc[i-window:i-1, 'high']) and \
               all(data.loc[i, 'high'] > data.loc[i+1:i+window, 'high']):
                data.loc[i, 'pivot_high'] = 1
                
            # Check if this bar's low is lower than all bars in the window before and after
            if all(data.loc[i, 'low'] < data.loc[i-window:i-1, 'low']) and \
               all(data.loc[i, 'low'] < data.loc[i+1:i+window, 'low']):
                data.loc[i, 'pivot_low'] = 1
                
        # Store pivot point values
        data['pivot_high_value'] = data['high'] * data['pivot_high']
        data['pivot_low_value'] = data['low'] * data['pivot_low']
        
        # Replace zeros with NaN for easier manipulation
        data['pivot_high_value'].replace(0, np.nan, inplace=True)
        data['pivot_low_value'].replace(0, np.nan, inplace=True)
        
    def _detect_head_and_shoulders(self, data: pd.DataFrame) -> None:
        """
        Detect Head and Shoulders pattern (bearish reversal).
        
        Args:
            data: DataFrame with OHLCV and pivot data
        """
        data['head_and_shoulders'] = 0
        data['head_and_shoulders_quality'] = 0
        data['head_and_shoulders_target'] = 0
        
        # Find pivot highs
        pivot_highs = data[data['pivot_high'] == 1].copy()
        
        # Check each window of 5 consecutives pivot highs
        for i in range(len(pivot_highs) - 4):
            # Get the potential left shoulder, head, and right shoulder
            left_shoulder_idx = pivot_highs.index[i]
            head_idx = pivot_highs.index[i+2]  # Middle pivot
            right_shoulder_idx = pivot_highs.index[i+4]
            
            # Pattern must be contained within our max lookback
            if head_idx - left_shoulder_idx > self.max_pattern_bars:
                continue
            
            if right_shoulder_idx - head_idx > self.max_pattern_bars:
                continue
                
            # Get the values
            left_shoulder = pivot_highs.loc[left_shoulder_idx, 'pivot_high_value']
            head = pivot_highs.loc[head_idx, 'pivot_high_value']
            right_shoulder = pivot_highs.loc[right_shoulder_idx, 'pivot_high_value']
            
            # Check the pattern criteria
            if (head > left_shoulder and head > right_shoulder and 
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.1):
                
                # Find neckline between the troughs
                left_trough_idx = pivot_highs.index[i+1]
                right_trough_idx = pivot_highs.index[i+3]
                
                left_trough = data.loc[left_trough_idx:head_idx, 'low'].min()
                right_trough = data.loc[head_idx:right_trough_idx, 'low'].min()
                
                # Neckline should be relatively flat
                if abs(left_trough - right_trough) / left_trough < 0.05:
                    neckline = (left_trough + right_trough) / 2
                    
                    # Confirm the pattern with a neckline break
                    for j in range(right_shoulder_idx, min(right_shoulder_idx + self.max_pattern_bars, len(data))):
                        if data.loc[j, 'close'] < neckline:
                            # Pattern confirmed at index j
                            data.loc[j, 'head_and_shoulders'] = 1
                            
                            # Price target: distance from head to neckline projected below
                            height = head - neckline
                            data.loc[j, 'head_and_shoulders_target'] = neckline - height
                            
                            # Quality calculation done in _calculate_pattern_ratings method
                            break
    
    def _detect_inverse_head_and_shoulders(self, data: pd.DataFrame) -> None:
        """
        Detect Inverse Head and Shoulders pattern (bullish reversal).
        
        Args:
            data: DataFrame with OHLCV and pivot data
        """
        data['inverse_head_and_shoulders'] = 0
        data['inverse_head_and_shoulders_quality'] = 0
        data['inverse_head_and_shoulders_target'] = 0
        
        # Find pivot lows
        pivot_lows = data[data['pivot_low'] == 1].copy()
        
        # Check each window of 5 consecutives pivot lows
        for i in range(len(pivot_lows) - 4):
            # Get the potential left shoulder, head, and right shoulder
            left_shoulder_idx = pivot_lows.index[i]
            head_idx = pivot_lows.index[i+2]  # Middle pivot
            right_shoulder_idx = pivot_lows.index[i+4]
            
            # Pattern must be contained within our max lookback
            if head_idx - left_shoulder_idx > self.max_pattern_bars:
                continue
            
            if right_shoulder_idx - head_idx > self.max_pattern_bars:
                continue
                
            # Get the values
            left_shoulder = pivot_lows.loc[left_shoulder_idx, 'pivot_low_value']
            head = pivot_lows.loc[head_idx, 'pivot_low_value']
            right_shoulder = pivot_lows.loc[right_shoulder_idx, 'pivot_low_value']
            
            # Check the pattern criteria
            if (head < left_shoulder and head < right_shoulder and 
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.1):
                
                # Find neckline between the peaks
                left_peak_idx = pivot_lows.index[i+1]
                right_peak_idx = pivot_lows.index[i+3]
                
                left_peak = data.loc[left_shoulder_idx:head_idx, 'high'].max()
                right_peak = data.loc[head_idx:right_shoulder_idx, 'high'].max()
                
                # Neckline should be relatively flat
                if abs(left_peak - right_peak) / left_peak < 0.05:
                    neckline = (left_peak + right_peak) / 2
                    
                    # Confirm the pattern with a neckline break
                    for j in range(right_shoulder_idx, min(right_shoulder_idx + self.max_pattern_bars, len(data))):
                        if data.loc[j, 'close'] > neckline:
                            # Pattern confirmed at index j
                            data.loc[j, 'inverse_head_and_shoulders'] = 1
                            
                            # Price target: distance from head to neckline projected above
                            height = neckline - head
                            data.loc[j, 'inverse_head_and_shoulders_target'] = neckline + height
                            
                            # Quality calculation done in _calculate_pattern_ratings method
                            break
    
    def _detect_double_top(self, data: pd.DataFrame) -> None:
        """
        Detect Double Top pattern (bearish reversal).
        
        Args:
            data: DataFrame with OHLCV and pivot data
        """
        data['double_top'] = 0
        data['double_top_quality'] = 0
        data['double_top_target'] = 0
        
        # Find pivot highs
        pivot_highs = data[data['pivot_high'] == 1].copy()
        
        # Check each window of 3 consecutive pivot highs
        for i in range(len(pivot_highs) - 2):
            # Get the potential double top points
            first_top_idx = pivot_highs.index[i]
            second_top_idx = pivot_highs.index[i+2]
            
            # Pattern must be contained within our max lookback
            if second_top_idx - first_top_idx > self.max_pattern_bars:
                continue
                
            # Get the values
            first_top = pivot_highs.loc[first_top_idx, 'pivot_high_value']
            second_top = pivot_highs.loc[second_top_idx, 'pivot_high_value']
            
            # Middle trough
            trough_idx = pivot_highs.index[i+1]
            trough = data.loc[trough_idx, 'low']
            
            # Check the pattern criteria
            if abs(first_top - second_top) / first_top < 0.05:  # Tops should be at similar levels
                # Neckline is at the trough
                neckline = trough
                
                # Confirm the pattern with a neckline break
                for j in range(second_top_idx, min(second_top_idx + self.max_pattern_bars, len(data))):
                    if data.loc[j, 'close'] < neckline:
                        # Pattern confirmed at index j
                        data.loc[j, 'double_top'] = 1
                        
                        # Price target: distance from tops to neckline projected below
                        height = ((first_top + second_top) / 2) - neckline
                        data.loc[j, 'double_top_target'] = neckline - height
                        
                        # Quality calculation done in _calculate_pattern_ratings method
                        break
    
    def _detect_double_bottom(self, data: pd.DataFrame) -> None:
        """
        Detect Double Bottom pattern (bullish reversal).
        
        Args:
            data: DataFrame with OHLCV and pivot data
        """
        data['double_bottom'] = 0
        data['double_bottom_quality'] = 0
        data['double_bottom_target'] = 0
        
        # Find pivot lows
        pivot_lows = data[data['pivot_low'] == 1].copy()
        
        # Check each window of 3 consecutive pivot lows
        for i in range(len(pivot_lows) - 2):
            # Get the potential double bottom points
            first_bottom_idx = pivot_lows.index[i]
            second_bottom_idx = pivot_lows.index[i+2]
            
            # Pattern must be contained within our max lookback
            if second_bottom_idx - first_bottom_idx > self.max_pattern_bars:
                continue
                
            # Get the values
            first_bottom = pivot_lows.loc[first_bottom_idx, 'pivot_low_value']
            second_bottom = pivot_lows.loc[second_bottom_idx, 'pivot_low_value']
            
            # Middle peak
            peak_idx = pivot_lows.index[i+1]
            peak = data.loc[peak_idx, 'high']
            
            # Check the pattern criteria
            if abs(first_bottom - second_bottom) / first_bottom < 0.05:  # Bottoms should be at similar levels
                # Neckline is at the peak
                neckline = peak
                
                # Confirm the pattern with a neckline break
                for j in range(second_bottom_idx, min(second_bottom_idx + self.max_pattern_bars, len(data))):
                    if data.loc[j, 'close'] > neckline:
                        # Pattern confirmed at index j
                        data.loc[j, 'double_bottom'] = 1
                        
                        # Price target: distance from bottoms to neckline projected above
                        height = neckline - ((first_bottom + second_bottom) / 2)
                        data.loc[j, 'double_bottom_target'] = neckline + height
                        
                        # Quality calculation done in _calculate_pattern_ratings method
                        break
    
    def _detect_ascending_triangle(self, data: pd.DataFrame) -> None:
        """
        Detect Ascending Triangle pattern (typically bullish).
        
        Args:
            data: DataFrame with OHLCV and pivot data
        """
        data['ascending_triangle'] = 0
        data['ascending_triangle_quality'] = 0
        data['ascending_triangle_target'] = 0
        
        # Need at least 2 higher lows and 2 similar highs
        # Find pivot points
        pivot_highs = data[data['pivot_high'] == 1].copy()
        pivot_lows = data[data['pivot_low'] == 1].copy()
        
        # First check if we have enough pivot points
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return
            
        # Check for ascending triangle
        for i in range(len(pivot_highs) - 1):
            high1_idx = pivot_highs.index[i]
            high2_idx = pivot_highs.index[i+1]
            
            # Check for similar highs
            high1 = pivot_highs.loc[high1_idx, 'pivot_high_value']
            high2 = pivot_highs.loc[high2_idx, 'pivot_high_value']
            
            # Horizontal resistance line
            if abs(high1 - high2) / high1 < 0.03:
                resistance = (high1 + high2) / 2
                
                # Find lows between these highs
                lows_between = pivot_lows[(pivot_lows.index > high1_idx) & 
                                        (pivot_lows.index < high2_idx)]
                
                if len(lows_between) >= 2:
                    # Check if lows are ascending
                    is_ascending = True
                    for j in range(len(lows_between) - 1):
                        low1 = lows_between.iloc[j]['pivot_low_value']
                        low2 = lows_between.iloc[j+1]['pivot_low_value']
                        
                        if low2 <= low1:
                            is_ascending = False
                            break
                            
                    if is_ascending:
                        # Calculate the slope of the ascending trendline
                        first_low_idx = lows_between.index[0]
                        last_low_idx = lows_between.index[-1]
                        first_low = lows_between.loc[first_low_idx, 'pivot_low_value']
                        last_low = lows_between.loc[last_low_idx, 'pivot_low_value']
                        
                        # Confirm the pattern with a breakout
                        for j in range(high2_idx, min(high2_idx + self.max_pattern_bars, len(data))):
                            if data.loc[j, 'close'] > resistance:
                                # Pattern confirmed at index j
                                data.loc[j, 'ascending_triangle'] = 1
                                
                                # Price target: height of the triangle added to the breakout point
                                height = resistance - first_low
                                data.loc[j, 'ascending_triangle_target'] = resistance + height
                                
                                # Quality calculation done in _calculate_pattern_ratings method
                                break
    
    def _detect_descending_triangle(self, data: pd.DataFrame) -> None:
        """
        Detect Descending Triangle pattern (typically bearish).
        
        Args:
            data: DataFrame with OHLCV and pivot data
        """
        data['descending_triangle'] = 0
        data['descending_triangle_quality'] = 0
        data['descending_triangle_target'] = 0
        
        # Need at least 2 lower highs and 2 similar lows
        # Find pivot points
        pivot_highs = data[data['pivot_high'] == 1].copy()
        pivot_lows = data[data['pivot_low'] == 1].copy()
        
        # First check if we have enough pivot points
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return
            
        # Check for descending triangle
        for i in range(len(pivot_lows) - 1):
            low1_idx = pivot_lows.index[i]
            low2_idx = pivot_lows.index[i+1]
            
            # Check for similar lows
            low1 = pivot_lows.loc[low1_idx, 'pivot_low_value']
            low2 = pivot_lows.loc[low2_idx, 'pivot_low_value']
            
            # Horizontal support line
            if abs(low1 - low2) / low1 < 0.03:
                support = (low1 + low2) / 2
                
                # Find highs between these lows
                highs_between = pivot_highs[(pivot_highs.index > low1_idx) & 
                                         (pivot_highs.index < low2_idx)]
                
                if len(highs_between) >= 2:
                    # Check if highs are descending
                    is_descending = True
                    for j in range(len(highs_between) - 1):
                        high1 = highs_between.iloc[j]['pivot_high_value']
                        high2 = highs_between.iloc[j+1]['pivot_high_value']
                        
                        if high2 >= high1:
                            is_descending = False
                            break
                            
                    if is_descending:
                        # Calculate the slope of the descending trendline
                        first_high_idx = highs_between.index[0]
                        last_high_idx = highs_between.index[-1]
                        first_high = highs_between.loc[first_high_idx, 'pivot_high_value']
                        last_high = highs_between.loc[last_high_idx, 'pivot_high_value']
                        
                        # Confirm the pattern with a breakdown
                        for j in range(low2_idx, min(low2_idx + self.max_pattern_bars, len(data))):
                            if data.loc[j, 'close'] < support:
                                # Pattern confirmed at index j
                                data.loc[j, 'descending_triangle'] = 1
                                
                                # Price target: height of the triangle subtracted from the breakdown point
                                height = first_high - support
                                data.loc[j, 'descending_triangle_target'] = support - height
                                
                                # Quality calculation done in _calculate_pattern_ratings method
                                break
    
    def _detect_symmetrical_triangle(self, data: pd.DataFrame) -> None:
        """
        Detect Symmetrical Triangle pattern (continuation pattern).
        
        Args:
            data: DataFrame with OHLCV and pivot data
        """
        data['symmetrical_triangle'] = 0
        data['symmetrical_triangle_quality'] = 0
        data['symmetrical_triangle_target'] = 0
        data['symmetrical_triangle_direction'] = 0  # 1 for bullish, -1 for bearish
        
        # Need at least 2 lower highs and 2 higher lows
        # Find pivot points
        pivot_highs = data[data['pivot_high'] == 1].copy()
        pivot_lows = data[data['pivot_low'] == 1].copy()
        
        # First check if we have enough pivot points
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return
            
        # Check each segment with multiple highs and lows
        for i in range(len(pivot_highs) - 1):
            start_idx = pivot_highs.index[i]
            end_idx = start_idx + self.max_pattern_bars
            
            if end_idx >= len(data):
                end_idx = len(data) - 1
                
            # Get highs and lows in this window
            window_highs = pivot_highs[(pivot_highs.index >= start_idx) & 
                                     (pivot_highs.index <= end_idx)]
            window_lows = pivot_lows[(pivot_lows.index >= start_idx) & 
                                   (pivot_lows.index <= end_idx)]
            
            if len(window_highs) < 2 or len(window_lows) < 2:
                continue
                
            # Check if highs are descending and lows are ascending
            highs_descending = True
            lows_ascending = True
            
            for j in range(len(window_highs) - 1):
                if window_highs.iloc[j+1]['pivot_high_value'] >= window_highs.iloc[j]['pivot_high_value']:
                    highs_descending = False
                    break
                    
            for j in range(len(window_lows) - 1):
                if window_lows.iloc[j+1]['pivot_low_value'] <= window_lows.iloc[j]['pivot_low_value']:
                    lows_ascending = False
                    break
            
            if highs_descending and lows_ascending:
                # Calculate the converging trendlines
                first_high = window_highs.iloc[0]['pivot_high_value']
                last_high = window_highs.iloc[-1]['pivot_high_value']
                first_low = window_lows.iloc[0]['pivot_low_value']
                last_low = window_lows.iloc[-1]['pivot_low_value']
                
                # Last point in the pattern
                last_idx = max(window_highs.index[-1], window_lows.index[-1])
                
                # Check for breakout after the pattern
                for j in range(last_idx, min(last_idx + self.max_pattern_bars, len(data))):
                    # Determine if breakout is up or down
                    if data.loc[j, 'close'] > data.loc[j-1:j, 'high'].max():
                        # Bullish breakout
                        data.loc[j, 'symmetrical_triangle'] = 1
                        data.loc[j, 'symmetrical_triangle_direction'] = 1
                        
                        # Price target: height of the triangle added to the breakout point
                        height = first_high - first_low
                        data.loc[j, 'symmetrical_triangle_target'] = data.loc[j, 'close'] + height
                        break
                        
                    elif data.loc[j, 'close'] < data.loc[j-1:j, 'low'].min():
                        # Bearish breakout
                        data.loc[j, 'symmetrical_triangle'] = 1
                        data.loc[j, 'symmetrical_triangle_direction'] = -1
                        
                        # Price target: height of the triangle subtracted from the breakdown point
                        height = first_high - first_low
                        data.loc[j, 'symmetrical_triangle_target'] = data.loc[j, 'close'] - height
                        break
    
    def _detect_bull_flag(self, data: pd.DataFrame) -> None:
        """
        Detect Bullish Flag pattern (continuation pattern).
        
        Args:
            data: DataFrame with OHLCV data
        """
        data['bull_flag'] = 0
        data['bull_flag_quality'] = 0
        data['bull_flag_target'] = 0
        
        # Look for strong upward move followed by consolidation
        for i in range(self.min_pattern_bars, len(data) - self.min_pattern_bars):
            # Check for a strong upward move (flag pole)
            pole_start = i - self.min_pattern_bars
            pole_end = i
            
            price_change = data.loc[pole_end, 'close'] - data.loc[pole_start, 'close']
            percent_change = price_change / data.loc[pole_start, 'close']
            
            # Pole should be substantial rise
            if percent_change > 0.05:  # Minimum 5% rise
                # Calculate the highest high in the pole
                pole_high = data.loc[pole_start:pole_end, 'high'].max()
                
                # Check for flag consolidation (slight downward channel)
                flag_start = pole_end
                flag_end = min(flag_start + self.max_pattern_bars, len(data) - 1)
                
                # Flag should not retrace more than 50% of the pole
                max_retrace = data.loc[flag_start:flag_end, 'low'].min()
                retrace_percent = (pole_high - max_retrace) / price_change
                
                if retrace_percent < 0.5 and retrace_percent > 0:
                    # Check for breakout above the flag
                    for j in range(flag_start + self.min_pattern_bars, flag_end):
                        # Calculate the flag's upper trendline at this point
                        flag_highs = data.loc[flag_start:j, 'high']
                        if len(flag_highs) < 3:
                            continue
                            
                        if data.loc[j, 'close'] > flag_highs.max():
                            # Confirmed breakout
                            data.loc[j, 'bull_flag'] = 1
                            
                            # Price target: pole height added to breakout point
                            data.loc[j, 'bull_flag_target'] = data.loc[j, 'close'] + price_change
                            break
    
    def _detect_bear_flag(self, data: pd.DataFrame) -> None:
        """
        Detect Bearish Flag pattern (continuation pattern).
        
        Args:
            data: DataFrame with OHLCV data
        """
        data['bear_flag'] = 0
        data['bear_flag_quality'] = 0
        data['bear_flag_target'] = 0
        
        # Look for strong downward move followed by consolidation
        for i in range(self.min_pattern_bars, len(data) - self.min_pattern_bars):
            # Check for a strong downward move (flag pole)
            pole_start = i - self.min_pattern_bars
            pole_end = i
            
            price_change = data.loc[pole_start, 'close'] - data.loc[pole_end, 'close']
            percent_change = price_change / data.loc[pole_start, 'close']
            
            # Pole should be substantial drop
            if percent_change > 0.05:  # Minimum 5% drop
                # Calculate the lowest low in the pole
                pole_low = data.loc[pole_start:pole_end, 'low'].min()
                
                # Check for flag consolidation (slight upward channel)
                flag_start = pole_end
                flag_end = min(flag_start + self.max_pattern_bars, len(data) - 1)
                
                # Flag should not retrace more than 50% of the pole
                max_retrace = data.loc[flag_start:flag_end, 'high'].max()
                retrace_percent = (max_retrace - pole_low) / price_change
                
                if retrace_percent < 0.5 and retrace_percent > 0:
                    # Check for breakdown below the flag
                    for j in range(flag_start + self.min_pattern_bars, flag_end):
                        # Calculate the flag's lower trendline at this point
                        flag_lows = data.loc[flag_start:j, 'low']
                        if len(flag_lows) < 3:
                            continue
                            
                        if data.loc[j, 'close'] < flag_lows.min():
                            # Confirmed breakdown
                            data.loc[j, 'bear_flag'] = 1
                            
                            # Price target: pole height subtracted from breakdown point
                            data.loc[j, 'bear_flag_target'] = data.loc[j, 'close'] - price_change
                            break
    
    def _detect_pennant(self, data: pd.DataFrame) -> None:
        """
        Detect Pennant pattern (continuation pattern).
        
        Args:
            data: DataFrame with OHLCV data
        """
        data['pennant'] = 0
        data['pennant_quality'] = 0
        data['pennant_target'] = 0
        data['pennant_direction'] = 0  # 1 for bullish, -1 for bearish
        
        # Look for strong move followed by convergence pattern
        for i in range(self.min_pattern_bars, len(data) - self.min_pattern_bars):
            # Check for a strong move (pennant pole)
            pole_start = i - self.min_pattern_bars
            pole_end = i
            
            price_change = data.loc[pole_end, 'close'] - data.loc[pole_start, 'close']
            percent_change = abs(price_change) / data.loc[pole_start, 'close']
            
            # Pole should be substantial move
            if percent_change > 0.05:  # Minimum 5% move
                # Direction of the pole (bullish or bearish)
                is_bullish = price_change > 0
                
                # Check for pennant consolidation
                pennant_start = pole_end
                pennant_end = min(pennant_start + self.max_pattern_bars, len(data) - 1)
                
                # Calculate price range in the pennant
                for j in range(pennant_start + self.min_pattern_bars, pennant_end):
                    pennant_segment = data.loc[pennant_start:j]
                    initial_range = pennant_segment.iloc[0:5]['high'].max() - pennant_segment.iloc[0:5]['low'].min()
                    final_range = pennant_segment.iloc[-5:]['high'].max() - pennant_segment.iloc[-5:]['low'].min()
                    
                    # Pennant should show converging range (narrowing)
                    if final_range < 0.5 * initial_range:
                        # Check for breakout in the direction of the pole
                        if is_bullish:
                            if data.loc[j, 'close'] > pennant_segment.iloc[-6:-1]['high'].max():
                                data.loc[j, 'pennant'] = 1
                                data.loc[j, 'pennant_direction'] = 1
                                
                                # Price target: pole height added to breakout point
                                data.loc[j, 'pennant_target'] = data.loc[j, 'close'] + abs(price_change)
                                break
                        else:  # Bearish
                            if data.loc[j, 'close'] < pennant_segment.iloc[-6:-1]['low'].min():
                                data.loc[j, 'pennant'] = 1
                                data.loc[j, 'pennant_direction'] = -1
                                
                                # Price target: pole height subtracted from breakdown point
                                data.loc[j, 'pennant_target'] = data.loc[j, 'close'] - abs(price_change)
                                break
    
    def _detect_rectangle(self, data: pd.DataFrame) -> None:
        """
        Detect Rectangle pattern (consolidation pattern).
        
        Args:
            data: DataFrame with OHLCV data
        """
        data['rectangle'] = 0
        data['rectangle_quality'] = 0
        data['rectangle_target'] = 0
        data['rectangle_direction'] = 0  # 1 for bullish, -1 for bearish
        
        # Find segments with clear horizontal support and resistance
        for i in range(self.min_pattern_bars, len(data) - self.min_pattern_bars):
            # Define rectangle window
            window_start = i - self.min_pattern_bars
            window_end = i
            
            window_data = data.loc[window_start:window_end]
            
            # Calculate support and resistance levels
            support_level = window_data['low'].quantile(0.1)  # Lower 10% of lows
            resistance_level = window_data['high'].quantile(0.9)  # Upper 10% of highs
            
            range_percent = (resistance_level - support_level) / support_level
            
            # Rectangle range should be reasonable (not too tight, not too wide)
            if range_percent > 0.02 and range_percent < 0.15:
                # Count touches of support and resistance
                support_touches = sum((window_data['low'] - support_level).abs() < 0.01 * support_level)
                resistance_touches = sum((window_data['high'] - resistance_level).abs() < 0.01 * resistance_level)
                
                # Good rectangles have multiple touches of both support and resistance
                if support_touches >= 2 and resistance_touches >= 2:
                    # Check for breakout after the rectangle
                    for j in range(window_end + 1, min(window_end + 20, len(data))):
                        # Bullish breakout
                        if data.loc[j, 'close'] > resistance_level * 1.01:
                            data.loc[j, 'rectangle'] = 1
                            data.loc[j, 'rectangle_direction'] = 1
                            
                            # Price target: height of rectangle added to breakout point
                            height = resistance_level - support_level
                            data.loc[j, 'rectangle_target'] = resistance_level + height
                            break
                            
                        # Bearish breakout
                        elif data.loc[j, 'close'] < support_level * 0.99:
                            data.loc[j, 'rectangle'] = 1
                            data.loc[j, 'rectangle_direction'] = -1
                            
                            # Price target: height of rectangle subtracted from breakdown point
                            height = resistance_level - support_level
                            data.loc[j, 'rectangle_target'] = support_level - height
                            break
    
    def _calculate_pattern_ratings(self, data: pd.DataFrame) -> None:
        """
        Calculate quality ratings for detected patterns.
        
        Args:
            data: DataFrame with detected patterns
        """
        # Get all pattern columns
        pattern_cols = [col for col in data.columns if any(pattern in col for pattern in [
            'head_and_shoulders', 'inverse_head_and_shoulders', 'double_top', 'double_bottom',
            'ascending_triangle', 'descending_triangle', 'symmetrical_triangle', 
            'bull_flag', 'bear_flag', 'pennant', 'rectangle'
        ]) and not any(suffix in col for suffix in ['quality', 'target', 'direction'])]
        
        # For each pattern, calculate quality rating
        for pattern_col in pattern_cols:
            quality_col = f"{pattern_col}_quality"
            
            # Get indices where the pattern is detected
            pattern_indices = data[data[pattern_col] == 1].index
            
            for idx in pattern_indices:
                # Calculate quality rating based on pattern-specific criteria
                if 'head_and_shoulders' in pattern_col or 'inverse_head_and_shoulders' in pattern_col:
                    # Quality based on symmetry and clarity
                    left_shoulder_idx = idx - data.loc[:idx, 'pivot_high'].iloc[::-1].argmax()
                    left_shoulder = data.loc[left_shoulder_idx, 'high']
                    
                    right_shoulder_idx = idx - data.loc[:idx, 'pivot_high'].iloc[::-1].argmax() * 2
                    if right_shoulder_idx < 0:
                        right_shoulder_idx = 0
                    right_shoulder = data.loc[right_shoulder_idx, 'high']
                    
                    # Symmetry
                    symmetry = 1 - min(abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder), 1)
                    
                    # Calculate quality (1-5)
                    quality = symmetry * 5
                    data.loc[idx, quality_col] = max(1, min(5, round(quality)))
                    
                elif 'double_top' in pattern_col or 'double_bottom' in pattern_col:
                    # Quality based on symmetry and time between tops/bottoms
                    first_peak_idx = idx - data.loc[:idx, 'pivot_high'].iloc[::-1].argmax()
                    second_peak_idx = idx
                    
                    # Time duration
                    duration_quality = min((second_peak_idx - first_peak_idx) / 10, 1)
                    
                    # Calculate quality (1-5)
                    quality = duration_quality * 5
                    data.loc[idx, quality_col] = max(1, min(5, round(quality)))
                    
                elif 'triangle' in pattern_col:
                    # Quality based on number of touches of trendlines
                    # This is simplistic - in real implementation, count actual touches
                    data.loc[idx, quality_col] = int(PatternQuality.GOOD.value)
                    
                elif 'flag' in pattern_col or 'pennant' in pattern_col:
                    # Quality based on pole strength and consolidation clarity
                    pole_start = idx - 10
                    pole_end = idx - 5
                    
                    if pole_start < 0:
                        pole_start = 0
                        
                    price_change = abs(data.loc[pole_end, 'close'] - data.loc[pole_start, 'close'])
                    percent_change = price_change / data.loc[pole_start, 'close']
                    
                    # Strong pole is high quality
                    pole_quality = min(percent_change * 50, 5)
                    data.loc[idx, quality_col] = max(1, min(5, round(pole_quality)))
                    
                elif 'rectangle' in pattern_col:
                    # Quality based on number of touches and clarity of boundaries
                    data.loc[idx, quality_col] = int(PatternQuality.GOOD.value)  # Default to GOOD
    
    def _calculate_price_targets(self, data: pd.DataFrame) -> None:
        """
        Calculate price targets for detected patterns.
        Price targets have already been calculated in individual detection methods.
        This method could be expanded to include secondary targets and stop levels.
        
        Args:
            data: DataFrame with detected patterns
        """
        pass
"""
