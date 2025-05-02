"""
Classic Chart Patterns Analysis Module

This module provides comprehensive chart pattern recognition including:
- Head and Shoulders / Inverse Head and Shoulders
- Double Top / Double Bottom
- Triple Top / Triple Bottom
- Ascending / Descending / Symmetrical Triangles
- Rectangle / Channel Patterns
- Flag and Pennant Patterns
- Wedge Patterns
- Cup and Handle Patterns

Implementations include both standard and incremental calculation approaches.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime
import math
from enum import Enum

from analysis_engine.analysis.advanced_ta.base import (
    AdvancedAnalysisBase,
    PatternRecognitionBase,
    PatternResult,
    ConfidenceLevel,
    MarketDirection,
    AnalysisTimeframe,
    detect_swings,
    normalize_price_series
)


class ChartPatternType(Enum):
    """Types of classic chart patterns"""
    HEAD_AND_SHOULDERS = "Head and Shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "Inverse Head and Shoulders"
    DOUBLE_TOP = "Double Top"
    DOUBLE_BOTTOM = "Double Bottom"
    TRIPLE_TOP = "Triple Top"
    TRIPLE_BOTTOM = "Triple Bottom"
    ASCENDING_TRIANGLE = "Ascending Triangle"
    DESCENDING_TRIANGLE = "Descending Triangle"
    SYMMETRICAL_TRIANGLE = "Symmetrical Triangle"
    RECTANGLE = "Rectangle"
    RISING_CHANNEL = "Rising Channel"
    FALLING_CHANNEL = "Falling Channel"
    BULL_FLAG = "Bull Flag"
    BEAR_FLAG = "Bear Flag"
    PENNANT = "Pennant"
    RISING_WEDGE = "Rising Wedge"
    FALLING_WEDGE = "Falling Wedge"
    CUP_AND_HANDLE = "Cup and Handle"
    UNKNOWN = "Unknown"


class ChartPattern(PatternResult):
    """Extended PatternResult with Chart Pattern specific attributes"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pattern_type = kwargs.get("pattern_type", ChartPatternType.UNKNOWN)
        self.points = kwargs.get("points", {})  # Dict mapping point names to (time, price)
        self.measurements = kwargs.get("measurements", {})  # Dict of pattern measurements
        self.formation_time = kwargs.get("formation_time", 0)  # Time taken to form in bars
        self.neckline = kwargs.get("neckline", None)  # (slope, intercept) for patterns with neckline
        self.breakout = kwargs.get("breakout", None)  # (time, price) of breakout point
        self.volume_confirms = kwargs.get("volume_confirms", False)  # Whether volume confirms pattern


class ChartPatternRecognizer(PatternRecognitionBase):
    """
    Classic Chart Pattern Recognition Engine
    
    This class implements detection of classic chart patterns that have
    been used by technical analysts for decades.
    """
    
    def __init__(self, price_column: str = "close", volume_column: str = "volume",
                 lookback_period: int = 100, min_pattern_bars: int = 10,
                 pattern_types: List[str] = None, 
                 min_pattern_height: float = 0.01,
                 use_volume_confirmation: bool = True):
        """
        Initialize the Chart Pattern Recognizer
        
        Args:
            price_column: Column name for price data
            volume_column: Column name for volume data
            lookback_period: Maximum number of bars to analyze
            min_pattern_bars: Minimum bars required for valid pattern
            pattern_types: List of pattern types to detect (default: all)
            min_pattern_height: Minimum height of pattern as % of price
            use_volume_confirmation: Whether to use volume for confirmation
        """
        pattern_types = pattern_types or [p.value for p in ChartPatternType if p != ChartPatternType.UNKNOWN]
        
        parameters = {
            "price_column": price_column,
            "volume_column": volume_column,
            "lookback_period": lookback_period,
            "min_pattern_bars": min_pattern_bars,
            "pattern_types": pattern_types,
            "min_pattern_height": min_pattern_height,
            "use_volume_confirmation": use_volume_confirmation
        }
        super().__init__("Chart Patterns", parameters)
    
    def find_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        Find chart patterns in price data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of ChartPattern objects
        """
        # Verify we have enough data
        if len(df) < self.parameters["lookback_period"]:
            return []
        
        # Use only the lookback period
        analysis_df = df.iloc[-self.parameters["lookback_period"]:]
        
        # Detect swing points for pattern recognition
        swing_df = detect_swings(analysis_df, lookback=5, price_col=self.parameters["price_column"])
        
        # Check for each pattern type
        patterns = []
        
        # Dictionary mapping pattern types to detection methods
        detection_methods = {
            ChartPatternType.HEAD_AND_SHOULDERS: self._find_head_and_shoulders,
            ChartPatternType.INVERSE_HEAD_AND_SHOULDERS: self._find_inverse_head_and_shoulders,
            ChartPatternType.DOUBLE_TOP: self._find_double_top,
            ChartPatternType.DOUBLE_BOTTOM: self._find_double_bottom,
            ChartPatternType.TRIPLE_TOP: self._find_triple_top,
            ChartPatternType.TRIPLE_BOTTOM: self._find_triple_bottom,
            ChartPatternType.ASCENDING_TRIANGLE: self._find_ascending_triangle,
            ChartPatternType.DESCENDING_TRIANGLE: self._find_descending_triangle,
            ChartPatternType.SYMMETRICAL_TRIANGLE: self._find_symmetrical_triangle,
            ChartPatternType.RECTANGLE: self._find_rectangle,
            # Add other patterns as needed
        }
        
        # Run each detection method
        for pattern_type_name in self.parameters["pattern_types"]:
            try:
                pattern_type = next(
                    p for p in ChartPatternType 
                    if p.value == pattern_type_name
                )
                
                detection_method = detection_methods.get(pattern_type)
                
                if detection_method:
                    found_patterns = detection_method(analysis_df, swing_df)
                    patterns.extend(found_patterns)
                
            except Exception as e:
                print(f"Error detecting {pattern_type_name}: {str(e)}")
                continue
        
        return patterns
    
    def _find_head_and_shoulders(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> List[ChartPattern]:
        """
        Find Head and Shoulders patterns
        
        A Head and Shoulders pattern consists of:
        - A left shoulder (minor high)
        - A head (higher high)
        - A right shoulder (minor high, similar to left shoulder)
        - A neckline connecting the troughs between the shoulders and head
        
        Args:
            df: DataFrame with OHLCV data
            swing_df: DataFrame with swing high/low points
            
        Returns:
            List of ChartPattern objects
        """
        patterns = []
        price_col = self.parameters["price_column"]
        min_bars = self.parameters["min_pattern_bars"]
        
        # Extract swing highs and lows
        swing_highs = swing_df[swing_df["swing_high"]].copy()
        swing_lows = swing_df[swing_df["swing_low"]].copy()
        
        # Need at least 3 swing highs and 2 swing lows for H&S
        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return patterns
            
        # Iterate potential combinations
        for i in range(len(swing_highs) - 2):
            # Get three consecutive swing highs
            left_shoulder_idx = swing_highs.index[i]
            head_idx = swing_highs.index[i+1]
            right_shoulder_idx = swing_highs.index[i+2]
            
            # Get the prices
            left_shoulder = swing_highs.loc[left_shoulder_idx][price_col]
            head = swing_highs.loc[head_idx][price_col]
            right_shoulder = swing_highs.loc[right_shoulder_idx][price_col]
            
            # Check if the pattern is formed properly
            # Head must be higher than both shoulders
            if not (head > left_shoulder and head > right_shoulder):
                continue
                
            # Shoulders should be roughly at same level (allow 30% difference)
            shoulder_diff_pct = abs(left_shoulder - right_shoulder) / left_shoulder
            if shoulder_diff_pct > 0.3:
                continue
                
            # Find the lows between the highs
            left_trough_candidates = swing_lows[
                (swing_lows.index > left_shoulder_idx) & 
                (swing_lows.index < head_idx)
            ]
            
            right_trough_candidates = swing_lows[
                (swing_lows.index > head_idx) & 
                (swing_lows.index < right_shoulder_idx)
            ]
            
            # If we don't have both troughs, skip
            if len(left_trough_candidates) == 0 or len(right_trough_candidates) == 0:
                continue
                
            # Take the lowest points as our troughs
            left_trough_idx = left_trough_candidates[price_col].idxmin()
            right_trough_idx = right_trough_candidates[price_col].idxmin()
            
            left_trough = swing_lows.loc[left_trough_idx][price_col]
            right_trough = swing_lows.loc[right_trough_idx][price_col]
            
            # Calculate the neckline (linear regression through two points)
            x1 = df.index.get_loc(left_trough_idx)
            x2 = df.index.get_loc(right_trough_idx)
            y1 = left_trough
            y2 = right_trough
            
            if x2 - x1 == 0:  # Avoid division by zero
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            # Calculate the neckline at the right shoulder
            x3 = df.index.get_loc(right_shoulder_idx)
            neckline_at_right_shoulder = slope * x3 + intercept
            
            # Check pattern height (as % of price)
            pattern_height = (head - min(left_trough, right_trough)) / head
            if pattern_height < self.parameters["min_pattern_height"]:
                continue
                
            # Check if the pattern has enough bars
            pattern_width = df.index.get_loc(right_shoulder_idx) - df.index.get_loc(left_shoulder_idx)
            if pattern_width < min_bars:
                continue
            
            # Check for pattern completion (right shoulder completed)
            # For a full pattern, price should break below the neckline after right shoulder
            breakout_idx = None
            breakout_price = None
            
            # Look for a breakout beyond the right shoulder
            for j in range(df.index.get_loc(right_shoulder_idx) + 1, len(df)):
                curr_price = df.iloc[j][price_col]
                curr_idx = df.index[j]
                neckline_at_curr = slope * j + intercept
                
                if curr_price < neckline_at_curr:
                    breakout_idx = curr_idx
                    breakout_price = curr_price
                    break
            
            # Volume confirmation (optional)
            volume_confirms = False
            if self.parameters["use_volume_confirmation"] and self.parameters["volume_column"] in df.columns:
                # Volume should increase on the breakout
                if breakout_idx is not None:
                    breakout_loc = df.index.get_loc(breakout_idx)
                    if breakout_loc > 0:
                        prev_volume = df.iloc[breakout_loc - 1][self.parameters["volume_column"]]
                        breakout_volume = df.iloc[breakout_loc][self.parameters["volume_column"]]
                        volume_confirms = breakout_volume > prev_volume
            
            # Create the pattern
            points = {
                'left_shoulder': (left_shoulder_idx, left_shoulder),
                'head': (head_idx, head),
                'right_shoulder': (right_shoulder_idx, right_shoulder),
                'left_trough': (left_trough_idx, left_trough),
                'right_trough': (right_trough_idx, right_trough)
            }
            
            measurements = {
                'pattern_height': pattern_height,
                'pattern_width': pattern_width,
                'shoulder_ratio': left_shoulder / right_shoulder,
                'trough_ratio': left_trough / right_trough,
                'price_target': right_trough - pattern_height  # Measured move target
            }
            
            # Calculate confidence level
            confidence = self._calculate_pattern_confidence(
                pattern_type=ChartPatternType.HEAD_AND_SHOULDERS,
                measurements=measurements,
                has_breakout=breakout_idx is not None,
                volume_confirms=volume_confirms
            )
            
            # Create pattern result
            pattern = ChartPattern(
                pattern_name="Head and Shoulders",
                pattern_type=ChartPatternType.HEAD_AND_SHOULDERS,
                timeframe=AnalysisTimeframe.D1,  # Would be determined by input data
                direction=MarketDirection.BEARISH,  # H&S is bearish
                confidence=confidence,
                start_time=left_shoulder_idx,
                end_time=right_shoulder_idx if breakout_idx is None else breakout_idx,
                start_price=left_shoulder,
                end_price=right_shoulder if breakout_idx is None else breakout_price,
                points=points,
                measurements=measurements,
                formation_time=pattern_width,
                neckline=(slope, intercept),
                breakout=(breakout_idx, breakout_price) if breakout_idx is not None else None,
                volume_confirms=volume_confirms
            )
            
            # Calculate target prices
            # Traditional target is the height of the pattern projected from the neckline
            if breakout_idx is not None:
                height = head - (left_trough + right_trough) / 2
                target1 = breakout_price - height * 0.618  # Conservative target
                target2 = breakout_price - height  # Standard measured move
                target3 = breakout_price - height * 1.618  # Extended target
                
                # Calculate stop loss (above the right shoulder)
                stop_loss = max(right_shoulder, head * 0.05 + right_shoulder)
                
                pattern.target_prices = [target1, target2, target3]
                pattern.stop_loss = stop_loss
            
            patterns.append(pattern)
        
        return patterns
    
    def _find_inverse_head_and_shoulders(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> List[ChartPattern]:
        """
        Find Inverse Head and Shoulders patterns
        
        An Inverse H&S pattern is the mirror image of H&S, showing a reversal from downtrend to uptrend.
        
        Args:
            df: DataFrame with OHLCV data
            swing_df: DataFrame with swing high/low points
            
        Returns:
            List of ChartPattern objects
        """
        patterns = []
        price_col = self.parameters["price_column"]
        min_bars = self.parameters["min_pattern_bars"]
        
        # Extract swing highs and lows
        swing_highs = swing_df[swing_df["swing_high"]].copy()
        swing_lows = swing_df[swing_df["swing_low"]].copy()
        
        # Need at least 3 swing lows and 2 swing highs for inverse H&S
        if len(swing_lows) < 3 or len(swing_highs) < 2:
            return patterns
            
        # Iterate potential combinations
        for i in range(len(swing_lows) - 2):
            # Get three consecutive swing lows
            left_shoulder_idx = swing_lows.index[i]
            head_idx = swing_lows.index[i+1]
            right_shoulder_idx = swing_lows.index[i+2]
            
            # Get the prices
            left_shoulder = swing_lows.loc[left_shoulder_idx][price_col]
            head = swing_lows.loc[head_idx][price_col]
            right_shoulder = swing_lows.loc[right_shoulder_idx][price_col]
            
            # Check if the pattern is formed properly
            # Head must be lower than both shoulders
            if not (head < left_shoulder and head < right_shoulder):
                continue
                
            # Shoulders should be roughly at same level (allow 30% difference)
            shoulder_diff_pct = abs(left_shoulder - right_shoulder) / left_shoulder
            if shoulder_diff_pct > 0.3:
                continue
                
            # Find the highs between the lows
            left_peak_candidates = swing_highs[
                (swing_highs.index > left_shoulder_idx) & 
                (swing_highs.index < head_idx)
            ]
            
            right_peak_candidates = swing_highs[
                (swing_highs.index > head_idx) & 
                (swing_highs.index < right_shoulder_idx)
            ]
            
            # If we don't have both peaks, skip
            if len(left_peak_candidates) == 0 or len(right_peak_candidates) == 0:
                continue
                
            # Take the highest points as our peaks
            left_peak_idx = left_peak_candidates[price_col].idxmax()
            right_peak_idx = right_peak_candidates[price_col].idxmax()
            
            left_peak = swing_highs.loc[left_peak_idx][price_col]
            right_peak = swing_highs.loc[right_peak_idx][price_col]
            
            # Calculate the neckline (linear regression through two points)
            x1 = df.index.get_loc(left_peak_idx)
            x2 = df.index.get_loc(right_peak_idx)
            y1 = left_peak
            y2 = right_peak
            
            if x2 - x1 == 0:  # Avoid division by zero
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            # Calculate the neckline at the right shoulder
            x3 = df.index.get_loc(right_shoulder_idx)
            neckline_at_right_shoulder = slope * x3 + intercept
            
            # Check pattern height (as % of price)
            pattern_height = (max(left_peak, right_peak) - head) / head
            if pattern_height < self.parameters["min_pattern_height"]:
                continue
                
            # Check if the pattern has enough bars
            pattern_width = df.index.get_loc(right_shoulder_idx) - df.index.get_loc(left_shoulder_idx)
            if pattern_width < min_bars:
                continue
            
            # Check for pattern completion (breakout above neckline after right shoulder)
            breakout_idx = None
            breakout_price = None
            
            # Look for a breakout beyond the right shoulder
            for j in range(df.index.get_loc(right_shoulder_idx) + 1, len(df)):
                curr_price = df.iloc[j][price_col]
                curr_idx = df.index[j]
                neckline_at_curr = slope * j + intercept
                
                if curr_price > neckline_at_curr:
                    breakout_idx = curr_idx
                    breakout_price = curr_price
                    break
            
            # Volume confirmation (optional)
            volume_confirms = False
            if self.parameters["use_volume_confirmation"] and self.parameters["volume_column"] in df.columns:
                # Volume should increase on the breakout
                if breakout_idx is not None:
                    breakout_loc = df.index.get_loc(breakout_idx)
                    if breakout_loc > 0:
                        prev_volume = df.iloc[breakout_loc - 1][self.parameters["volume_column"]]
                        breakout_volume = df.iloc[breakout_loc][self.parameters["volume_column"]]
                        volume_confirms = breakout_volume > prev_volume
            
            # Create the pattern
            points = {
                'left_shoulder': (left_shoulder_idx, left_shoulder),
                'head': (head_idx, head),
                'right_shoulder': (right_shoulder_idx, right_shoulder),
                'left_peak': (left_peak_idx, left_peak),
                'right_peak': (right_peak_idx, right_peak)
            }
            
            measurements = {
                'pattern_height': pattern_height,
                'pattern_width': pattern_width,
                'shoulder_ratio': left_shoulder / right_shoulder,
                'peak_ratio': left_peak / right_peak,
                'price_target': right_peak + pattern_height  # Measured move target
            }
            
            # Calculate confidence level
            confidence = self._calculate_pattern_confidence(
                pattern_type=ChartPatternType.INVERSE_HEAD_AND_SHOULDERS,
                measurements=measurements,
                has_breakout=breakout_idx is not None,
                volume_confirms=volume_confirms
            )
            
            # Create pattern result
            pattern = ChartPattern(
                pattern_name="Inverse Head and Shoulders",
                pattern_type=ChartPatternType.INVERSE_HEAD_AND_SHOULDERS,
                timeframe=AnalysisTimeframe.D1,  # Would be determined by input data
                direction=MarketDirection.BULLISH,  # Inverse H&S is bullish
                confidence=confidence,
                start_time=left_shoulder_idx,
                end_time=right_shoulder_idx if breakout_idx is None else breakout_idx,
                start_price=left_shoulder,
                end_price=right_shoulder if breakout_idx is None else breakout_price,
                points=points,
                measurements=measurements,
                formation_time=pattern_width,
                neckline=(slope, intercept),
                breakout=(breakout_idx, breakout_price) if breakout_idx is not None else None,
                volume_confirms=volume_confirms
            )
            
            # Calculate target prices
            # Traditional target is the height of the pattern projected from the neckline
            if breakout_idx is not None:
                height = (left_peak + right_peak) / 2 - head
                target1 = breakout_price + height * 0.618  # Conservative target
                target2 = breakout_price + height  # Standard measured move
                target3 = breakout_price + height * 1.618  # Extended target
                
                # Calculate stop loss (below the right shoulder)
                stop_loss = min(right_shoulder, right_shoulder - head * 0.05)
                
                pattern.target_prices = [target1, target2, target3]
                pattern.stop_loss = stop_loss
            
            patterns.append(pattern)
        
        return patterns
        
    def _find_double_top(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> List[ChartPattern]:
        """
        Find Double Top patterns
        
        A Double Top consists of two peaks at roughly the same price level
        with a trough in between, and a break below the trough confirms the pattern.
        
        Args:
            df: DataFrame with OHLCV data
            swing_df: DataFrame with swing high/low points
            
        Returns:
            List of ChartPattern objects
        """
        patterns = []
        price_col = self.parameters["price_column"]
        min_bars = self.parameters["min_pattern_bars"]
        
        # Extract swing highs and lows
        swing_highs = swing_df[swing_df["swing_high"]].copy()
        
        # Need at least 2 swing highs for a double top
        if len(swing_highs) < 2:
            return patterns
            
        # Iterate potential combinations
        for i in range(len(swing_highs) - 1):
            # Get two consecutive swing highs
            first_peak_idx = swing_highs.index[i]
            second_peak_idx = swing_highs.index[i+1]
            
            # Get the prices
            first_peak = swing_highs.loc[first_peak_idx][price_col]
            second_peak = swing_highs.loc[second_peak_idx][price_col]
            
            # Check if the peaks are at similar levels (within 3%)
            peak_diff_pct = abs(first_peak - second_peak) / first_peak
            if peak_diff_pct > 0.03:
                continue
                
            # Find the low between the two peaks
            trough_candidates = swing_df[
                (swing_df.index > first_peak_idx) & 
                (swing_df.index < second_peak_idx) &
                (swing_df["swing_low"] == True)
            ]
            
            # If we don't have a trough, skip
            if len(trough_candidates) == 0:
                continue
                
            # Take the lowest point as our trough
            trough_idx = trough_candidates[price_col].idxmin()
            trough = swing_df.loc[trough_idx][price_col]
            
            # Check pattern height (as % of price)
            pattern_height = (first_peak - trough) / first_peak
            if pattern_height < self.parameters["min_pattern_height"]:
                continue
                
            # Check if the pattern has enough bars
            pattern_width = df.index.get_loc(second_peak_idx) - df.index.get_loc(first_peak_idx)
            if pattern_width < min_bars:
                continue
            
            # Check for pattern confirmation (break below the trough)
            breakout_idx = None
            breakout_price = None
            
            # Look for a breakout after the second peak
            for j in range(df.index.get_loc(second_peak_idx) + 1, len(df)):
                curr_price = df.iloc[j][price_col]
                curr_idx = df.index[j]
                
                if curr_price < trough:
                    breakout_idx = curr_idx
                    breakout_price = curr_price
                    break
            
            # Volume confirmation (optional)
            volume_confirms = False
            if self.parameters["use_volume_confirmation"] and self.parameters["volume_column"] in df.columns:
                # Volume should increase on the breakout and decrease on second peak
                if breakout_idx is not None:
                    breakout_loc = df.index.get_loc(breakout_idx)
                    if breakout_loc > 0:
                        # Breakout volume increase
                        prev_volume = df.iloc[breakout_loc - 1][self.parameters["volume_column"]]
                        breakout_volume = df.iloc[breakout_loc][self.parameters["volume_column"]]
                        
                        # Second peak volume compared to first peak
                        first_peak_loc = df.index.get_loc(first_peak_idx)
                        second_peak_loc = df.index.get_loc(second_peak_idx)
                        first_peak_volume = df.iloc[first_peak_loc][self.parameters["volume_column"]]
                        second_peak_volume = df.iloc[second_peak_loc][self.parameters["volume_column"]]
                        
                        volume_confirms = (breakout_volume > prev_volume and 
                                          second_peak_volume < first_peak_volume)
            
            # Create the pattern
            points = {
                'first_peak': (first_peak_idx, first_peak),
                'trough': (trough_idx, trough),
                'second_peak': (second_peak_idx, second_peak)
            }
            
            measurements = {
                'pattern_height': pattern_height,
                'pattern_width': pattern_width,
                'peak_ratio': first_peak / second_peak,
                'price_target': trough - (first_peak - trough)  # Measured move target
            }
            
            # Calculate confidence level
            confidence = self._calculate_pattern_confidence(
                pattern_type=ChartPatternType.DOUBLE_TOP,
                measurements=measurements,
                has_breakout=breakout_idx is not None,
                volume_confirms=volume_confirms
            )
            
            # Create pattern result
            pattern = ChartPattern(
                pattern_name="Double Top",
                pattern_type=ChartPatternType.DOUBLE_TOP,
                timeframe=AnalysisTimeframe.D1,  # Would be determined by input data
                direction=MarketDirection.BEARISH,  # Double Top is bearish
                confidence=confidence,
                start_time=first_peak_idx,
                end_time=second_peak_idx if breakout_idx is None else breakout_idx,
                start_price=first_peak,
                end_price=second_peak if breakout_idx is None else breakout_price,
                points=points,
                measurements=measurements,
                formation_time=pattern_width,
                breakout=(breakout_idx, breakout_price) if breakout_idx is not None else None,
                volume_confirms=volume_confirms
            )
            
            # Calculate target prices
            # Traditional target is the height of the pattern projected from the break
            if breakout_idx is not None:
                height = first_peak - trough
                target1 = trough - height * 0.618  # Conservative target
                target2 = trough - height  # Standard measured move
                target3 = trough - height * 1.618  # Extended target
                
                # Calculate stop loss (above the second peak)
                stop_loss = max(second_peak, second_peak * 1.02)
                
                pattern.target_prices = [target1, target2, target3]
                pattern.stop_loss = stop_loss
            
            patterns.append(pattern)
        
        return patterns
    
    def _find_double_bottom(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> List[ChartPattern]:
        """
        Find Double Bottom patterns
        
        A Double Bottom consists of two troughs at roughly the same price level
        with a peak in between, and a break above the peak confirms the pattern.
        
        Args:
            df: DataFrame with OHLCV data
            swing_df: DataFrame with swing high/low points
            
        Returns:
            List of ChartPattern objects
        """
        patterns = []
        price_col = self.parameters["price_column"]
        min_bars = self.parameters["min_pattern_bars"]
        
        # Extract swing lows
        swing_lows = swing_df[swing_df["swing_low"]].copy()
        
        # Need at least 2 swing lows for a double bottom
        if len(swing_lows) < 2:
            return patterns
            
        # Iterate potential combinations
        for i in range(len(swing_lows) - 1):
            # Get two consecutive swing lows
            first_trough_idx = swing_lows.index[i]
            second_trough_idx = swing_lows.index[i+1]
            
            # Get the prices
            first_trough = swing_lows.loc[first_trough_idx][price_col]
            second_trough = swing_lows.loc[second_trough_idx][price_col]
            
            # Check if the troughs are at similar levels (within 3%)
            trough_diff_pct = abs(first_trough - second_trough) / first_trough
            if trough_diff_pct > 0.03:
                continue
                
            # Find the high between the two troughs
            peak_candidates = swing_df[
                (swing_df.index > first_trough_idx) & 
                (swing_df.index < second_trough_idx) &
                (swing_df["swing_high"] == True)
            ]
            
            # If we don't have a peak, skip
            if len(peak_candidates) == 0:
                continue
                
            # Take the highest point as our peak
            peak_idx = peak_candidates[price_col].idxmax()
            peak = swing_df.loc[peak_idx][price_col]
            
            # Check pattern height (as % of price)
            pattern_height = (peak - first_trough) / first_trough
            if pattern_height < self.parameters["min_pattern_height"]:
                continue
                
            # Check if the pattern has enough bars
            pattern_width = df.index.get_loc(second_trough_idx) - df.index.get_loc(first_trough_idx)
            if pattern_width < min_bars:
                continue
            
            # Check for pattern confirmation (break above the peak)
            breakout_idx = None
            breakout_price = None
            
            # Look for a breakout after the second trough
            for j in range(df.index.get_loc(second_trough_idx) + 1, len(df)):
                curr_price = df.iloc[j][price_col]
                curr_idx = df.index[j]
                
                if curr_price > peak:
                    breakout_idx = curr_idx
                    breakout_price = curr_price
                    break
            
            # Volume confirmation (optional)
            volume_confirms = False
            if self.parameters["use_volume_confirmation"] and self.parameters["volume_column"] in df.columns:
                # Volume should increase on the breakout and decrease on second trough
                if breakout_idx is not None:
                    breakout_loc = df.index.get_loc(breakout_idx)
                    if breakout_loc > 0:
                        # Breakout volume increase
                        prev_volume = df.iloc[breakout_loc - 1][self.parameters["volume_column"]]
                        breakout_volume = df.iloc[breakout_loc][self.parameters["volume_column"]]
                        
                        # Second trough volume compared to first trough
                        first_trough_loc = df.index.get_loc(first_trough_idx)
                        second_trough_loc = df.index.get_loc(second_trough_idx)
                        first_trough_volume = df.iloc[first_trough_loc][self.parameters["volume_column"]]
                        second_trough_volume = df.iloc[second_trough_loc][self.parameters["volume_column"]]
                        
                        volume_confirms = (breakout_volume > prev_volume and 
                                          second_trough_volume < first_trough_volume)
            
            # Create the pattern
            points = {
                'first_trough': (first_trough_idx, first_trough),
                'peak': (peak_idx, peak),
                'second_trough': (second_trough_idx, second_trough)
            }
            
            measurements = {
                'pattern_height': pattern_height,
                'pattern_width': pattern_width,
                'trough_ratio': first_trough / second_trough,
                'price_target': peak + (peak - first_trough)  # Measured move target
            }
            
            # Calculate confidence level
            confidence = self._calculate_pattern_confidence(
                pattern_type=ChartPatternType.DOUBLE_BOTTOM,
                measurements=measurements,
                has_breakout=breakout_idx is not None,
                volume_confirms=volume_confirms
            )
            
            # Create pattern result
            pattern = ChartPattern(
                pattern_name="Double Bottom",
                pattern_type=ChartPatternType.DOUBLE_BOTTOM,
                timeframe=AnalysisTimeframe.D1,  # Would be determined by input data
                direction=MarketDirection.BULLISH,  # Double Bottom is bullish
                confidence=confidence,
                start_time=first_trough_idx,
                end_time=second_trough_idx if breakout_idx is None else breakout_idx,
                start_price=first_trough,
                end_price=second_trough if breakout_idx is None else breakout_price,
                points=points,
                measurements=measurements,
                formation_time=pattern_width,
                breakout=(breakout_idx, breakout_price) if breakout_idx is not None else None,
                volume_confirms=volume_confirms
            )
            
            # Calculate target prices
            # Traditional target is the height of the pattern projected from the break
            if breakout_idx is not None:
                height = peak - first_trough
                target1 = peak + height * 0.618  # Conservative target
                target2 = peak + height  # Standard measured move
                target3 = peak + height * 1.618  # Extended target
                
                # Calculate stop loss (below the second trough)
                stop_loss = min(second_trough, second_trough * 0.98)
                
                pattern.target_prices = [target1, target2, target3]
                pattern.stop_loss = stop_loss
            
            patterns.append(pattern)
        
        return patterns
    
    def _find_triple_top(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> List[ChartPattern]:
        """Find Triple Top patterns"""
        # Implementation similar to double top but with three peaks
        return []  # Placeholder
    
    def _find_triple_bottom(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> List[ChartPattern]:
        """Find Triple Bottom patterns"""
        # Implementation similar to double bottom but with three troughs
        return []  # Placeholder
    
    def _find_ascending_triangle(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> List[ChartPattern]:
        """Find Ascending Triangle patterns"""
        # Implementation for ascending triangles
        return []  # Placeholder
    
    def _find_descending_triangle(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> List[ChartPattern]:
        """Find Descending Triangle patterns"""
        # Implementation for descending triangles
        return []  # Placeholder
    
    def _find_symmetrical_triangle(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> List[ChartPattern]:
        """Find Symmetrical Triangle patterns"""
        # Implementation for symmetrical triangles
        return []  # Placeholder
    
    def _find_rectangle(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> List[ChartPattern]:
        """Find Rectangle/Channel patterns"""
        # Implementation for rectangle patterns
        return []  # Placeholder
    
    def _calculate_pattern_confidence(self, pattern_type: ChartPatternType, 
                                     measurements: Dict[str, float],
                                     has_breakout: bool,
                                     volume_confirms: bool) -> ConfidenceLevel:
        """
        Calculate confidence level for a pattern based on its characteristics
        
        Args:
            pattern_type: Type of chart pattern
            measurements: Dictionary of pattern measurements
            has_breakout: Whether pattern has confirmed with a breakout
            volume_confirms: Whether volume confirms the pattern
            
        Returns:
            ConfidenceLevel enum value
        """
        # Start with medium confidence
        confidence = ConfidenceLevel.MEDIUM
        
        # Adjust based on pattern completeness
        if not has_breakout:
            confidence = ConfidenceLevel.LOW
        
        # Adjust based on pattern size
        pattern_height = measurements.get('pattern_height', 0)
        if pattern_height > 0.1:  # Large pattern (>10% of price)
            confidence = ConfidenceLevel(min(confidence.value + 1, ConfidenceLevel.VERY_HIGH.value))
        elif pattern_height < 0.02:  # Small pattern (<2% of price)
            confidence = ConfidenceLevel(max(confidence.value - 1, ConfidenceLevel.VERY_LOW.value))
        
        # Adjust based on pattern symmetry (for applicable patterns)
        if pattern_type in [ChartPatternType.HEAD_AND_SHOULDERS, 
                           ChartPatternType.INVERSE_HEAD_AND_SHOULDERS]:
            shoulder_ratio = measurements.get('shoulder_ratio', 1.0)
            if 0.9 <= shoulder_ratio <= 1.1:  # Very symmetrical
                confidence = ConfidenceLevel(min(confidence.value + 1, ConfidenceLevel.VERY_HIGH.value))
            elif shoulder_ratio < 0.7 or shoulder_ratio > 1.3:  # Very asymmetrical
                confidence = ConfidenceLevel(max(confidence.value - 1, ConfidenceLevel.VERY_LOW.value))
                
        elif pattern_type in [ChartPatternType.DOUBLE_TOP, ChartPatternType.DOUBLE_BOTTOM]:
            peak_ratio = measurements.get('peak_ratio', 1.0)
            if 0.95 <= peak_ratio <= 1.05:  # Very even peaks/troughs
                confidence = ConfidenceLevel(min(confidence.value + 1, ConfidenceLevel.VERY_HIGH.value))
        
        # Adjust based on volume confirmation
        if volume_confirms:
            confidence = ConfidenceLevel(min(confidence.value + 1, ConfidenceLevel.VERY_HIGH.value))
        
        return confidence
    
    def initialize_incremental(self) -> Dict[str, Any]:
        """Initialize state for incremental pattern detection"""
        return {
            "price_buffer": [],
            "swing_points": [],  # List of (time, price, is_high)
            "potential_patterns": {},  # Dictionary of partial patterns being tracked
            "complete_patterns": [],  # List of completed patterns
            "lookback_period": self.parameters["lookback_period"],
            "price_column": self.parameters["price_column"],
        }
    
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Update chart pattern detection with new data
        
        Args:
            state: Current state dictionary
            new_data: New data point
            
        Returns:
            Updated state and any newly detected patterns
        """
        # Add new price to buffer
        if self.parameters["price_column"] in new_data:
            price = new_data[self.parameters["price_column"]]
            time = new_data.get("time", datetime.now())
            state["price_buffer"].append((time, price))
        else:
            return state  # Can't process without the required price data
        
        # Keep buffer within lookback size
        if len(state["price_buffer"]) > state["lookback_period"]:
            state["price_buffer"] = state["price_buffer"][-state["lookback_period"]:]
        
        # For pattern detection we need significant historical data
        if len(state["price_buffer"]) < 20:  # Arbitrary minimum
            return state
            
        # Simplified swing detection (could be enhanced with a proper algorithm)
        # Check if we can detect a new swing point with the latest data
        if len(state["price_buffer"]) >= 11:
            mid_idx = len(state["price_buffer"]) - 6
            mid_point = state["price_buffer"][mid_idx]
            
            is_swing_high = True
            is_swing_low = True
            
            # Check 5 points before and after
            for i in range(1, 6):
                before_price = state["price_buffer"][mid_idx - i][1]
                after_price = state["price_buffer"][mid_idx + i][1]
                
                if mid_point[1] <= before_price or mid_point[1] <= after_price:
                    is_swing_high = False
                    
                if mid_point[1] >= before_price or mid_point[1] >= after_price:
                    is_swing_low = False
            
            # Add swing point if detected
            if is_swing_high:
                state["swing_points"].append((mid_point[0], mid_point[1], True))
            elif is_swing_low:
                state["swing_points"].append((mid_point[0], mid_point[1], False))
                
            # Sort swing points by time
            state["swing_points"].sort(key=lambda x: x[0])
            
            # Keep only recent swings (arbitrary limit)
            if len(state["swing_points"]) > 20:
                state["swing_points"] = state["swing_points"][-20:]
                
            # Attempt to identify patterns with the updated swing points
            # For actual implementation, call pattern-specific detection methods
            self._identify_incremental_patterns(state)
        
        return state
    
    def _identify_incremental_patterns(self, state: Dict[str, Any]) -> None:
        """
        Attempt to identify chart patterns from current swing points
        
        Args:
            state: Current state dictionary with swing points
            
        Returns:
            None, updates state["complete_patterns"] in-place
        """
        # This is a simplified stub implementation
        # In a complete implementation, you would:
        # 1. Create mock DataFrames from the swing point data
        # 2. Call pattern-specific detection methods 
        # 3. Add any newly detected patterns to state["complete_patterns"]
        
        # Example implementation for head and shoulders:
        high_swings = [(t, p) for t, p, is_high in state["swing_points"] if is_high]
        low_swings = [(t, p) for t, p, is_high in state["swing_points"] if not is_high]
        
        # Need at least 3 highs and 2 lows for H&S
        if len(high_swings) >= 3 and len(low_swings) >= 2:
            # Create mock DataFrames and call the detection method
            # Implementation would be pattern-specific
            pass
