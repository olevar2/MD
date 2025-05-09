"""
Base module for advanced pattern recognition.

This module provides base classes and enums for advanced pattern recognition.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import abstractmethod
import pandas as pd
import numpy as np

from feature_store_service.indicators.base_indicator import BaseIndicator


class PatternDirection(Enum):
    """Direction of a pattern."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class PatternStrength(Enum):
    """Strength of a pattern."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    UNKNOWN = "unknown"


class AdvancedPatternType(Enum):
    """Types of advanced patterns."""
    # Renko patterns
    RENKO_REVERSAL = "renko_reversal"
    RENKO_BREAKOUT = "renko_breakout"
    RENKO_DOUBLE_TOP = "renko_double_top"
    RENKO_DOUBLE_BOTTOM = "renko_double_bottom"
    
    # Ichimoku patterns
    ICHIMOKU_TK_CROSS = "ichimoku_tk_cross"
    ICHIMOKU_KUMO_BREAKOUT = "ichimoku_kumo_breakout"
    ICHIMOKU_KUMO_TWIST = "ichimoku_kumo_twist"
    ICHIMOKU_CHIKOU_CROSS = "ichimoku_chikou_cross"
    
    # Wyckoff patterns
    WYCKOFF_ACCUMULATION = "wyckoff_accumulation"
    WYCKOFF_DISTRIBUTION = "wyckoff_distribution"
    WYCKOFF_SPRING = "wyckoff_spring"
    WYCKOFF_UPTHRUST = "wyckoff_upthrust"
    
    # Heikin-Ashi patterns
    HEIKIN_ASHI_REVERSAL = "heikin_ashi_reversal"
    HEIKIN_ASHI_CONTINUATION = "heikin_ashi_continuation"
    
    # VSA patterns
    VSA_NO_DEMAND = "vsa_no_demand"
    VSA_NO_SUPPLY = "vsa_no_supply"
    VSA_STOPPING_VOLUME = "vsa_stopping_volume"
    VSA_CLIMACTIC_VOLUME = "vsa_climactic_volume"
    VSA_EFFORT_VS_RESULT = "vsa_effort_vs_result"
    
    # Market Profile patterns
    MARKET_PROFILE_VALUE_AREA = "market_profile_value_area"
    MARKET_PROFILE_SINGLE_PRINT = "market_profile_single_print"
    MARKET_PROFILE_IB_BREAKOUT = "market_profile_ib_breakout"
    
    # Point and Figure patterns
    PNF_DOUBLE_TOP = "pnf_double_top"
    PNF_DOUBLE_BOTTOM = "pnf_double_bottom"
    PNF_TRIPLE_TOP = "pnf_triple_top"
    PNF_TRIPLE_BOTTOM = "pnf_triple_bottom"
    PNF_BULLISH_CATAPULT = "pnf_bullish_catapult"
    PNF_BEARISH_CATAPULT = "pnf_bearish_catapult"
    
    # Wolfe Wave patterns
    WOLFE_WAVE_BULLISH = "wolfe_wave_bullish"
    WOLFE_WAVE_BEARISH = "wolfe_wave_bearish"
    
    # Pitchfork patterns
    PITCHFORK_MEDIAN_LINE_BOUNCE = "pitchfork_median_line_bounce"
    PITCHFORK_MEDIAN_LINE_BREAK = "pitchfork_median_line_break"
    
    # Divergence patterns
    DIVERGENCE_REGULAR_BULLISH = "divergence_regular_bullish"
    DIVERGENCE_REGULAR_BEARISH = "divergence_regular_bearish"
    DIVERGENCE_HIDDEN_BULLISH = "divergence_hidden_bullish"
    DIVERGENCE_HIDDEN_BEARISH = "divergence_hidden_bearish"
    DIVERGENCE_TRIPLE = "divergence_triple"


class AdvancedPatternRecognizer(BaseIndicator):
    """
    Base class for advanced pattern recognizers.
    
    This class provides common functionality for all advanced pattern recognizers.
    """
    
    category = "pattern"
    
    def __init__(self, **kwargs):
        """
        Initialize the pattern recognizer.
        
        Args:
            **kwargs: Additional parameters
        """
        self.name = kwargs.get("name", self.__class__.__name__)
        self.lookback_period = kwargs.get("lookback_period", 100)
        self.min_pattern_size = kwargs.get("min_pattern_size", 5)
        self.max_pattern_size = kwargs.get("max_pattern_size", 50)
        self.sensitivity = kwargs.get("sensitivity", 0.75)
        self.pattern_types = kwargs.get("pattern_types", None)  # None means all patterns
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pattern recognition values
        """
        raise NotImplementedError("Subclasses must implement calculate()")
    
    def find_patterns(self, data: pd.DataFrame, pattern_types: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data
            pattern_types: List of pattern types to look for (None = all patterns)
            
        Returns:
            Dictionary of pattern types and their occurrences
        """
        raise NotImplementedError("Subclasses must implement find_patterns()")
    
    def _calculate_pattern_strength(self, data: pd.DataFrame) -> None:
        """
        Calculate and add pattern strength metrics to the DataFrame.
        
        Args:
            data: DataFrame with pattern recognition columns
        """
        # Get all pattern columns
        pattern_cols = [col for col in data.columns if col.startswith("pattern_") and
                        "strength" not in col and
                        "direction" not in col and
                        "target" not in col and
                        "stop" not in col]
        
        # Skip if no patterns found or strength column exists
        if not pattern_cols or "pattern_strength" in data.columns:
            # Initialize strength column if it doesn't exist
            if "pattern_strength" not in data.columns:
                data["pattern_strength"] = 0
            return
        
        # Initialize strength column
        data["pattern_strength"] = 0
        
        # Calculate strength for each pattern type found
        for col in pattern_cols:
            # Find contiguous pattern regions
            pattern_regions = self._find_contiguous_regions(data[col])
            
            for start_idx, end_idx in pattern_regions:
                if end_idx < start_idx:
                    continue  # Skip invalid regions
                
                # Calculate pattern strength based on various factors
                length = end_idx - start_idx + 1
                
                # Ensure indices are valid before slicing
                if start_idx < 0 or end_idx >= len(data):
                    continue
                
                pattern_slice = data.iloc[start_idx:end_idx+1]
                if pattern_slice.empty:
                    continue
                
                price_range = pattern_slice['high'].max() - pattern_slice['low'].min()
                avg_price = pattern_slice['close'].mean()
                
                # Calculate volume increase (handle potential missing volume or start index)
                volume_increase = 1.0
                if 'volume' in data.columns:
                    prev_volume_start = max(0, start_idx - 10)
                    if start_idx > prev_volume_start:  # Ensure there's a previous period
                        prev_volume_slice = data['volume'].iloc[prev_volume_start:start_idx]
                        current_volume_slice = pattern_slice['volume']
                        if not prev_volume_slice.empty and prev_volume_slice.mean() != 0:
                            volume_increase = current_volume_slice.mean() / prev_volume_slice.mean()
                        elif current_volume_slice.mean() > 0:  # Handle case where previous volume is zero
                            volume_increase = 2.0  # Assign a default high increase factor
                
                # Normalize factors (avoid division by zero)
                normalized_length = min(1.0, length / self.max_pattern_size) if self.max_pattern_size > 0 else 0
                normalized_range = min(1.0, price_range / (avg_price * 0.1)) if avg_price > 0 else 0
                normalized_volume = min(1.0, max(0, volume_increase - 1))  # Strength from volume *increase*
                
                # Calculate strength (0-100) - weighted average
                # Weights can be adjusted based on importance
                length_weight = 0.4
                range_weight = 0.4
                volume_weight = 0.2
                
                pattern_strength = int(
                    (normalized_length * length_weight +
                     normalized_range * range_weight +
                     normalized_volume * volume_weight) * 100
                )
                
                # Update strength in the DataFrame, taking the max if multiple patterns overlap
                current_strength = data["pattern_strength"].iloc[start_idx:end_idx+1]
                data.iloc[start_idx:end_idx+1, data.columns.get_loc("pattern_strength")] = np.maximum(current_strength, pattern_strength)
    
    def _find_contiguous_regions(self, series: pd.Series) -> List[Tuple[int, int]]:
        """
        Find contiguous regions where series values are 1.
        
        Args:
            series: Series with pattern markers (0 or 1)
            
        Returns:
            List of (start_idx, end_idx) tuples for each contiguous region
        """
        regions = []
        in_region = False
        start_idx = 0
        
        for i in range(len(series)):
            if series.iloc[i] == 1 and not in_region:
                # Start of a new region
                in_region = True
                start_idx = i
            elif series.iloc[i] != 1 and in_region:
                # End of a region
                in_region = False
                regions.append((start_idx, i - 1))
        
        # Handle case where pattern extends to the end
        if in_region:
            regions.append((start_idx, len(series) - 1))
        
        return regions
    
    def _find_peaks_troughs(self, data: pd.DataFrame, window: int = 5, find_peaks: bool = True) -> List[int]:
        """
        Find peaks or troughs in price data.
        
        Args:
            data: DataFrame with price data
            window: Window size for peak/trough detection
            find_peaks: If True, find peaks (high points), otherwise find troughs (low points)
            
        Returns:
            List of indices of peaks or troughs
        """
        # Use high for peaks and low for troughs
        price_series = data['high'] if find_peaks else data['low']
        
        # Find peaks or troughs
        result_indices = []
        
        # We need at least 2*window + 1 data points
        if len(data) < (2 * window + 1):
            return result_indices
        
        for i in range(window, len(data) - window):
            is_extremum = True
            for j in range(1, window + 1):
                if find_peaks:
                    # Check if current point is higher than neighbors
                    if not (price_series.iloc[i] > price_series.iloc[i-j] and price_series.iloc[i] > price_series.iloc[i+j]):
                        is_extremum = False
                        break
                else:
                    # Check if current point is lower than neighbors
                    if not (price_series.iloc[i] < price_series.iloc[i-j] and price_series.iloc[i] < price_series.iloc[i+j]):
                        is_extremum = False
                        break
            
            if is_extremum:
                result_indices.append(i)
        
        return result_indices
    
    def _calculate_pattern_direction(self, data: pd.DataFrame, pattern_col: str, window: int = 10) -> pd.Series:
        """
        Calculate the direction of a pattern.
        
        Args:
            data: DataFrame with pattern data
            pattern_col: Column name of the pattern
            window: Window size for trend determination
            
        Returns:
            Series with pattern direction values
        """
        direction = pd.Series(index=data.index, dtype='object')
        
        # Find contiguous pattern regions
        pattern_regions = self._find_contiguous_regions(data[pattern_col])
        
        for start_idx, end_idx in pattern_regions:
            if end_idx < start_idx or start_idx < 0 or end_idx >= len(data):
                continue
            
            # Get the pattern slice
            pattern_slice = data.iloc[start_idx:end_idx+1]
            if pattern_slice.empty:
                continue
            
            # Calculate the trend before the pattern
            pre_start = max(0, start_idx - window)
            pre_pattern = data.iloc[pre_start:start_idx]
            
            if not pre_pattern.empty:
                pre_trend = pre_pattern['close'].iloc[-1] - pre_pattern['close'].iloc[0]
                
                # Determine direction based on pre-trend and pattern structure
                if pre_trend > 0:
                    # Previous trend was up
                    if pattern_slice['close'].iloc[-1] > pattern_slice['close'].iloc[0]:
                        # Pattern continues up trend
                        direction.iloc[start_idx:end_idx+1] = PatternDirection.BULLISH.value
                    else:
                        # Pattern reverses up trend
                        direction.iloc[start_idx:end_idx+1] = PatternDirection.BEARISH.value
                else:
                    # Previous trend was down
                    if pattern_slice['close'].iloc[-1] < pattern_slice['close'].iloc[0]:
                        # Pattern continues down trend
                        direction.iloc[start_idx:end_idx+1] = PatternDirection.BEARISH.value
                    else:
                        # Pattern reverses down trend
                        direction.iloc[start_idx:end_idx+1] = PatternDirection.BULLISH.value
            else:
                # Not enough data for pre-trend, use pattern direction
                if pattern_slice['close'].iloc[-1] > pattern_slice['close'].iloc[0]:
                    direction.iloc[start_idx:end_idx+1] = PatternDirection.BULLISH.value
                elif pattern_slice['close'].iloc[-1] < pattern_slice['close'].iloc[0]:
                    direction.iloc[start_idx:end_idx+1] = PatternDirection.BEARISH.value
                else:
                    direction.iloc[start_idx:end_idx+1] = PatternDirection.NEUTRAL.value
        
        # Fill NaN values with UNKNOWN
        direction = direction.fillna(PatternDirection.UNKNOWN.value)
        
        return direction
    
    def _calculate_pattern_targets(self, data: pd.DataFrame, pattern_col: str, direction_col: str) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate price targets and stop levels for a pattern.
        
        Args:
            data: DataFrame with pattern data
            pattern_col: Column name of the pattern
            direction_col: Column name of the pattern direction
            
        Returns:
            Tuple of (target_series, stop_series)
        """
        target = pd.Series(index=data.index, dtype='float')
        stop = pd.Series(index=data.index, dtype='float')
        
        # Find contiguous pattern regions
        pattern_regions = self._find_contiguous_regions(data[pattern_col])
        
        for start_idx, end_idx in pattern_regions:
            if end_idx < start_idx or start_idx < 0 or end_idx >= len(data):
                continue
            
            # Get the pattern slice
            pattern_slice = data.iloc[start_idx:end_idx+1]
            if pattern_slice.empty:
                continue
            
            # Get pattern direction
            direction = data[direction_col].iloc[end_idx]
            
            # Calculate pattern height
            pattern_high = pattern_slice['high'].max()
            pattern_low = pattern_slice['low'].min()
            pattern_height = pattern_high - pattern_low
            
            # Calculate targets and stops based on direction
            if direction == PatternDirection.BULLISH.value:
                # Bullish pattern
                entry_price = pattern_slice['close'].iloc[-1]
                target.iloc[start_idx:end_idx+1] = entry_price + pattern_height
                stop.iloc[start_idx:end_idx+1] = pattern_low - (pattern_height * 0.1)  # 10% below pattern low
            elif direction == PatternDirection.BEARISH.value:
                # Bearish pattern
                entry_price = pattern_slice['close'].iloc[-1]
                target.iloc[start_idx:end_idx+1] = entry_price - pattern_height
                stop.iloc[start_idx:end_idx+1] = pattern_high + (pattern_height * 0.1)  # 10% above pattern high
            else:
                # Neutral or unknown direction
                target.iloc[start_idx:end_idx+1] = np.nan
                stop.iloc[start_idx:end_idx+1] = np.nan
        
        return target, stop