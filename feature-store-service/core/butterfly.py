"""
Butterfly Pattern Module.

This module provides implementation of the Butterfly harmonic pattern recognition.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from core.base_1 import BasePatternDetector, PatternType


class ButterflyPattern(BasePatternDetector):
    """
    Butterfly Pattern Detector.
    
    The Butterfly pattern is a harmonic pattern that uses Fibonacci retracement levels
    to identify potential reversal zones. It consists of five points (X, A, B, C, D)
    with specific Fibonacci relationships between them.
    
    Bullish Butterfly:
    - XA: Initial leg
    - AB: 0.786 retracement of XA
    - BC: 0.382-0.886 retracement of AB
    - CD: 1.618-2.618 extension of BC
    - D: 1.27 extension of XA
    
    Bearish Butterfly:
    - XA: Initial leg
    - AB: 0.786 retracement of XA
    - BC: 0.382-0.886 retracement of AB
    - CD: 1.618-2.618 extension of BC
    - D: 1.27 extension of XA
    """
    
    def __init__(
        self, 
        lookback_period: int = 100,
        tolerance: float = 0.05,
        **kwargs
    ):
        """
        Initialize Butterfly Pattern Detector.
        
        Args:
            lookback_period: Number of bars to look back for pattern recognition
            tolerance: Tolerance for Fibonacci ratio matches (0.01-0.10)
            **kwargs: Additional parameters
        """
        self.name = "butterfly_pattern"
        self.lookback_period = lookback_period
        self.tolerance = max(0.01, min(0.10, tolerance))
        
        # Fibonacci ratios for Butterfly pattern
        self.ab_ratio = 0.786  # AB should be 0.786 retracement of XA
        self.bc_ratio_min = 0.382  # BC should be 0.382-0.886 retracement of AB
        self.bc_ratio_max = 0.886
        self.cd_ratio_min = 1.618  # CD should be 1.618-2.618 extension of BC
        self.cd_ratio_max = 2.618
        self.xd_ratio = 1.27  # D should be 1.27 extension of XA
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Butterfly pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Butterfly pattern values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Initialize pattern columns with zeros
        result['pattern_butterfly_bullish'] = 0
        result['pattern_butterfly_bearish'] = 0
        
        # Find potential swing points (highs and lows)
        swing_highs, swing_lows = self._find_swing_points(data, window=5)
        
        # Look for bullish Butterfly patterns (forms at lows)
        bullish_patterns = self._find_bullish_butterfly(data, swing_highs, swing_lows)
        
        # Look for bearish Butterfly patterns (forms at highs)
        bearish_patterns = self._find_bearish_butterfly(data, swing_highs, swing_lows)
        
        # Mark pattern locations in the result DataFrame
        for pattern in bullish_patterns:
            d_point = pattern['points']['D']['idx']
            if 0 <= d_point < len(result):
                result.loc[result.index[d_point], 'pattern_butterfly_bullish'] = 1
        
        for pattern in bearish_patterns:
            d_point = pattern['points']['D']['idx']
            if 0 <= d_point < len(result):
                result.loc[result.index[d_point], 'pattern_butterfly_bearish'] = 1
        
        return result
    
    def _find_swing_points(self, data: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """
        Find swing highs and lows in the data.
        
        Args:
            data: DataFrame with OHLCV data
            window: Window size for swing point detection
            
        Returns:
            Tuple of (swing_highs, swing_lows) where each is a list of indices
        """
        highs = data['high'].values
        lows = data['low'].values
        
        swing_highs = []
        swing_lows = []
        
        # Find swing highs
        for i in range(window, len(data) - window):
            if all(highs[i] > highs[i-j] for j in range(1, window+1)) and \
               all(highs[i] > highs[i+j] for j in range(1, window+1)):
                swing_highs.append(i)
        
        # Find swing lows
        for i in range(window, len(data) - window):
            if all(lows[i] < lows[i-j] for j in range(1, window+1)) and \
               all(lows[i] < lows[i+j] for j in range(1, window+1)):
                swing_lows.append(i)
        
        return swing_highs, swing_lows
    
    def _find_bullish_butterfly(self, data: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> List[Dict[str, Any]]:
        """
        Find bullish Butterfly patterns in the data.
        
        Args:
            data: DataFrame with OHLCV data
            swing_highs: List of swing high indices
            swing_lows: List of swing low indices
            
        Returns:
            List of bullish Butterfly patterns
        """
        patterns = []
        window = 5  # Window size for swing point detection
        
        # Need at least 4 swing points to form a Butterfly pattern
        if len(swing_lows) < 2 or len(swing_highs) < 2:
            return patterns
        
        # Look for potential X points (must be a swing low)
        for x_idx in swing_lows:
            # Skip if too close to the beginning or end
            if x_idx < window or x_idx > len(data) - window:
                continue
            
            # Find potential A points (must be a swing high after X)
            for a_idx in [h for h in swing_highs if h > x_idx]:
                # Skip if too far away
                if a_idx - x_idx > self.lookback_period:
                    continue
                
                x_price = data['low'].iloc[x_idx]
                a_price = data['high'].iloc[a_idx]
                xa_diff = a_price - x_price
                
                # Find potential B points (must be a swing low after A)
                for b_idx in [l for l in swing_lows if l > a_idx]:
                    # Skip if too far away
                    if b_idx - a_idx > self.lookback_period:
                        continue
                    
                    b_price = data['low'].iloc[b_idx]
                    ab_diff = a_price - b_price
                    
                    # Check if AB is approximately 0.786 retracement of XA
                    ab_ratio = ab_diff / xa_diff
                    if not self._is_within_tolerance(ab_ratio, self.ab_ratio):
                        continue
                    
                    # Find potential C points (must be a swing high after B)
                    for c_idx in [h for h in swing_highs if h > b_idx]:
                        # Skip if too far away
                        if c_idx - b_idx > self.lookback_period:
                            continue
                        
                        c_price = data['high'].iloc[c_idx]
                        bc_diff = c_price - b_price
                        
                        # Check if BC is within 0.382-0.886 retracement of AB
                        bc_ratio = bc_diff / ab_diff
                        if not (self._is_within_tolerance(bc_ratio, self.bc_ratio_min, self.bc_ratio_max)):
                            continue
                        
                        # Find potential D points (must be a swing low after C)
                        for d_idx in [l for l in swing_lows if l > c_idx]:
                            # Skip if too far away
                            if d_idx - c_idx > self.lookback_period:
                                continue
                            
                            d_price = data['low'].iloc[d_idx]
                            cd_diff = c_price - d_price
                            xd_diff = x_price - d_price
                            
                            # Check if CD is within 1.618-2.618 extension of BC
                            cd_ratio = cd_diff / bc_diff
                            if not (self._is_within_tolerance(cd_ratio, self.cd_ratio_min, self.cd_ratio_max)):
                                continue
                            
                            # Check if D is approximately 1.27 extension of XA
                            xd_ratio = abs(xd_diff / xa_diff)
                            if not self._is_within_tolerance(xd_ratio, self.xd_ratio):
                                continue
                            
                            # All conditions met, we have a bullish Butterfly pattern
                            pattern = {
                                'start_idx': x_idx,
                                'end_idx': d_idx,
                                'pattern_type': 'butterfly',
                                'direction': 'bullish',
                                'strength': self._calculate_pattern_strength(
                                    ab_ratio, self.ab_ratio,
                                    bc_ratio, (self.bc_ratio_min + self.bc_ratio_max) / 2,
                                    cd_ratio, (self.cd_ratio_min + self.cd_ratio_max) / 2,
                                    xd_ratio, self.xd_ratio
                                ),
                                'points': {
                                    'X': {'idx': x_idx, 'price': x_price},
                                    'A': {'idx': a_idx, 'price': a_price},
                                    'B': {'idx': b_idx, 'price': b_price},
                                    'C': {'idx': c_idx, 'price': c_price},
                                    'D': {'idx': d_idx, 'price': d_price}
                                }
                            }
                            
                            patterns.append(pattern)
        
        return patterns
    
    def _find_bearish_butterfly(self, data: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> List[Dict[str, Any]]:
        """
        Find bearish Butterfly patterns in the data.
        
        Args:
            data: DataFrame with OHLCV data
            swing_highs: List of swing high indices
            swing_lows: List of swing low indices
            
        Returns:
            List of bearish Butterfly patterns
        """
        patterns = []
        window = 5  # Window size for swing point detection
        
        # Need at least 4 swing points to form a Butterfly pattern
        if len(swing_lows) < 2 or len(swing_highs) < 2:
            return patterns
        
        # Look for potential X points (must be a swing high)
        for x_idx in swing_highs:
            # Skip if too close to the beginning or end
            if x_idx < window or x_idx > len(data) - window:
                continue
            
            # Find potential A points (must be a swing low after X)
            for a_idx in [l for l in swing_lows if l > x_idx]:
                # Skip if too far away
                if a_idx - x_idx > self.lookback_period:
                    continue
                
                x_price = data['high'].iloc[x_idx]
                a_price = data['low'].iloc[a_idx]
                xa_diff = x_price - a_price
                
                # Find potential B points (must be a swing high after A)
                for b_idx in [h for h in swing_highs if h > a_idx]:
                    # Skip if too far away
                    if b_idx - a_idx > self.lookback_period:
                        continue
                    
                    b_price = data['high'].iloc[b_idx]
                    ab_diff = b_price - a_price
                    
                    # Check if AB is approximately 0.786 retracement of XA
                    ab_ratio = ab_diff / xa_diff
                    if not self._is_within_tolerance(ab_ratio, self.ab_ratio):
                        continue
                    
                    # Find potential C points (must be a swing low after B)
                    for c_idx in [l for l in swing_lows if l > b_idx]:
                        # Skip if too far away
                        if c_idx - b_idx > self.lookback_period:
                            continue
                        
                        c_price = data['low'].iloc[c_idx]
                        bc_diff = b_price - c_price
                        
                        # Check if BC is within 0.382-0.886 retracement of AB
                        bc_ratio = bc_diff / ab_diff
                        if not (self._is_within_tolerance(bc_ratio, self.bc_ratio_min, self.bc_ratio_max)):
                            continue
                        
                        # Find potential D points (must be a swing high after C)
                        for d_idx in [h for h in swing_highs if h > c_idx]:
                            # Skip if too far away
                            if d_idx - c_idx > self.lookback_period:
                                continue
                            
                            d_price = data['high'].iloc[d_idx]
                            cd_diff = d_price - c_price
                            xd_diff = d_price - x_price
                            
                            # Check if CD is within 1.618-2.618 extension of BC
                            cd_ratio = cd_diff / bc_diff
                            if not (self._is_within_tolerance(cd_ratio, self.cd_ratio_min, self.cd_ratio_max)):
                                continue
                            
                            # Check if D is approximately 1.27 extension of XA
                            xd_ratio = abs(xd_diff / xa_diff)
                            if not self._is_within_tolerance(xd_ratio, self.xd_ratio):
                                continue
                            
                            # All conditions met, we have a bearish Butterfly pattern
                            pattern = {
                                'start_idx': x_idx,
                                'end_idx': d_idx,
                                'pattern_type': 'butterfly',
                                'direction': 'bearish',
                                'strength': self._calculate_pattern_strength(
                                    ab_ratio, self.ab_ratio,
                                    bc_ratio, (self.bc_ratio_min + self.bc_ratio_max) / 2,
                                    cd_ratio, (self.cd_ratio_min + self.cd_ratio_max) / 2,
                                    xd_ratio, self.xd_ratio
                                ),
                                'points': {
                                    'X': {'idx': x_idx, 'price': x_price},
                                    'A': {'idx': a_idx, 'price': a_price},
                                    'B': {'idx': b_idx, 'price': b_price},
                                    'C': {'idx': c_idx, 'price': c_price},
                                    'D': {'idx': d_idx, 'price': d_price}
                                }
                            }
                            
                            patterns.append(pattern)
        
        return patterns
    
    def _is_within_tolerance(self, value: float, target: float, target_max: Optional[float] = None) -> bool:
        """
        Check if a value is within tolerance of a target or range.
        
        Args:
            value: The value to check
            target: The target value or minimum of range
            target_max: The maximum of range (if None, only check against target)
            
        Returns:
            True if value is within tolerance, False otherwise
        """
        if target_max is None:
            return abs(value - target) <= self.tolerance
        else:
            return target - self.tolerance <= value <= target_max + self.tolerance
    
    def _calculate_pattern_strength(self, ab_ratio: float, ab_target: float,
                                   bc_ratio: float, bc_target: float,
                                   cd_ratio: float, cd_target: float,
                                   xd_ratio: float, xd_target: float) -> float:
        """
        Calculate the strength of a pattern based on how closely it matches the ideal ratios.
        
        Args:
            ab_ratio: Actual AB ratio
            ab_target: Target AB ratio
            bc_ratio: Actual BC ratio
            bc_target: Target BC ratio
            cd_ratio: Actual CD ratio
            cd_target: Target CD ratio
            xd_ratio: Actual XD ratio
            xd_target: Target XD ratio
            
        Returns:
            Pattern strength as a value between 0 and 1
        """
        # Calculate how closely each ratio matches its target
        ab_match = 1.0 - min(abs(ab_ratio - ab_target) / ab_target, 1.0)
        bc_match = 1.0 - min(abs(bc_ratio - bc_target) / bc_target, 1.0)
        cd_match = 1.0 - min(abs(cd_ratio - cd_target) / cd_target, 1.0)
        xd_match = 1.0 - min(abs(xd_ratio - xd_target) / xd_target, 1.0)
        
        # Calculate overall strength as weighted average
        # XD ratio is most important for Butterfly pattern
        strength = (ab_match * 0.2 + bc_match * 0.2 + cd_match * 0.2 + xd_match * 0.4)
        
        return strength
