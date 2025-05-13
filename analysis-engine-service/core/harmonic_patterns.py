"""
Harmonic Pattern Indicators Module

This module provides implementations of various harmonic pattern indicators:
- HarmonicPatternFinder: Base class for finding harmonic patterns
- ButterflyPattern: Implementation of the Butterfly harmonic pattern
- GartleyPattern: Implementation of the Gartley harmonic pattern
- CrabPattern: Implementation of the Crab harmonic pattern

These indicators identify potential reversal zones based on specific Fibonacci ratio relationships
between price swings that form specific geometric patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase, PatternRecognitionBase
from analysis_engine.utils.validation import validate_dataframe

logger = logging.getLogger(__name__)


@dataclass
class HarmonicMeasurement:
    """Stores measurements for a harmonic pattern."""
    pattern_type: str
    points: Dict[str, Tuple[int, float]]  # Map of point name to (index, price)
    ratios: Dict[str, float]  # Map of ratio name to value
    confidence: float
    target_zone: Tuple[float, float]  # (support, resistance)
    completion_index: int


class HarmonicPatternFinder(PatternRecognitionBase):
    """
    Base class for finding harmonic patterns in price data.
    
    Harmonic patterns are geometric price patterns that use Fibonacci retracement
    and extension levels to identify potential reversal zones. They're characterized
    by specific leg measurements that conform to Fibonacci ratios.
    """
    
    def __init__(self, tolerance: float = 0.1, min_pattern_bars: int = 10, 
                 max_pattern_bars: int = 100, min_leg_size: float = 0.01,
                 fibonacci_tolerance: float = 0.05):
        """
        Initialize the HarmonicPatternFinder.
        
        Args:
            tolerance: Tolerance for the Fibonacci ratio matching (default: 0.1)
            min_pattern_bars: Minimum number of bars for a valid pattern (default: 10)
            max_pattern_bars: Maximum number of bars for a valid pattern (default: 100)
            min_leg_size: Minimum size of a leg as a percentage of price (default: 0.01)
            fibonacci_tolerance: Tolerance for Fibonacci ratio validation (default: 0.05)
        """
        super().__init__(name="HarmonicPatternFinder")
        self.tolerance = tolerance
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.min_leg_size = min_leg_size
        self.fibonacci_tolerance = fibonacci_tolerance
        
        # Common Fibonacci ratios used in harmonic patterns
        self.fib_ratios = {
            "0.382": 0.382,
            "0.5": 0.5,
            "0.618": 0.618,
            "0.786": 0.786,
            "0.886": 0.886,
            "1.13": 1.13,
            "1.27": 1.27,
            "1.414": 1.414,
            "1.618": 1.618,
            "2.0": 2.0,
            "2.24": 2.24,
            "2.618": 2.618,
            "3.14": 3.14,
            "3.618": 3.618
        }
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Find harmonic patterns in the price data.
        
        Args:
            data: DataFrame with price data (must contain 'high', 'low', 'close' columns)
            
        Returns:
            DataFrame with harmonic pattern information
        """
        validate_dataframe(data, required_columns=['high', 'low', 'close'])
        
        # Create a copy of the data
        result = data.copy()
        
        # Find potential swing points
        swing_points = self._find_swing_points(data)
        
        # Find patterns using swing points
        patterns = self._find_patterns(data, swing_points)
        
        # Add pattern information to the result DataFrame
        result['harmonic_pattern'] = None
        result['harmonic_pattern_strength'] = 0.0
        result['potential_reversal_zone_low'] = np.nan
        result['potential_reversal_zone_high'] = np.nan
        
        for pattern in patterns:
            completion_idx = pattern.completion_index
            if 0 <= completion_idx < len(result):
                result.loc[completion_idx, 'harmonic_pattern'] = pattern.pattern_type
                result.loc[completion_idx, 'harmonic_pattern_strength'] = pattern.confidence
                result.loc[completion_idx, 'potential_reversal_zone_low'] = pattern.target_zone[0]
                result.loc[completion_idx, 'potential_reversal_zone_high'] = pattern.target_zone[1]
        
        return result
    
    def _find_swing_points(self, data: pd.DataFrame) -> List[Tuple[int, str, float]]:
        """
        Find swing highs and swing lows in the price data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            List of (index, type, price) tuples where type is 'high' or 'low'
        """
        # Use a simple swing point detection algorithm
        high = data['high'].values
        low = data['low'].values
        
        swing_points = []
        
        # Look for swing highs and lows with a window of 2 bars on each side
        window = 2
        
        for i in range(window, len(data) - window):
            # Check if this is a swing high
            if all(high[i] > high[i - j] for j in range(1, window + 1)) and \
               all(high[i] > high[i + j] for j in range(1, window + 1)):
                swing_points.append((i, 'high', high[i]))
            
            # Check if this is a swing low
            if all(low[i] < low[i - j] for j in range(1, window + 1)) and \
               all(low[i] < low[i + j] for j in range(1, window + 1)):
                swing_points.append((i, 'low', low[i]))
        
        return swing_points
    
    def _find_patterns(self, data: pd.DataFrame, swing_points: List[Tuple[int, str, float]]) -> List[HarmonicMeasurement]:
        """
        Base method for finding patterns. Subclasses should override this.
        
        Args:
            data: DataFrame with price data
            swing_points: List of swing points
            
        Returns:
            List of HarmonicMeasurement objects representing found patterns
        """
        # This is a base implementation that subclasses will override
        return []
    
    def _validate_fibonacci_ratio(self, actual: float, target: float) -> bool:
        """
        Validate if a ratio is within tolerance of a target Fibonacci ratio.
        
        Args:
            actual: The actual ratio
            target: The target Fibonacci ratio
            
        Returns:
            True if the ratio is within tolerance of the target, False otherwise
        """
        return abs(actual - target) <= self.fibonacci_tolerance
    
    def _calculate_ratio(self, leg1: float, leg2: float) -> float:
        """
        Calculate the ratio between two legs.
        
        Args:
            leg1: First leg size
            leg2: Second leg size
            
        Returns:
            The ratio leg2 / leg1
        """
        if abs(leg1) < 1e-10:  # Avoid division by zero
            return 0.0
        return abs(leg2 / leg1)
    
    def _calculate_target_zone(self, pattern_type: str, points: Dict[str, Tuple[int, float]]) -> Tuple[float, float]:
        """
        Calculate the potential reversal zone based on the pattern type and points.
        
        Args:
            pattern_type: The type of the pattern
            points: Dictionary of points in the pattern
            
        Returns:
            A tuple of (support level, resistance level)
        """
        # Default implementation - subclasses will override with specific PRZ calculations
        if 'D' in points:
            d_price = points['D'][1]
            # Default PRZ is +/- 1% around the D point
            return (d_price * 0.99, d_price * 1.01)
        return (0.0, 0.0)
    
    def _calculate_pattern_confidence(self, ratios: Dict[str, float], target_ratios: Dict[str, float]) -> float:
        """
        Calculate the confidence score for a pattern based on how closely it matches ideal ratios.
        
        Args:
            ratios: The actual ratios calculated from the pattern
            target_ratios: The ideal target ratios for the pattern type
            
        Returns:
            A confidence score between 0.0 and 1.0
        """
        if not ratios or not target_ratios:
            return 0.0
        
        total_deviation = 0.0
        for ratio_name, actual in ratios.items():
            if ratio_name in target_ratios:
                target = target_ratios[ratio_name]
                deviation = abs(actual - target) / target
                total_deviation += deviation
        
        avg_deviation = total_deviation / len(ratios)
        confidence = max(0.0, 1.0 - avg_deviation * 5)  # Scale to penalize deviations
        return min(1.0, confidence)  # Cap at 1.0
    
    def _is_pattern_valid(self, points: Dict[str, Tuple[int, float]], min_move_percent: float = 0.01) -> bool:
        """
        Check if a pattern is valid based on minimum move sizes and other criteria.
        
        Args:
            points: Dictionary of points in the pattern
            min_move_percent: Minimum percentage move required between consecutive points
            
        Returns:
            True if the pattern is valid, False otherwise
        """
        # Ensure we have all required points
        required_points = ['X', 'A', 'B', 'C', 'D']
        if not all(point in points for point in required_points):
            return False
        
        # Get prices
        x_price = points['X'][1]
        a_price = points['A'][1]
        b_price = points['B'][1]
        c_price = points['C'][1]
        d_price = points['D'][1]
        
        # Calculate percentage moves
        xa_move = abs(a_price - x_price) / x_price
        ab_move = abs(b_price - a_price) / a_price
        bc_move = abs(c_price - b_price) / b_price
        cd_move = abs(d_price - c_price) / c_price
        
        # Check minimum move sizes
        if any(move < min_move_percent for move in [xa_move, ab_move, bc_move, cd_move]):
            return False
        
        # Check for proper alternating direction
        xa_direction = 1 if a_price > x_price else -1
        ab_direction = 1 if b_price > a_price else -1
        bc_direction = 1 if c_price > b_price else -1
        cd_direction = 1 if d_price > c_price else -1
        
        # Check for alternating directions in the pattern
        if xa_direction == ab_direction or ab_direction == bc_direction or bc_direction == cd_direction:
            return False
        
        # All checks passed
        return True


class ButterflyPattern(HarmonicPatternFinder):
    """
    Implementation of the Butterfly harmonic pattern.
    
    The Butterfly pattern has the following characteristics:
    - XA: Initial leg
    - AB: Retracement of XA by 0.786
    - BC: Extension of AB by 0.382 to 0.886
    - CD: Extension of BC beyond point A
    - D: Completion point at 1.27 or 1.618 extension of XA
    """
    
    def __init__(self, tolerance: float = 0.1, min_pattern_bars: int = 10, 
                 max_pattern_bars: int = 100, min_leg_size: float = 0.01):
        """
        Initialize the ButterflyPattern detector.
        
        Args:
            tolerance: Tolerance for the Fibonacci ratio matching (default: 0.1)
            min_pattern_bars: Minimum number of bars for a valid pattern (default: 10)
            max_pattern_bars: Maximum number of bars for a valid pattern (default: 100)
            min_leg_size: Minimum size of a leg as a percentage of price (default: 0.01)
        """
        super().__init__(tolerance=tolerance, min_pattern_bars=min_pattern_bars,
                         max_pattern_bars=max_pattern_bars, min_leg_size=min_leg_size)
        self.name = "ButterflyPattern"
        
        # Define the ideal ratios for a Butterfly pattern
        self.ideal_ratios = {
            "XB": 0.786,  # B is a 0.786 retracement of XA
            "AC": 0.382,  # C is a 0.382 retracement of AB
            "XD": 1.27,   # D is a 1.27 or 1.618 extension of XA
            "BD": 1.618   # D is a 1.618 extension of BC
        }
    
    def _find_patterns(self, data: pd.DataFrame, swing_points: List[Tuple[int, str, float]]) -> List[HarmonicMeasurement]:
        """
        Find Butterfly patterns in the swing points.
        
        Args:
            data: DataFrame with price data
            swing_points: List of swing points
            
        Returns:
            List of HarmonicMeasurement objects representing found Butterfly patterns
        """
        patterns = []
        
        # Need at least 4 swing points for a valid pattern
        if len(swing_points) < 4:
            return patterns
        
        # Look for potential X points
        for i in range(len(swing_points) - 4):
            x_idx, x_type, x_price = swing_points[i]
            
            # Look for potential A points
            for j in range(i+1, min(i+self.max_pattern_bars, len(swing_points) - 3)):
                a_idx, a_type, a_price = swing_points[j]
                
                # A should be opposite direction of X
                if a_type == x_type:
                    continue
                
                # Calculate XA leg
                xa_leg = a_price - x_price
                
                # Look for potential B points
                for k in range(j+1, min(j+self.max_pattern_bars, len(swing_points) - 2)):
                    b_idx, b_type, b_price = swing_points[k]
                    
                    # B should be opposite direction of A
                    if b_type == a_type:
                        continue
                    
                    # Calculate AB leg
                    ab_leg = b_price - a_price
                    
                    # Check if B is a 0.786 retracement of XA
                    xb_ratio = abs((b_price - x_price) / xa_leg)
                    if not self._validate_fibonacci_ratio(xb_ratio, self.ideal_ratios["XB"]):
                        continue
                    
                    # Look for potential C points
                    for l in range(k+1, min(k+self.max_pattern_bars, len(swing_points) - 1)):
                        c_idx, c_type, c_price = swing_points[l]
                        
                        # C should be opposite direction of B
                        if c_type == b_type:
                            continue
                        
                        # Calculate BC leg
                        bc_leg = c_price - b_price
                        
                        # Check if C is around a 0.382 to 0.886 retracement of AB
                        ac_ratio = abs((c_price - a_price) / ab_leg)
                        if not (0.382 - self.tolerance <= ac_ratio <= 0.886 + self.tolerance):
                            continue
                        
                        # Look for potential D points
                        for m in range(l+1, min(l+self.max_pattern_bars, len(swing_points))):
                            d_idx, d_type, d_price = swing_points[m]
                            
                            # D should be opposite direction of C
                            if d_type == c_type:
                                continue
                            
                            # Calculate CD leg
                            cd_leg = d_price - c_price
                            
                            # Check if D is a 1.27 or 1.618 extension of XA
                            xd_ratio = abs((d_price - x_price) / xa_leg)
                            if not (self._validate_fibonacci_ratio(xd_ratio, 1.27) or 
                                    self._validate_fibonacci_ratio(xd_ratio, 1.618)):
                                continue
                            
                            # Check if D is around a 1.618 extension of BC
                            bd_ratio = abs((d_price - b_price) / bc_leg)
                            if not self._validate_fibonacci_ratio(bd_ratio, 1.618):
                                continue
                            
                            # We found a potential Butterfly pattern
                            points = {
                                'X': (x_idx, x_price),
                                'A': (a_idx, a_price),
                                'B': (b_idx, b_price),
                                'C': (c_idx, c_price),
                                'D': (d_idx, d_price)
                            }
                            
                            if self._is_pattern_valid(points):
                                ratios = {
                                    "XB": xb_ratio,
                                    "AC": ac_ratio,
                                    "XD": xd_ratio,
                                    "BD": bd_ratio
                                }
                                
                                target_zone = self._calculate_target_zone("Butterfly", points)
                                confidence = self._calculate_pattern_confidence(ratios, self.ideal_ratios)
                                
                                patterns.append(HarmonicMeasurement(
                                    pattern_type="Butterfly",
                                    points=points,
                                    ratios=ratios,
                                    confidence=confidence,
                                    target_zone=target_zone,
                                    completion_index=d_idx
                                ))
        
        return patterns
    
    def _calculate_target_zone(self, pattern_type: str, points: Dict[str, Tuple[int, float]]) -> Tuple[float, float]:
        """
        Calculate the potential reversal zone for a Butterfly pattern.
        
        Args:
            pattern_type: The type of the pattern
            points: Dictionary of points in the pattern
            
        Returns:
            A tuple of (support level, resistance level)
        """
        x_price = points['X'][1]
        a_price = points['A'][1]
        d_price = points['D'][1]
        
        # For Butterfly, PRZ is typically around the 1.27 to 1.618 extension of XA
        xa_move = abs(a_price - x_price)
        
        # Calculate Fibonacci projection levels
        fib_127 = x_price + (1.27 * xa_move * (1 if a_price > x_price else -1))
        fib_1618 = x_price + (1.618 * xa_move * (1 if a_price > x_price else -1))
        
        # PRZ is between these two levels
        return (min(fib_127, fib_1618), max(fib_127, fib_1618))


class GartleyPattern(HarmonicPatternFinder):
    """
    Implementation of the Gartley harmonic pattern.
    
    The Gartley pattern has the following characteristics:
    - XA: Initial leg
    - AB: Retracement of XA by 0.618
    - BC: Retracement of AB by 0.382 to 0.886
    - CD: Retracement of BC by 1.27 to 1.618
    - D: Completion point at 0.786 retracement of XA
    """
    
    def __init__(self, tolerance: float = 0.1, min_pattern_bars: int = 10, 
                 max_pattern_bars: int = 100, min_leg_size: float = 0.01):
        """
        Initialize the GartleyPattern detector.
        
        Args:
            tolerance: Tolerance for the Fibonacci ratio matching (default: 0.1)
            min_pattern_bars: Minimum number of bars for a valid pattern (default: 10)
            max_pattern_bars: Maximum number of bars for a valid pattern (default: 100)
            min_leg_size: Minimum size of a leg as a percentage of price (default: 0.01)
        """
        super().__init__(tolerance=tolerance, min_pattern_bars=min_pattern_bars,
                         max_pattern_bars=max_pattern_bars, min_leg_size=min_leg_size)
        self.name = "GartleyPattern"
        
        # Define the ideal ratios for a Gartley pattern
        self.ideal_ratios = {
            "AB": 0.618,  # B is a 0.618 retracement of XA
            "BC": 0.382,  # C is a 0.382 to 0.886 retracement of AB
            "CD": 1.27,   # D is a 1.27 to 1.618 extension of BC
            "XD": 0.786   # D is a 0.786 retracement of XA
        }
    
    def _find_patterns(self, data: pd.DataFrame, swing_points: List[Tuple[int, str, float]]) -> List[HarmonicMeasurement]:
        """
        Find Gartley patterns in the swing points.
        
        Args:
            data: DataFrame with price data
            swing_points: List of swing points
            
        Returns:
            List of HarmonicMeasurement objects representing found Gartley patterns
        """
        patterns = []
        
        # Need at least 4 swing points for a valid pattern
        if len(swing_points) < 4:
            return patterns
        
        # Look for potential X points
        for i in range(len(swing_points) - 4):
            x_idx, x_type, x_price = swing_points[i]
            
            # Look for potential A points
            for j in range(i+1, min(i+self.max_pattern_bars, len(swing_points) - 3)):
                a_idx, a_type, a_price = swing_points[j]
                
                # A should be opposite direction of X
                if a_type == x_type:
                    continue
                
                # Calculate XA leg
                xa_leg = a_price - x_price
                
                # Look for potential B points
                for k in range(j+1, min(j+self.max_pattern_bars, len(swing_points) - 2)):
                    b_idx, b_type, b_price = swing_points[k]
                    
                    # B should be opposite direction of A
                    if b_type == a_type:
                        continue
                    
                    # Calculate AB leg
                    ab_leg = b_price - a_price
                    
                    # Check if B is a 0.618 retracement of XA
                    ab_ratio = abs(ab_leg / xa_leg)
                    if not self._validate_fibonacci_ratio(ab_ratio, self.ideal_ratios["AB"]):
                        continue
                    
                    # Look for potential C points
                    for l in range(k+1, min(k+self.max_pattern_bars, len(swing_points) - 1)):
                        c_idx, c_type, c_price = swing_points[l]
                        
                        # C should be opposite direction of B
                        if c_type == b_type:
                            continue
                        
                        # Calculate BC leg
                        bc_leg = c_price - b_price
                        
                        # Check if C is around a 0.382 to 0.886 retracement of AB
                        bc_ratio = abs(bc_leg / ab_leg)
                        if not (0.382 - self.tolerance <= bc_ratio <= 0.886 + self.tolerance):
                            continue
                        
                        # Look for potential D points
                        for m in range(l+1, min(l+self.max_pattern_bars, len(swing_points))):
                            d_idx, d_type, d_price = swing_points[m]
                            
                            # D should be opposite direction of C
                            if d_type == c_type:
                                continue
                            
                            # Calculate CD leg
                            cd_leg = d_price - c_price
                            
                            # Check if D is a 1.27 to 1.618 extension of BC
                            cd_ratio = abs(cd_leg / bc_leg)
                            if not (1.27 - self.tolerance <= cd_ratio <= 1.618 + self.tolerance):
                                continue
                            
                            # Check if D is a 0.786 retracement of XA
                            xd_ratio = abs((d_price - x_price) / xa_leg)
                            if not self._validate_fibonacci_ratio(xd_ratio, self.ideal_ratios["XD"]):
                                continue
                            
                            # We found a potential Gartley pattern
                            points = {
                                'X': (x_idx, x_price),
                                'A': (a_idx, a_price),
                                'B': (b_idx, b_price),
                                'C': (c_idx, c_price),
                                'D': (d_idx, d_price)
                            }
                            
                            if self._is_pattern_valid(points):
                                ratios = {
                                    "AB": ab_ratio,
                                    "BC": bc_ratio,
                                    "CD": cd_ratio,
                                    "XD": xd_ratio
                                }
                                
                                target_zone = self._calculate_target_zone("Gartley", points)
                                confidence = self._calculate_pattern_confidence(ratios, self.ideal_ratios)
                                
                                patterns.append(HarmonicMeasurement(
                                    pattern_type="Gartley",
                                    points=points,
                                    ratios=ratios,
                                    confidence=confidence,
                                    target_zone=target_zone,
                                    completion_index=d_idx
                                ))
        
        return patterns
    
    def _calculate_target_zone(self, pattern_type: str, points: Dict[str, Tuple[int, float]]) -> Tuple[float, float]:
        """
        Calculate the potential reversal zone for a Gartley pattern.
        
        Args:
            pattern_type: The type of the pattern
            points: Dictionary of points in the pattern
            
        Returns:
            A tuple of (support level, resistance level)
        """
        x_price = points['X'][1]
        a_price = points['A'][1]
        d_price = points['D'][1]
        
        # For Gartley, PRZ is typically around the 0.786 retracement of XA
        xa_move = abs(a_price - x_price)
        
        # Calculate Fibonacci retracement levels
        fib_618 = x_price + (0.618 * xa_move * (1 if a_price > x_price else -1))
        fib_786 = x_price + (0.786 * xa_move * (1 if a_price > x_price else -1))
        
        # PRZ is between these two levels
        return (min(fib_618, fib_786), max(fib_618, fib_786))


class CrabPattern(HarmonicPatternFinder):
    """
    Implementation of the Crab harmonic pattern.
    
    The Crab pattern has the following characteristics:
    - XA: Initial leg
    - AB: Retracement of XA by 0.382 to 0.618
    - BC: Extension of AB by 0.382 to 0.886
    - CD: Extension of BC by 2.24 to 3.618
    - D: Completion point at 1.618 extension of XA
    """
    
    def __init__(self, tolerance: float = 0.1, min_pattern_bars: int = 10, 
                 max_pattern_bars: int = 100, min_leg_size: float = 0.01):
        """
        Initialize the CrabPattern detector.
        
        Args:
            tolerance: Tolerance for the Fibonacci ratio matching (default: 0.1)
            min_pattern_bars: Minimum number of bars for a valid pattern (default: 10)
            max_pattern_bars: Maximum number of bars for a valid pattern (default: 100)
            min_leg_size: Minimum size of a leg as a percentage of price (default: 0.01)
        """
        super().__init__(tolerance=tolerance, min_pattern_bars=min_pattern_bars,
                         max_pattern_bars=max_pattern_bars, min_leg_size=min_leg_size)
        self.name = "CrabPattern"
        
        # Define the ideal ratios for a Crab pattern
        self.ideal_ratios = {
            "AB": 0.382,  # B is a 0.382 to 0.618 retracement of XA
            "BC": 0.382,  # C is a 0.382 to 0.886 retracement of AB
            "CD": 2.24,   # D is a 2.24 to 3.618 extension of BC
            "XD": 1.618   # D is a 1.618 extension of XA
        }
    
    def _find_patterns(self, data: pd.DataFrame, swing_points: List[Tuple[int, str, float]]) -> List[HarmonicMeasurement]:
        """
        Find Crab patterns in the swing points.
        
        Args:
            data: DataFrame with price data
            swing_points: List of swing points
            
        Returns:
            List of HarmonicMeasurement objects representing found Crab patterns
        """
        patterns = []
        
        # Need at least 4 swing points for a valid pattern
        if len(swing_points) < 4:
            return patterns
        
        # Look for potential X points
        for i in range(len(swing_points) - 4):
            x_idx, x_type, x_price = swing_points[i]
            
            # Look for potential A points
            for j in range(i+1, min(i+self.max_pattern_bars, len(swing_points) - 3)):
                a_idx, a_type, a_price = swing_points[j]
                
                # A should be opposite direction of X
                if a_type == x_type:
                    continue
                
                # Calculate XA leg
                xa_leg = a_price - x_price
                
                # Look for potential B points
                for k in range(j+1, min(j+self.max_pattern_bars, len(swing_points) - 2)):
                    b_idx, b_type, b_price = swing_points[k]
                    
                    # B should be opposite direction of A
                    if b_type == a_type:
                        continue
                    
                    # Calculate AB leg
                    ab_leg = b_price - a_price
                    
                    # Check if B is a 0.382 to 0.618 retracement of XA
                    ab_ratio = abs(ab_leg / xa_leg)
                    if not (0.382 - self.tolerance <= ab_ratio <= 0.618 + self.tolerance):
                        continue
                    
                    # Look for potential C points
                    for l in range(k+1, min(k+self.max_pattern_bars, len(swing_points) - 1)):
                        c_idx, c_type, c_price = swing_points[l]
                        
                        # C should be opposite direction of B
                        if c_type == b_type:
                            continue
                        
                        # Calculate BC leg
                        bc_leg = c_price - b_price
                        
                        # Check if C is around a 0.382 to 0.886 retracement of AB
                        bc_ratio = abs(bc_leg / ab_leg)
                        if not (0.382 - self.tolerance <= bc_ratio <= 0.886 + self.tolerance):
                            continue
                        
                        # Look for potential D points
                        for m in range(l+1, min(l+self.max_pattern_bars, len(swing_points))):
                            d_idx, d_type, d_price = swing_points[m]
                            
                            # D should be opposite direction of C
                            if d_type == c_type:
                                continue
                            
                            # Calculate CD leg
                            cd_leg = d_price - c_price
                            
                            # Check if D is a 2.24 to 3.618 extension of BC
                            cd_ratio = abs(cd_leg / bc_leg)
                            if not (2.24 - self.tolerance <= cd_ratio <= 3.618 + self.tolerance):
                                continue
                            
                            # Check if D is a 1.618 extension of XA
                            xd_ratio = abs((d_price - x_price) / xa_leg)
                            if not self._validate_fibonacci_ratio(xd_ratio, self.ideal_ratios["XD"]):
                                continue
                            
                            # We found a potential Crab pattern
                            points = {
                                'X': (x_idx, x_price),
                                'A': (a_idx, a_price),
                                'B': (b_idx, b_price),
                                'C': (c_idx, c_price),
                                'D': (d_idx, d_price)
                            }
                            
                            if self._is_pattern_valid(points):
                                ratios = {
                                    "AB": ab_ratio,
                                    "BC": bc_ratio,
                                    "CD": cd_ratio,
                                    "XD": xd_ratio
                                }
                                
                                target_zone = self._calculate_target_zone("Crab", points)
                                confidence = self._calculate_pattern_confidence(ratios, self.ideal_ratios)
                                
                                patterns.append(HarmonicMeasurement(
                                    pattern_type="Crab",
                                    points=points,
                                    ratios=ratios,
                                    confidence=confidence,
                                    target_zone=target_zone,
                                    completion_index=d_idx
                                ))
        
        return patterns
    
    def _calculate_target_zone(self, pattern_type: str, points: Dict[str, Tuple[int, float]]) -> Tuple[float, float]:
        """
        Calculate the potential reversal zone for a Crab pattern.
        
        Args:
            pattern_type: The type of the pattern
            points: Dictionary of points in the pattern
            
        Returns:
            A tuple of (support level, resistance level)
        """
        x_price = points['X'][1]
        a_price = points['A'][1]
        d_price = points['D'][1]
        
        # For Crab, PRZ is typically around the 1.618 extension of XA
        xa_move = abs(a_price - x_price)
        
        # Calculate Fibonacci extension levels
        fib_1618 = x_price + (1.618 * xa_move * (1 if a_price > x_price else -1))
        
        # For Crab patterns, also include the 2.24 extension as part of the PRZ
        fib_224 = x_price + (2.24 * xa_move * (1 if a_price > x_price else -1))
        
        # PRZ is between these two levels
        return (min(fib_1618, fib_224), max(fib_1618, fib_224))
