"""
Harmonic Pattern Recognizer Module

This module provides comprehensive harmonic pattern recognition capabilities including:
- Gartley Pattern
- Butterfly Pattern
- Bat Pattern
- Crab Pattern
- Shark Pattern
- Cypher Pattern

Implementation includes both batch and incremental calculation approaches.
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
    AnalysisTimeframe
)


class PatternPoint:
    """Represents a pivotal point in a harmonic pattern"""
    
    def __init__(self, index: int, time: datetime, price: float, is_high: bool):
        """
        Initialize a pattern point
        
        Args:
            index: The index in the DataFrame
            time: The time of the point
            price: The price at the point
            is_high: Whether the point is a high (True) or low (False)
        """
        self.index = index
        self.time = time
        self.price = price
        self.is_high = is_high
        
    def __str__(self):
        return f"{'High' if self.is_high else 'Low'} @ {self.time}: {self.price:.5f}"


class HarmonicPatternType(Enum):
    """Types of harmonic patterns"""
    GARTLEY = "Gartley"
    BUTTERFLY = "Butterfly"
    BAT = "Bat"
    CRAB = "Crab"
    SHARK = "Shark"
    CYPHER = "Cypher"
    THREE_DRIVE = "Three Drive"
    ABCD = "ABCD"
    UNKNOWN = "Unknown"


class PatternDirection(Enum):
    """Pattern direction: bullish or bearish"""
    BULLISH = "bullish"
    BEARISH = "bearish"


class HarmonicPattern:
    """Represents a discovered harmonic pattern"""
    
    def __init__(
        self,
        pattern_type: HarmonicPatternType,
        direction: PatternDirection,
        points: Dict[str, Tuple[datetime, float]],
        ratios: Dict[str, float],
        completion: float = 1.0,
        confidence: float = 0.0
    ):
        """
        Initialize a harmonic pattern
        
        Args:
            pattern_type: The type of harmonic pattern
            direction: Whether the pattern is bullish or bearish
            points: Dictionary mapping point names to (time, price) tuples
            ratios: Dictionary of measured ratios between points
            completion: Percentage of completion (0.0-1.0)
            confidence: Confidence score (0.0-1.0)
        """
        self.pattern_type = pattern_type
        self.direction = direction
        self.points = points
        self.ratios = ratios
        self.completion = completion
        self.confidence = confidence
        self.potential_reversal_zone = self._calculate_prz()
        self.target_prices = self._calculate_targets()
        self.stop_loss = self._calculate_stop_loss()
        
    def _calculate_prz(self) -> Tuple[float, float]:
        """Calculate the potential reversal zone"""
        # This would depend on the pattern type
        # For simplicity, using a placeholder implementation
        if not self.points.get('D'):
            return (0, 0)
        
        d_price = self.points['D'][1]
        d_time = self.points['D'][0]
        
        # In a real implementation, the PRZ would be calculated 
        # based on Fibonacci projections from the pattern points
        prz_min = d_price * 0.99
        prz_max = d_price * 1.01
        
        return (prz_min, prz_max)
        
    def _calculate_targets(self) -> List[float]:
        """Calculate target price levels"""
        # Simplified implementation
        targets = []
        if self.pattern_type in [HarmonicPatternType.GARTLEY, HarmonicPatternType.BUTTERFLY, 
                                HarmonicPatternType.BAT, HarmonicPatternType.CRAB]:
            if self.points.get('D') and self.points.get('A'):
                d_price = self.points['D'][1]
                a_price = self.points['A'][1]
                
                if self.direction == PatternDirection.BULLISH:
                    # Add Fibonacci projection targets
                    targets.append(d_price + (d_price - a_price) * 0.382)  # 38.2% projection
                    targets.append(d_price + (d_price - a_price) * 0.618)  # 61.8% projection
                    targets.append(d_price + (d_price - a_price))          # 100% projection
                else:
                    targets.append(d_price - (a_price - d_price) * 0.382)
                    targets.append(d_price - (a_price - d_price) * 0.618)
                    targets.append(d_price - (a_price - d_price))
                    
        return targets
    
    def _calculate_stop_loss(self) -> float:
        """Calculate recommended stop loss level"""
        # Simplified implementation
        if not self.points.get('D'):
            return 0
        
        d_price = self.points['D'][1]
        
        if self.direction == PatternDirection.BULLISH:
            # For bullish patterns, stop loss is below point D
            return d_price * 0.98  # 2% below D
        else:
            # For bearish patterns, stop loss is above point D
            return d_price * 1.02  # 2% above D
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for API responses"""
        return {
            "type": self.pattern_type.value,
            "direction": self.direction.value,
            "points": {k: {"time": v[0].isoformat(), "price": v[1]} 
                      for k, v in self.points.items()},
            "ratios": self.ratios,
            "completion_percentage": self.completion * 100,
            "confidence_score": self.confidence,
            "potential_reversal_zone": {
                "min": self.potential_reversal_zone[0],
                "max": self.potential_reversal_zone[1]
            },
            "targets": self.target_prices,
            "stop_loss": self.stop_loss
        }


class HarmonicPatternRecognizer(PatternRecognitionBase):
    """
    Harmonic Pattern Recognition implementation
    
    This class implements detection of harmonic price patterns which are based on
    specific Fibonacci ratios between price movements.
    """
    
    def __init__(
        self,
        name: str = "HarmonicPatternRecognizer",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize the Harmonic Pattern Recognizer
        
        Args:
            name: Name identifier for this analyzer
            parameters: Dictionary of parameters including:
                - price_column: Column name for price data
                - min_pattern_bars: Minimum bars for valid pattern
                - pattern_types: Types of patterns to look for
                - tolerance: Tolerance for Fibonacci ratio matches
        """
        default_params = {
            "price_column": "close",
            "min_pattern_bars": 10,
            "pattern_types": [p.value for p in HarmonicPatternType if p != HarmonicPatternType.UNKNOWN],
            "tolerance": 0.05  # 5% tolerance for Fibonacci ratios
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
        
        # Set up Fibonacci ratio definitions for each pattern
        self.pattern_definitions = {
            HarmonicPatternType.GARTLEY: {
                "min_points": 5,  # X, A, B, C, D points
                "XA_retracement": 0.618,  # B should retrace 61.8% of XA
                "AB_retracement": 0.382,  # C should retrace 38.2% of AB
                "BC_projection": 0.886,   # BC projection against AB
                "XA_projection": 0.786    # D should be 78.6% of XA
            },
            HarmonicPatternType.BUTTERFLY: {
                "min_points": 5,
                "XA_retracement": 0.786,
                "AB_retracement": 0.382,
                "BC_projection": 1.618,
                "XA_projection": 1.272
            },
            HarmonicPatternType.BAT: {
                "min_points": 5,
                "XA_retracement": 0.382, 
                "AB_retracement": 0.382,
                "BC_projection": 1.618,
                "XA_projection": 0.886
            },
            HarmonicPatternType.CRAB: {
                "min_points": 5,
                "XA_retracement": 0.382,
                "AB_retracement": 0.618,
                "BC_projection": 2.618,
                "XA_projection": 1.618
            },
            HarmonicPatternType.SHARK: {
                "min_points": 5,
                "XA_retracement": 0.446,
                "AB_retracement": 0.618,
                "BC_projection": 1.618,
                "XA_projection": 0.886
            },
            HarmonicPatternType.CYPHER: {
                "min_points": 5,
                "XA_retracement": 0.382,
                "AB_retracement": 0.618,
                "BC_projection": 1.272,
                "XA_projection": 0.786
            },
            HarmonicPatternType.ABCD: {
                "min_points": 4,  # A, B, C, D points
                "AB_retracement": 0.618,
                "BC_projection": 1.272
            }
        }
        
    def find_patterns(self, df: pd.DataFrame) -> List[HarmonicPattern]:
        """
        Find harmonic patterns in the provided price data
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            List of detected harmonic patterns
        """
        # Verify we have enough data
        if len(df) < self.parameters.get("min_pattern_bars", 10):
            return []
            
        # Detect swing points (peaks and troughs)
        swing_points = self._identify_swing_points(df)
        
        patterns = []
        
        # Try to identify each type of harmonic pattern
        for pattern_type_name in self.parameters.get("pattern_types", []):
            try:
                pattern_type = HarmonicPatternType(pattern_type_name)
                pattern_def = self.pattern_definitions.get(pattern_type)
                
                if pattern_def:
                    # Find all instances of this pattern
                    new_patterns = self._find_pattern_instances(df, swing_points, pattern_type, pattern_def)
                    patterns.extend(new_patterns)
                    
            except ValueError:
                # Skip invalid pattern types
                continue
                
        return patterns
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate harmonic patterns and add corresponding columns to the DataFrame
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with added pattern columns
        """
        patterns = self.find_patterns(df)
        
        # Create a copy to avoid SettingWithCopyWarning
        result_df = df.copy()
        
        # Initialize pattern columns
        for pattern_type in HarmonicPatternType:
            if pattern_type != HarmonicPatternType.UNKNOWN:
                bull_col = f"harmonic_{pattern_type.value.lower()}_bullish"
                bear_col = f"harmonic_{pattern_type.value.lower()}_bearish"
                result_df[bull_col] = 0
                result_df[bear_col] = 0
                
        # Add confidence columns
        result_df['harmonic_pattern_confidence'] = 0.0
        
        # Mark points where patterns are detected
        for pattern in patterns:
            # Get the D point (pattern completion point)
            d_point = pattern.points.get('D')
            if d_point and d_point[0] in result_df.index:
                # Find the index location for this timestamp
                d_idx_loc = result_df.index.get_loc(d_point[0])
                
                # Mark the pattern
                pattern_col = f"harmonic_{pattern.pattern_type.value.lower()}"
                if pattern.direction == PatternDirection.BULLISH:
                    result_df.iloc[d_idx_loc, result_df.columns.get_loc(f"{pattern_col}_bullish")] = 1
                else:
                    result_df.iloc[d_idx_loc, result_df.columns.get_loc(f"{pattern_col}_bearish")] = 1
                    
                # Set confidence
                result_df.iloc[d_idx_loc, result_df.columns.get_loc('harmonic_pattern_confidence')] = pattern.confidence
        
        return result_df
        
    def _identify_swing_points(self, df: pd.DataFrame) -> List[PatternPoint]:
        """
        Identify swing high and swing low points in the data
        
        Args:
            df: DataFrame with price data
            
        Returns:
            List of PatternPoint objects sorted by time
        """
        price_col = self.parameters.get("price_column", "close")
        high_col = "high"
        low_col = "low"
        
        # Check if we have high/low columns
        has_hl = high_col in df.columns and low_col in df.columns
        
        lookback = 2  # Number of bars to look back/forward for swings
        
        swing_points = []
        
        # Find swing highs
        for i in range(lookback, len(df) - lookback):
            if has_hl:
                # A swing high is formed when the high is higher than surrounding bars
                is_swing_high = True
                for j in range(1, lookback + 1):
                    if df[high_col].iloc[i] <= df[high_col].iloc[i - j] or \
                       df[high_col].iloc[i] <= df[high_col].iloc[i + j]:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    swing_points.append(PatternPoint(
                        index=i,
                        time=df.index[i],
                        price=df[high_col].iloc[i],
                        is_high=True
                    ))
                    
            # A swing low is formed when the low is lower than surrounding bars
            is_swing_low = True
            for j in range(1, lookback + 1):
                if df[low_col].iloc[i] >= df[low_col].iloc[i - j] or \
                   df[low_col].iloc[i] >= df[low_col].iloc[i + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_points.append(PatternPoint(
                    index=i,
                    time=df.index[i],
                    price=df[low_col].iloc[i],
                    is_high=False
                ))
        
        # Sort by time (index)
        swing_points.sort(key=lambda p: p.index)
        return swing_points
        
    def _find_pattern_instances(
        self,
        df: pd.DataFrame,
        swing_points: List[PatternPoint],
        pattern_type: HarmonicPatternType,
        pattern_def: Dict[str, Any]
    ) -> List[HarmonicPattern]:
        """
        Find all instances of a specific harmonic pattern
        
        Args:
            df: DataFrame with price data
            swing_points: List of detected swing points
            pattern_type: The type of harmonic pattern to find
            pattern_def: Pattern definition with expected ratios
            
        Returns:
            List of harmonic patterns
        """
        min_points = pattern_def.get("min_points", 5)
        tolerance = self.parameters.get("tolerance", 0.05)
        
        patterns = []
        
        # We need at least min_points swing points to form a pattern
        if len(swing_points) < min_points:
            return []
            
        # Check different starting points
        for i in range(len(swing_points) - min_points + 1):
            # For a 5-point pattern (X,A,B,C,D), we need alternating highs/lows
            # For a 4-point pattern (A,B,C,D), we need alternating points
            
            if min_points == 5:  # 5-point pattern (X,A,B,C,D)
                x = swing_points[i]
                a = swing_points[i+1]
                b = swing_points[i+2]
                c = swing_points[i+3]
                d = swing_points[i+4]
                
                # Check for alternating high/low pattern
                if not (x.is_high != a.is_high and 
                        a.is_high != b.is_high and 
                        b.is_high != c.is_high and 
                        c.is_high != d.is_high):
                    continue
                    
                # Determine if pattern is bullish (final point D is a low) or bearish
                is_bullish = not d.is_high
                
                # Calculate price movements
                xa_move = abs(a.price - x.price)
                ab_move = abs(b.price - a.price)
                bc_move = abs(c.price - b.price)
                cd_move = abs(d.price - c.price)
                xd_move = abs(d.price - x.price)
                
                # Calculate key ratios
                ab_xa_ratio = ab_move / xa_move if xa_move else 0
                bc_ab_ratio = bc_move / ab_move if ab_move else 0
                cd_bc_ratio = cd_move / bc_move if bc_move else 0
                xd_xa_ratio = xd_move / xa_move if xa_move else 0
                
                # Check if ratios match the expected pattern with tolerance
                expected_ab_xa = pattern_def.get("XA_retracement", 0)
                expected_bc_ab = pattern_def.get("AB_retracement", 0)
                expected_bc_projection = pattern_def.get("BC_projection", 0)
                expected_xd_xa = pattern_def.get("XA_projection", 0)
                
                # Basic check for ABCD - point B retraces to correct level
                if abs(ab_xa_ratio - expected_ab_xa) <= tolerance:
                    # BC movement matches expected projection/retracement
                    if (abs(bc_ab_ratio - expected_bc_ab) <= tolerance or
                        abs(bc_ab_ratio - expected_bc_projection) <= tolerance):
                        # XD matches expected projection
                        if abs(xd_xa_ratio - expected_xd_xa) <= tolerance:
                            # Create pattern with calculated ratios
                            pattern = HarmonicPattern(
                                pattern_type=pattern_type,
                                direction=PatternDirection.BULLISH if is_bullish else PatternDirection.BEARISH,
                                points={
                                    'X': (x.time, x.price),
                                    'A': (a.time, a.price),
                                    'B': (b.time, b.price),
                                    'C': (c.time, c.price),
                                    'D': (d.time, d.price)
                                },
                                ratios={
                                    'AB/XA': ab_xa_ratio,
                                    'BC/AB': bc_ab_ratio,
                                    'CD/BC': cd_bc_ratio,
                                    'XD/XA': xd_xa_ratio
                                },
                                completion=1.0,  # Fully formed pattern
                                confidence=self._calculate_pattern_confidence(
                                    expected_ab_xa, ab_xa_ratio,
                                    expected_bc_ab, bc_ab_ratio,
                                    expected_xd_xa, xd_xa_ratio,
                                    tolerance
                                )
                            )
                            patterns.append(pattern)
                            
            elif min_points == 4:  # 4-point pattern (A,B,C,D)
                a = swing_points[i]
                b = swing_points[i+1]
                c = swing_points[i+2]
                d = swing_points[i+3]
                
                # Check for alternating high/low pattern
                if not (a.is_high != b.is_high and 
                        b.is_high != c.is_high and 
                        c.is_high != d.is_high):
                    continue
                    
                is_bullish = not d.is_high
                
                # Calculate price movements
                ab_move = abs(b.price - a.price)
                bc_move = abs(c.price - b.price)
                cd_move = abs(d.price - c.price)
                
                # Calculate key ratios for ABCD pattern
                bc_ab_ratio = bc_move / ab_move if ab_move else 0
                cd_bc_ratio = cd_move / bc_move if bc_move else 0
                
                # Check expected ratios for ABCD pattern
                expected_ab_retracement = pattern_def.get("AB_retracement", 0.618)
                expected_bc_projection = pattern_def.get("BC_projection", 1.272)
                
                # In ABCD pattern, BC should be around 0.618 of AB, and CD should be around 1.272 of BC
                if abs(bc_ab_ratio - expected_ab_retracement) <= tolerance:
                    if abs(cd_bc_ratio - expected_bc_projection) <= tolerance:
                        pattern = HarmonicPattern(
                            pattern_type=pattern_type,
                            direction=PatternDirection.BULLISH if is_bullish else PatternDirection.BEARISH,
                            points={
                                'A': (a.time, a.price),
                                'B': (b.time, b.price),
                                'C': (c.time, c.price),
                                'D': (d.time, d.price)
                            },
                            ratios={
                                'BC/AB': bc_ab_ratio,
                                'CD/BC': cd_bc_ratio
                            },
                            completion=1.0,
                            confidence=self._calculate_pattern_confidence(
                                expected_ab_retracement, bc_ab_ratio,
                                expected_bc_projection, cd_bc_ratio,
                                0, 0, tolerance
                            )
                        )
                        patterns.append(pattern)
        
        return patterns

    def _calculate_pattern_confidence(
        self,
        expected_ratio1: float,
        actual_ratio1: float,
        expected_ratio2: float,
        actual_ratio2: float,
        expected_ratio3: float,
        actual_ratio3: float,
        tolerance: float
    ) -> float:
        """
        Calculate confidence score based on how closely ratios match expected values
        
        Args:
            expected_ratio1: First expected ratio
            actual_ratio1: First actual ratio
            expected_ratio2: Second expected ratio
            actual_ratio2: Second actual ratio
            expected_ratio3: Third expected ratio (may be 0 for ABCD pattern)
            actual_ratio3: Third actual ratio (may be 0 for ABCD pattern)
            tolerance: Ratio tolerance
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Calculate how precisely each ratio matches the expected value
        # 1.0 = perfect match, 0.0 = at tolerance boundary
        precision1 = 1.0 - abs(actual_ratio1 - expected_ratio1) / tolerance
        precision2 = 1.0 - abs(actual_ratio2 - expected_ratio2) / tolerance
        
        # For patterns with 3 ratios
        if expected_ratio3 > 0:
            precision3 = 1.0 - abs(actual_ratio3 - expected_ratio3) / tolerance
            avg_precision = (precision1 + precision2 + precision3) / 3
        else:
            # For patterns with 2 ratios (like ABCD)
            avg_precision = (precision1 + precision2) / 2
            
        # Clamp to valid range
        return max(0.0, min(1.0, avg_precision))
        
    def initialize_incremental(self) -> Dict[str, Any]:
        """Initialize state for incremental pattern detection"""
        return {
            "swing_points": [],  # List of detected swing points
            "complete_patterns": [],  # List of detected patterns
            "last_bar": None  # Last processed bar
        }
        
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Update state with new price data for incremental pattern detection
        
        Args:
            state: Current state dictionary
            new_data: New price bar data
            
        Returns:
            Updated state dictionary
        """
        # Extract price data
        high = new_data.get("high", new_data.get("close"))
        low = new_data.get("low", new_data.get("close"))
        close = new_data.get("close")
        timestamp = new_data.get("timestamp", datetime.now())
        
        # Check if this is a new high or low compared to recent points
        lookback = 2 # Same as in _identify_swing_points
        
        # If we have fewer than 2*lookback+1 points, just add the current data point
        if len(state.get("swing_points", [])) < 2 * lookback + 1:
            # Just add the current bar to the state
            state["last_bar"] = {
                "high": high,
                "low": low,
                "close": close,
                "timestamp": timestamp,
                "index": len(state.get("swing_points", []))
            }
            return state
        
        # Get recent swing points
        recent_points = state.get("swing_points", [])[-2*lookback:]
        
        # Check if new high is a swing high
        is_swing_high = True
        for p in recent_points:
            if p.is_high and p.price >= high:
                is_swing_high = False
                break
        
        # Check if new low is a swing low
        is_swing_low = True
        for p in recent_points:
            if not p.is_high and p.price <= low:
                is_swing_low = False
                break
        
        # Add new swing points if detected
        current_index = len(state.get("swing_points", []))
        if is_swing_high:
            state["swing_points"].append(PatternPoint(
                index=current_index,
                time=timestamp,
                price=high,
                is_high=True
            ))
        
        if is_swing_low:
            state["swing_points"].append(PatternPoint(
                index=current_index,
                time=timestamp,
                price=low,
                is_high=False
            ))
        
        # Update last bar
        state["last_bar"] = {
            "high": high,
            "low": low,
            "close": close,
            "timestamp": timestamp,
            "index": current_index
        }
        
        # Try to identify patterns with the updated swing points
        self._identify_incremental_patterns(state)
        
        return state
        
    def _identify_incremental_patterns(self, state: Dict[str, Any]) -> None:
        """
        Attempt to identify harmonic patterns from current swing points
        
        Args:
            state: Current state dictionary with swing points
            
        Returns:
            None, updates state["complete_patterns"] in-place
        """
        swing_points = state.get("swing_points", [])
        
        # We need at least 5 swing points for a full harmonic pattern
        if len(swing_points) < 5:
            return
            
        # Check for each pattern type
        for pattern_type_name in self.parameters.get("pattern_types", []):
            try:
                pattern_type = HarmonicPatternType(pattern_type_name)
                pattern_def = self.pattern_definitions.get(pattern_type)
                
                if pattern_def:
                    min_points = pattern_def.get("min_points", 5)
                    tolerance = self.parameters.get("tolerance", 0.05)
                    
                    # Look for patterns in the most recent swing points
                    # For simplicity, just check the last possible pattern
                    if len(swing_points) >= min_points:
                        # Get the last min_points swing points
                        recent_points = swing_points[-min_points:]
                        
                        # Do the same validation as in _find_pattern_instances
                        if min_points == 5:  # 5-point pattern
                            x, a, b, c, d = recent_points
                            
                            # Check alternating
                            if (x.is_high != a.is_high and 
                                a.is_high != b.is_high and 
                                b.is_high != c.is_high and 
                                c.is_high != d.is_high):
                                
                                # Same validation logic as in _find_pattern_instances
                                is_bullish = not d.is_high
                                
                                # Calculate ratios
                                xa_move = abs(a.price - x.price)
                                ab_move = abs(b.price - a.price)
                                bc_move = abs(c.price - b.price)
                                cd_move = abs(d.price - c.price)
                                xd_move = abs(d.price - x.price)
                                
                                ab_xa_ratio = ab_move / xa_move if xa_move else 0
                                bc_ab_ratio = bc_move / ab_move if ab_move else 0
                                cd_bc_ratio = cd_move / bc_move if bc_move else 0
                                xd_xa_ratio = xd_move / xa_move if xa_move else 0
                                
                                expected_ab_xa = pattern_def.get("XA_retracement", 0)
                                expected_bc_ab = pattern_def.get("AB_retracement", 0)
                                expected_bc_projection = pattern_def.get("BC_projection", 0)
                                expected_xd_xa = pattern_def.get("XA_projection", 0)
                                
                                # Check the ratios
                                if (abs(ab_xa_ratio - expected_ab_xa) <= tolerance and
                                    (abs(bc_ab_ratio - expected_bc_ab) <= tolerance or
                                     abs(bc_ab_ratio - expected_bc_projection) <= tolerance) and
                                    abs(xd_xa_ratio - expected_xd_xa) <= tolerance):
                                    
                                    # Create the pattern
                                    pattern = HarmonicPattern(
                                        pattern_type=pattern_type,
                                        direction=PatternDirection.BULLISH if is_bullish else PatternDirection.BEARISH,
                                        points={
                                            'X': (x.time, x.price),
                                            'A': (a.time, a.price),
                                            'B': (b.time, b.price),
                                            'C': (c.time, c.price),
                                            'D': (d.time, d.price)
                                        },
                                        ratios={
                                            'AB/XA': ab_xa_ratio,
                                            'BC/AB': bc_ab_ratio,
                                            'CD/BC': cd_bc_ratio,
                                            'XD/XA': xd_xa_ratio
                                        },
                                        completion=1.0,
                                        confidence=self._calculate_pattern_confidence(
                                            expected_ab_xa, ab_xa_ratio,
                                            expected_bc_ab, bc_ab_ratio,
                                            expected_xd_xa, xd_xa_ratio,
                                            tolerance
                                        )
                                    )
                                    
                                    # Check if this is a new pattern
                                    is_new = True
                                    for existing in state.get("complete_patterns", []):
                                        # Compare key attributes to see if it's the same pattern
                                        if (existing.pattern_type == pattern.pattern_type and
                                            existing.direction == pattern.direction and
                                            existing.points['D'][0] == pattern.points['D'][0]):
                                            is_new = False
                                            break
                                            
                                    if is_new:
                                        if "complete_patterns" not in state:
                                            state["complete_patterns"] = []
                                        state["complete_patterns"].append(pattern)
                            
                        elif min_points == 4:  # 4-point ABCD pattern
                            a, b, c, d = recent_points
                            
                            # Similar validation for ABCD pattern
                            if (a.is_high != b.is_high and 
                                b.is_high != c.is_high and 
                                c.is_high != d.is_high):
                                
                                # Add ABCD pattern logic here (similar to above)
                                pass
                                
            except ValueError:
                continue
        
        # Limit stored patterns to avoid memory issues
        if len(state.get("complete_patterns", [])) > 20:
            state["complete_patterns"] = state["complete_patterns"][-20:]
            
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get information about this indicator"""
        return {
            'name': 'HarmonicPatternRecognizer',
            'description': 'Recognizes harmonic price patterns based on Fibonacci ratios',
            'category': 'pattern',
            'parameters': [
                {
                    'name': 'price_column',
                    'description': 'Column name for price data',
                    'type': 'str',
                    'default': 'close'
                },
                {
                    'name': 'min_pattern_bars',
                    'description': 'Minimum bars required for valid pattern',
                    'type': 'int',
                    'default': 10
                },
                {
                    'name': 'pattern_types',
                    'description': 'Types of patterns to detect',
                    'type': 'list',
                    'default': '[Gartley, Butterfly, Bat, Crab, Shark, Cypher, ABCD]'
                },
                {
                    'name': 'tolerance',
                    'description': 'Tolerance for Fibonacci ratio matches (0.01-0.10)',
                    'type': 'float',
                    'default': 0.05
                }
            ]
        }
