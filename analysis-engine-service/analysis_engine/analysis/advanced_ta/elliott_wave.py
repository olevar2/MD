"""
Elliott Wave Analysis Module

This module provides implementation of Elliott Wave pattern detection and analysis,
including wave identification, labeling, and related Fibonacci measurements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime
from enum import Enum
import math

from analysis_engine.analysis.advanced_ta.base import (
    AdvancedAnalysisBase,
    PatternRecognitionBase,
    PatternResult,
    ConfidenceLevel,
    MarketDirection
)


class WaveType(Enum):
    """Types of Elliott Waves"""
    IMPULSE = "impulse"  # 5-wave pattern in direction of larger trend
    CORRECTION = "correction"  # 3-wave pattern against the larger trend
    DIAGONAL = "diagonal"  # 5-wave wedge-shaped pattern
    EXTENSION = "extension"  # Extended wave within an impulse wave
    TRIANGLE = "triangle"  # Complex correction in form of a triangle
    UNKNOWN = "unknown"


class WavePosition(Enum):
    """Position in the Elliott Wave sequence"""
    # Impulse wave positions
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    
    # Corrective wave positions
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    
    # Sub-waves
    SUB_ONE = "i"
    SUB_TWO = "ii"
    SUB_THREE = "iii"
    SUB_FOUR = "iv"
    SUB_FIVE = "v"
    
    # Sub-corrective waves
    SUB_A = "a"
    SUB_B = "b"
    SUB_C = "c"
    SUB_D = "d"
    SUB_E = "e"


class WaveDegree(Enum):
    """Degrees of trend in Elliott Wave Theory"""
    GRAND_SUPERCYCLE = "Grand Supercycle"
    SUPERCYCLE = "Supercycle"
    CYCLE = "Cycle"
    PRIMARY = "Primary"
    INTERMEDIATE = "Intermediate"
    MINOR = "Minor"
    MINUTE = "Minute"
    MINUETTE = "Minuette"
    SUBMINUETTE = "Subminuette"


class ElliottWavePattern(PatternResult):
    """
    Represents a detected Elliott Wave pattern
    
    Extends PatternResult with Elliott Wave specific attributes.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wave_type = kwargs.get("wave_type", WaveType.UNKNOWN)
        self.wave_degree = kwargs.get("wave_degree", WaveDegree.INTERMEDIATE)
        self.waves = kwargs.get("waves", {})  # Dict mapping wave positions to (time, price) tuples
        self.sub_waves = kwargs.get("sub_waves", {})  # Sub-waves if identified
        self.fibonacci_levels = kwargs.get("fibonacci_levels", {})  # Fibonacci projections
        self.completion_percentage = kwargs.get("completion_percentage", 100.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        pattern_dict = super().to_dict()
        pattern_dict.update({
            "wave_type": self.wave_type.value,
            "wave_degree": self.wave_degree.value,
            "waves": {str(k.value): {"time": v[0].isoformat(), "price": v[1]} 
                    for k, v in self.waves.items()},
            "completion_percentage": self.completion_percentage
        })
        
        # Add fibonacci levels if present
        if self.fibonacci_levels:
            pattern_dict["fibonacci_levels"] = {
                k: {"price": v} for k, v in self.fibonacci_levels.items()
            }
            
        return pattern_dict


class ElliottWaveAnalyzer(PatternRecognitionBase):
    """
    Elliott Wave Analysis Engine
    
    Provides detection and analysis of Elliott Wave patterns in price data.
    """
    
    def __init__(
        self,
        name: str = "ElliottWaveAnalyzer",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize Elliott Wave analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {
            "price_column": "close",
            "high_column": "high",
            "low_column": "low",
            "lookback_period": 200,
            "max_wave_count": 5,  # Maximum number of waves to identify
            "min_wave_size": 5,  # Minimum number of bars for a valid wave
            "use_zigzag_filter": True,  # Use zigzag to identify waves
            "zigzag_threshold": 0.05,  # 5% threshold for zigzag
            "fibonacci_levels": [0.382, 0.5, 0.618, 1.0, 1.618, 2.618],
            "wave_degree": "Intermediate"  # Default wave degree for analysis
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def find_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        Find Elliott Wave patterns in price data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of ElliottWavePattern objects
        """
        # Verify we have enough data
        if len(df) < self.parameters["lookback_period"]:
            return []
        
        # Use the most recent data within lookback period
        analysis_df = df.iloc[-self.parameters["lookback_period"]:]
        
        # Detect significant zigzag points
        zigzag_points = self._detect_zigzag_points(analysis_df)
        
        # Group points into potential Elliott Wave patterns
        wave_patterns = self._identify_wave_patterns(analysis_df, zigzag_points)
        
        return wave_patterns
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Elliott Wave analysis and add to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Elliott Wave analysis
        """
        result_df = df.copy()
        
        # Find Elliott Wave patterns
        patterns = self.find_patterns(result_df)
        
        # Add zigzag pivot points to DataFrame 
        result_df['elliott_zigzag'] = 0
        
        # Add wave identification columns
        wave_types = {wave_type.value for wave_type in WaveType 
                     if wave_type != WaveType.UNKNOWN}
        
        for wave_type in wave_types:
            result_df[f'elliott_{wave_type}'] = 0
        
        # Add wave position markers
        for position in WavePosition:
            result_df[f'elliott_wave_{position.value}'] = np.nan
        
        # Add Fibonacci projections column
        result_df['elliott_fib_projection'] = np.nan
        
        # Mark patterns in DataFrame
        for pattern in patterns:
            # Mark wave type
            pattern_range = (result_df.index >= pattern.start_time) & (result_df.index <= pattern.end_time)
            pattern_type_col = f'elliott_{pattern.wave_type.value}'
            
            if pattern_type_col in result_df.columns:
                result_df.loc[pattern_range, pattern_type_col] = 1
            
            # Mark wave positions
            for position, (time, price) in pattern.waves.items():
                position_col = f'elliott_wave_{position.value}'
                if position_col in result_df.columns and time in result_df.index:
                    result_df.loc[time, position_col] = price
                    
                    # Mark as zigzag point too
                    result_df.loc[time, 'elliott_zigzag'] = 1
            
            # Add Fibonacci projections for wave 3 and wave 5
            if WavePosition.ONE in pattern.waves and WavePosition.TWO in pattern.waves:
                # Wave 1 to 2 retracement
                w1_time, w1_price = pattern.waves[WavePosition.ONE]
                w2_time, w2_price = pattern.waves[WavePosition.TWO]
                
                # Only add projections after wave 2
                if w2_time in result_df.index:
                    w2_idx = result_df.index.get_loc(w2_time)
                    wave_1_size = abs(w1_price - w2_price)
                    
                    # Calculate Fibonacci projections from wave 2
                    for fib_level in self.parameters["fibonacci_levels"]:
                        if pattern.direction == MarketDirection.BULLISH:
                            fib_price = w2_price + wave_1_size * fib_level
                        else:
                            fib_price = w2_price - wave_1_size * fib_level
                            
                        # Only project forward from wave 2
                        projection_range = result_df.index[w2_idx:]
                        fib_col = f'elliott_fib_{str(fib_level).replace(".", "_")}'
                        
                        # Add column if not exists
                        if fib_col not in result_df.columns:
                            result_df[fib_col] = np.nan
                            
                        result_df.loc[projection_range, fib_col] = fib_price
        
        return result_df
    
    def _detect_zigzag_points(self, df: pd.DataFrame) -> List[Tuple[datetime, float, bool]]:
        """
        Detect significant turning points using zigzag algorithm
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of (timestamp, price, is_high) tuples for zigzag points
        """
        # Extract price columns
        high_col = self.parameters.get("high_column", self.parameters["price_column"])
        low_col = self.parameters.get("low_column", self.parameters["price_column"])
        price_col = self.parameters["price_column"]
        
        # If high and low columns are available, use those for better accuracy
        if high_col in df.columns and low_col in df.columns:
            price = df[price_col].values
            highs = df[high_col].values
            lows = df[low_col].values
        else:
            # Use price column for both high and low
            price = df[price_col].values
            highs = price
            lows = price
        
        timestamps = df.index.to_pydatetime()
        zigzag_threshold = self.parameters["zigzag_threshold"]
        
        zigzag_points = []
        in_uptrend = None
        last_price = None
        swing_high = None
        swing_low = None
        swing_high_price = -float('inf')
        swing_low_price = float('inf')
        
        for i in range(len(price)):
            current_price = price[i]
            current_high = highs[i]
            current_low = lows[i]
            
            # Initial point
            if last_price is None:
                last_price = current_price
                continue
            
            # Determine initial trend direction
            if in_uptrend is None:
                in_uptrend = current_price > last_price
                if in_uptrend:
                    swing_low = (timestamps[i-1], last_price, False)
                    zigzag_points.append(swing_low)
                else:
                    swing_high = (timestamps[i-1], last_price, True)
                    zigzag_points.append(swing_high)
                
                swing_high_price = current_high
                swing_low_price = current_low
                continue
            
            # Update swing points
            if in_uptrend:
                # Looking for higher highs
                if current_high > swing_high_price:
                    swing_high_price = current_high
                
                # Check for reversal
                if current_price < (swing_high_price * (1 - zigzag_threshold)):
                    # Reversal from uptrend to downtrend
                    swing_high = (timestamps[i-1], swing_high_price, True)
                    zigzag_points.append(swing_high)
                    in_uptrend = False
                    swing_low_price = current_low
            else:
                # Looking for lower lows
                if current_low < swing_low_price:
                    swing_low_price = current_low
                
                # Check for reversal
                if current_price > (swing_low_price * (1 + zigzag_threshold)):
                    # Reversal from downtrend to uptrend
                    swing_low = (timestamps[i-1], swing_low_price, False)
                    zigzag_points.append(swing_low)
                    in_uptrend = True
                    swing_high_price = current_high
            
            last_price = current_price
        
        # Add the most recent swing point if it hasn't been added
        if in_uptrend and swing_high_price > -float('inf'):
            swing_high = (timestamps[-1], swing_high_price, True)
            zigzag_points.append(swing_high)
        elif not in_uptrend and swing_low_price < float('inf'):
            swing_low = (timestamps[-1], swing_low_price, False)
            zigzag_points.append(swing_low)
        
        return zigzag_points
    
    def _identify_wave_patterns(self, df: pd.DataFrame, zigzag_points: List[Tuple[datetime, float, bool]]) -> List[ElliottWavePattern]:
        """
        Identify Elliott Wave patterns from zigzag points
        
        Args:
            df: DataFrame with OHLCV data
            zigzag_points: List of zigzag points as (time, price, is_high) tuples
            
        Returns:
            List of identified Elliott Wave patterns
        """
        patterns = []
        
        # Need at least 5 points for a valid pattern
        if len(zigzag_points) < 5:
            return patterns
        
        # Check for impulse wave patterns
        impulse_patterns = self._find_impulse_patterns(df, zigzag_points)
        patterns.extend(impulse_patterns)
        
        # Check for corrective wave patterns
        correction_patterns = self._find_correction_patterns(df, zigzag_points)
        patterns.extend(correction_patterns)
        
        return patterns
    
    def _find_impulse_patterns(self, df: pd.DataFrame, zigzag_points: List[Tuple[datetime, float, bool]]) -> List[ElliottWavePattern]:
        """
        Find impulse wave patterns in zigzag points
        
        Args:
            df: DataFrame with OHLCV data
            zigzag_points: List of zigzag points
            
        Returns:
            List of impulse wave patterns
        """
        patterns = []
        
        # Try to find patterns with different starting points
        for start_idx in range(len(zigzag_points) - 4):  # Need at least 5 points
            # Extract potential wave points
            potential_waves = zigzag_points[start_idx:start_idx+5]
            
            # Verify alternating high/low pattern for impulse waves
            alternating = True
            for i in range(1, len(potential_waves)):
                if potential_waves[i][2] == potential_waves[i-1][2]:  # is_high is the same
                    alternating = False
                    break
            
            if not alternating:
                continue
            
            # Get wave points
            w0_time, w0_price, w0_is_high = potential_waves[0]
            w1_time, w1_price, w1_is_high = potential_waves[1]
            w2_time, w2_price, w2_is_high = potential_waves[2]
            w3_time, w3_price, w3_is_high = potential_waves[3]
            w4_time, w4_price, w4_is_high = potential_waves[4]
            
            # Determine if this is a valid impulse pattern
            
            # For uptrend impulse:
            # Wave 1: UP, Wave 2: DOWN (not below start), Wave 3: UP (longer than 1), Wave 4: DOWN (not below 1), Wave 5: UP
            if (w0_is_high == False and w1_is_high == True and 
                w2_is_high == False and w3_is_high == True and 
                w4_is_high == False):
                
                # Check wave 2 doesn't go below start
                if w2_price < w0_price:
                    continue
                
                # Check wave 3 is longer than wave 1
                w1_size = w1_price - w0_price
                w3_size = w3_price - w2_price
                if w3_size <= w1_size:
                    continue
                
                # Check wave 4 doesn't go below end of wave 1
                if w4_price < w1_price:
                    continue
                
                # This looks like a valid bullish impulse pattern
                pattern = ElliottWavePattern(
                    pattern_name="Impulse Wave (Bullish)",
                    pattern_type=WaveType.IMPULSE,
                    wave_degree=WaveDegree(self.parameters.get("wave_degree", "Intermediate")),
                    confidence=ConfidenceLevel.MEDIUM, 
                    direction=MarketDirection.BULLISH,
                    start_time=w0_time,
                    end_time=w4_time,
                    start_price=w0_price,
                    end_price=w4_price,
                    waves={
                        WavePosition.ONE: (w1_time, w1_price),
                        WavePosition.TWO: (w2_time, w2_price),
                        WavePosition.THREE: (w3_time, w3_price),
                        WavePosition.FOUR: (w4_time, w4_price),
                        # Wave 5 is projected (not completed)
                        # Using simple 1:1 projection with wave 1
                        WavePosition.FIVE: (df.index[-1], w4_price + w1_size)
                    },
                    completion_percentage=80.0  # Assume it's mostly complete
                )
                
                # Calculate Fibonacci projections for Wave 3 and Wave 5
                pattern.fibonacci_levels = self._calculate_impulse_fibonacci_levels(
                    w0_price, w1_price, w2_price, w3_price, w4_price,
                    MarketDirection.BULLISH
                )
                
                patterns.append(pattern)
            
            # For downtrend impulse:
            # Wave 1: DOWN, Wave 2: UP (not above start), Wave 3: DOWN (longer than 1), Wave 4: UP (not above 1), Wave 5: DOWN
            elif (w0_is_high == True and w1_is_high == False and 
                  w2_is_high == True and w3_is_high == False and 
                  w4_is_high == True):
                
                # Check wave 2 doesn't go above start
                if w2_price > w0_price:
                    continue
                
                # Check wave 3 is longer than wave 1
                w1_size = w0_price - w1_price
                w3_size = w2_price - w3_price
                if w3_size <= w1_size:
                    continue
                
                # Check wave 4 doesn't go above end of wave 1
                if w4_price > w1_price:
                    continue
                
                # This looks like a valid bearish impulse pattern
                pattern = ElliottWavePattern(
                    pattern_name="Impulse Wave (Bearish)",
                    pattern_type=WaveType.IMPULSE,
                    wave_degree=WaveDegree(self.parameters.get("wave_degree", "Intermediate")),
                    confidence=ConfidenceLevel.MEDIUM, 
                    direction=MarketDirection.BEARISH,
                    start_time=w0_time,
                    end_time=w4_time,
                    start_price=w0_price,
                    end_price=w4_price,
                    waves={
                        WavePosition.ONE: (w1_time, w1_price),
                        WavePosition.TWO: (w2_time, w2_price),
                        WavePosition.THREE: (w3_time, w3_price),
                        WavePosition.FOUR: (w4_time, w4_price),
                        # Wave 5 is projected (not completed)
                        # Using simple 1:1 projection with wave 1
                        WavePosition.FIVE: (df.index[-1], w4_price - w1_size)
                    },
                    completion_percentage=80.0  # Assume it's mostly complete
                )
                
                # Calculate Fibonacci projections
                pattern.fibonacci_levels = self._calculate_impulse_fibonacci_levels(
                    w0_price, w1_price, w2_price, w3_price, w4_price,
                    MarketDirection.BEARISH
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _find_correction_patterns(self, df: pd.DataFrame, zigzag_points: List[Tuple[datetime, float, bool]]) -> List[ElliottWavePattern]:
        """
        Find correction (ABC) wave patterns in zigzag points
        
        Args:
            df: DataFrame with OHLCV data
            zigzag_points: List of zigzag points
            
        Returns:
            List of correction wave patterns
        """
        patterns = []
        
        # Try to find correction patterns with different starting points
        for start_idx in range(len(zigzag_points) - 2):  # Need at least 3 points
            # Extract potential wave points
            potential_waves = zigzag_points[start_idx:start_idx+3]
            
            # Verify alternating high/low pattern for ABC corrections
            alternating = True
            for i in range(1, len(potential_waves)):
                if potential_waves[i][2] == potential_waves[i-1][2]:  # is_high is the same
                    alternating = False
                    break
            
            if not alternating:
                continue
            
            # Get wave points
            w0_time, w0_price, w0_is_high = potential_waves[0]
            wA_time, wA_price, wA_is_high = potential_waves[1]
            wB_time, wB_price, wB_is_high = potential_waves[2]
            
            # Project wave C based on Fibonacci relationships
            # For this simple implementation, project C as 100% of A
            if wB_is_high:  # B is a high, C will be a low
                wC_price = wB_price - abs(wA_price - w0_price)
                direction = MarketDirection.BEARISH
            else:  # B is a low, C will be a high
                wC_price = wB_price + abs(wA_price - w0_price)
                direction = MarketDirection.BULLISH
                
            wC_time = df.index[-1]  # Project to end of data
            
            # Check if correction is valid
            if direction == MarketDirection.BULLISH:
                # For bullish correction: A is down, B is up (not above start), C is down
                if not (w0_is_high and not wA_is_high and wB_is_high and wB_price <= w0_price):
                    continue
            else:
                # For bearish correction: A is up, B is down (not below start), C is up
                if not (not w0_is_high and wA_is_high and not wB_is_high and wB_price >= w0_price):
                    continue
            
            # Looks like a valid correction pattern
            pattern = ElliottWavePattern(
                pattern_name=f"Correction Wave ({direction.value})",
                pattern_type=WaveType.CORRECTION,
                wave_degree=WaveDegree(self.parameters.get("wave_degree", "Intermediate")),
                confidence=ConfidenceLevel.MEDIUM, 
                direction=direction,
                start_time=w0_time,
                end_time=wB_time,  # Only up to B since C is projected
                start_price=w0_price,
                end_price=wB_price,
                waves={
                    WavePosition.A: (wA_time, wA_price),
                    WavePosition.B: (wB_time, wB_price),
                    WavePosition.C: (wC_time, wC_price)
                },
                completion_percentage=66.7  # 2 of 3 waves complete
            )
            
            # Calculate Fibonacci projections for Wave C
            pattern.fibonacci_levels = self._calculate_correction_fibonacci_levels(
                w0_price, wA_price, wB_price, direction
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_impulse_fibonacci_levels(
        self,
        w0_price: float,
        w1_price: float,
        w2_price: float,
        w3_price: float,
        w4_price: float,
        direction: MarketDirection
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci projections for impulse waves
        
        Args:
            w0_price: Price at start (wave 0)
            w1_price: Price at wave 1
            w2_price: Price at wave 2
            w3_price: Price at wave 3
            w4_price: Price at wave 4
            direction: Market direction (bullish or bearish)
            
        Returns:
            Dictionary of Fibonacci levels
        """
        fib_levels = {}
        
        # Calculate wave sizes
        if direction == MarketDirection.BULLISH:
            wave1_size = w1_price - w0_price
            wave3_proj_base = w2_price
            wave5_proj_base = w4_price
            
            # Wave 3 projections from Wave 1
            fib_levels["wave3_0.618"] = wave3_proj_base + 0.618 * wave1_size
            fib_levels["wave3_1.000"] = wave3_proj_base + 1.000 * wave1_size
            fib_levels["wave3_1.618"] = wave3_proj_base + 1.618 * wave1_size
            fib_levels["wave3_2.618"] = wave3_proj_base + 2.618 * wave1_size
            
            # Wave 5 projections from Wave 1
            fib_levels["wave5_0.618"] = wave5_proj_base + 0.618 * wave1_size
            fib_levels["wave5_1.000"] = wave5_proj_base + 1.000 * wave1_size
            fib_levels["wave5_1.618"] = wave5_proj_base + 1.618 * wave1_size
            
            # Wave 5 projections from Wave 1+3
            wave13_size = (w1_price - w0_price) + (w3_price - w2_price)
            fib_levels["wave5_0.382_13"] = wave5_proj_base + 0.382 * wave13_size
            fib_levels["wave5_0.618_13"] = wave5_proj_base + 0.618 * wave13_size
        else:
            wave1_size = w0_price - w1_price
            wave3_proj_base = w2_price
            wave5_proj_base = w4_price
            
            # Wave 3 projections from Wave 1
            fib_levels["wave3_0.618"] = wave3_proj_base - 0.618 * wave1_size
            fib_levels["wave3_1.000"] = wave3_proj_base - 1.000 * wave1_size
            fib_levels["wave3_1.618"] = wave3_proj_base - 1.618 * wave1_size
            fib_levels["wave3_2.618"] = wave3_proj_base - 2.618 * wave1_size
            
            # Wave 5 projections from Wave 1
            fib_levels["wave5_0.618"] = wave5_proj_base - 0.618 * wave1_size
            fib_levels["wave5_1.000"] = wave5_proj_base - 1.000 * wave1_size
            fib_levels["wave5_1.618"] = wave5_proj_base - 1.618 * wave1_size
            
            # Wave 5 projections from Wave 1+3
            wave13_size = (w0_price - w1_price) + (w2_price - w3_price)
            fib_levels["wave5_0.382_13"] = wave5_proj_base - 0.382 * wave13_size
            fib_levels["wave5_0.618_13"] = wave5_proj_base - 0.618 * wave13_size
        
        return fib_levels
    
    def _calculate_correction_fibonacci_levels(
        self,
        w0_price: float,
        wA_price: float,
        wB_price: float,
        direction: MarketDirection
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci projections for corrective waves
        
        Args:
            w0_price: Price at start
            wA_price: Price at wave A
            wB_price: Price at wave B
            direction: Market direction
            
        Returns:
            Dictionary of Fibonacci levels
        """
        fib_levels = {}
        
        # Calculate wave sizes
        if direction == MarketDirection.BULLISH:
            waveA_size = abs(w0_price - wA_price)
            waveC_proj_base = wB_price
            
            # Wave C projections from Wave A
            fib_levels["waveC_0.618"] = waveC_proj_base - 0.618 * waveA_size
            fib_levels["waveC_1.000"] = waveC_proj_base - 1.000 * waveA_size
            fib_levels["waveC_1.272"] = waveC_proj_base - 1.272 * waveA_size
            fib_levels["waveC_1.618"] = waveC_proj_base - 1.618 * waveA_size
            
        else:
            waveA_size = abs(w0_price - wA_price)
            waveC_proj_base = wB_price
            
            # Wave C projections from Wave A
            fib_levels["waveC_0.618"] = waveC_proj_base + 0.618 * waveA_size
            fib_levels["waveC_1.000"] = waveC_proj_base + 1.000 * waveA_size
            fib_levels["waveC_1.272"] = waveC_proj_base + 1.272 * waveA_size
            fib_levels["waveC_1.618"] = waveC_proj_base + 1.618 * waveA_size
        
        return fib_levels
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Elliott Wave Analyzer',
            'description': 'Identifies Elliott Wave patterns and projects price targets',
            'category': 'pattern',
            'parameters': [
                {
                    'name': 'price_column',
                    'description': 'Column to use for price data',
                    'type': 'str',
                    'default': 'close'
                },
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for pattern detection',
                    'type': 'int',
                    'default': 200
                },
                {
                    'name': 'use_zigzag_filter',
                    'description': 'Use zigzag filter to identify waves',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'zigzag_threshold',
                    'description': 'Percentage threshold for zigzag reversals',
                    'type': 'float',
                    'default': 0.05
                },
                {
                    'name': 'wave_degree',
                    'description': 'Default wave degree for analysis',
                    'type': 'str',
                    'default': 'Intermediate',
                    'options': [d.value for d in WaveDegree]
                }
            ]
        }


class NestedElliottWaveAnalyzer(ElliottWaveAnalyzer):
    """
    Nested Elliott Wave Analysis
    
    Extended version of Elliott Wave Analyzer that identifies nested wave patterns
    across multiple degrees, providing more detailed wave counts.
    """
    
    def __init__(
        self,
        name: str = "NestedElliottWaveAnalyzer",
        parameters: Dict[str, Any] = None
    ):
        """Initialize Nested Elliott Wave Analyzer"""
        default_params = {
            "price_column": "close",
            "high_column": "high",
            "low_column": "low",
            "lookback_period": 200,
            "max_wave_count": 5,
            "min_wave_size": 5,
            "use_zigzag_filter": True,
            "zigzag_threshold": 0.05,
            "fibonacci_levels": [0.382, 0.5, 0.618, 1.0, 1.618, 2.618],
            "wave_degree": "Intermediate",
            "analyze_subwaves": True,  # Analyze nested waves
            "max_subwave_depth": 2,    # Maximum nesting level
            "subwave_zigzag_threshold": 0.03  # Lower threshold for subwaves
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def find_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        Find nested Elliott Wave patterns
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of ElliottWavePattern objects with nested subwaves
        """
        # First get the primary wave patterns
        primary_patterns = super().find_patterns(df)
        
        # If not analyzing subwaves, return primary patterns
        if not self.parameters["analyze_subwaves"]:
            return primary_patterns
        
        # Analyze subwaves for each primary pattern
        current_depth = 1
        max_depth = self.parameters["max_subwave_depth"]
        
        enhanced_patterns = []
        
        for pattern in primary_patterns:
            # Get the DataFrame slice for this pattern
            pattern_start_loc = df.index.get_loc(pattern.start_time)
            pattern_end_loc = df.index.get_loc(pattern.end_time)
            pattern_df = df.iloc[pattern_start_loc:pattern_end_loc+1]
            
            # Skip too small patterns
            if len(pattern_df) < self.parameters["min_wave_size"] * 3:
                enhanced_patterns.append(pattern)
                continue
            
            # Analyze subwaves with reduced threshold
            subwave_params = self.parameters.copy()
            subwave_params["zigzag_threshold"] = self.parameters["subwave_zigzag_threshold"]
            
            # Create a subwave analyzer with reduced threshold
            subwave_analyzer = ElliottWaveAnalyzer(
                name=f"SubwaveAnalyzer_{current_depth}",
                parameters=subwave_params
            )
            
            # Find subwave patterns
            subwave_patterns = subwave_analyzer.find_patterns(pattern_df)
            
            # Match subwaves to appropriate positions
            sub_waves = {}
            for sw_pattern in subwave_patterns:
                # Check if this is a subwave of an appropriate position
                if pattern.wave_type == WaveType.IMPULSE:
                    # For impulse patterns, check for subwaves within wave 1, 3, 5
                    for pos in [WavePosition.ONE, WavePosition.THREE, WavePosition.FIVE]:
                        if pos in pattern.waves:
                            pos_time, pos_price = pattern.waves[pos]
                            # If this subwave corresponds to this position
                            if (sw_pattern.start_time >= pattern.start_time and 
                                sw_pattern.end_time <= pos_time):
                                sub_waves[pos] = sw_pattern
                
                elif pattern.wave_type == WaveType.CORRECTION:
                    # For correction patterns, check for subwaves within A and C
                    for pos in [WavePosition.A, WavePosition.C]:
                        if pos in pattern.waves:
                            pos_time, pos_price = pattern.waves[pos]
                            # If this subwave corresponds to this position
                            if (sw_pattern.start_time >= pattern.start_time and 
                                sw_pattern.end_time <= pos_time):
                                sub_waves[pos] = sw_pattern
            
            # Update the pattern with subwave information
            if sub_waves:
                pattern.sub_waves = sub_waves
                pattern.confidence = ConfidenceLevel.HIGH  # Increase confidence when subwaves are identified
            
            enhanced_patterns.append(pattern)
        
        return enhanced_patterns
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate extended Elliott Wave analysis with nested waves
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with nested Elliott Wave analysis
        """
        # First calculate primary waves
        result_df = super().calculate(df)
        
        # If not analyzing subwaves, return primary analysis
        if not self.parameters["analyze_subwaves"]:
            return result_df
            
        # Get patterns with subwaves
        patterns = self.find_patterns(df)
        
        # Add subwave columns
        for position in WavePosition:
            for sub_position in [WavePosition.ONE, WavePosition.TWO, WavePosition.THREE,
                                WavePosition.FOUR, WavePosition.FIVE, 
                                WavePosition.A, WavePosition.B, WavePosition.C]:
                col_name = f'elliott_wave_{position.value}_{sub_position.value}'
                result_df[col_name] = np.nan
        
        # Mark subwaves in DataFrame
        for pattern in patterns:
            if hasattr(pattern, 'sub_waves') and pattern.sub_waves:
                for position, subwave_pattern in pattern.sub_waves.items():
                    for sub_pos, (time, price) in subwave_pattern.waves.items():
                        col_name = f'elliott_wave_{position.value}_{sub_pos.value}'
                        if col_name in result_df.columns and time in result_df.index:
                            result_df.loc[time, col_name] = price
        
        return result_df
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Nested Elliott Wave Analyzer',
            'description': 'Identifies Elliott Wave patterns with nested wave structure',
            'category': 'pattern',
            'parameters': [
                {
                    'name': 'price_column',
                    'description': 'Column to use for price data',
                    'type': 'str',
                    'default': 'close'
                },
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for pattern detection',
                    'type': 'int',
                    'default': 200
                },
                {
                    'name': 'use_zigzag_filter',
                    'description': 'Use zigzag filter to identify waves',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'zigzag_threshold',
                    'description': 'Percentage threshold for zigzag reversals',
                    'type': 'float',
                    'default': 0.05
                },
                {
                    'name': 'analyze_subwaves',
                    'description': 'Analyze nested waves within primary waves',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'max_subwave_depth',
                    'description': 'Maximum nesting level for subwave analysis',
                    'type': 'int',
                    'default': 2
                },
                {
                    'name': 'subwave_zigzag_threshold',
                    'description': 'Percentage threshold for subwave zigzag reversals',
                    'type': 'float',
                    'default': 0.03
                }
            ]
        }


class ElliottWaveAnalyzer(PatternRecognitionBase):
    """
    Elliott Wave Detector
    
    Identifies potential Elliott Wave patterns in price data, including impulse waves,
    corrective waves, and validates them against Elliott Wave rules and guidelines.
    The detector applies multiple filters to identify high-probability Elliott Wave patterns.
    """
    
    def __init__(
        self, 
        name: str = "ElliottWaveAnalyzer",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize Elliott Wave Detector
        
        Args:
            name: Name of the detector
            parameters: Dictionary of parameters for the detector
        """
        default_params = {
            "price_column": "close",
            "high_column": "high",
            "low_column": "low",
            "min_wave_length": 5,  # Minimum number of bars for a valid wave
            "max_wave_retracement": 0.78,  # Maximum retracement for wave 2 (78.6%)
            "min_third_wave_extension": 1.0,  # Wave 3 should be at least 100% of wave 1
            "typical_fifth_wave_ratio": 0.618,  # Typical ratio of wave 5 to wave 1
            "confluence_threshold": 0.05,  # Price distance for swing confluences
            "wave_degree": WaveDegree.INTERMEDIATE.value,  # Default wave degree
            "use_zigzag": True,  # Use zigzag algorithm for swing detection
            "zigzag_threshold": 0.05,  # 5% minimum move for zigzag
            "strict_rules": True,  # Enforce strict Elliott Wave rules
            "alternation_check": True,  # Check for alternation between waves 2 & 4
            "detect_extensions": True,  # Detect extended waves
            "detect_truncations": True,  # Detect truncated 5th waves
            "detect_diagonals": True,  # Detect diagonal patterns
            "confidence_threshold": 0.6  # Minimum confidence for a valid pattern
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def detect(self, df: pd.DataFrame) -> List[ElliottWavePattern]:
        """
        Detect Elliott Wave patterns in the given data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of detected Elliott Wave patterns
        """
        if len(df) < 30:  # Need sufficient data for wave detection
            return []
            
        price_col = self.parameters["price_column"]
        high_col = self.parameters["high_column"]
        low_col = self.parameters["low_column"]
        
        # 1. Detect significant swing points using zigzag or local extrema
        if self.parameters["use_zigzag"]:
            zigzag_threshold = self.parameters["zigzag_threshold"]
            swing_points = self._zigzag_algorithm(df[high_col], df[low_col], zigzag_threshold)
        else:
            swing_points = self._detect_swing_points(df[price_col], window=5)
            
        if len(swing_points) < 6:  # Need at least 6 swing points for a 5-wave pattern
            return []
            
        # 2. Generate candidate wave counts
        candidates = self._generate_wave_candidates(df, swing_points)
        
        # 3. Validate candidates against Elliott Wave rules
        valid_patterns = []
        for candidate in candidates:
            valid, confidence = self._validate_elliott_rules(df, candidate)
            if valid and confidence >= self.parameters["confidence_threshold"]:
                # Create ElliottWavePattern object
                pattern = self._create_pattern(df, candidate, confidence)
                valid_patterns.append(pattern)
        
        # 4. Sort by confidence level and return
        valid_patterns.sort(key=lambda p: p.confidence.value, reverse=True)
        return valid_patterns
    
    def _zigzag_algorithm(self, high: pd.Series, low: pd.Series, threshold: float) -> List[Tuple[int, float, str]]:
        """
        Implement zigzag algorithm to find significant swing points
        
        Args:
            high: Series of high prices
            low: Series of low prices
            threshold: Minimum percentage move for zigzag point
            
        Returns:
            List of (index, price, "high"/"low") tuples
        """
        if high.empty or low.empty:
            return []
            
        # Initialize with first point as a pivot
        swings = []
        is_uptrend = True
        last_extreme_idx = 0
        last_extreme_price = low.iloc[0]
        
        for i in range(1, len(high)):
            current_price = high.iloc[i]
            current_high = high.iloc[i]
            current_low = low.iloc[i]
            
            if is_uptrend:
                # In uptrend, track higher highs
                if current_high > last_extreme_price:
                    # New higher high
                    last_extreme_idx = i
                    last_extreme_price = current_high
                
                # Check for trend reversal (significant drop)
                price_drop = (last_extreme_price - current_low) / last_extreme_price
                if price_drop > threshold:
                    # Trend reversal - record the last extreme as high
                    swings.append((last_extreme_idx, last_extreme_price, "high"))
                    # Start tracking downtrend
                    is_uptrend = False
                    last_extreme_idx = i
                    last_extreme_price = current_low
                    
            else:
                # In downtrend, track lower lows
                if current_low < last_extreme_price:
                    # New lower low
                    last_extreme_idx = i
                    last_extreme_price = current_low
                
                # Check for reversal (significant rise)
                price_rise = (current_high - last_extreme_price) / last_extreme_price
                if price_rise > threshold:
                    # Trend reversal - record the last extreme as low
                    swings.append((last_extreme_idx, last_extreme_price, "low"))
                    # Start tracking uptrend
                    is_uptrend = True
                    last_extreme_idx = i
                    last_extreme_price = current_high
        
        # Add the last extreme point
        if len(swings) > 0:
            last_type = "low" if swings[-1][2] == "high" else "high"
            swings.append((last_extreme_idx, last_extreme_price, last_type))
            
        return swings
    
    def _detect_swing_points(self, prices: pd.Series, window: int = 5) -> List[Tuple[int, float, str]]:
        """
        Detect swing highs and lows using local extrema
        
        Args:
            prices: Series of price data
            window: Window size for local extrema detection
            
        Returns:
            List of (index, price, "high"/"low") tuples
        """
        swing_points = []
        
        for i in range(window, len(prices) - window):
            window_left = prices.iloc[i - window:i]
            window_right = prices.iloc[i + 1:i + window + 1]
            current_price = prices.iloc[i]
            
            # Check for swing high
            if current_price > max(window_left.max(), window_right.max()):
                swing_points.append((i, current_price, "high"))
                
            # Check for swing low
            if current_price < min(window_left.min(), window_right.min()):
                swing_points.append((i, current_price, "low"))
        
        return sorted(swing_points, key=lambda x: x[0])
    
    def _generate_wave_candidates(self, df: pd.DataFrame, swing_points: List[Tuple[int, float, str]]
                               ) -> List[Dict[WavePosition, Tuple[int, float]]]:
        """
        Generate candidate wave counts from swing points
        
        Args:
            df: DataFrame with OHLCV data
            swing_points: List of swing points
            
        Returns:
            List of candidate wave counts
        """
        candidates = []
        
        # Need at least 9 swing points for a complete 5-wave structure (5 waves + 4 connecting points)
        if len(swing_points) < 9:
            return candidates
        
        # The algorithm tests different combinations of swing points to find valid Elliott Wave counts
        for i in range(len(swing_points) - 8):
            # For each starting point, try to find a 5-wave pattern
            for j in range(i + 1, len(swing_points) - 7):
                # Ensure correct alternation of highs and lows
                if swing_points[i][2] == swing_points[j][2]:
                    continue
                    
                wave1_start = (swing_points[i][0], swing_points[i][1])
                wave1_end = (swing_points[j][0], swing_points[j][1])
                
                # Try different possibilities for wave 2
                for k in range(j + 1, len(swing_points) - 5):
                    if swing_points[k][2] == swing_points[j][2]:
                        continue
                        
                    wave2_end = (swing_points[k][0], swing_points[k][1])
                    
                    # Check wave 2 doesn't go below start
                    wave1_range = abs(wave1_end[1] - wave1_start[1])
                    wave2_retracement = abs(wave2_end[1] - wave1_end[1]) / wave1_range
                    
                    if wave2_retracement > self.parameters["max_wave_retracement"]:
                        continue
                    
                    # Try different possibilities for wave 3
                    for l in range(k + 1, len(swing_points) - 3):
                        if swing_points[l][2] == swing_points[k][2]:
                            continue
                            
                        wave3_end = (swing_points[l][0], swing_points[l][1])
                        
                        # Wave 3 should be in same direction as wave 1
                        if ((wave1_end[1] > wave1_start[1]) != (wave3_end[1] > wave2_end[1])):
                            continue
                            
                        # Wave 3 should not be the shortest among waves 1, 3, 5
                        wave3_range = abs(wave3_end[1] - wave2_end[1])
                        if wave3_range < wave1_range:
                            # Wave 3 is shorter than wave 1 - could be valid, but less common
                            pass
                        
                        # Try different possibilities for wave 4
                        for m in range(l + 1, len(swing_points) - 1):
                            if swing_points[m][2] == swing_points[l][2]:
                                continue
                                
                            wave4_end = (swing_points[m][0], swing_points[m][1])
                            
                            # Wave 4 should not overlap wave 1
                            if ((wave1_end[1] > wave1_start[1]) and (wave4_end[1] < wave1_end[1])) or \
                               ((wave1_end[1] < wave1_start[1]) and (wave4_end[1] > wave1_end[1])):
                                continue
                                
                            # Try different possibilities for wave 5
                            for n in range(m + 1, len(swing_points)):
                                if swing_points[n][2] == swing_points[m][2]:
                                    continue
                                    
                                wave5_end = (swing_points[n][0], swing_points[n][1])
                                
                                # Wave 5 should be in same direction as waves 1 and 3
                                if ((wave1_end[1] > wave1_start[1]) != (wave5_end[1] > wave4_end[1])):
                                    continue
                                    
                                # Create candidate wave count
                                candidate = {
                                    WavePosition.ONE: (swing_points[i][0], swing_points[i][1], swing_points[j][0], swing_points[j][1]),
                                    WavePosition.TWO: (swing_points[j][0], swing_points[j][1], swing_points[k][0], swing_points[k][1]),
                                    WavePosition.THREE: (swing_points[k][0], swing_points[k][1], swing_points[l][0], swing_points[l][1]),
                                    WavePosition.FOUR: (swing_points[l][0], swing_points[l][1], swing_points[m][0], swing_points[m][1]),
                                    WavePosition.FIVE: (swing_points[m][0], swing_points[m][1], swing_points[n][0], swing_points[n][1])
                                }
                                
                                candidates.append(candidate)
        
        return candidates
    
    def _validate_elliott_rules(self, df: pd.DataFrame, candidate: Dict[WavePosition, Tuple[int, float, int, float]]
                           ) -> Tuple[bool, float]:
        """
        Validate a candidate wave count against Elliott Wave rules
        
        Args:
            df: DataFrame with OHLCV data
            candidate: Candidate wave count
            
        Returns:
            Tuple of (is_valid, confidence)
        """
        # Initialize confidence score
        confidence_score = 1.0
        price_col = self.parameters["price_column"]
        
        # Wave 1: Extracting data
        wave1_start_idx, wave1_start_price, wave1_end_idx, wave1_end_price = candidate[WavePosition.ONE]
        wave2_start_idx, wave2_start_price, wave2_end_idx, wave2_end_price = candidate[WavePosition.TWO]
        wave3_start_idx, wave3_start_price, wave3_end_idx, wave3_end_price = candidate[WavePosition.THREE]
        wave4_start_idx, wave4_start_price, wave4_end_idx, wave4_end_price = candidate[WavePosition.FOUR]
        wave5_start_idx, wave5_start_price, wave5_end_idx, wave5_end_price = candidate[WavePosition.FIVE]
        
        # Calculate wave lengths (in price units)
        wave1_length = abs(wave1_end_price - wave1_start_price)
        wave2_length = abs(wave2_end_price - wave2_start_price)
        wave3_length = abs(wave3_end_price - wave3_start_price)
        wave4_length = abs(wave4_end_price - wave4_start_price)
        wave5_length = abs(wave5_end_price - wave5_start_price)
        
        # Calculate wave durations (in bars)
        wave1_duration = wave1_end_idx - wave1_start_idx
        wave2_duration = wave2_end_idx - wave2_start_idx
        wave3_duration = wave3_end_idx - wave3_start_idx
        wave4_duration = wave4_end_idx - wave4_start_idx
        wave5_duration = wave5_end_idx - wave5_start_idx
        
        # Check if uptrend or downtrend
        is_uptrend = wave1_end_price > wave1_start_price
        
        # Rule 1: Wave 2 should not retrace more than 100% of wave 1
        wave2_retracement = wave2_length / wave1_length
        if wave2_retracement > 1.0:
            return False, 0.0
        
        # Rule 2: Wave 3 should never be the shortest of waves 1, 3 and 5
        if wave3_length < wave1_length and wave3_length < wave5_length:
            return False, 0.0
            
        # Rule 3: Wave 4 should not overlap wave 1, except in diagonal patterns
        if is_uptrend:
            if wave4_end_price < wave1_end_price and not self._check_if_diagonal(candidate):
                return False, 0.0
        else:
            if wave4_end_price > wave1_end_price and not self._check_if_diagonal(candidate):
                return False, 0.0
                
        # Rule 4: Minimum length requirements for each wave
        min_wave_length = self.parameters["min_wave_length"]
        if (wave1_duration < min_wave_length or
            wave2_duration < min_wave_length or
            wave3_duration < min_wave_length or
            wave4_duration < min_wave_length or
            wave5_duration < min_wave_length):
            confidence_score *= 0.7  # Reduce confidence if waves are short
            
        # Guideline 1: Wave 3 is often extended (longest and strongest)
        if not (wave3_length > wave1_length and wave3_length > wave5_length):
            confidence_score *= 0.9
            
        # Guideline 2: Wave relationships - common Fibonacci ratios
        # Wave 3 is often 1.618 * wave 1
        ideal_wave3 = wave1_length * 1.618
        wave3_ratio_diff = abs(wave3_length - ideal_wave3) / ideal_wave3
        if wave3_ratio_diff > 0.2:  # Allow 20% tolerance
            confidence_score *= 0.9
            
        # Wave 5 is often 0.618 * wave 1
        ideal_wave5 = wave1_length * 0.618
        wave5_ratio_diff = abs(wave5_length - ideal_wave5) / ideal_wave5
        if wave5_ratio_diff > 0.3:  # Allow 30% tolerance
            confidence_score *= 0.9
            
        # Guideline 3: Alternation between waves 2 and 4
        if self.parameters["alternation_check"]:
            # Check if wave 2 and wave 4 are different in structure (sharp vs flat)
            wave2_sharpness = self._calculate_wave_sharpness(df, wave2_start_idx, wave2_end_idx)
            wave4_sharpness = self._calculate_wave_sharpness(df, wave4_start_idx, wave4_end_idx)
            
            if abs(wave2_sharpness - wave4_sharpness) < 0.2:  # Not much alternation
                confidence_score *= 0.8
        
        # Adjust final confidence based on pattern complexity
        if self._check_if_extended(candidate):
            confidence_score *= 1.1  # Bonus for detecting extended waves
            
        if self._check_if_diagonal(candidate):
            confidence_score *= 0.9  # Slightly reduce confidence for diagonal patterns
            
        # Cap confidence at 1.0
        confidence_score = min(confidence_score, 1.0)
            
        return True, confidence_score
    
    def _check_if_extended(self, candidate: Dict[WavePosition, Tuple[int, float, int, float]]) -> bool:
        """Check if any wave is extended (significantly larger than the others)"""
        if not self.parameters["detect_extensions"]:
            return False
            
        wave1_start_idx, wave1_start_price, wave1_end_idx, wave1_end_price = candidate[WavePosition.ONE]
        wave3_start_idx, wave3_start_price, wave3_end_idx, wave3_end_price = candidate[WavePosition.THREE]
        wave5_start_idx, wave5_start_price, wave5_end_idx, wave5_end_price = candidate[WavePosition.FIVE]
        
        wave1_length = abs(wave1_end_price - wave1_start_price)
        wave3_length = abs(wave3_end_price - wave3_start_price)
        wave5_length = abs(wave5_end_price - wave5_start_price)
        
        # Extended wave is typically at least 1.618 times the next largest wave
        if wave3_length > (1.618 * max(wave1_length, wave5_length)):
            return True
            
        if wave5_length > (1.618 * max(wave1_length, wave3_length)):
            return True
            
        if wave1_length > (1.618 * max(wave3_length, wave5_length)):
            return True
        
        return False
    
    def _check_if_diagonal(self, candidate: Dict[WavePosition, Tuple[int, float, int, float]]) -> bool:
        """Check if the pattern forms a diagonal (wedge-shaped) pattern"""
        if not self.parameters["detect_diagonals"]:
            return False
            
        # Extract wave end points
        wave1_end_idx, wave1_end_price = candidate[WavePosition.ONE][2:4]
        wave2_end_idx, wave2_end_price = candidate[WavePosition.TWO][2:4]
        wave3_end_idx, wave3_end_price = candidate[WavePosition.THREE][2:4]
        wave4_end_idx, wave4_end_price = candidate[WavePosition.FOUR][2:4]
        wave5_end_idx, wave5_end_price = candidate[WavePosition.FIVE][2:4]
        
        # Calculate slopes of trendlines connecting wave 1-3-5 and 2-4
        try:
            # Upper trendline: connects waves 1, 3, 5 ends
            upper_slope1 = (wave3_end_price - wave1_end_price) / (wave3_end_idx - wave1_end_idx)
            upper_slope2 = (wave5_end_price - wave3_end_price) / (wave5_end_idx - wave3_end_idx)
            
            # Lower trendline: connects waves 2, 4 ends
            lower_slope = (wave4_end_price - wave2_end_price) / (wave4_end_idx - wave2_end_idx)
            
            # In a diagonal, the slopes should be in the same direction (converging)
            if (upper_slope1 > 0 and upper_slope2 > 0 and lower_slope > 0) or \
               (upper_slope1 < 0 and upper_slope2 < 0 and lower_slope < 0):
                
                # Check for convergence: slopes should be getting less steep
                if abs(upper_slope2) < abs(upper_slope1):
                    return True
        except:
            pass
            
        return False
    
    def _calculate_wave_sharpness(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """
        Calculate the sharpness of a wave by measuring its direct route vs actual route
        
        Args:
            df: DataFrame with price data
            start_idx: Starting index of the wave
            end_idx: Ending index of the wave
            
        Returns:
            Sharpness value (higher = sharper wave)
        """
        if start_idx >= end_idx:
            return 0.0
            
        # Extract price data for the wave
        price_col = self.parameters["price_column"]
        wave_slice = df[price_col].iloc[start_idx:end_idx+1]
        
        # Calculate direct distance (start point to end point)
        direct_distance = abs(wave_slice.iloc[-1] - wave_slice.iloc[0])
        
        # Calculate actual traveled distance (sum of all moves)
        traveled_distance = np.sum(np.abs(wave_slice.diff().dropna()))
        
        # Sharpness = direct / traveled (higher value means more direct/sharp move)
        if traveled_distance > 0:
            return direct_distance / traveled_distance
        else:
            return 0.0
    
    def _create_pattern(self, df: pd.DataFrame, candidate: Dict[WavePosition, Tuple[int, float, int, float]],
                      confidence: float) -> ElliottWavePattern:
        """
        Create an ElliottWavePattern from a validated candidate
        
        Args:
            df: DataFrame with OHLCV data
            candidate: Validated wave candidate
            confidence: Confidence score
            
        Returns:
            ElliottWavePattern object
        """
        # Determine wave type (impulse, diagonal, etc.)
        wave_type = WaveType.IMPULSE
        if self._check_if_diagonal(candidate):
            wave_type = WaveType.DIAGONAL
            
        # Map confidence score to ConfidenceLevel enum
        confidence_level = ConfidenceLevel.MEDIUM
        if confidence >= 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
            
        # Extract wave data
        waves = {}
        for position, (start_idx, start_price, end_idx, end_price) in candidate.items():
            # Convert index to datetime (if df.index is a DatetimeIndex)
            if isinstance(df.index, pd.DatetimeIndex):
                start_time = df.index[start_idx]
                end_time = df.index[end_idx]
            else:
                start_time = start_idx
                end_time = end_idx
                
            waves[position] = (end_time, end_price)
            
        # Determine trend direction
        first_wave_start_price = candidate[WavePosition.ONE][1]
        last_wave_end_price = candidate[WavePosition.FIVE][3]
        trend = MarketDirection.BULLISH if last_wave_end_price > first_wave_start_price else MarketDirection.BEARISH
        
        # Calculate Fibonacci projection levels for potential wave targets
        fib_levels = self._calculate_fibonacci_levels(candidate, trend)
        
        # Create pattern result
        pattern = ElliottWavePattern(
            pattern_name="Elliott 5-Wave",
            start_idx=candidate[WavePosition.ONE][0],
            end_idx=candidate[WavePosition.FIVE][2],
            confidence=confidence_level,
            direction=trend,
            wave_type=wave_type,
            wave_degree=WaveDegree(self.parameters["wave_degree"]),
            waves=waves,
            fibonacci_levels=fib_levels,
            completion_percentage=100.0  # Assuming complete pattern
        )
        
        return pattern
    
    def _calculate_fibonacci_levels(self, candidate: Dict[WavePosition, Tuple[int, float, int, float]],
                                 trend: MarketDirection) -> Dict[str, float]:
        """Calculate Fibonacci projection levels for potential future price targets"""
        wave1_start_price = candidate[WavePosition.ONE][1]
        wave1_end_price = candidate[WavePosition.ONE][3]
        wave1_length = abs(wave1_end_price - wave1_start_price)
        
        last_wave_end_price = candidate[WavePosition.FIVE][3]
        
        fib_levels = {}
        
        # Direction of projection
        direction = 1 if trend == MarketDirection.BULLISH else -1
        
        # Common Fibonacci extension levels
        extensions = [1.0, 1.618, 2.0, 2.618]
        for ext in extensions:
            projection = last_wave_end_price + (direction * wave1_length * ext)
            fib_levels[f"extension_{ext}"] = projection
        
        return fib_levels


class ElliottWaveCounter(AdvancedAnalysisBase):
    """
    Elliott Wave Counter
    
    Identifies and labels the current wave count in the market based on Elliott Wave Principle.
    This indicator helps traders understand which wave of the Elliott cycle the market is 
    currently in, providing context for trading decisions.
    """
    
    def __init__(
        self, 
        name: str = "ElliottWaveCounter",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize Elliott Wave Counter
        
        Args:
            name: Name of the counter
            parameters: Dictionary of parameters for the counter
        """
        default_params = {
            "price_column": "close",
            "high_column": "high",
            "low_column": "low",
            "use_zigzag": True,  # Use zigzag for swing detection
            "zigzag_threshold": 0.03,  # 3% minimum move for zigzag
            "max_wave_count": 3,  # Maximum number of wave counts to return
            "lookback_period": 200,  # Number of bars to consider
            "wave_degrees": [WaveDegree.MINOR.value, WaveDegree.MINUTE.value],  # Wave degrees to analyze
            "min_confidence": 0.5,  # Minimum confidence for valid wave count
            "validate_rules": True,  # Validate against Elliott Wave rules
            "projection_bars": 20,  # Number of bars to project into the future
            "detect_nested_waves": True  # Detect waves within waves
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Elliott Wave count and labeling
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Elliott Wave counts and labels
        """
        result_df = df.copy()
        
        # Check if we have enough data
        if len(df) < 30:
            return result_df
            
        # Use the most recent lookback_period bars
        lookback = min(self.parameters["lookback_period"], len(df))
        analysis_df = df.iloc[-lookback:]
        
        # Detect wave counts using ElliottWaveAnalyzer
        detector = ElliottWaveAnalyzer(parameters={
            "price_column": self.parameters["price_column"],
            "high_column": self.parameters["high_column"],
            "low_column": self.parameters["low_column"],
            "use_zigzag": self.parameters["use_zigzag"],
            "zigzag_threshold": self.parameters["zigzag_threshold"],
            "confidence_threshold": self.parameters["min_confidence"]
        })
        
        wave_patterns = detector.detect(analysis_df)
        
        if not wave_patterns:
            return result_df
            
        # Get the most confident wave pattern
        wave_pattern = wave_patterns[0]
        
        # Add wave labels to the dataframe
        result_df = self._add_wave_labels(result_df, wave_pattern)
        
        # Add wave count confidence
        result_df["ew_count_confidence"] = wave_pattern.confidence.value
        
        # Add current wave position (which wave we're in)
        result_df["ew_current_wave"] = self._determine_current_wave(wave_pattern)
        
        # Add wave degree
        result_df["ew_wave_degree"] = wave_pattern.wave_degree.value
        
        # Add projected targets
        projections = self._calculate_projections(wave_pattern)
        for target, value in projections.items():
            result_df[f"ew_target_{target}"] = value
        
        return result_df
        
    def _add_wave_labels(self, df: pd.DataFrame, wave_pattern: ElliottWavePattern) -> pd.DataFrame:
        """
        Add wave labels to the dataframe
        
        Args:
            df: DataFrame to add labels to
            wave_pattern: ElliottWavePattern object
            
        Returns:
            DataFrame with wave labels
        """
        # Initialize wave label column
        df["ew_wave_label"] = None
        
        # For each wave, label the end points
        for position, (time, price) in wave_pattern.waves.items():
            # Find the closest index for this time
            if isinstance(time, (pd.Timestamp, datetime)):
                try:
                    idx = df.index.get_indexer([time], method='nearest')[0]
                    df.loc[df.index[idx], "ew_wave_label"] = position.value
                except Exception as e:
                    # Log error but continue
                    logging.warning(f"Error labeling wave {position}: {str(e)}")
        
        return df
        
    def _determine_current_wave(self, wave_pattern: ElliottWavePattern) -> str:
        """
        Determine which wave the market is currently in
        
        Args:
            wave_pattern: ElliottWavePattern object
            
        Returns:
            String representing current wave position
        """
        # Get the highest wave number detected
        wave_positions = list(wave_pattern.waves.keys())
        if not wave_positions:
            return "unknown"
            
        # Sort waves in sequence order
        try:
            if wave_positions[0] in [WavePosition.ONE, WavePosition.TWO, WavePosition.THREE, 
                                  WavePosition.FOUR, WavePosition.FIVE]:
                # Impulse wave
                ordered_positions = [WavePosition.ONE, WavePosition.TWO, WavePosition.THREE, 
                                   WavePosition.FOUR, WavePosition.FIVE]
            else:
                # Corrective wave
                ordered_positions = [WavePosition.A, WavePosition.B, WavePosition.C]
                
            current_wave = None
            for pos in ordered_positions:
                if pos in wave_pattern.waves:
                    current_wave = pos
                    
            # If we found all 5 waves, we may be in a new cycle
            if current_wave == WavePosition.FIVE:
                # Check pattern completion percentage
                if wave_pattern.completion_percentage >= 100.0:
                    # Complete pattern may indicate start of corrective wave
                    return WavePosition.A.value
            
            return current_wave.value if current_wave else "unknown"
        except Exception as e:
            logging.warning(f"Error determining current wave: {str(e)}")
            return "unknown"
            
    def _calculate_projections(self, wave_pattern: ElliottWavePattern) -> Dict[str, float]:
        """
        Calculate price projections for future waves
        
        Args:
            wave_pattern: ElliottWavePattern object
            
        Returns:
            Dictionary of projection names and values
        """
        projections = {}
        
        # Get existing Fibonacci projections from the pattern if available
        if hasattr(wave_pattern, "fibonacci_levels") and wave_pattern.fibonacci_levels:
            projections.update(wave_pattern.fibonacci_levels)
            return projections
            
        # Calculate projections manually
        if wave_pattern.wave_type == WaveType.IMPULSE:
            # For impulse patterns
            if (WavePosition.ONE in wave_pattern.waves and 
                WavePosition.TWO in wave_pattern.waves and 
                WavePosition.THREE not in wave_pattern.waves):
                # Projecting wave 3 target
                w1_start_time, w1_start_price = next(iter(wave_pattern.waves.values()))
                w1_end_time, w1_end_price = wave_pattern.waves[WavePosition.ONE]
                w2_end_time, w2_end_price = wave_pattern.waves[WavePosition.TWO]
                
                w1_range = abs(w1_end_price - w1_start_price)
                
                # Common wave 3 targets based on wave 1
                # Wave 3 is commonly 1.618, 2.618, or 4.236 times wave 1
                direction = 1 if w1_end_price > w1_start_price else -1
                
                projections["wave3_min"] = w2_end_price + (direction * w1_range)
                projections["wave3_161"] = w2_end_price + (direction * w1_range * 1.618)
                projections["wave3_261"] = w2_end_price + (direction * w1_range * 2.618)
                
        elif wave_pattern.wave_type == WaveType.CORRECTION:
            # For corrective patterns
            if (WavePosition.A in wave_pattern.waves and 
                WavePosition.B in wave_pattern.waves and 
                WavePosition.C not in wave_pattern.waves):
                # Projecting wave C target
                wa_start_time, wa_start_price = next(iter(wave_pattern.waves.values()))
                wa_end_time, wa_end_price = wave_pattern.waves[WavePosition.A]
                wb_end_time, wb_end_price = wave_pattern.waves[WavePosition.B]
                
                wa_range = abs(wa_end_price - wa_start_price)
                
                # Wave C is commonly equal to wave A, or 1.618 times wave A
                direction = -1 if wa_end_price > wa_start_price else 1
                
                projections["waveC_100"] = wb_end_price + (direction * wa_range)
                projections["waveC_161"] = wb_end_price + (direction * wa_range * 1.618)
                
        return projections