"""
Elliott Wave Analyzer Module.

This module provides the ElliottWaveAnalyzer class for detecting Elliott Wave patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime

from analysis_engine.analysis.advanced_ta.base import (
    PatternRecognitionBase,
    PatternResult,
    ConfidenceLevel,
    MarketDirection
)
from analysis_engine.analysis.advanced_ta.elliott_wave.models import (
    WaveType, WavePosition, WaveDegree
)
from analysis_engine.analysis.advanced_ta.elliott_wave.pattern import ElliottWavePattern
from analysis_engine.analysis.advanced_ta.elliott_wave.utils import (
    detect_zigzag_points, detect_swing_points, calculate_wave_sharpness
)
from analysis_engine.analysis.advanced_ta.elliott_wave.fibonacci import (
    calculate_impulse_fibonacci_levels, calculate_correction_fibonacci_levels, calculate_fibonacci_levels
)
from analysis_engine.analysis.advanced_ta.elliott_wave.validators import (
    validate_elliott_rules, check_if_extended, check_if_diagonal, map_confidence_to_level
)


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
        zigzag_points = detect_zigzag_points(
            analysis_df, 
            self.parameters["price_column"],
            self.parameters["high_column"],
            self.parameters["low_column"],
            self.parameters["zigzag_threshold"]
        )
        
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
                pattern.fibonacci_levels = calculate_impulse_fibonacci_levels(
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
                pattern.fibonacci_levels = calculate_impulse_fibonacci_levels(
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
            pattern.fibonacci_levels = calculate_correction_fibonacci_levels(
                w0_price, wA_price, wB_price, direction
            )
            
            patterns.append(pattern)
        
        return patterns
    
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