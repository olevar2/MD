"""
Elliott Wave Counter Module.

This module provides the ElliottWaveCounter class for identifying and labeling
the current wave count in the market.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from analysis_engine.analysis.advanced_ta.base import (
    AdvancedAnalysisBase,
    ConfidenceLevel,
    MarketDirection
)
from analysis_engine.analysis.advanced_ta.elliott_wave.models import (
    WaveType, WavePosition, WaveDegree
)
from analysis_engine.analysis.advanced_ta.elliott_wave.pattern import ElliottWavePattern
from analysis_engine.analysis.advanced_ta.elliott_wave.analyzer import ElliottWaveAnalyzer


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
            "use_zigzag_filter": self.parameters["use_zigzag"],
            "zigzag_threshold": self.parameters["zigzag_threshold"],
            "confidence_threshold": self.parameters["min_confidence"]
        })
        
        wave_patterns = detector.find_patterns(analysis_df)
        
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