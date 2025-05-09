"""
Flag and Pennant Pattern Recognition Module.

This module provides functionality to detect flag and pennant patterns in price data.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from feature_store_service.indicators.chart_patterns.base import PatternRecognitionBase


class FlagPennantPattern(PatternRecognitionBase):
    """
    Flag and Pennant Pattern Detector.
    
    Identifies flag and pennant patterns in price data.
    """
    
    def __init__(
        self, 
        lookback_period: int = 100,
        min_pattern_size: int = 10,
        max_pattern_size: int = 50,
        sensitivity: float = 0.75,
        **kwargs
    ):
        """
        Initialize Flag and Pennant Pattern Detector.
        
        Args:
            lookback_period: Number of bars to look back for pattern recognition
            min_pattern_size: Minimum size of patterns to recognize (in bars)
            max_pattern_size: Maximum size of patterns to recognize (in bars)
            sensitivity: Sensitivity of pattern detection (0.0-1.0)
            **kwargs: Additional parameters
        """
        super().__init__(
            lookback_period=lookback_period,
            min_pattern_size=min_pattern_size,
            max_pattern_size=max_pattern_size,
            sensitivity=sensitivity,
            **kwargs
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate flag and pennant pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pattern recognition values
        """
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Initialize pattern columns with zeros
        result["pattern_flag"] = 0
        result["pattern_pennant"] = 0
        
        # Find flag and pennant patterns
        result = self._find_flag_pennant_patterns(result)
        
        return result
    
    def _find_flag_pennant_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Find Flag and Pennant patterns. Requires 'volume' column.
        
        Args:
            data: DataFrame with price data (must include 'volume')
        
        Returns:
            DataFrame with flag/pennant pattern recognition markers
        """
        result = data.copy()
        
        if 'volume' not in result.columns:
            # If volume is missing, cannot reliably detect flags/pennants
            # logger.warning("Volume column missing, cannot detect Flag/Pennant patterns.")
            return result 

        # Initialize pattern columns if not already done
        for pattern in ["flag", "pennant"]:
            if f"pattern_{pattern}" not in result.columns:
                result[f"pattern_{pattern}"] = 0
                
        # We need at least min_pattern_size bars for a valid pattern
        if len(result) < self.min_pattern_size:
            return result
            
        # Parameters
        flagpole_min_bars = max(3, int(self.min_pattern_size * 0.2)) # Min bars for flagpole
        flagpole_min_change_pct = 0.03 * max(self.sensitivity, 0.1) # Min price change for flagpole
        consolidation_min_bars = max(5, int(self.min_pattern_size * 0.5)) # Min bars for consolidation
        consolidation_max_bars = self.max_pattern_size - flagpole_min_bars
        parallel_threshold = 0.005 * (1 / max(self.sensitivity, 0.1)) # Threshold for parallel lines (flag)
        converging_threshold = 0.005 * max(self.sensitivity, 0.1) # Threshold for converging lines (pennant)
        volume_decrease_factor = 0.5 # Volume should decrease during consolidation

        # Iterate through potential pattern end points
        for end_loc in range(self.min_pattern_size, len(result)):
            # Look back for potential flagpole + consolidation
            max_lookback = flagpole_min_bars + consolidation_max_bars
            start_loc = max(0, end_loc - max_lookback)
            lookback_data = result.iloc[start_loc:end_loc]

            if len(lookback_data) < self.min_pattern_size: continue

            # 1. Identify potential flagpole
            for fp_end_loc_rel in range(flagpole_min_bars, len(lookback_data) - consolidation_min_bars):
                fp_start_loc_rel = 0
                flagpole_data = lookback_data.iloc[fp_start_loc_rel:fp_end_loc_rel]
                
                if len(flagpole_data) < flagpole_min_bars: continue

                fp_start_price = flagpole_data['close'].iloc[0]
                fp_end_price = flagpole_data['close'].iloc[-1]
                if fp_start_price == 0: continue

                price_change_pct = (fp_end_price - fp_start_price) / fp_start_price
                is_strong_move = abs(price_change_pct) >= flagpole_min_change_pct
                
                if not is_strong_move: continue

                is_uptrend_pole = price_change_pct > 0
                
                # 2. Look for consolidation after the flagpole
                cons_start_loc_rel = fp_end_loc_rel
                cons_end_loc_rel = len(lookback_data) # End of current lookback window
                consolidation_data = lookback_data.iloc[cons_start_loc_rel:cons_end_loc_rel]

                if len(consolidation_data) < consolidation_min_bars: continue

                # Check volume decrease during consolidation
                avg_volume_pole = flagpole_data['volume'].mean()
                avg_volume_cons = consolidation_data['volume'].mean()
                if avg_volume_pole == 0 or avg_volume_cons / avg_volume_pole > volume_decrease_factor:
                    continue # Volume did not decrease significantly

                # Calculate upper and lower trendlines of consolidation using relative locations
                highs = consolidation_data['high']
                lows = consolidation_data['low']
                x_range = np.arange(len(consolidation_data))

                try:
                    high_slope, high_intercept = np.polyfit(x_range, highs, 1)
                    low_slope, low_intercept = np.polyfit(x_range, lows, 1)
                except (np.linalg.LinAlgError, ValueError):
                    continue # Could not fit lines

                # Check for Flag: Parallel trendlines, sloping against the trend
                is_parallel = abs(high_slope - low_slope) < parallel_threshold
                is_counter_trend = (is_uptrend_pole and high_slope < -parallel_threshold / 2) or \
                                   (not is_uptrend_pole and high_slope > parallel_threshold / 2)
                is_flag = is_parallel and is_counter_trend

                # Check for Pennant: Converging trendlines
                is_converging = (high_slope < -converging_threshold and low_slope > converging_threshold) or \
                                (high_slope > converging_threshold and low_slope < -converging_threshold) # Allow both directions
                is_pennant = is_converging

                pattern_type = None
                if is_flag:
                    pattern_type = "flag"
                elif is_pennant:
                    pattern_type = "pennant"

                if pattern_type:
                    # Mark pattern using absolute locations
                    pattern_start_loc_abs = start_loc + fp_start_loc_rel
                    pattern_end_loc_abs = start_loc + cons_end_loc_rel - 1 # End loc is inclusive

                    result.iloc[pattern_start_loc_abs:pattern_end_loc_abs + 1, result.columns.get_loc(f"pattern_{pattern_type}")] = 1
                    # Potentially break here if we only want the first match ending at end_loc
                    # break # Found a pattern ending here

        return result
