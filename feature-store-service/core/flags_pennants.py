"""
Flag and Pennant Pattern Recognition Module.

This module provides functionality to detect flag and pennant patterns in price data.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from core.base_1 import PatternRecognitionBase


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FlagPennantPattern(PatternRecognitionBase):
    """
    Flag and Pennant Pattern Detector.
    
    Identifies flag and pennant patterns in price data.
    """

    def __init__(self, lookback_period: int=100, min_pattern_size: int=10,
        max_pattern_size: int=50, sensitivity: float=0.75, **kwargs):
        """
        Initialize Flag and Pennant Pattern Detector.
        
        Args:
            lookback_period: Number of bars to look back for pattern recognition
            min_pattern_size: Minimum size of patterns to recognize (in bars)
            max_pattern_size: Maximum size of patterns to recognize (in bars)
            sensitivity: Sensitivity of pattern detection (0.0-1.0)
            **kwargs: Additional parameters
        """
        super().__init__(lookback_period=lookback_period, min_pattern_size=
            min_pattern_size, max_pattern_size=max_pattern_size,
            sensitivity=sensitivity, **kwargs)

    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
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
        result = data.copy()
        result['pattern_flag'] = 0
        result['pattern_pennant'] = 0
        result = self._find_flag_pennant_patterns(result)
        return result

    @with_exception_handling
    def _find_flag_pennant_patterns(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Find Flag and Pennant patterns. Requires 'volume' column.
        
        Args:
            data: DataFrame with price data (must include 'volume')
        
        Returns:
            DataFrame with flag/pennant pattern recognition markers
        """
        result = data.copy()
        if 'volume' not in result.columns:
            return result
        for pattern in ['flag', 'pennant']:
            if f'pattern_{pattern}' not in result.columns:
                result[f'pattern_{pattern}'] = 0
        if len(result) < self.min_pattern_size:
            return result
        flagpole_min_bars = max(3, int(self.min_pattern_size * 0.2))
        flagpole_min_change_pct = 0.03 * max(self.sensitivity, 0.1)
        consolidation_min_bars = max(5, int(self.min_pattern_size * 0.5))
        consolidation_max_bars = self.max_pattern_size - flagpole_min_bars
        parallel_threshold = 0.005 * (1 / max(self.sensitivity, 0.1))
        converging_threshold = 0.005 * max(self.sensitivity, 0.1)
        volume_decrease_factor = 0.5
        for end_loc in range(self.min_pattern_size, len(result)):
            max_lookback = flagpole_min_bars + consolidation_max_bars
            start_loc = max(0, end_loc - max_lookback)
            lookback_data = result.iloc[start_loc:end_loc]
            if len(lookback_data) < self.min_pattern_size:
                continue
            for fp_end_loc_rel in range(flagpole_min_bars, len(
                lookback_data) - consolidation_min_bars):
                fp_start_loc_rel = 0
                flagpole_data = lookback_data.iloc[fp_start_loc_rel:
                    fp_end_loc_rel]
                if len(flagpole_data) < flagpole_min_bars:
                    continue
                fp_start_price = flagpole_data['close'].iloc[0]
                fp_end_price = flagpole_data['close'].iloc[-1]
                if fp_start_price == 0:
                    continue
                price_change_pct = (fp_end_price - fp_start_price
                    ) / fp_start_price
                is_strong_move = abs(price_change_pct
                    ) >= flagpole_min_change_pct
                if not is_strong_move:
                    continue
                is_uptrend_pole = price_change_pct > 0
                cons_start_loc_rel = fp_end_loc_rel
                cons_end_loc_rel = len(lookback_data)
                consolidation_data = lookback_data.iloc[cons_start_loc_rel:
                    cons_end_loc_rel]
                if len(consolidation_data) < consolidation_min_bars:
                    continue
                avg_volume_pole = flagpole_data['volume'].mean()
                avg_volume_cons = consolidation_data['volume'].mean()
                if (avg_volume_pole == 0 or avg_volume_cons /
                    avg_volume_pole > volume_decrease_factor):
                    continue
                highs = consolidation_data['high']
                lows = consolidation_data['low']
                x_range = np.arange(len(consolidation_data))
                try:
                    high_slope, high_intercept = np.polyfit(x_range, highs, 1)
                    low_slope, low_intercept = np.polyfit(x_range, lows, 1)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                is_parallel = abs(high_slope - low_slope) < parallel_threshold
                is_counter_trend = (is_uptrend_pole and high_slope < -
                    parallel_threshold / 2 or not is_uptrend_pole and 
                    high_slope > parallel_threshold / 2)
                is_flag = is_parallel and is_counter_trend
                is_converging = (high_slope < -converging_threshold and 
                    low_slope > converging_threshold or high_slope >
                    converging_threshold and low_slope < -converging_threshold)
                is_pennant = is_converging
                pattern_type = None
                if is_flag:
                    pattern_type = 'flag'
                elif is_pennant:
                    pattern_type = 'pennant'
                if pattern_type:
                    pattern_start_loc_abs = start_loc + fp_start_loc_rel
                    pattern_end_loc_abs = start_loc + cons_end_loc_rel - 1
                    result.iloc[pattern_start_loc_abs:pattern_end_loc_abs +
                        1, result.columns.get_loc(f'pattern_{pattern_type}')
                        ] = 1
        return result
