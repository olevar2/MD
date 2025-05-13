"""
Ichimoku Pattern Recognizer Module.

This module provides pattern recognition capabilities for Ichimoku Cloud analysis.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from core.base import AdvancedPatternRecognizer
from models.models_1 import (
    IchimokuPatternType,
    IchimokuComponents,
    IchimokuPattern
)
from utils.utils_1 import (
    detect_tk_cross,
    detect_kumo_breakout,
    detect_kumo_twist,
    detect_chikou_cross
)


class IchimokuPatternRecognizer(AdvancedPatternRecognizer):
    """
    Recognizes patterns in Ichimoku Cloud analysis.
    
    This class identifies common Ichimoku patterns like TK crosses,
    Kumo breakouts, Kumo twists, and Chikou crosses.
    """
    
    category = "pattern"
    
    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
        pattern_types: Optional[List[str]] = None,
        lookback_period: int = 100,
        sensitivity: float = 0.75,
        **kwargs
    ):
        """
        Initialize the Ichimoku pattern recognizer.
        
        Args:
            tenkan_period: Period for Tenkan-sen calculation
            kijun_period: Period for Kijun-sen calculation
            senkou_b_period: Period for Senkou Span B calculation
            displacement: Displacement period for Senkou Span and Chikou Span
            pattern_types: List of pattern types to look for (None = all patterns)
            lookback_period: Number of bars to look back for pattern recognition
            sensitivity: Sensitivity of pattern detection (0.0-1.0)
            **kwargs: Additional parameters
        """
        super().__init__(
            lookback_period=lookback_period,
            sensitivity=sensitivity,
            pattern_types=pattern_types,
            **kwargs
        )
        
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        
        # Set pattern types to recognize
        all_patterns = [pt.value for pt in IchimokuPatternType]
        
        if pattern_types is None:
            self.pattern_types = all_patterns
        else:
            self.pattern_types = [pt for pt in pattern_types if pt in all_patterns]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Ichimoku pattern recognition values
        """
        # Validate input data
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate Ichimoku components if not already present
        if 'ichimoku_tenkan' not in result.columns:
            result = self._calculate_ichimoku_components(result)
        
        # Initialize pattern columns
        for pattern_type in IchimokuPatternType:
            result[f"pattern_{pattern_type.value}"] = 0
        
        # Add direction and strength columns
        result["pattern_ichimoku_direction"] = ""
        result["pattern_ichimoku_strength"] = 0.0
        result["pattern_ichimoku_target"] = np.nan
        result["pattern_ichimoku_stop"] = np.nan
        
        # Detect TK crosses
        if IchimokuPatternType.TK_CROSS.value in self.pattern_types:
            tk_patterns = detect_tk_cross(
                result,
                tenkan_col="ichimoku_tenkan",
                kijun_col="ichimoku_kijun",
                price_col="close",
                lookback=int(5 * self.sensitivity)
            )
            
            # Map patterns to DataFrame
            for pattern in tk_patterns:
                # Find the rows corresponding to this pattern
                pattern_rows = result.iloc[pattern.start_index:pattern.end_index+1]
                
                if not pattern_rows.empty:
                    # Set pattern values
                    result.loc[pattern_rows.index, f"pattern_{pattern.pattern_type.value}"] = 1
                    result.loc[pattern_rows.index, "pattern_ichimoku_direction"] = pattern.direction
                    result.loc[pattern_rows.index, "pattern_ichimoku_strength"] = pattern.strength
                    
                    if pattern.target_price is not None:
                        result.loc[pattern_rows.index, "pattern_ichimoku_target"] = pattern.target_price
                    
                    if pattern.stop_price is not None:
                        result.loc[pattern_rows.index, "pattern_ichimoku_stop"] = pattern.stop_price
        
        # Detect Kumo breakouts
        if IchimokuPatternType.KUMO_BREAKOUT.value in self.pattern_types:
            kumo_breakout_patterns = detect_kumo_breakout(
                result,
                senkou_a_col="ichimoku_senkou_a",
                senkou_b_col="ichimoku_senkou_b",
                price_col="close",
                lookback=int(10 * self.sensitivity)
            )
            
            # Map patterns to DataFrame
            for pattern in kumo_breakout_patterns:
                # Find the rows corresponding to this pattern
                pattern_rows = result.iloc[pattern.start_index:pattern.end_index+1]
                
                if not pattern_rows.empty:
                    # Set pattern values
                    result.loc[pattern_rows.index, f"pattern_{pattern.pattern_type.value}"] = 1
                    result.loc[pattern_rows.index, "pattern_ichimoku_direction"] = pattern.direction
                    result.loc[pattern_rows.index, "pattern_ichimoku_strength"] = pattern.strength
                    
                    if pattern.target_price is not None:
                        result.loc[pattern_rows.index, "pattern_ichimoku_target"] = pattern.target_price
                    
                    if pattern.stop_price is not None:
                        result.loc[pattern_rows.index, "pattern_ichimoku_stop"] = pattern.stop_price
        
        # Detect Kumo twists
        if IchimokuPatternType.KUMO_TWIST.value in self.pattern_types:
            kumo_twist_patterns = detect_kumo_twist(
                result,
                senkou_a_col="ichimoku_senkou_a",
                senkou_b_col="ichimoku_senkou_b",
                price_col="close",
                lookback=int(5 * self.sensitivity)
            )
            
            # Map patterns to DataFrame
            for pattern in kumo_twist_patterns:
                # Find the rows corresponding to this pattern
                pattern_rows = result.iloc[pattern.start_index:pattern.end_index+1]
                
                if not pattern_rows.empty:
                    # Set pattern values
                    result.loc[pattern_rows.index, f"pattern_{pattern.pattern_type.value}"] = 1
                    result.loc[pattern_rows.index, "pattern_ichimoku_direction"] = pattern.direction
                    result.loc[pattern_rows.index, "pattern_ichimoku_strength"] = pattern.strength
                    
                    if pattern.target_price is not None:
                        result.loc[pattern_rows.index, "pattern_ichimoku_target"] = pattern.target_price
                    
                    if pattern.stop_price is not None:
                        result.loc[pattern_rows.index, "pattern_ichimoku_stop"] = pattern.stop_price
        
        # Detect Chikou crosses
        if IchimokuPatternType.CHIKOU_CROSS.value in self.pattern_types:
            chikou_patterns = detect_chikou_cross(
                result,
                chikou_col="ichimoku_chikou",
                price_col="close",
                lookback=int(5 * self.sensitivity)
            )
            
            # Map patterns to DataFrame
            for pattern in chikou_patterns:
                # Find the rows corresponding to this pattern
                pattern_rows = result.iloc[pattern.start_index:pattern.end_index+1]
                
                if not pattern_rows.empty:
                    # Set pattern values
                    result.loc[pattern_rows.index, f"pattern_{pattern.pattern_type.value}"] = 1
                    result.loc[pattern_rows.index, "pattern_ichimoku_direction"] = pattern.direction
                    result.loc[pattern_rows.index, "pattern_ichimoku_strength"] = pattern.strength
                    
                    if pattern.target_price is not None:
                        result.loc[pattern_rows.index, "pattern_ichimoku_target"] = pattern.target_price
                    
                    if pattern.stop_price is not None:
                        result.loc[pattern_rows.index, "pattern_ichimoku_stop"] = pattern.stop_price
        
        return result
    
    def find_patterns(
        self,
        data: pd.DataFrame,
        pattern_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find Ichimoku patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data
            pattern_types: List of pattern types to look for (None = all patterns)
            
        Returns:
            Dictionary of pattern types and their occurrences
        """
        # Determine which patterns to look for
        if pattern_types is None:
            patterns_to_find = self.pattern_types
        else:
            patterns_to_find = [pt for pt in pattern_types if pt in self.pattern_types]
        
        # Calculate Ichimoku patterns
        result = self.calculate(data)
        
        # Initialize the patterns dictionary
        patterns_dict = {pattern_type: [] for pattern_type in patterns_to_find}
        
        # Extract patterns from the DataFrame
        for pattern_type in patterns_to_find:
            pattern_col = f"pattern_{pattern_type}"
            
            if pattern_col in result.columns:
                # Find contiguous pattern regions
                pattern_regions = self._find_contiguous_regions(result[pattern_col])
                
                for start_idx, end_idx in pattern_regions:
                    if end_idx < start_idx or start_idx < 0 or end_idx >= len(result):
                        continue
                    
                    # Get the pattern slice
                    pattern_slice = result.iloc[start_idx:end_idx+1]
                    if pattern_slice.empty:
                        continue
                    
                    # Get pattern properties
                    direction = pattern_slice["pattern_ichimoku_direction"].iloc[-1]
                    strength = pattern_slice["pattern_ichimoku_strength"].iloc[-1]
                    target_price = pattern_slice["pattern_ichimoku_target"].iloc[-1]
                    stop_price = pattern_slice["pattern_ichimoku_stop"].iloc[-1]
                    
                    # Create pattern dictionary
                    pattern_dict = {
                        'pattern_type': pattern_type,
                        'start_index': start_idx,
                        'end_index': end_idx,
                        'direction': direction,
                        'strength': strength,
                        'target_price': target_price,
                        'stop_price': stop_price
                    }
                    
                    # Add to patterns dictionary
                    patterns_dict[pattern_type].append(pattern_dict)
        
        return patterns_dict
    
    def _calculate_ichimoku_components(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added Ichimoku components
        """
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
        high_tenkan = result['high'].rolling(window=self.tenkan_period).max()
        low_tenkan = result['low'].rolling(window=self.tenkan_period).min()
        result['ichimoku_tenkan'] = (high_tenkan + low_tenkan) / 2
        
        # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
        high_kijun = result['high'].rolling(window=self.kijun_period).max()
        low_kijun = result['low'].rolling(window=self.kijun_period).min()
        result['ichimoku_kijun'] = (high_kijun + low_kijun) / 2
        
        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 displaced forward by displacement periods
        result['ichimoku_senkou_a'] = ((result['ichimoku_tenkan'] + result['ichimoku_kijun']) / 2).shift(self.displacement)
        
        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past senkou_b_period, displaced forward by displacement periods
        high_senkou_b = result['high'].rolling(window=self.senkou_b_period).max()
        low_senkou_b = result['low'].rolling(window=self.senkou_b_period).min()
        result['ichimoku_senkou_b'] = ((high_senkou_b + low_senkou_b) / 2).shift(self.displacement)
        
        # Calculate Chikou Span (Lagging Span): Current closing price displaced backward by displacement periods
        result['ichimoku_chikou'] = result['close'].shift(-self.displacement)
        
        return result