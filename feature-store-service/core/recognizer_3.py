"""
Volume Spread Analysis (VSA) Pattern Recognizer Module.

This module provides pattern recognition capabilities for VSA methodology.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from core.base import AdvancedPatternRecognizer
from models.models_4 import (
    VSAPatternType,
    VSADirection,
    VSABar,
    VSAPattern
)
from utils.utils_3 import (
    prepare_vsa_data,
    extract_vsa_bars,
    detect_no_demand,
    detect_no_supply,
    detect_stopping_volume,
    detect_climactic_volume,
    detect_effort_vs_result,
    detect_trap_move
)


class VSAPatternRecognizer(AdvancedPatternRecognizer):
    """
    Recognizes patterns based on Volume Spread Analysis (VSA) methodology.
    
    This class identifies VSA patterns like No Demand, No Supply, Stopping Volume,
    Climactic Volume, and Effort vs Result imbalances.
    """
    
    category = "pattern"
    
    def __init__(
        self,
        pattern_types: Optional[List[str]] = None,
        lookback_period: int = 100,
        sensitivity: float = 0.75,
        **kwargs
    ):
        """
        Initialize the VSA pattern recognizer.
        
        Args:
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
        
        # Set pattern types to recognize
        all_patterns = [pt.value for pt in VSAPatternType]
        
        if pattern_types is None:
            self.pattern_types = all_patterns
        else:
            self.pattern_types = [pt for pt in pattern_types if pt in all_patterns]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VSA pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with VSA pattern recognition values
        """
        # Validate input data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Prepare VSA data
        result = prepare_vsa_data(result)
        
        # Initialize pattern columns
        for pattern_type in VSAPatternType:
            result[f"pattern_{pattern_type.value}"] = 0
        
        # Add direction and strength columns
        result["pattern_vsa_direction"] = ""
        result["pattern_vsa_strength"] = 0.0
        result["pattern_vsa_target"] = np.nan
        result["pattern_vsa_stop"] = np.nan
        
        # Extract VSA bars
        bars = extract_vsa_bars(result)
        
        # Find patterns
        patterns = self._find_patterns_in_bars(bars)
        
        # Map patterns to DataFrame
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            
            if pattern_type in self.pattern_types:
                # Find the rows corresponding to this pattern
                pattern_indices = list(range(pattern.start_index, pattern.end_index + 1))
                
                if pattern_indices:
                    # Set pattern values
                    result.iloc[pattern_indices, result.columns.get_loc(f"pattern_{pattern_type}")] = 1
                    result.iloc[pattern_indices, result.columns.get_loc("pattern_vsa_direction")] = pattern.direction.value
                    result.iloc[pattern_indices, result.columns.get_loc("pattern_vsa_strength")] = pattern.strength
                    
                    if pattern.target_price is not None:
                        result.iloc[pattern_indices, result.columns.get_loc("pattern_vsa_target")] = pattern.target_price
                    
                    if pattern.stop_price is not None:
                        result.iloc[pattern_indices, result.columns.get_loc("pattern_vsa_stop")] = pattern.stop_price
        
        return result
    
    def find_patterns(
        self,
        data: pd.DataFrame,
        pattern_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find VSA patterns in the given data.
        
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
        
        # Validate input data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # Prepare VSA data
        vsa_data = prepare_vsa_data(data)
        
        # Extract VSA bars
        bars = extract_vsa_bars(vsa_data)
        
        # Find patterns
        patterns = self._find_patterns_in_bars(bars)
        
        # Initialize the patterns dictionary
        patterns_dict = {pattern_type: [] for pattern_type in patterns_to_find}
        
        # Convert patterns to dictionary format
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            
            if pattern_type in patterns_to_find:
                # Convert pattern to dictionary
                pattern_dict = pattern.to_dict()
                
                # Add to patterns dictionary
                patterns_dict[pattern_type].append(pattern_dict)
        
        return patterns_dict
    
    def _find_patterns_in_bars(self, bars: List[VSABar]) -> List[VSAPattern]:
        """
        Find all VSA patterns in the given bars.
        
        Args:
            bars: List of VSA bars
            
        Returns:
            List of VSAPattern objects
        """
        patterns = []
        
        # Detect No Demand patterns
        if VSAPatternType.NO_DEMAND.value in self.pattern_types:
            no_demand_patterns = detect_no_demand(
                bars,
                lookback=self.lookback_period,
                sensitivity=self.sensitivity
            )
            patterns.extend(no_demand_patterns)
        
        # Detect No Supply patterns
        if VSAPatternType.NO_SUPPLY.value in self.pattern_types:
            no_supply_patterns = detect_no_supply(
                bars,
                lookback=self.lookback_period,
                sensitivity=self.sensitivity
            )
            patterns.extend(no_supply_patterns)
        
        # Detect Stopping Volume patterns
        if VSAPatternType.STOPPING_VOLUME.value in self.pattern_types:
            stopping_volume_patterns = detect_stopping_volume(
                bars,
                lookback=self.lookback_period,
                sensitivity=self.sensitivity
            )
            patterns.extend(stopping_volume_patterns)
        
        # Detect Climactic Volume patterns
        if VSAPatternType.CLIMACTIC_VOLUME.value in self.pattern_types:
            climactic_volume_patterns = detect_climactic_volume(
                bars,
                lookback=self.lookback_period,
                sensitivity=self.sensitivity
            )
            patterns.extend(climactic_volume_patterns)
        
        # Detect Effort vs Result patterns
        if VSAPatternType.EFFORT_VS_RESULT.value in self.pattern_types:
            effort_vs_result_patterns = detect_effort_vs_result(
                bars,
                lookback=self.lookback_period,
                sensitivity=self.sensitivity
            )
            patterns.extend(effort_vs_result_patterns)
        
        # Detect Trap Move patterns
        if VSAPatternType.TRAP_MOVE.value in self.pattern_types:
            trap_move_patterns = detect_trap_move(
                bars,
                lookback=self.lookback_period,
                sensitivity=self.sensitivity
            )
            patterns.extend(trap_move_patterns)
        
        return patterns