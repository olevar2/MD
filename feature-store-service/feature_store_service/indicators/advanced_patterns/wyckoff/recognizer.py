"""
Wyckoff Pattern Recognizer Module.

This module provides pattern recognition capabilities for Wyckoff methodology.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from feature_store_service.indicators.advanced_patterns.base import AdvancedPatternRecognizer
from feature_store_service.indicators.advanced_patterns.wyckoff.models import (
    WyckoffPatternType,
    WyckoffPhase,
    WyckoffSchematic
)
from feature_store_service.indicators.advanced_patterns.wyckoff.utils import (
    detect_accumulation_phase,
    detect_distribution_phase,
    detect_spring,
    detect_upthrust
)


class WyckoffPatternRecognizer(AdvancedPatternRecognizer):
    """
    Recognizes patterns based on Wyckoff methodology.
    
    This class identifies Wyckoff accumulation and distribution phases,
    as well as specific Wyckoff events like springs and upthrusts.
    """
    
    category = "pattern"
    
    def __init__(
        self,
        pattern_types: Optional[List[str]] = None,
        lookback_period: int = 100,
        sensitivity: float = 0.75,
        volume_weight: float = 0.6,
        price_weight: float = 0.4,
        **kwargs
    ):
        """
        Initialize the Wyckoff pattern recognizer.
        
        Args:
            pattern_types: List of pattern types to look for (None = all patterns)
            lookback_period: Number of bars to look back for pattern recognition
            sensitivity: Sensitivity of pattern detection (0.0-1.0)
            volume_weight: Weight of volume confirmation in pattern strength
            price_weight: Weight of price action in pattern strength
            **kwargs: Additional parameters
        """
        super().__init__(
            lookback_period=lookback_period,
            sensitivity=sensitivity,
            pattern_types=pattern_types,
            **kwargs
        )
        
        self.volume_weight = volume_weight
        self.price_weight = price_weight
        
        # Set pattern types to recognize
        all_patterns = [pt.value for pt in WyckoffPatternType]
        
        if pattern_types is None:
            self.pattern_types = all_patterns
        else:
            self.pattern_types = [pt for pt in pattern_types if pt in all_patterns]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Wyckoff pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Wyckoff pattern recognition values
        """
        # Validate input data
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # Check if volume data is available
        has_volume = 'volume' in data.columns
        if not has_volume:
            # Add a dummy volume column if not available
            data['volume'] = 1.0
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Initialize pattern columns
        for pattern_type in WyckoffPatternType:
            result[f"pattern_{pattern_type.value}"] = 0
        
        # Add phase column
        result["wyckoff_phase"] = ""
        
        # Add direction and strength columns
        result["pattern_wyckoff_direction"] = ""
        result["pattern_wyckoff_strength"] = 0.0
        result["pattern_wyckoff_target"] = np.nan
        result["pattern_wyckoff_stop"] = np.nan
        
        # Find patterns
        patterns = self._find_patterns(result)
        
        # Map patterns to DataFrame
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            
            if pattern_type in self.pattern_types:
                # Find the rows corresponding to this pattern
                pattern_rows = result.iloc[pattern.start_index:pattern.end_index+1]
                
                if not pattern_rows.empty:
                    # Set pattern values
                    result.loc[pattern_rows.index, f"pattern_{pattern_type}"] = 1
                    result.loc[pattern_rows.index, "pattern_wyckoff_direction"] = pattern.direction
                    result.loc[pattern_rows.index, "pattern_wyckoff_strength"] = pattern.strength
                    
                    # Set phase values
                    for phase, (phase_start, phase_end) in pattern.phases.items():
                        phase_rows = result.iloc[phase_start:phase_end+1]
                        result.loc[phase_rows.index, "wyckoff_phase"] = phase.value
                    
                    if pattern.target_price is not None:
                        result.loc[pattern_rows.index, "pattern_wyckoff_target"] = pattern.target_price
                    
                    if pattern.stop_price is not None:
                        result.loc[pattern_rows.index, "pattern_wyckoff_stop"] = pattern.stop_price
        
        # Remove dummy volume column if it was added
        if not has_volume:
            result = result.drop(columns=['volume'])
        
        return result
    
    def find_patterns(
        self,
        data: pd.DataFrame,
        pattern_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find Wyckoff patterns in the given data.
        
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
        
        # Check if volume data is available
        has_volume = 'volume' in data.columns
        if not has_volume:
            # Add a dummy volume column if not available
            data = data.copy()
            data['volume'] = 1.0
        
        # Find patterns
        patterns = self._find_patterns(data)
        
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
    
    def _find_patterns(self, data: pd.DataFrame) -> List[WyckoffSchematic]:
        """
        Find all Wyckoff patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            List of WyckoffSchematic objects
        """
        patterns = []
        
        # Detect accumulation phases
        if WyckoffPatternType.ACCUMULATION.value in self.pattern_types:
            accumulation_patterns = detect_accumulation_phase(
                data,
                lookback=self.lookback_period,
                volume_weight=self.volume_weight,
                price_weight=self.price_weight,
                sensitivity=self.sensitivity
            )
            patterns.extend(accumulation_patterns)
        
        # Detect distribution phases
        if WyckoffPatternType.DISTRIBUTION.value in self.pattern_types:
            distribution_patterns = detect_distribution_phase(
                data,
                lookback=self.lookback_period,
                volume_weight=self.volume_weight,
                price_weight=self.price_weight,
                sensitivity=self.sensitivity
            )
            patterns.extend(distribution_patterns)
        
        # Detect springs
        if WyckoffPatternType.SPRING.value in self.pattern_types:
            spring_patterns = detect_spring(
                data,
                lookback=min(50, self.lookback_period),
                sensitivity=self.sensitivity
            )
            patterns.extend(spring_patterns)
        
        # Detect upthrusts
        if WyckoffPatternType.UPTHRUST.value in self.pattern_types:
            upthrust_patterns = detect_upthrust(
                data,
                lookback=min(50, self.lookback_period),
                sensitivity=self.sensitivity
            )
            patterns.extend(upthrust_patterns)
        
        return patterns