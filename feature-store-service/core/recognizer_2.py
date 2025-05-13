"""
Renko Pattern Recognizer Module.

This module provides pattern recognition capabilities for Renko charts.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from core.base import AdvancedPatternRecognizer
from models.models_3 import (
    RenkoPatternType,
    RenkoBrick,
    RenkoDirection,
    RenkoPattern
)
from core.builder import RenkoChartBuilder
from utils.utils_2 import (
    detect_renko_reversal,
    detect_renko_breakout,
    detect_renko_double_formation
)


class RenkoPatternRecognizer(AdvancedPatternRecognizer):
    """
    Recognizes patterns in Renko charts.
    
    This class identifies common Renko chart patterns like reversals,
    breakouts, and double tops/bottoms.
    """
    
    category = "pattern"
    
    def __init__(
        self,
        brick_size: Optional[float] = None,
        brick_method: str = "atr",
        atr_period: int = 14,
        min_trend_length: int = 3,
        min_consolidation_length: int = 4,
        pattern_types: Optional[List[str]] = None,
        lookback_period: int = 100,
        sensitivity: float = 0.75,
        **kwargs
    ):
        """
        Initialize the Renko pattern recognizer.
        
        Args:
            brick_size: Size of each brick (None = auto-calculate using brick_method)
            brick_method: Method to calculate brick size ('atr', 'fixed', 'percentage')
            atr_period: Period for ATR calculation when brick_method='atr'
            min_trend_length: Minimum number of bricks in the same direction before a reversal
            min_consolidation_length: Minimum number of bricks in consolidation
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
        
        self.brick_size = brick_size
        self.brick_method = brick_method
        self.atr_period = atr_period
        self.min_trend_length = min_trend_length
        self.min_consolidation_length = min_consolidation_length
        
        # Initialize Renko chart builder
        self.renko_builder = RenkoChartBuilder(
            brick_size=brick_size,
            brick_method=brick_method,
            atr_period=atr_period
        )
        
        # Set pattern types to recognize
        all_patterns = [pt.value for pt in RenkoPatternType]
        
        if pattern_types is None:
            self.pattern_types = all_patterns
        else:
            self.pattern_types = [pt for pt in pattern_types if pt in all_patterns]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Renko pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Renko pattern recognition values
        """
        # Validate input data
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Build Renko chart
        result = self.renko_builder.calculate(result)
        
        # Initialize pattern columns
        for pattern_type in RenkoPatternType:
            result[f"pattern_{pattern_type.value}"] = 0
        
        # Add direction and strength columns
        result["pattern_renko_direction"] = ""
        result["pattern_renko_strength"] = 0.0
        result["pattern_renko_target"] = np.nan
        result["pattern_renko_stop"] = np.nan
        
        # Extract Renko bricks from the DataFrame
        bricks = self._extract_bricks_from_dataframe(result)
        
        # Find patterns
        patterns = self._find_patterns_in_bricks(bricks)
        
        # Map patterns to DataFrame
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            
            if pattern_type in self.pattern_types:
                # Find the rows corresponding to this pattern
                pattern_indices = [brick.index for brick in pattern.bricks if brick.index is not None]
                
                if pattern_indices:
                    # Find the corresponding rows in the DataFrame
                    pattern_rows = result[result['renko_brick_index'].isin(pattern_indices)]
                    
                    if not pattern_rows.empty:
                        # Set pattern values
                        result.loc[pattern_rows.index, f"pattern_{pattern_type}"] = 1
                        result.loc[pattern_rows.index, "pattern_renko_direction"] = pattern.direction
                        result.loc[pattern_rows.index, "pattern_renko_strength"] = pattern.strength
                        
                        if pattern.target_price is not None:
                            result.loc[pattern_rows.index, "pattern_renko_target"] = pattern.target_price
                        
                        if pattern.stop_price is not None:
                            result.loc[pattern_rows.index, "pattern_renko_stop"] = pattern.stop_price
        
        return result
    
    def find_patterns(
        self,
        data: pd.DataFrame,
        pattern_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find Renko patterns in the given data.
        
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
        
        # Calculate Renko patterns
        result = self.calculate(data)
        
        # Initialize the patterns dictionary
        patterns_dict = {pattern_type: [] for pattern_type in patterns_to_find}
        
        # Extract Renko bricks from the DataFrame
        bricks = self._extract_bricks_from_dataframe(result)
        
        # Find patterns
        patterns = self._find_patterns_in_bricks(bricks)
        
        # Convert patterns to dictionary format
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            
            if pattern_type in patterns_to_find:
                # Convert pattern to dictionary
                pattern_dict = {
                    'pattern_type': pattern_type,
                    'start_index': pattern.start_index,
                    'end_index': pattern.end_index,
                    'direction': pattern.direction,
                    'strength': pattern.strength,
                    'target_price': pattern.target_price,
                    'stop_price': pattern.stop_price
                }
                
                # Add to patterns dictionary
                patterns_dict[pattern_type].append(pattern_dict)
        
        return patterns_dict
    
    def _extract_bricks_from_dataframe(self, data: pd.DataFrame) -> List[RenkoBrick]:
        """
        Extract Renko bricks from the DataFrame.
        
        Args:
            data: DataFrame with Renko data
            
        Returns:
            List of RenkoBrick objects
        """
        bricks = []
        
        # Filter rows with brick data
        brick_rows = data[data['renko_brick_index'].notna()]
        
        for _, row in brick_rows.iterrows():
            direction = RenkoDirection.UP if row['renko_brick_direction'] == 1 else RenkoDirection.DOWN
            
            brick = RenkoBrick(
                direction=direction,
                open_price=row['renko_brick_open'],
                close_price=row['renko_brick_close'],
                open_time=row.name,  # Timestamp is the index
                close_time=row.name,
                index=int(row['renko_brick_index'])
            )
            
            bricks.append(brick)
        
        return bricks
    
    def _find_patterns_in_bricks(self, bricks: List[RenkoBrick]) -> List[RenkoPattern]:
        """
        Find patterns in Renko bricks.
        
        Args:
            bricks: List of Renko bricks
            
        Returns:
            List of RenkoPattern objects
        """
        patterns = []
        
        # Detect reversal patterns
        if RenkoPatternType.REVERSAL.value in self.pattern_types:
            reversal_patterns = detect_renko_reversal(
                bricks,
                min_trend_length=self.min_trend_length,
                lookback=self.lookback_period
            )
            patterns.extend(reversal_patterns)
        
        # Detect breakout patterns
        if RenkoPatternType.BREAKOUT.value in self.pattern_types:
            breakout_patterns = detect_renko_breakout(
                bricks,
                min_consolidation_length=self.min_consolidation_length,
                lookback=self.lookback_period
            )
            patterns.extend(breakout_patterns)
        
        # Detect double top/bottom patterns
        if RenkoPatternType.DOUBLE_TOP.value in self.pattern_types or RenkoPatternType.DOUBLE_BOTTOM.value in self.pattern_types:
            double_patterns = detect_renko_double_formation(
                bricks,
                max_deviation=0.1 * (1.0 / self.sensitivity),  # Adjust based on sensitivity
                lookback=self.lookback_period
            )
            patterns.extend(double_patterns)
        
        return patterns