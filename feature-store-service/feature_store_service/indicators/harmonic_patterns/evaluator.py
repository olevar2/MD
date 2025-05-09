"""
Harmonic Pattern Evaluator Module.

This module provides functions for evaluating harmonic patterns.
"""

import pandas as pd
import numpy as np
from typing import List


def evaluate_patterns(data: pd.DataFrame, pattern_cols: List[str]) -> pd.DataFrame:
    """
    Calculate comprehensive pattern evaluation metrics.
    
    Args:
        data: DataFrame with detected patterns
        pattern_cols: List of pattern column names
        
    Returns:
        DataFrame with pattern evaluations
    """
    # Make a copy to avoid modifying the input data
    result = data.copy()
    
    # Add pattern count column
    result['total_harmonic_patterns'] = result[pattern_cols].sum(axis=1)
    
    # Add pattern quality ratings (1-10 scale)
    for pattern_col in pattern_cols:
        quality_col = f"{pattern_col}_quality"
        
        # Skip patterns that weren't detected
        if result[pattern_col].sum() == 0:
            continue
            
        # Get indices where the pattern is detected
        pattern_indices = result[result[pattern_col] == 1].index
        
        for idx in pattern_indices:
            # Base quality factors (can be expanded based on additional criteria)
            clarity_score = np.random.randint(7, 11)  # Placeholder for clarity calculation
            
            # Market context factors
            volume_factor = 1.0  # Placeholder for volume significance calculation
            trend_alignment = 1.0  # Placeholder for trend alignment calculation
            
            # Final quality score (1-10 scale)
            quality = min(10, max(1, round(clarity_score * volume_factor * trend_alignment)))
            result.loc[idx, quality_col] = quality
    
    # Add pattern confluence column 
    # (higher values indicate multiple patterns pointing in same direction)
    result['pattern_confluence'] = 0
    
    for i in range(len(result)):
        bullish_count = 0
        bearish_count = 0
        
        for pattern_col in pattern_cols:
            direction_col = f"{pattern_col}_direction"
            
            if direction_col in result.columns and i < len(result):
                if result.loc[i, direction_col] == 1:  # Bullish
                    bullish_count += 1
                elif result.loc[i, direction_col] == -1:  # Bearish
                    bearish_count += 1
                    
        # Confluence is the difference between bullish and bearish patterns
        result.loc[i, 'pattern_confluence'] = abs(bullish_count - bearish_count)
        
        # Direction of confluence
        if bullish_count > bearish_count:
            result.loc[i, 'pattern_direction'] = 1
        elif bearish_count > bullish_count:
            result.loc[i, 'pattern_direction'] = -1
        else:
            result.loc[i, 'pattern_direction'] = 0
    
    return result