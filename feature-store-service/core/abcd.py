"""
ABCD Pattern Detector Module.

This module provides the ABCDPatternDetector class for detecting ABCD harmonic patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

from feature_store_service.indicators.harmonic_patterns.detectors import BasePatternDetector
from utils.utils_6 import (
    calculate_ratio, ratio_matches
)


class ABCDPatternDetector(BasePatternDetector):
    """
    ABCD Pattern Detector.
    
    This class detects ABCD harmonic patterns in price data.
    """
    
    def __init__(self, max_pattern_bars: int = 100, pattern_template: dict = None):
        """
        Initialize ABCD pattern detector.
        
        Args:
            max_pattern_bars: Maximum number of bars to look back for pattern detection
            pattern_template: Template with ratio requirements for the pattern
        """
        super().__init__(max_pattern_bars, pattern_template)
        self.pattern_name = "abcd"
        
    def detect(self, data: pd.DataFrame, pivot_indices: pd.Index) -> pd.DataFrame:
        """
        Detect ABCD harmonic pattern.
        
        Args:
            data: DataFrame with OHLCV data and pivot points
            pivot_indices: Index of pivot points
            
        Returns:
            DataFrame with detected patterns
        """
        # Need at least 4 pivot points for ABCD pattern
        if len(pivot_indices) < 4:
            return data
            
        # Check each potential pattern
        for i in range(len(pivot_indices) - 3):
            # Get the ABCD points
            a_idx = pivot_indices[i]
            b_idx = pivot_indices[i+1]
            c_idx = pivot_indices[i+2]
            d_idx = pivot_indices[i+3]
            
            # Skip if pattern is too long
            if d_idx - a_idx > self.max_pattern_bars:
                continue
                
            # Get point values (could be high or low depending on pivot type)
            a_val = data.loc[a_idx, 'pivot_high_value'] if not np.isnan(data.loc[a_idx, 'pivot_high_value']) else data.loc[a_idx, 'pivot_low_value']
            b_val = data.loc[b_idx, 'pivot_high_value'] if not np.isnan(data.loc[b_idx, 'pivot_high_value']) else data.loc[b_idx, 'pivot_low_value']
            c_val = data.loc[c_idx, 'pivot_high_value'] if not np.isnan(data.loc[c_idx, 'pivot_high_value']) else data.loc[c_idx, 'pivot_low_value']
            d_val = data.loc[d_idx, 'pivot_high_value'] if not np.isnan(data.loc[d_idx, 'pivot_high_value']) else data.loc[d_idx, 'pivot_low_value']
            
            # Calculate legs
            ab = abs(a_val - b_val)
            bc = abs(b_val - c_val)
            cd = abs(c_val - d_val)
            
            # Calculate ratios
            bc_ab_ratio = calculate_ratio(bc, ab)
            cd_bc_ratio = calculate_ratio(cd, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                ratio_matches(bc_ab_ratio, self.pattern_template["BC_AB"]["ratio"], self.pattern_template["BC_AB"]["tolerance"]) and
                ratio_matches(cd_bc_ratio, self.pattern_template["CD_BC"]["ratio"], self.pattern_template["CD_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = b_val < a_val
                
                # Mark the pattern at D point
                data.loc[d_idx, self.pattern_name] = 1
                data.loc[d_idx, f"{self.pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target (projection beyond D)
                # For ABCD pattern, target is often a 0.618 retracement of the CD leg
                target = d_val + (0.618 * cd) * (-1 if is_bullish else 1)
                data.loc[d_idx, f"{self.pattern_name}_target"] = target
                
                # Set stop loss (beyond C)
                data.loc[d_idx, f"{self.pattern_name}_stop"] = c_val
                
        return data