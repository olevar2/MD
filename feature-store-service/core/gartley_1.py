"""
Gartley Pattern Detector Module.

This module provides the GartleyPatternDetector class for detecting Gartley harmonic patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

from feature_store_service.indicators.harmonic_patterns.detectors import BasePatternDetector
from utils.utils_6 import (
    calculate_ratio, ratio_matches
)


class GartleyPatternDetector(BasePatternDetector):
    """
    Gartley Pattern Detector.
    
    This class detects Gartley harmonic patterns in price data.
    """
    
    def __init__(self, max_pattern_bars: int = 100, pattern_template: dict = None):
        """
        Initialize Gartley pattern detector.
        
        Args:
            max_pattern_bars: Maximum number of bars to look back for pattern detection
            pattern_template: Template with ratio requirements for the pattern
        """
        super().__init__(max_pattern_bars, pattern_template)
        self.pattern_name = "gartley"
        
    def detect(self, data: pd.DataFrame, pivot_indices: pd.Index) -> pd.DataFrame:
        """
        Detect Gartley harmonic pattern.
        
        Args:
            data: DataFrame with OHLCV data and pivot points
            pivot_indices: Index of pivot points
            
        Returns:
            DataFrame with detected patterns
        """
        # Need at least 5 pivot points for XABCD pattern
        if len(pivot_indices) < 5:
            return data
            
        # Check each potential pattern
        for i in range(len(pivot_indices) - 4):
            # Get the XABCD points
            x_idx = pivot_indices[i]
            a_idx = pivot_indices[i+1]
            b_idx = pivot_indices[i+2]
            c_idx = pivot_indices[i+3]
            d_idx = pivot_indices[i+4]
            
            # Skip if pattern is too long
            if d_idx - x_idx > self.max_pattern_bars:
                continue
                
            # Get point values (could be high or low depending on pivot type)
            x_val = data.loc[x_idx, 'pivot_high_value'] if not np.isnan(data.loc[x_idx, 'pivot_high_value']) else data.loc[x_idx, 'pivot_low_value']
            a_val = data.loc[a_idx, 'pivot_high_value'] if not np.isnan(data.loc[a_idx, 'pivot_high_value']) else data.loc[a_idx, 'pivot_low_value']
            b_val = data.loc[b_idx, 'pivot_high_value'] if not np.isnan(data.loc[b_idx, 'pivot_high_value']) else data.loc[b_idx, 'pivot_low_value']
            c_val = data.loc[c_idx, 'pivot_high_value'] if not np.isnan(data.loc[c_idx, 'pivot_high_value']) else data.loc[c_idx, 'pivot_low_value']
            d_val = data.loc[d_idx, 'pivot_high_value'] if not np.isnan(data.loc[d_idx, 'pivot_high_value']) else data.loc[d_idx, 'pivot_low_value']
            
            # Calculate legs
            xa = abs(x_val - a_val)
            ab = abs(a_val - b_val)
            bc = abs(b_val - c_val)
            cd = abs(c_val - d_val)
            
            # Calculate ratios
            ab_xa_ratio = calculate_ratio(ab, xa)
            bc_ab_ratio = calculate_ratio(bc, ab)
            cd_bc_ratio = calculate_ratio(cd, bc)
            xa_bc_ratio = calculate_ratio(xa, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                ratio_matches(ab_xa_ratio, self.pattern_template["AB_XA"]["ratio"], self.pattern_template["AB_XA"]["tolerance"]) and
                ratio_matches(bc_ab_ratio, self.pattern_template["BC_AB"]["ratio"], self.pattern_template["BC_AB"]["tolerance"]) and
                ratio_matches(cd_bc_ratio, self.pattern_template["CD_BC"]["ratio"], self.pattern_template["CD_BC"]["tolerance"]) and
                ratio_matches(xa_bc_ratio, self.pattern_template["XA_BC"]["ratio"], self.pattern_template["XA_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = a_val > x_val
                
                # Mark the pattern at D point
                data.loc[d_idx, self.pattern_name] = 1
                data.loc[d_idx, f"{self.pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target (projection beyond D)
                target = d_val + (0.618 * xa) * (1 if is_bullish else -1)
                data.loc[d_idx, f"{self.pattern_name}_target"] = target
                
                # Set stop loss (beyond X)
                data.loc[d_idx, f"{self.pattern_name}_stop"] = x_val
                
        return data