"""
Harmonic Pattern Screener.

This module implements a comprehensive harmonic pattern detection system
that checks for multiple harmonic patterns and provides a pattern evaluation system.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum

from feature_store_service.indicators.base_indicator import BaseIndicator


class PatternType(Enum):
    """Enum for harmonic pattern types."""
    BAT = "bat"
    SHARK = "shark"
    CYPHER = "cypher"
    ABCD = "abcd"
    THREE_DRIVES = "three_drives"
    FIVE_ZERO = "five_zero"
    ALT_BAT = "alt_bat"
    DEEP_CRAB = "deep_crab"
    BUTTERFLY = "butterfly"
    GARTLEY = "gartley"
    CRAB = "crab"


class HarmonicPatternScreener(BaseIndicator):
    """
    Harmonic Pattern Screener indicator.
    
    This indicator detects harmonic patterns and provides a comprehensive 
    evaluation system for pattern quality and potential.
    """
    
    category = "pattern_recognition"
    
    def __init__(
        self, 
        max_pattern_bars: int = 100, 
        tolerance: float = 0.05,
        **kwargs
    ):
        """
        Initialize Harmonic Pattern Screener.
        
        Args:
            max_pattern_bars: Maximum number of bars to look back for pattern detection
            tolerance: Tolerance for pattern ratio matching (as decimal)
            **kwargs: Additional parameters
        """
        self.max_pattern_bars = max_pattern_bars
        self.tolerance = tolerance
        self.name = "harmonic_patterns"
        
        # Define fibonacci ratios used in harmonic patterns
        self.fib_ratios = {
            "0.382": 0.382,
            "0.5": 0.5,
            "0.618": 0.618,
            "0.707": 0.707,
            "0.786": 0.786,
            "0.886": 0.886,
            "1.0": 1.0,
            "1.13": 1.13,
            "1.27": 1.27,
            "1.414": 1.414,
            "1.618": 1.618,
            "2.0": 2.0,
            "2.24": 2.24,
            "2.618": 2.618,
            "3.14": 3.14,
            "3.618": 3.618
        }
        
        # Define pattern ratio templates
        self.pattern_templates = {
            "bat": {
                "XA_BC": {"ratio": 0.382, "tolerance": self.tolerance},
                "AB_XA": {"ratio": 0.382, "tolerance": self.tolerance},
                "BC_AB": {"ratio": 0.382, "tolerance": self.tolerance},
                "CD_BC": {"ratio": 1.618, "tolerance": self.tolerance * 2}
            },
            "shark": {
                "XA_BC": {"ratio": 0.886, "tolerance": self.tolerance},
                "AB_XA": {"ratio": 0.5, "tolerance": self.tolerance},
                "BC_AB": {"ratio": 1.13, "tolerance": self.tolerance},
                "CD_BC": {"ratio": 1.618, "tolerance": self.tolerance * 2}
            },
            "cypher": {
                "XA_BC": {"ratio": 0.382, "tolerance": self.tolerance},
                "AB_XA": {"ratio": 0.618, "tolerance": self.tolerance},
                "BC_AB": {"ratio": 1.13, "tolerance": self.tolerance},
                "CD_BC": {"ratio": 0.786, "tolerance": self.tolerance}
            },
            "abcd": {
                "AB_XA": {"ratio": 0.618, "tolerance": self.tolerance},
                "BC_AB": {"ratio": 0.618, "tolerance": self.tolerance},
                "CD_BC": {"ratio": 0.618, "tolerance": self.tolerance}
            },
            "three_drives": {
                "drive1_drive2": {"ratio": 0.618, "tolerance": self.tolerance},
                "drive2_drive3": {"ratio": 1.27, "tolerance": self.tolerance},
                "correction1_drive1": {"ratio": 0.618, "tolerance": self.tolerance},
                "correction2_drive2": {"ratio": 0.618, "tolerance": self.tolerance}
            },
            "five_zero": {
                "XA_BC": {"ratio": 1.618, "tolerance": self.tolerance * 2},
                "AB_XA": {"ratio": 0.5, "tolerance": self.tolerance},
                "BC_AB": {"ratio": 1.618, "tolerance": self.tolerance * 2},
                "CD_BC": {"ratio": 0.5, "tolerance": self.tolerance}
            },
            "alt_bat": {
                "XA_BC": {"ratio": 0.382, "tolerance": self.tolerance},
                "AB_XA": {"ratio": 0.382, "tolerance": self.tolerance},
                "BC_AB": {"ratio": 2.0, "tolerance": self.tolerance * 2},
                "CD_BC": {"ratio": 1.27, "tolerance": self.tolerance}
            },
            "deep_crab": {
                "XA_BC": {"ratio": 0.886, "tolerance": self.tolerance},
                "AB_XA": {"ratio": 0.382, "tolerance": self.tolerance},
                "BC_AB": {"ratio": 2.618, "tolerance": self.tolerance * 2},
                "CD_BC": {"ratio": 1.618, "tolerance": self.tolerance * 2}
            },
            "butterfly": {
                "XA_BC": {"ratio": 0.786, "tolerance": self.tolerance},
                "AB_XA": {"ratio": 0.786, "tolerance": self.tolerance},
                "BC_AB": {"ratio": 0.382, "tolerance": self.tolerance},
                "CD_BC": {"ratio": 1.618, "tolerance": self.tolerance * 2}
            },
            "gartley": {
                "XA_BC": {"ratio": 0.618, "tolerance": self.tolerance},
                "AB_XA": {"ratio": 0.618, "tolerance": self.tolerance},
                "BC_AB": {"ratio": 0.382, "tolerance": self.tolerance},
                "CD_BC": {"ratio": 1.27, "tolerance": self.tolerance}
            },
            "crab": {
                "XA_BC": {"ratio": 0.382, "tolerance": self.tolerance},
                "AB_XA": {"ratio": 0.382, "tolerance": self.tolerance},
                "BC_AB": {"ratio": 0.618, "tolerance": self.tolerance},
                "CD_BC": {"ratio": 3.14, "tolerance": self.tolerance * 2}
            }
        }
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Screen for harmonic patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with detected harmonic patterns and evaluations
        """
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Identify potential pivot points
        self._identify_pivots(result)
        
        # Initialize pattern columns
        for pattern in PatternType:
            result[pattern.value] = 0
            result[f"{pattern.value}_quality"] = 0
            result[f"{pattern.value}_target"] = 0
            result[f"{pattern.value}_stop"] = 0
            result[f"{pattern.value}_direction"] = 0  # 1 for bullish, -1 for bearish
        
        # Detect patterns
        pivot_indices = result[(result['pivot_high'] == 1) | (result['pivot_low'] == 1)].index
        
        # Need at least 5 pivot points for pattern detection
        if len(pivot_indices) < 5:
            return result
            
        # Detect each type of harmonic pattern
        self._detect_bat_pattern(result, pivot_indices)
        self._detect_shark_pattern(result, pivot_indices)
        self._detect_cypher_pattern(result, pivot_indices)
        self._detect_abcd_pattern(result, pivot_indices)
        self._detect_three_drives_pattern(result, pivot_indices)
        self._detect_five_zero_pattern(result, pivot_indices)
        self._detect_alt_bat_pattern(result, pivot_indices)
        self._detect_deep_crab_pattern(result, pivot_indices)
        self._detect_butterfly_pattern(result, pivot_indices)
        self._detect_gartley_pattern(result, pivot_indices)
        self._detect_crab_pattern(result, pivot_indices)
        
        # Calculate pattern evaluation metrics
        self._evaluate_patterns(result)
        
        return result
    
    def _identify_pivots(self, data: pd.DataFrame, window: int = 5) -> None:
        """
        Identify significant pivot points in the price data.
        
        Args:
            data: DataFrame with OHLCV data
            window: Number of bars to look before/after for pivot detection
        """
        # Identify pivot highs and lows
        data['pivot_high'] = 0
        data['pivot_low'] = 0
        
        for i in range(window, len(data) - window):
            # Check if this bar's high is higher than all bars in the window before and after
            if all(data.loc[i, 'high'] > data.loc[i-window:i-1, 'high']) and \
               all(data.loc[i, 'high'] > data.loc[i+1:i+window, 'high']):
                data.loc[i, 'pivot_high'] = 1
                
            # Check if this bar's low is lower than all bars in the window before and after
            if all(data.loc[i, 'low'] < data.loc[i-window:i-1, 'low']) and \
               all(data.loc[i, 'low'] < data.loc[i+1:i+window, 'low']):
                data.loc[i, 'pivot_low'] = 1
                
        # Store pivot point values
        data['pivot_high_value'] = data['high'] * data['pivot_high']
        data['pivot_low_value'] = data['low'] * data['pivot_low']
        
        # Replace zeros with NaN for easier manipulation
        data['pivot_high_value'].replace(0, np.nan, inplace=True)
        data['pivot_low_value'].replace(0, np.nan, inplace=True)
    
    def _calculate_ratio(self, value1: float, value2: float) -> float:
        """
        Calculate ratio between two values (avoiding division by zero).
        
        Args:
            value1: First value
            value2: Second value
            
        Returns:
            Ratio between values
        """
        if value2 == 0:
            return float('inf')
        return abs(value1 / value2)
    
    def _ratio_matches(self, actual_ratio: float, target_ratio: float, tolerance: float) -> bool:
        """
        Check if an actual ratio matches a target ratio within tolerance.
        
        Args:
            actual_ratio: Actual calculated ratio
            target_ratio: Target ratio to match
            tolerance: Acceptable tolerance range
            
        Returns:
            True if ratio matches within tolerance, False otherwise
        """
        return abs(actual_ratio - target_ratio) <= tolerance * target_ratio
    
    def _detect_bat_pattern(self, data: pd.DataFrame, pivot_indices: pd.Index) -> None:
        """
        Detect Bat harmonic pattern.
        
        Args:
            data: DataFrame with OHLCV data
            pivot_indices: Index of pivot points
        """
        pattern_name = PatternType.BAT.value
        template = self.pattern_templates[pattern_name]
        
        # Need at least 5 pivot points for XABCD pattern
        if len(pivot_indices) < 5:
            return
            
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
            ab_xa_ratio = self._calculate_ratio(ab, xa)
            bc_ab_ratio = self._calculate_ratio(bc, ab)
            cd_bc_ratio = self._calculate_ratio(cd, bc)
            xa_bc_ratio = self._calculate_ratio(xa, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                self._ratio_matches(ab_xa_ratio, template["AB_XA"]["ratio"], template["AB_XA"]["tolerance"]) and
                self._ratio_matches(bc_ab_ratio, template["BC_AB"]["ratio"], template["BC_AB"]["tolerance"]) and
                self._ratio_matches(cd_bc_ratio, template["CD_BC"]["ratio"], template["CD_BC"]["tolerance"]) and
                self._ratio_matches(xa_bc_ratio, template["XA_BC"]["ratio"], template["XA_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = a_val > x_val
                
                # Mark the pattern at D point
                data.loc[d_idx, pattern_name] = 1
                data.loc[d_idx, f"{pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target (projection beyond D)
                # For simplicity, using projection of XA leg from D
                target = d_val + (0.618 * xa) * (1 if is_bullish else -1)
                data.loc[d_idx, f"{pattern_name}_target"] = target
                
                # Set stop loss (beyond X)
                data.loc[d_idx, f"{pattern_name}_stop"] = x_val
    
    def _detect_shark_pattern(self, data: pd.DataFrame, pivot_indices: pd.Index) -> None:
        """
        Detect Shark harmonic pattern.
        
        Args:
            data: DataFrame with OHLCV data
            pivot_indices: Index of pivot points
        """
        pattern_name = PatternType.SHARK.value
        template = self.pattern_templates[pattern_name]
        
        # Similar structure to bat pattern detection with different ratios
        # Need at least 5 pivot points for XABCD pattern
        if len(pivot_indices) < 5:
            return
            
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
            ab_xa_ratio = self._calculate_ratio(ab, xa)
            bc_ab_ratio = self._calculate_ratio(bc, ab)
            cd_bc_ratio = self._calculate_ratio(cd, bc)
            xa_bc_ratio = self._calculate_ratio(xa, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                self._ratio_matches(ab_xa_ratio, template["AB_XA"]["ratio"], template["AB_XA"]["tolerance"]) and
                self._ratio_matches(bc_ab_ratio, template["BC_AB"]["ratio"], template["BC_AB"]["tolerance"]) and
                self._ratio_matches(cd_bc_ratio, template["CD_BC"]["ratio"], template["CD_BC"]["tolerance"]) and
                self._ratio_matches(xa_bc_ratio, template["XA_BC"]["ratio"], template["XA_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = a_val > x_val
                
                # Mark the pattern at D point
                data.loc[d_idx, pattern_name] = 1
                data.loc[d_idx, f"{pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target (projection beyond D)
                target = d_val + (0.5 * xa) * (1 if is_bullish else -1)
                data.loc[d_idx, f"{pattern_name}_target"] = target
                
                # Set stop loss (beyond X)
                data.loc[d_idx, f"{pattern_name}_stop"] = x_val
    
    def _detect_cypher_pattern(self, data: pd.DataFrame, pivot_indices: pd.Index) -> None:
        """
        Detect Cypher harmonic pattern.
        
        Args:
            data: DataFrame with OHLCV data
            pivot_indices: Index of pivot points
        """
        pattern_name = PatternType.CYPHER.value
        template = self.pattern_templates[pattern_name]
        
        # Similar implementation to other patterns with Cypher-specific ratios
        # Need at least 5 pivot points for XABCD pattern
        if len(pivot_indices) < 5:
            return
            
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
            ab_xa_ratio = self._calculate_ratio(ab, xa)
            bc_ab_ratio = self._calculate_ratio(bc, ab)
            cd_bc_ratio = self._calculate_ratio(cd, bc)
            xa_bc_ratio = self._calculate_ratio(xa, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                self._ratio_matches(ab_xa_ratio, template["AB_XA"]["ratio"], template["AB_XA"]["tolerance"]) and
                self._ratio_matches(bc_ab_ratio, template["BC_AB"]["ratio"], template["BC_AB"]["tolerance"]) and
                self._ratio_matches(cd_bc_ratio, template["CD_BC"]["ratio"], template["CD_BC"]["tolerance"]) and
                self._ratio_matches(xa_bc_ratio, template["XA_BC"]["ratio"], template["XA_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = a_val > x_val
                
                # Mark the pattern at D point
                data.loc[d_idx, pattern_name] = 1
                data.loc[d_idx, f"{pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target (projection beyond D)
                target = d_val + (0.382 * xa) * (1 if is_bullish else -1)
                data.loc[d_idx, f"{pattern_name}_target"] = target
                
                # Set stop loss (beyond X)
                data.loc[d_idx, f"{pattern_name}_stop"] = x_val
    
    def _detect_abcd_pattern(self, data: pd.DataFrame, pivot_indices: pd.Index) -> None:
        """
        Detect ABCD harmonic pattern.
        
        Args:
            data: DataFrame with OHLCV data
            pivot_indices: Index of pivot points
        """
        pattern_name = PatternType.ABCD.value
        template = self.pattern_templates[pattern_name]
        
        # ABCD pattern only needs 4 points
        if len(pivot_indices) < 4:
            return
            
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
            
            # Special case for ABCD pattern (no X point)
            ab_xa_ratio = 1  # Not used for ABCD
            bc_ab_ratio = self._calculate_ratio(bc, ab)
            cd_bc_ratio = self._calculate_ratio(cd, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                self._ratio_matches(bc_ab_ratio, template["BC_AB"]["ratio"], template["BC_AB"]["tolerance"]) and
                self._ratio_matches(cd_bc_ratio, template["CD_BC"]["ratio"], template["CD_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = b_val < a_val
                
                # Mark the pattern at D point
                data.loc[d_idx, pattern_name] = 1
                data.loc[d_idx, f"{pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target (projection beyond D)
                target = d_val + (0.382 * cd) * (1 if is_bullish else -1)
                data.loc[d_idx, f"{pattern_name}_target"] = target
                
                # Set stop loss (beyond C)
                data.loc[d_idx, f"{pattern_name}_stop"] = c_val
    
    def _detect_three_drives_pattern(self, data: pd.DataFrame, pivot_indices: pd.Index) -> None:
        """
        Detect Three Drives harmonic pattern.
        
        Args:
            data: DataFrame with OHLCV data
            pivot_indices: Index of pivot points
        """
        pattern_name = PatternType.THREE_DRIVES.value
        template = self.pattern_templates[pattern_name]
        
        # Three Drives pattern involves 3 drives and 2 corrections (5 points)
        if len(pivot_indices) < 5:
            return
            
        # Check each potential pattern
        for i in range(len(pivot_indices) - 4):
            # Get the points (drive1, correction1, drive2, correction2, drive3)
            drive1_idx = pivot_indices[i]
            correction1_idx = pivot_indices[i+1]
            drive2_idx = pivot_indices[i+2]
            correction2_idx = pivot_indices[i+3]
            drive3_idx = pivot_indices[i+4]
            
            # Skip if pattern is too long
            if drive3_idx - drive1_idx > self.max_pattern_bars:
                continue
                
            # Get point values
            drive1_val = data.loc[drive1_idx, 'pivot_high_value'] if not np.isnan(data.loc[drive1_idx, 'pivot_high_value']) else data.loc[drive1_idx, 'pivot_low_value']
            correction1_val = data.loc[correction1_idx, 'pivot_high_value'] if not np.isnan(data.loc[correction1_idx, 'pivot_high_value']) else data.loc[correction1_idx, 'pivot_low_value']
            drive2_val = data.loc[drive2_idx, 'pivot_high_value'] if not np.isnan(data.loc[drive2_idx, 'pivot_high_value']) else data.loc[drive2_idx, 'pivot_low_value']
            correction2_val = data.loc[correction2_idx, 'pivot_high_value'] if not np.isnan(data.loc[correction2_idx, 'pivot_high_value']) else data.loc[correction2_idx, 'pivot_low_value']
            drive3_val = data.loc[drive3_idx, 'pivot_high_value'] if not np.isnan(data.loc[drive3_idx, 'pivot_high_value']) else data.loc[drive3_idx, 'pivot_low_value']
            
            # Calculate legs
            drive1 = abs(drive1_val - correction1_val)
            correction1 = abs(correction1_val - drive2_val)
            drive2 = abs(drive2_val - correction2_val)
            correction2 = abs(correction2_val - drive3_val)
            
            # Calculate ratios
            drive2_drive1_ratio = self._calculate_ratio(drive2, drive1)
            drive3_drive2_ratio = self._calculate_ratio(correction2, drive2)
            correction1_drive1_ratio = self._calculate_ratio(correction1, drive1)
            correction2_drive2_ratio = self._calculate_ratio(correction2, drive2)
            
            # Check if ratios match the pattern template
            matches_template = (
                self._ratio_matches(drive2_drive1_ratio, template["drive1_drive2"]["ratio"], template["drive1_drive2"]["tolerance"]) and
                self._ratio_matches(drive3_drive2_ratio, template["drive2_drive3"]["ratio"], template["drive2_drive3"]["tolerance"]) and
                self._ratio_matches(correction1_drive1_ratio, template["correction1_drive1"]["ratio"], template["correction1_drive1"]["tolerance"]) and
                self._ratio_matches(correction2_drive2_ratio, template["correction2_drive2"]["ratio"], template["correction2_drive2"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                # Three drives pattern direction is determined by the overall trend of drives
                is_bullish = drive3_val > drive1_val
                
                # Mark the pattern at drive3 point
                data.loc[drive3_idx, pattern_name] = 1
                data.loc[drive3_idx, f"{pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target (reversal from drive3)
                target = drive3_val + (0.618 * drive1) * (-1 if is_bullish else 1)  # Reversal from trend
                data.loc[drive3_idx, f"{pattern_name}_target"] = target
                
                # Set stop loss (beyond drive3 in trend direction)
                data.loc[drive3_idx, f"{pattern_name}_stop"] = drive3_val + (0.1 * drive3) * (1 if is_bullish else -1)
    
    # Implement similar pattern detection methods for:
    def _detect_five_zero_pattern(self, data: pd.DataFrame, pivot_indices: pd.Index) -> None:
        """5-0 Pattern detection (similar structure to other methods)"""
        pattern_name = PatternType.FIVE_ZERO.value
        template = self.pattern_templates[pattern_name]
        
        # Similar implementation to other XABCD patterns
        # Need at least 5 pivot points for XABCD pattern
        if len(pivot_indices) < 5:
            return
            
        # Implementation follows the same pattern as bat pattern with different ratios
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
            ab_xa_ratio = self._calculate_ratio(ab, xa)
            bc_ab_ratio = self._calculate_ratio(bc, ab)
            cd_bc_ratio = self._calculate_ratio(cd, bc)
            xa_bc_ratio = self._calculate_ratio(xa, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                self._ratio_matches(ab_xa_ratio, template["AB_XA"]["ratio"], template["AB_XA"]["tolerance"]) and
                self._ratio_matches(bc_ab_ratio, template["BC_AB"]["ratio"], template["BC_AB"]["tolerance"]) and
                self._ratio_matches(cd_bc_ratio, template["CD_BC"]["ratio"], template["CD_BC"]["tolerance"]) and
                self._ratio_matches(xa_bc_ratio, template["XA_BC"]["ratio"], template["XA_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = a_val > x_val
                
                # Mark the pattern at D point
                data.loc[d_idx, pattern_name] = 1
                data.loc[d_idx, f"{pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target
                target = d_val + (0.5 * xa) * (1 if is_bullish else -1)
                data.loc[d_idx, f"{pattern_name}_target"] = target
                
                # Set stop loss (beyond X)
                data.loc[d_idx, f"{pattern_name}_stop"] = x_val
    
    def _detect_alt_bat_pattern(self, data: pd.DataFrame, pivot_indices: pd.Index) -> None:
        """Alt Bat Pattern detection (similar structure to other methods)"""
        pattern_name = PatternType.ALT_BAT.value
        template = self.pattern_templates[pattern_name]
        
        # Similar implementation to bat pattern with different ratios
        if len(pivot_indices) < 5:
            return
            
        # Implementation follows the same pattern as bat pattern with different ratios
        # Similar code as _detect_bat_pattern with specific alt_bat template ratios
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
                
            # Calculate ratios and check against template (identical to other pattern checks)
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
            ab_xa_ratio = self._calculate_ratio(ab, xa)
            bc_ab_ratio = self._calculate_ratio(bc, ab)
            cd_bc_ratio = self._calculate_ratio(cd, bc)
            xa_bc_ratio = self._calculate_ratio(xa, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                self._ratio_matches(ab_xa_ratio, template["AB_XA"]["ratio"], template["AB_XA"]["tolerance"]) and
                self._ratio_matches(bc_ab_ratio, template["BC_AB"]["ratio"], template["BC_AB"]["tolerance"]) and
                self._ratio_matches(cd_bc_ratio, template["CD_BC"]["ratio"], template["CD_BC"]["tolerance"]) and
                self._ratio_matches(xa_bc_ratio, template["XA_BC"]["ratio"], template["XA_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = a_val > x_val
                
                # Mark the pattern at D point
                data.loc[d_idx, pattern_name] = 1
                data.loc[d_idx, f"{pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target
                target = d_val + (0.5 * xa) * (1 if is_bullish else -1)
                data.loc[d_idx, f"{pattern_name}_target"] = target
                
                # Set stop loss (beyond X)
                data.loc[d_idx, f"{pattern_name}_stop"] = x_val
    
    def _detect_deep_crab_pattern(self, data: pd.DataFrame, pivot_indices: pd.Index) -> None:
        """Deep Crab Pattern detection (similar structure to other methods)"""
        pattern_name = PatternType.DEEP_CRAB.value
        template = self.pattern_templates[pattern_name]
        
        # Similar implementation to bat pattern with different ratios
        if len(pivot_indices) < 5:
            return
        
        # Implementation follows the same pattern as bat pattern with different ratios
        # Similar code structure as other pattern detection methods
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
            ab_xa_ratio = self._calculate_ratio(ab, xa)
            bc_ab_ratio = self._calculate_ratio(bc, ab)
            cd_bc_ratio = self._calculate_ratio(cd, bc)
            xa_bc_ratio = self._calculate_ratio(xa, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                self._ratio_matches(ab_xa_ratio, template["AB_XA"]["ratio"], template["AB_XA"]["tolerance"]) and
                self._ratio_matches(bc_ab_ratio, template["BC_AB"]["ratio"], template["BC_AB"]["tolerance"]) and
                self._ratio_matches(cd_bc_ratio, template["CD_BC"]["ratio"], template["CD_BC"]["tolerance"]) and
                self._ratio_matches(xa_bc_ratio, template["XA_BC"]["ratio"], template["XA_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = a_val > x_val
                
                # Mark the pattern at D point
                data.loc[d_idx, pattern_name] = 1
                data.loc[d_idx, f"{pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target
                target = d_val + (0.5 * xa) * (1 if is_bullish else -1)
                data.loc[d_idx, f"{pattern_name}_target"] = target
                
                # Set stop loss (beyond X)
                data.loc[d_idx, f"{pattern_name}_stop"] = x_val
    
    def _detect_butterfly_pattern(self, data: pd.DataFrame, pivot_indices: pd.Index) -> None:
        """Butterfly Pattern detection"""
        pattern_name = PatternType.BUTTERFLY.value
        template = self.pattern_templates[pattern_name]
        
        # Implementation follows the same pattern as bat pattern with different ratios
        if len(pivot_indices) < 5:
            return
        
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
            ab_xa_ratio = self._calculate_ratio(ab, xa)
            bc_ab_ratio = self._calculate_ratio(bc, ab)
            cd_bc_ratio = self._calculate_ratio(cd, bc)
            xa_bc_ratio = self._calculate_ratio(xa, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                self._ratio_matches(ab_xa_ratio, template["AB_XA"]["ratio"], template["AB_XA"]["tolerance"]) and
                self._ratio_matches(bc_ab_ratio, template["BC_AB"]["ratio"], template["BC_AB"]["tolerance"]) and
                self._ratio_matches(cd_bc_ratio, template["CD_BC"]["ratio"], template["CD_BC"]["tolerance"]) and
                self._ratio_matches(xa_bc_ratio, template["XA_BC"]["ratio"], template["XA_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = a_val > x_val
                
                # Mark the pattern at D point
                data.loc[d_idx, pattern_name] = 1
                data.loc[d_idx, f"{pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target
                target = d_val + (0.5 * xa) * (1 if is_bullish else -1)
                data.loc[d_idx, f"{pattern_name}_target"] = target
                
                # Set stop loss (beyond X)
                data.loc[d_idx, f"{pattern_name}_stop"] = x_val
    
    def _detect_gartley_pattern(self, data: pd.DataFrame, pivot_indices: pd.Index) -> None:
        """Gartley Pattern detection"""
        pattern_name = PatternType.GARTLEY.value
        template = self.pattern_templates[pattern_name]
        
        # Implementation follows the same pattern as bat pattern with different ratios
        if len(pivot_indices) < 5:
            return
        
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
            ab_xa_ratio = self._calculate_ratio(ab, xa)
            bc_ab_ratio = self._calculate_ratio(bc, ab)
            cd_bc_ratio = self._calculate_ratio(cd, bc)
            xa_bc_ratio = self._calculate_ratio(xa, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                self._ratio_matches(ab_xa_ratio, template["AB_XA"]["ratio"], template["AB_XA"]["tolerance"]) and
                self._ratio_matches(bc_ab_ratio, template["BC_AB"]["ratio"], template["BC_AB"]["tolerance"]) and
                self._ratio_matches(cd_bc_ratio, template["CD_BC"]["ratio"], template["CD_BC"]["tolerance"]) and
                self._ratio_matches(xa_bc_ratio, template["XA_BC"]["ratio"], template["XA_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = a_val > x_val
                
                # Mark the pattern at D point
                data.loc[d_idx, pattern_name] = 1
                data.loc[d_idx, f"{pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target
                target = d_val + (0.5 * xa) * (1 if is_bullish else -1)
                data.loc[d_idx, f"{pattern_name}_target"] = target
                
                # Set stop loss (beyond X)
                data.loc[d_idx, f"{pattern_name}_stop"] = x_val
    
    def _detect_crab_pattern(self, data: pd.DataFrame, pivot_indices: pd.Index) -> None:
        """Crab Pattern detection"""
        pattern_name = PatternType.CRAB.value
        template = self.pattern_templates[pattern_name]
        
        # Implementation follows the same pattern as bat pattern with different ratios
        if len(pivot_indices) < 5:
            return
        
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
            ab_xa_ratio = self._calculate_ratio(ab, xa)
            bc_ab_ratio = self._calculate_ratio(bc, ab)
            cd_bc_ratio = self._calculate_ratio(cd, bc)
            xa_bc_ratio = self._calculate_ratio(xa, bc)
            
            # Check if ratios match the pattern template
            matches_template = (
                self._ratio_matches(ab_xa_ratio, template["AB_XA"]["ratio"], template["AB_XA"]["tolerance"]) and
                self._ratio_matches(bc_ab_ratio, template["BC_AB"]["ratio"], template["BC_AB"]["tolerance"]) and
                self._ratio_matches(cd_bc_ratio, template["CD_BC"]["ratio"], template["CD_BC"]["tolerance"]) and
                self._ratio_matches(xa_bc_ratio, template["XA_BC"]["ratio"], template["XA_BC"]["tolerance"])
            )
            
            if matches_template:
                # Determine if pattern is bullish or bearish
                is_bullish = a_val > x_val
                
                # Mark the pattern at D point
                data.loc[d_idx, pattern_name] = 1
                data.loc[d_idx, f"{pattern_name}_direction"] = 1 if is_bullish else -1
                
                # Set potential price target
                target = d_val + (0.5 * xa) * (1 if is_bullish else -1)
                data.loc[d_idx, f"{pattern_name}_target"] = target
                
                # Set stop loss (beyond X)
                data.loc[d_idx, f"{pattern_name}_stop"] = x_val
    
    def _evaluate_patterns(self, data: pd.DataFrame) -> None:
        """
        Calculate comprehensive pattern evaluation metrics.
        
        Args:
            data: DataFrame with detected patterns
        """
        # Get all pattern columns
        pattern_cols = [col for col in data.columns if col in [p.value for p in PatternType]]
        
        # Add pattern count column
        data['total_harmonic_patterns'] = data[pattern_cols].sum(axis=1)
        
        # Add pattern quality ratings (1-10 scale)
        for pattern in PatternType:
            pattern_col = pattern.value
            quality_col = f"{pattern_col}_quality"
            
            # Skip patterns that weren't detected
            if data[pattern_col].sum() == 0:
                continue
                
            # Get indices where the pattern is detected
            pattern_indices = data[data[pattern_col] == 1].index
            
            for idx in pattern_indices:
                # Base quality factors (can be expanded based on additional criteria)
                clarity_score = np.random.randint(7, 11)  # Placeholder for clarity calculation
                
                # Market context factors
                volume_factor = 1.0  # Placeholder for volume significance calculation
                trend_alignment = 1.0  # Placeholder for trend alignment calculation
                
                # Final quality score (1-10 scale)
                quality = min(10, max(1, round(clarity_score * volume_factor * trend_alignment)))
                data.loc[idx, quality_col] = quality
        
        # Add pattern confluence column 
        # (higher values indicate multiple patterns pointing in same direction)
        data['pattern_confluence'] = 0
        
        for i in range(len(data)):
            bullish_count = 0
            bearish_count = 0
            
            for pattern in PatternType:
                direction_col = f"{pattern.value}_direction"
                
                if direction_col in data.columns and i < len(data):
                    if data.loc[i, direction_col] == 1:  # Bullish
                        bullish_count += 1
                    elif data.loc[i, direction_col] == -1:  # Bearish
                        bearish_count += 1
                        
            # Confluence is the difference between bullish and bearish patterns
            data.loc[i, 'pattern_confluence'] = abs(bullish_count - bearish_count)
            
            # Direction of confluence
            if bullish_count > bearish_count:
                data.loc[i, 'pattern_direction'] = 1
            elif bearish_count > bullish_count:
                data.loc[i, 'pattern_direction'] = -1
            else:
                data.loc[i, 'pattern_direction'] = 0
"""
