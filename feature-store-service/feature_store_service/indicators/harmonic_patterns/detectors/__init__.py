"""
Harmonic Pattern Detectors Package.

This package provides detector classes for various harmonic patterns.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Tuple

from feature_store_service.indicators.harmonic_patterns.utils import (
    calculate_ratio, ratio_matches
)


class BasePatternDetector(ABC):
    """
    Base class for harmonic pattern detectors.
    
    This abstract class defines the interface for all harmonic pattern detectors.
    """
    
    def __init__(self, max_pattern_bars: int = 100, pattern_template: dict = None):
        """
        Initialize base pattern detector.
        
        Args:
            max_pattern_bars: Maximum number of bars to look back for pattern detection
            pattern_template: Template with ratio requirements for the pattern
        """
        self.max_pattern_bars = max_pattern_bars
        self.pattern_template = pattern_template
        
    @abstractmethod
    def detect(self, data: pd.DataFrame, pivot_indices: pd.Index) -> pd.DataFrame:
        """
        Detect pattern in the given data.
        
        Args:
            data: DataFrame with OHLCV data and pivot points
            pivot_indices: Index of pivot points
            
        Returns:
            DataFrame with detected patterns
        """
        pass
    
    def _check_alternating_pivots(self, potential_waves: List[Tuple]) -> bool:
        """
        Verify alternating high/low pattern for waves.
        
        Args:
            potential_waves: List of potential wave points
            
        Returns:
            True if alternating, False otherwise
        """
        for i in range(1, len(potential_waves)):
            if potential_waves[i][2] == potential_waves[i-1][2]:  # is_high is the same
                return False
        return True