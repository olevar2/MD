"""
Unit tests for harmonic patterns detectors.
"""

import unittest
import pandas as pd
import numpy as np
from feature_store_service.indicators.harmonic_patterns.models import get_pattern_templates
from feature_store_service.indicators.harmonic_patterns.utils import identify_pivots
from feature_store_service.indicators.harmonic_patterns.detectors.bat import BatPatternDetector
from feature_store_service.indicators.harmonic_patterns.detectors.butterfly import ButterflyPatternDetector
from feature_store_service.indicators.harmonic_patterns.detectors.gartley import GartleyPatternDetector


class TestHarmonicPatternsDetectors(unittest.TestCase):
    """Test cases for harmonic patterns detectors."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.df = pd.DataFrame({
            'open': np.random.normal(100, 1, 100),
            'high': np.random.normal(101, 1, 100),
            'low': np.random.normal(99, 1, 100),
            'close': np.random.normal(100, 1, 100)
        }, index=dates)
        
        # Ensure high is always >= close and open
        self.df['high'] = self.df[['high', 'open', 'close']].max(axis=1)
        
        # Ensure low is always <= close and open
        self.df['low'] = self.df[['low', 'open', 'close']].min(axis=1)
        
        # Create a pattern-like structure
        # X point (low)
        self.df.iloc[10, self.df.columns.get_loc('low')] = 95
        
        # A point (high)
        self.df.iloc[20, self.df.columns.get_loc('high')] = 105
        
        # B point (low)
        self.df.iloc[30, self.df.columns.get_loc('low')] = 98
        
        # C point (high)
        self.df.iloc[40, self.df.columns.get_loc('high')] = 102
        
        # D point (low)
        self.df.iloc[50, self.df.columns.get_loc('low')] = 96
        
        # Identify pivots
        self.df_with_pivots = identify_pivots(self.df)
        
        # Get pivot indices
        self.pivot_indices = self.df_with_pivots[
            (self.df_with_pivots['pivot_high'] == 1) | 
            (self.df_with_pivots['pivot_low'] == 1)
        ].index
        
        # Get pattern templates
        self.templates = get_pattern_templates(tolerance=0.1)  # Use higher tolerance for testing
    
    def test_bat_detector(self):
        """Test BatPatternDetector."""
        detector = BatPatternDetector(
            max_pattern_bars=50,
            pattern_template=self.templates["bat"]
        )
        
        # Run detection
        result = detector.detect(self.df_with_pivots, self.pivot_indices)
        
        # Check that the detector added the expected columns
        self.assertIn("bat", result.columns)
        self.assertIn("bat_direction", result.columns)
        self.assertIn("bat_target", result.columns)
        self.assertIn("bat_stop", result.columns)
    
    def test_butterfly_detector(self):
        """Test ButterflyPatternDetector."""
        detector = ButterflyPatternDetector(
            max_pattern_bars=50,
            pattern_template=self.templates["butterfly"]
        )
        
        # Run detection
        result = detector.detect(self.df_with_pivots, self.pivot_indices)
        
        # Check that the detector added the expected columns
        self.assertIn("butterfly", result.columns)
        self.assertIn("butterfly_direction", result.columns)
        self.assertIn("butterfly_target", result.columns)
        self.assertIn("butterfly_stop", result.columns)
    
    def test_gartley_detector(self):
        """Test GartleyPatternDetector."""
        detector = GartleyPatternDetector(
            max_pattern_bars=50,
            pattern_template=self.templates["gartley"]
        )
        
        # Run detection
        result = detector.detect(self.df_with_pivots, self.pivot_indices)
        
        # Check that the detector added the expected columns
        self.assertIn("gartley", result.columns)
        self.assertIn("gartley_direction", result.columns)
        self.assertIn("gartley_target", result.columns)
        self.assertIn("gartley_stop", result.columns)


if __name__ == "__main__":
    unittest.main()