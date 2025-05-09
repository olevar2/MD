"""
Unit tests for harmonic patterns models module.
"""

import unittest
from feature_store_service.indicators.harmonic_patterns.models import (
    PatternType, get_pattern_templates, get_fibonacci_ratios
)


class TestHarmonicPatternsModels(unittest.TestCase):
    """Test cases for harmonic patterns models."""
    
    def test_pattern_type_enum(self):
        """Test PatternType enum values."""
        self.assertEqual(PatternType.BAT.value, "bat")
        self.assertEqual(PatternType.SHARK.value, "shark")
        self.assertEqual(PatternType.CYPHER.value, "cypher")
        self.assertEqual(PatternType.ABCD.value, "abcd")
        self.assertEqual(PatternType.THREE_DRIVES.value, "three_drives")
        self.assertEqual(PatternType.FIVE_ZERO.value, "five_zero")
        self.assertEqual(PatternType.ALT_BAT.value, "alt_bat")
        self.assertEqual(PatternType.DEEP_CRAB.value, "deep_crab")
        self.assertEqual(PatternType.BUTTERFLY.value, "butterfly")
        self.assertEqual(PatternType.GARTLEY.value, "gartley")
        self.assertEqual(PatternType.CRAB.value, "crab")
    
    def test_get_pattern_templates(self):
        """Test pattern templates retrieval."""
        templates = get_pattern_templates()
        
        # Should have templates for all pattern types
        for pattern_type in PatternType:
            self.assertIn(pattern_type.value, templates)
        
        # Check structure of a specific template
        bat_template = templates["bat"]
        self.assertIn("XA_BC", bat_template)
        self.assertIn("AB_XA", bat_template)
        self.assertIn("BC_AB", bat_template)
        self.assertIn("CD_BC", bat_template)
        
        # Each ratio should have a ratio and tolerance
        for ratio_key in bat_template:
            self.assertIn("ratio", bat_template[ratio_key])
            self.assertIn("tolerance", bat_template[ratio_key])
    
    def test_get_fibonacci_ratios(self):
        """Test Fibonacci ratios retrieval."""
        ratios = get_fibonacci_ratios()
        
        # Should have common Fibonacci ratios
        self.assertIn("0.382", ratios)
        self.assertIn("0.5", ratios)
        self.assertIn("0.618", ratios)
        self.assertIn("1.0", ratios)
        self.assertIn("1.618", ratios)
        
        # Check specific values
        self.assertEqual(ratios["0.618"], 0.618)
        self.assertEqual(ratios["1.618"], 1.618)


if __name__ == "__main__":
    unittest.main()