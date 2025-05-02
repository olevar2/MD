"""
Unit tests for indicator registry functionality.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from feature_store_service.indicators.indicator_registry import IndicatorRegistry
from feature_store_service.indicators.base_indicator import BaseIndicator

class SimpleTestIndicator(BaseIndicator):
    """Simple indicator for testing."""
    category = "test"
    params = {
        "period": {"type": "int", "min": 1, "max": 100, "default": 14},
        "source": {"type": "str", "options": ["close", "open"], "default": "close"}
    }
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

class TestIndicatorRegistry(unittest.TestCase):
    """Test suite for IndicatorRegistry."""

    def setUp(self):
        """Set up test environment."""
        self.registry = IndicatorRegistry()
        self.test_indicator = SimpleTestIndicator

    def test_register_indicator(self):
        """Test registering a new indicator."""
        self.registry.register_indicator(self.test_indicator)
        registered = self.registry.get_indicator("SimpleTestIndicator")
        self.assertEqual(registered, self.test_indicator)

    def test_register_duplicate(self):
        """Test registering duplicate indicator."""
        self.registry.register_indicator(self.test_indicator)
        # Should not raise error, just log warning
        self.registry.register_indicator(self.test_indicator)
        registered = self.registry.get_indicator("SimpleTestIndicator")
        self.assertEqual(registered, self.test_indicator)

    def test_get_indicators_by_category(self):
        """Test retrieving indicators by category."""
        self.registry.register_indicator(self.test_indicator)
        indicators = self.registry.get_indicators_by_category("test")
        self.assertIn("SimpleTestIndicator", indicators)
        self.assertEqual(len(indicators), 1)

        # Test non-existent category
        empty_indicators = self.registry.get_indicators_by_category("non_existent")
        self.assertEqual(len(empty_indicators), 0)

    def test_create_indicator(self):
        """Test creating indicator instance."""
        self.registry.register_indicator(self.test_indicator)
        
        # Test with valid parameters
        indicator = self.registry.create_indicator(
            "SimpleTestIndicator",
            period=20,
            source="close"
        )
        self.assertIsInstance(indicator, SimpleTestIndicator)
        self.assertEqual(indicator.period, 20)
        self.assertEqual(indicator.source, "close")

        # Test with invalid indicator ID
        invalid_indicator = self.registry.create_indicator("NonExistentIndicator")
        self.assertIsNone(invalid_indicator)

    def test_validate_parameters(self):
        """Test parameter validation."""
        self.registry.register_indicator(self.test_indicator)
        
        # Test valid parameters
        valid_params = {"period": 20, "source": "close"}
        validated = self.registry.validate_parameters("SimpleTestIndicator", valid_params)
        self.assertEqual(validated["period"], 20)
        self.assertEqual(validated["source"], "close")

        # Test invalid parameters
        invalid_params = {"period": 0, "source": "invalid"}
        with self.assertRaises(ValueError):
            self.registry.validate_parameters("SimpleTestIndicator", invalid_params)

    def test_get_all_indicators(self):
        """Test getting all registered indicators."""
        self.registry.register_indicator(self.test_indicator)
        all_indicators = self.registry.get_all_indicators()
        self.assertIn("SimpleTestIndicator", all_indicators)
        self.assertEqual(len(all_indicators), 1)

    def test_get_categories(self):
        """Test getting all indicator categories."""
        self.registry.register_indicator(self.test_indicator)
        categories = self.registry.get_categories()
        self.assertIn("test", categories)
        self.assertEqual(len(categories), 1)

if __name__ == "__main__":
    unittest.main()
