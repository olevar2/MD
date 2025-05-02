"""
Tests for advanced indicator optimization and error handling.

This module tests the performance optimization and error handling capabilities
for advanced indicator integration.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import importlib

# Silence warning logs during tests
logging.basicConfig(level=logging.ERROR)

from feature_store_service.indicators.advanced_indicator_adapter import (
    AdvancedIndicatorAdapter,
    AdvancedIndicatorError
)
from feature_store_service.indicators.advanced_indicator_optimization import (
    graceful_fallback,
    performance_tracking,
    optimizer
)


class TestIndicatorOptimization(unittest.TestCase):
    """Test optimization and error handling for advanced indicators."""
    
    @classmethod
    def setUpClass(cls):
        """Create sample data for testing."""
        # Create sample OHLCV data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        cls.test_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(102, 5, 100),
            'low': np.random.normal(98, 5, 100),
            'close': np.random.normal(101, 5, 100),
            'volume': np.random.normal(10000, 1000, 100)
        }, index=dates)
        
        # Ensure high >= open/close and low <= open/close
        for i in range(len(cls.test_data)):
            max_val = max(cls.test_data.iloc[i]['open'], cls.test_data.iloc[i]['close'])
            min_val = min(cls.test_data.iloc[i]['open'], cls.test_data.iloc[i]['close'])
            cls.test_data.iloc[i, cls.test_data.columns.get_loc('high')] = max(cls.test_data.iloc[i]['high'], max_val)
            cls.test_data.iloc[i, cls.test_data.columns.get_loc('low')] = min(cls.test_data.iloc[i]['low'], min_val)
    
    def setUp(self):
        """Reset optimizer before each test."""
        optimizer.clear_cache()
    
    def test_optimizer_caching(self):
        """Test that the optimizer correctly caches calculation results."""
        # Create a simple mock indicator that tracks calls
        call_count = [0]
        
        class MockAdvancedIndicator:
            def calculate(self, data):
                call_count[0] += 1
                result = data.copy()
                result['test_value'] = data['close'].rolling(window=10).mean()
                return result
        
        # Create adapter for our mock indicator
        adapter = AdvancedIndicatorAdapter(MockAdvancedIndicator, name_prefix='test')
        
        # First calculation should increase the call count
        result1 = adapter.calculate(self.test_data)
        self.assertEqual(call_count[0], 1)
        
        # Second calculation with same data should use cache
        result2 = adapter.calculate(self.test_data)
        self.assertEqual(call_count[0], 1)  # Call count shouldn't increase
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        
        # Calculation with different data should increment call count
        different_data = self.test_data.copy() * 2
        adapter.calculate(different_data)
        self.assertEqual(call_count[0], 2)
    
    def test_performance_tracking(self):
        """Test that performance tracking correctly identifies slow calculations."""
        # Create a mock slow indicator
        class SlowIndicator:
            def calculate(self, data):
                # Simulate slow calculation
                time.sleep(0.1)
                result = data.copy()
                result['slow_value'] = data['close'].mean()
                return result
        
        # Apply our performance decorator directly for testing
        original_calculate = SlowIndicator.calculate
        SlowIndicator.calculate = performance_tracking(threshold_ms=10)(original_calculate)
        
        # Create adapter
        slow_adapter = AdvancedIndicatorAdapter(SlowIndicator)
        
        # This should execute without errors (performance warning is logged)
        result = slow_adapter.calculate(self.test_data)
        
        # Verify result contains our calculated column
        self.assertIn('slow_value', result.columns)
    
    def test_graceful_fallback(self):
        """Test that error handling provides graceful fallback."""
        # Create a faulty indicator
        class FaultyIndicator:
            def calculate(self, data):
                # Simulate calculation error
                raise ValueError("Simulated calculation failure")
        
        # Apply our graceful fallback decorator directly for testing
        original_calculate = FaultyIndicator.calculate
        FaultyIndicator.calculate = graceful_fallback(fallback_value=None)(original_calculate)
        
        # Create adapter and calculate
        faulty_adapter = AdvancedIndicatorAdapter(FaultyIndicator)
        
        # This should not raise an exception due to graceful fallback
        result = faulty_adapter.calculate(self.test_data)
        
        # Result should be None due to our fallback value
        self.assertIsNone(result)
    
    def test_adapter_error_handling(self):
        """Test that adapter correctly handles errors from advanced indicators."""
        # Create a faulty indicator
        class FaultyIndicator:
            def calculate(self, data):
                # Simulate calculation error
                raise ValueError("Simulated calculation failure")
        
        # Create adapter (which has built-in error handling)
        faulty_adapter = AdvancedIndicatorAdapter(FaultyIndicator)
        
        # This should not raise an exception due to built-in error handling
        result = faulty_adapter.calculate(self.test_data)
        
        # Result should be the original data due to graceful fallback
        pd.testing.assert_frame_equal(result, self.test_data)
    
    def test_missing_analysis_engine(self):
        """Test behavior when Analysis Engine is unavailable."""
        # We'll simulate this by attempting to create an adapter for a nonexistent class
        try:
            # This should fail, but gracefully
            AdvancedIndicatorAdapter.create_adapter(
                "nonexistent_module", 
                "NonexistentClass"
            )
            self.fail("Should have raised an exception")
        except ImportError:
            # Expected behavior
            pass


if __name__ == '__main__':
    unittest.main()
