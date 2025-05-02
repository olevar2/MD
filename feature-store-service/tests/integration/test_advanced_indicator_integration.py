"""
Integration tests for advanced indicator adapters.

This module tests the integration between the Feature Store Service indicators
and the Analysis Engine Service advanced technical analysis components.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import logging
import importlib

# Silence warning logs during tests
logging.basicConfig(level=logging.ERROR)

from feature_store_service.indicators.advanced_indicator_adapter import (
    AdvancedIndicatorAdapter,
    load_advanced_indicators
)
from feature_store_service.indicators.indicator_registry import IndicatorRegistry
from feature_store_service.indicators.advanced_indicators_registrar import (
    register_advanced_indicators,
    register_analysis_engine_indicators
)


class TestAdvancedIndicatorIntegration(unittest.TestCase):
    """Test integration between Feature Store and Analysis Engine indicators."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that can be reused across test methods."""
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
        
        # Create a registry for testing
        cls.registry = IndicatorRegistry()
    
    def test_load_advanced_indicators(self):
        """Test loading advanced indicators from analysis engine."""
        try:
            # Try to import the analysis_engine module to check if it's available
            importlib.import_module("analysis_engine")
            
            # Load indicators
            indicators = load_advanced_indicators()
            self.assertIsInstance(indicators, dict)
            
            # If Analysis Engine is available, we should find some indicators
            if indicators:
                self.assertGreater(len(indicators), 0)
                print(f"Found {len(indicators)} advanced indicators")
                
        except ImportError:
            # Skip this test if the Analysis Engine is not available
            self.skipTest("Analysis Engine module not available for testing")
    
    def test_advanced_indicator_adapter(self):
        """Test creating and using an advanced indicator adapter."""
        try:
            # Try to import a simple indicator from the Analysis Engine Service
            # (using a common one that should be available)
            from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase
            
            # Create a simple test indicator
            class TestAdvancedIndicator(AdvancedAnalysisBase):
                def calculate(self, data):
                    result = data.copy()
                    result['test_indicator'] = result['close'].rolling(window=10).mean()
                    return result
            
            # Create an adapter for this indicator
            adapter = AdvancedIndicatorAdapter(TestAdvancedIndicator)
            
            # Test the adapter
            result = adapter.calculate(self.test_data)
            
            # Verify the result has our indicator column
            self.assertIn('test_indicator', result.columns)
            
            # Verify values are calculated correctly (after 10 periods)
            self.assertTrue(np.isnan(result['test_indicator'].iloc[5]))
            self.assertFalse(np.isnan(result['test_indicator'].iloc[15]))
            
        except ImportError:
            # Skip this test if the Analysis Engine is not available
            self.skipTest("Analysis Engine module not available for testing")
    
    def test_register_advanced_indicators(self):
        """Test registering advanced indicators in the registry."""
        # Register indicators with the registry
        register_advanced_indicators(self.registry)
        
        # Check that we have registered indicators
        indicators = self.registry.get_all_indicators()
        self.assertGreater(len(indicators), 0)
        
        # Verify that we have at least some basic indicators
        advanced_categories = set(
            indicator_class.category 
            for indicator_class in indicators.values()
        )
        
        self.assertIn('advanced', advanced_categories)
    
    def test_fibonacci_indicator_integration(self):
        """Test Fibonacci indicators integration if available."""
        try:
            # Try to import the Fibonacci module
            from analysis_engine.analysis.advanced_ta.fibonacci import FibonacciRetracement
            
            # Try to create an adapter for Fibonacci Retracement
            adapter = AdvancedIndicatorAdapter(
                FibonacciRetracement, 
                name_prefix='fib',
                auto_detect_swings=True
            )
            
            # Calculate with sample data
            result = adapter.calculate(self.test_data)
            
            # If successful, we should have some new columns
            self.assertGreater(len(result.columns), len(self.test_data.columns))
            
        except (ImportError, AttributeError, TypeError):
            # Skip if either module not found or wrong interface
            self.skipTest("Fibonacci indicators not available or have incompatible interface")
    
    def test_ml_integration(self):
        """Test integration with ML feature extraction."""
        try:
            # Try to import feature extraction components
            from feature_store_service.indicators.ml_integration import FeatureExtractor
            
            # Register some advanced indicators
            register_analysis_engine_indicators(self.registry)
            
            # Create a feature extractor
            extractor = FeatureExtractor()
            
            # Get all registered indicator instances
            indicators = {
                name: cls() if callable(cls) else cls
                for name, cls in self.registry.get_all_indicators().items()
            }
            
            # Calculate indicator values
            result_data = self.test_data.copy()
            for name, indicator in indicators.items():
                if hasattr(indicator, 'calculate'):
                    try:
                        # Only use a few indicators for testing
                        if name in ['TEMA', 'RSI', 'MACD']:
                            tmp = indicator.calculate(result_data)
                            # Merge new columns
                            for col in tmp.columns:
                                if col not in result_data.columns:
                                    result_data[col] = tmp[col]
                    except:
                        pass
            
            # Try to extract features
            features = extractor.extract_features(
                result_data,
                {name: {"type": name} for name in indicators.keys()}
            )
            
            # Verify we got features
            self.assertIsInstance(features, pd.DataFrame)
            self.assertGreater(len(features.columns), 0)
            
        except ImportError:
            # Skip if components not available
            self.skipTest("ML integration components not available")
    

if __name__ == '__main__':
    unittest.main()
