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
logging.basicConfig(level=logging.ERROR)
from feature_store_service.adapters.advanced_indicator_adapter import AdvancedIndicatorAdapter, FibonacciAnalyzerAdapter, load_advanced_indicators
from feature_store_service.indicators.indicator_registry import IndicatorRegistry
from feature_store_service.indicators.advanced_indicators_registrar import register_advanced_indicators, register_analysis_engine_indicators
from common_lib.indicators.indicator_interfaces import IBaseIndicator, IAdvancedIndicator, IFibonacciAnalyzer, IndicatorCategory, IIndicatorAdapter


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TestAdvancedIndicatorIntegration(unittest.TestCase):
    """Test integration between Feature Store and Analysis Engine indicators."""

    @classmethod
    def setUpClass(cls):
        """Set up test data that can be reused across test methods."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        cls.test_data = pd.DataFrame({'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(102, 5, 100), 'low': np.random.normal(
            98, 5, 100), 'close': np.random.normal(101, 5, 100), 'volume':
            np.random.normal(10000, 1000, 100)}, index=dates)
        for i in range(len(cls.test_data)):
            max_val = max(cls.test_data.iloc[i]['open'], cls.test_data.iloc
                [i]['close'])
            min_val = min(cls.test_data.iloc[i]['open'], cls.test_data.iloc
                [i]['close'])
            cls.test_data.iloc[i, cls.test_data.columns.get_loc('high')] = max(
                cls.test_data.iloc[i]['high'], max_val)
            cls.test_data.iloc[i, cls.test_data.columns.get_loc('low')] = min(
                cls.test_data.iloc[i]['low'], min_val)
        cls.registry = IndicatorRegistry()

    @with_exception_handling
    def test_load_advanced_indicators(self):
        """Test loading advanced indicators from analysis engine."""
        try:
            importlib.import_module('analysis_engine')
            indicators = load_advanced_indicators()
            self.assertIsInstance(indicators, dict)
            if indicators:
                self.assertGreater(len(indicators), 0)
                print(f'Found {len(indicators)} advanced indicators')
        except ImportError:
            self.skipTest('Analysis Engine module not available for testing')

    @with_exception_handling
    def test_advanced_indicator_adapter(self):
        """Test creating and using an advanced indicator adapter."""
        try:
            from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase


            class TestAdvancedIndicator(AdvancedAnalysisBase):
    """
    TestAdvancedIndicator class that inherits from AdvancedAnalysisBase.
    
    Attributes:
        Add attributes here
    """


                def calculate(self, data):
    """
    Calculate.
    
    Args:
        data: Description of data
    
    """

                    result = data.copy()
                    result['test_indicator'] = result['close'].rolling(window
                        =10).mean()
                    return result
            adapter = AdvancedIndicatorAdapter(TestAdvancedIndicator)
            result = adapter.calculate(self.test_data)
            self.assertIn('test_indicator', result.columns)
            self.assertTrue(np.isnan(result['test_indicator'].iloc[5]))
            self.assertFalse(np.isnan(result['test_indicator'].iloc[15]))
        except ImportError:
            self.skipTest('Analysis Engine module not available for testing')

    def test_register_advanced_indicators(self):
        """Test registering advanced indicators in the registry."""
        register_advanced_indicators(self.registry)
        indicators = self.registry.get_all_indicators()
        self.assertGreater(len(indicators), 0)
        advanced_categories = set(indicator_class.category for
            indicator_class in indicators.values())
        self.assertIn('advanced', advanced_categories)

    @with_exception_handling
    def test_fibonacci_indicator_integration(self):
        """Test Fibonacci indicators integration if available."""
        adapter = FibonacciAnalyzerAdapter()
        result = adapter.calculate_retracements(data=self.test_data,
            high_col='high', low_col='low', levels=[0.236, 0.382, 0.5, 
            0.618, 0.786])
        self.assertIn('fib_retracement_0_236', result.columns)
        self.assertIn('fib_retracement_0_382', result.columns)
        self.assertIn('fib_retracement_0_5', result.columns)
        self.assertIn('fib_retracement_0_618', result.columns)
        self.assertIn('fib_retracement_0_786', result.columns)
        result = adapter.calculate_extensions(data=self.test_data, high_col
            ='high', low_col='low', close_col='close', levels=[1.0, 1.272, 
            1.618, 2.0])
        self.assertIn('fib_extension_1_0', result.columns)
        self.assertIn('fib_extension_1_272', result.columns)
        self.assertIn('fib_extension_1_618', result.columns)
        self.assertIn('fib_extension_2_0', result.columns)
        try:
            from analysis_engine.analysis.advanced_ta.fibonacci import FibonacciRetracement
            wrapped_adapter = FibonacciAnalyzerAdapter(FibonacciRetracement())
            wrapped_result = wrapped_adapter.calculate_retracements(self.
                test_data)
            self.assertGreater(len(wrapped_result.columns), len(self.
                test_data.columns))
        except (ImportError, AttributeError, TypeError):
            print('Using standalone FibonacciAnalyzerAdapter implementation')

    @with_exception_handling
    def test_ml_integration(self):
        """Test integration with ML feature extraction."""
        try:
            from feature_store_service.indicators.ml_integration import FeatureExtractor
            register_analysis_engine_indicators(self.registry)
            extractor = FeatureExtractor()
            indicators = {name: (cls() if callable(cls) else cls) for name,
                cls in self.registry.get_all_indicators().items()}
            result_data = self.test_data.copy()
            for name, indicator in indicators.items():
                if hasattr(indicator, 'calculate'):
                    try:
                        if name in ['TEMA', 'RSI', 'MACD']:
                            tmp = indicator.calculate(result_data)
                            for col in tmp.columns:
                                if col not in result_data.columns:
                                    result_data[col] = tmp[col]
                    except:
                        pass
            features = extractor.extract_features(result_data, {name: {
                'type': name} for name in indicators.keys()})
            self.assertIsInstance(features, pd.DataFrame)
            self.assertGreater(len(features.columns), 0)
        except ImportError:
            self.skipTest('ML integration components not available')


if __name__ == '__main__':
    unittest.main()
