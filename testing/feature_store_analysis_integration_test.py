"""
Integration Tests for Feature Store and Analysis Engine Services

This module provides tests to verify the proper interaction between
the feature-store-service and analysis-engine-service, ensuring that
indicators can be calculated and visualized correctly.
"""

import unittest
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import pytest
import json

# Add necessary paths to system path
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))

# Import services
from feature_store_service.indicators import indicator_registry
from feature_store_service.indicators.moving_averages import simple_moving_average, exponential_moving_average
from feature_store_service.indicators.oscillators import relative_strength_index, macd
from feature_store_service.indicators.volatility import bollinger_bands, average_true_range
from feature_store_service.services.indicator_service import IndicatorService
from feature_store_service.services.data_access_service import DataAccessService

from analysis_engine.analysis.signal_system import SignalSystem
from analysis_engine.analysis.indicator_interface import IndicatorInterface
from analysis_engine.analysis.pattern_recognition import PatternRecognitionService
from analysis_engine.services.analysis_service import AnalysisService

from ui_service.components.visualization_adapter import VisualizationAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestData:
    """Helper class to generate test data"""
    
    @staticmethod
    def generate_price_data(days=30, freq='1h'):
        """Generate synthetic price data for testing"""
        periods = 24 * days if freq == '1h' else days
        dates = pd.date_range(start='2023-01-01', periods=periods, freq=freq)
        
        # Create price data with trend and noise
        np.random.seed(42)  # For reproducibility
        base = 100
        trend = np.linspace(0, 20, periods)
        noise = np.random.normal(0, 2, periods)
        cycle = 10 * np.sin(np.linspace(0, 4*np.pi, periods))
        
        close_prices = base + trend + noise + cycle
        
        # Create OHLCV data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices - np.random.uniform(0, 1, periods),
            'high': close_prices + np.random.uniform(0.5, 1.5, periods),
            'low': close_prices - np.random.uniform(0.5, 1.5, periods),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, periods)
        })
        
        # Set timestamp as index
        data.set_index('timestamp', inplace=True)
        return data


class FeatureStoreAnalysisEngineIntegrationTest(unittest.TestCase):
    """Integration tests for feature store and analysis engine services"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Initialize services
        cls.registry = indicator_registry.IndicatorRegistry()
        cls.registry.register_common_indicators()
        
        cls.indicator_service = IndicatorService()
        cls.analysis_service = AnalysisService()
        cls.indicator_interface = IndicatorInterface()
        cls.visualization_adapter = VisualizationAdapter()
        
        # Generate test data
        cls.price_data = TestData.generate_price_data()
        cls.daily_data = TestData.generate_price_data(days=365, freq='1d')
        
        logger.info(f"Test setup complete with {len(cls.price_data)} hourly and {len(cls.daily_data)} daily data points")
    
    def test_indicator_calculation_integration(self):
        """Test that indicators can be calculated correctly from the feature store service"""
        # Calculate various indicators with indicator service
        indicators = [
            {'name': 'sma', 'params': {'period': 20}},
            {'name': 'ema', 'params': {'period': 20}},
            {'name': 'rsi', 'params': {'period': 14}},
            {'name': 'bollinger_bands', 'params': {'period': 20, 'deviations': 2.0}}
        ]
        
        for indicator in indicators:
            # Calculate using indicator service
            result = self.indicator_service.calculate_indicator(
                indicator['name'], 
                self.price_data, 
                **indicator['params']
            )
            
            self.assertIsNotNone(result, f"Failed to calculate {indicator['name']}")
            
            if isinstance(result, pd.DataFrame):
                self.assertFalse(result.empty, f"{indicator['name']} returned empty DataFrame")
                self.assertTrue(all(~result.isna().all()), f"{indicator['name']} contains all NaN columns")
            elif isinstance(result, pd.Series):
                self.assertFalse(result.empty, f"{indicator['name']} returned empty Series")
                self.assertFalse(result.isna().all(), f"{indicator['name']} contains all NaN values")
            
            logger.info(f"Successfully calculated {indicator['name']}")
    
    def test_indicator_interface_integration(self):
        """Test that the analysis engine can access indicators through the indicator interface"""
        # Define indicators to test
        indicators = [
            {'name': 'sma', 'params': {'period': 20}},
            {'name': 'rsi', 'params': {'period': 14}},
            {'name': 'macd', 'params': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}}
        ]
        
        for indicator in indicators:
            # Request indicator through interface
            result = self.indicator_interface.get_indicator(
                indicator['name'],
                'EURUSD',
                '1h',
                self.price_data,
                indicator['params']
            )
            
            self.assertIsNotNone(result, f"Indicator interface failed to get {indicator['name']}")
            
            # Verify result structure
            self.assertIn('data', result, f"Missing data in {indicator['name']} result")
            self.assertIn('metadata', result, f"Missing metadata in {indicator['name']} result")
            
            logger.info(f"Successfully retrieved {indicator['name']} through indicator interface")
    
    def test_signal_generation_with_indicators(self):
        """Test that the signal system can generate signals based on indicators"""
        # Initialize signal system
        signal_system = SignalSystem()
        
        # Calculate required indicators
        sma_20 = simple_moving_average(self.price_data, period=20)
        sma_50 = simple_moving_average(self.price_data, period=50)
        rsi_14 = relative_strength_index(self.price_data, period=14)
        
        # Create indicator data package
        indicators_data = {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi_14': rsi_14,
            'price': self.price_data['close']
        }
        
        # Generate signals with simple strategy: SMA crossover with RSI confirmation
        strategy_config = {
            'name': 'sma_crossover_with_rsi',
            'rules': [
                {'type': 'crossover', 'indicator1': 'sma_20', 'indicator2': 'sma_50', 'direction': 'above'},
                {'type': 'threshold', 'indicator': 'rsi_14', 'condition': 'above', 'value': 50}
            ]
        }
        
        signals = signal_system.generate_signals(indicators_data, [strategy_config])
        
        # Verify signals
        self.assertIsNotNone(signals, "Signal system failed to generate signals")
        self.assertIsInstance(signals, pd.DataFrame, "Signals should be a DataFrame")
        self.assertIn('signal', signals.columns, "Signals DataFrame should have 'signal' column")
        
        # Check if any signals were generated
        signal_count = (signals['signal'] != 0).sum()
        logger.info(f"Generated {signal_count} signals with SMA crossover and RSI confirmation strategy")
        
        # We expect at least a few signals in our test data
        self.assertGreater(signal_count, 0, "No signals were generated")
    
    def test_pattern_recognition_integration(self):
        """Test pattern recognition integrated with indicator data"""
        # Initialize pattern recognition service
        pattern_service = PatternRecognitionService()
        
        # Detect patterns
        patterns = pattern_service.detect_candlestick_patterns(self.daily_data)
        
        self.assertIsNotNone(patterns, "Pattern recognition failed to detect patterns")
        self.assertIsInstance(patterns, pd.DataFrame, "Patterns should be a DataFrame")
        
        # Verify integration with analysis service
        analysis_results = self.analysis_service.analyze_price_action(
            self.daily_data, 
            detect_patterns=True
        )
        
        self.assertIn('patterns', analysis_results, "Analysis results should include patterns")
        self.assertIn('support_resistance', analysis_results, "Analysis results should include support/resistance levels")
        
        logger.info("Successfully completed pattern recognition integration test")
    
    def test_visualization_adapter_integration(self):
        """Test that the visualization adapter can connect indicators to visualizations"""
        # Get available indicators
        indicators = self.visualization_adapter.get_indicator_metadata()
        self.assertGreater(len(indicators), 0, "No indicators available in visualization adapter")
        
        # Test indicator calculation and visualization preparation for a few indicators
        test_indicators = ['sma', 'rsi', 'bollinger_bands']
        
        for indicator_name in test_indicators:
            # Calculate indicator
            indicator_data = self.visualization_adapter.calculate_indicator(
                indicator_name, self.price_data, {'period': 14}
            )
            
            self.assertIsNotNone(indicator_data, f"Failed to calculate {indicator_name}")
            
            # Prepare visualization data
            viz_data = self.visualization_adapter.prepare_visualization_data(
                indicator_name, indicator_data, self.price_data, {'period': 14}
            )
            
            self.assertIsNotNone(viz_data, f"Failed to prepare visualization data for {indicator_name}")
            self.assertIn('type', viz_data, f"Visualization data for {indicator_name} missing 'type'")
            self.assertIn('data', viz_data, f"Visualization data for {indicator_name} missing 'data'")
            
            logger.info(f"Successfully prepared visualization data for {indicator_name}")
    
    def test_end_to_end_indicator_flow(self):
        """Test the complete flow from data to indicators to analysis to visualization"""
        # 1. Fetch data
        data = self.price_data.copy()
        
        # 2. Calculate indicators with feature store service
        indicators = {}
        indicators['sma_20'] = self.indicator_service.calculate_indicator('sma', data, period=20)
        indicators['sma_50'] = self.indicator_service.calculate_indicator('sma', data, period=50)
        indicators['rsi_14'] = self.indicator_service.calculate_indicator('rsi', data, period=14)
        
        # 3. Analyze with analysis engine
        analysis_results = self.analysis_service.analyze_indicator_combination(
            indicators, data, strategy_type='trend_following'
        )
        
        self.assertIsNotNone(analysis_results, "Analysis service failed to analyze indicators")
        self.assertIn('signals', analysis_results, "Analysis results missing signals")
        self.assertIn('confluence', analysis_results, "Analysis results missing confluence data")
        
        # 4. Prepare visualization through adapter
        visualization_results = {}
        for name, indicator_data in indicators.items():
            viz_data = self.visualization_adapter.prepare_visualization_data(
                name.split('_')[0],  # Extract base name (sma, rsi)
                indicator_data,
                data,
                {'period': int(name.split('_')[1]) if '_' in name else 14}
            )
            visualization_results[name] = viz_data
        
        # 5. Verify integration
        self.assertEqual(len(visualization_results), 3, "Visualization results incomplete")
        
        for name, viz_data in visualization_results.items():
            self.assertIsNotNone(viz_data, f"Visualization data missing for {name}")
            self.assertIn('data', viz_data, f"Visualization data for {name} missing 'data' key")
        
        logger.info("Successfully completed end-to-end indicator flow test")


@pytest.mark.asyncio
async def test_async_indicator_calculation():
    """Test asynchronous indicator calculation across services"""
    # Initialize services
    registry = indicator_registry.IndicatorRegistry()
    registry.register_common_indicators()
    
    # Generate test data
    price_data = TestData.generate_price_data()
    
    # Calculate multiple indicators asynchronously
    indicators = [
        {'name': 'sma', 'params': {'period': 10}},
        {'name': 'sma', 'params': {'period': 20}},
        {'name': 'sma', 'params': {'period': 50}},
        {'name': 'ema', 'params': {'period': 10}},
        {'name': 'ema', 'params': {'period': 20}},
        {'name': 'bollinger_bands', 'params': {'period': 20, 'deviations': 2.0}},
        {'name': 'rsi', 'params': {'period': 14}}
    ]
    
    # Simulate async calculation
    import asyncio
    
    async def calculate_indicator(name, params):
        # In real implementation, this would call an async service
        await asyncio.sleep(0.1)  # Simulate async operation
        return registry.calculate(name, price_data, **params)
    
    # Calculate all indicators simultaneously
    tasks = [calculate_indicator(ind['name'], ind['params']) for ind in indicators]
    results = await asyncio.gather(*tasks)
    
    # Verify results
    assert len(results) == len(indicators), "Not all indicators were calculated"
    assert all(result is not None for result in results), "Some indicators failed to calculate"
    
    logger.info(f"Successfully calculated {len(results)} indicators asynchronously")
    return results


def test_large_data_performance():
    """Test performance with large datasets"""
    # Generate large dataset
    large_data = TestData.generate_price_data(days=365, freq='1h')
    
    # Initialize services
    registry = indicator_registry.IndicatorRegistry()
    registry.register_common_indicators()
    
    # Define indicators to test
    indicators = [
        {'name': 'sma', 'params': {'period': 20}},
        {'name': 'ema', 'params': {'period': 20}},
        {'name': 'bollinger_bands', 'params': {'period': 20, 'deviations': 2.0}},
        {'name': 'rsi', 'params': {'period': 14}}
    ]
    
    # Measure performance
    import time
    
    results = {}
    for indicator in indicators:
        start_time = time.time()
        result = registry.calculate(indicator['name'], large_data, **indicator['params'])
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        results[indicator['name']] = {
            'execution_time_ms': execution_time,
            'data_points': len(result)
        }
        
        # Basic validation
        assert result is not None, f"Failed to calculate {indicator['name']}"
        
        logger.info(f"Calculated {indicator['name']} with {len(large_data)} data points in {execution_time:.2f} ms")
    
    # Assert reasonable performance (adjust thresholds based on your expectations)
    for indicator_name, perf in results.items():
        assert perf['execution_time_ms'] < 5000, f"{indicator_name} took too long: {perf['execution_time_ms']:.2f} ms"
    
    return results


if __name__ == "__main__":
    unittest.main()
""""""
