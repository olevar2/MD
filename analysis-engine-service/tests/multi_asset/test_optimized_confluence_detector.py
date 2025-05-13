"""
Unit tests for the optimized confluence detector.

This module contains tests for the OptimizedConfluenceDetector class.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import asyncio
from unittest.mock import MagicMock, patch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.
    abspath(__file__)))))
try:
    from analysis_engine.multi_asset.optimized_confluence_detector import OptimizedConfluenceDetector
    from common_lib.caching import AdaptiveCacheManager, cached, get_cache_manager
    from common_lib.parallel import ParallelProcessor, get_parallel_processor
    from analysis_engine.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame
except ImportError as e:
    print(f'Error importing modules: {e}')
    try:
        sys.path.insert(0, 'D:\\MD\\forex_trading_platform')
        from analysis_engine.multi_asset.optimized_confluence_detector import OptimizedConfluenceDetector
        from common_lib.caching import AdaptiveCacheManager, cached, get_cache_manager
        from common_lib.parallel import ParallelProcessor, get_parallel_processor
        from analysis_engine.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame
    except ImportError as e:
        print(f'Error importing modules with full path: {e}')
        sys.exit(1)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TestOptimizedConfluenceDetector(unittest.TestCase):
    """Test the optimized confluence detector."""

    def setUp(self):
        """Set up test fixtures."""
        self.correlation_service = MagicMock()
        self.currency_strength_analyzer = MagicMock()
        self.detector = OptimizedConfluenceDetector(correlation_service=
            self.correlation_service, currency_strength_analyzer=self.
            currency_strength_analyzer, correlation_threshold=0.7,
            lookback_periods=20, cache_ttl_minutes=60, max_workers=4)
        self.price_data = self._create_test_price_data()
        self.related_pairs = {'GBPUSD': 0.85, 'AUDUSD': 0.75, 'USDCAD': -
            0.65, 'USDJPY': -0.55}

        async def mock_get_all_correlations():
    """
    Mock get all correlations.
    
    """

            return {'EURUSD': {'GBPUSD': 0.85, 'AUDUSD': 0.75, 'USDCAD': -
                0.65, 'USDJPY': -0.55}}
        self.correlation_service.get_all_correlations = (
            mock_get_all_correlations)
        (self.currency_strength_analyzer.calculate_currency_strength.
            return_value) = ({'EUR': 0.8, 'USD': -0.3, 'GBP': 0.6, 'AUD': 
            0.4, 'CAD': -0.2, 'JPY': -0.5})

    def _create_test_price_data(self):
        """Create test price data for multiple pairs."""
        pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDJPY']
        data = {}
        for pair in pairs:
            dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
            close = np.random.normal(1.2, 0.02, 100)
            if pair in ['EURUSD', 'GBPUSD', 'AUDUSD']:
                close = close + np.linspace(0, 0.05, 100)
            else:
                close = close - np.linspace(0, 0.05, 100)
            close = close + np.random.normal(0, 0.005, 100)
            df = pd.DataFrame({'open': close - np.random.normal(0, 0.002, 
                100), 'high': close + np.random.normal(0.003, 0.001, 100),
                'low': close - np.random.normal(0.003, 0.001, 100), 'close':
                close, 'volume': np.random.normal(1000, 200, 100)}, index=dates
                )
            data[pair] = df
        return data

    @with_exception_handling
    def test_find_related_pairs(self):
        """Test finding related pairs."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            related_pairs = loop.run_until_complete(self.detector.
                find_related_pairs('EURUSD'))
            self.assertGreaterEqual(len(related_pairs), 1)
            found_expected_pair = False
            for pair in ['GBPUSD', 'AUDUSD', 'USDCAD', 'USDJPY']:
                if pair in related_pairs:
                    found_expected_pair = True
                    break
            self.assertTrue(found_expected_pair,
                'No expected related pairs found')
            start_time = time.time()
            related_pairs = loop.run_until_complete(self.detector.
                find_related_pairs('EURUSD'))
            execution_time = time.time() - start_time
            self.assertLess(execution_time, 0.01)
        finally:
            loop.close()

    def test_detect_confluence_optimized(self):
        """Test optimized confluence detection."""
        result = self.detector.detect_confluence_optimized(symbol='EURUSD',
            price_data=self.price_data, signal_type='trend',
            signal_direction='bullish', related_pairs=self.related_pairs)
        self.assertIn('symbol', result)
        self.assertIn('signal_type', result)
        self.assertIn('signal_direction', result)
        self.assertIn('confirmations', result)
        self.assertIn('contradictions', result)
        self.assertIn('confluence_score', result)
        self.assertEqual(result['symbol'], 'EURUSD')
        self.assertEqual(result['signal_type'], 'trend')
        self.assertEqual(result['signal_direction'], 'bullish')
        self.assertGreaterEqual(result['confluence_score'], 0.0)
        self.assertLessEqual(result['confluence_score'], 1.0)
        start_time = time.time()
        result2 = self.detector.detect_confluence_optimized(symbol='EURUSD',
            price_data=self.price_data, signal_type='trend',
            signal_direction='bullish', related_pairs=self.related_pairs)
        execution_time = time.time() - start_time
        self.assertLess(execution_time, 0.01)

    def test_analyze_divergence_optimized(self):
        """Test optimized divergence analysis."""
        result = self.detector.analyze_divergence_optimized(symbol='EURUSD',
            price_data=self.price_data, related_pairs=self.related_pairs)
        self.assertIn('symbol', result)
        self.assertIn('divergences', result)
        self.assertIn('divergences_found', result)
        self.assertIn('divergence_score', result)
        self.assertEqual(result['symbol'], 'EURUSD')
        self.assertIsInstance(result['divergences'], list)
        self.assertIsInstance(result['divergences_found'], int)
        self.assertGreaterEqual(result['divergence_score'], 0.0)
        self.assertLessEqual(result['divergence_score'], 1.0)
        start_time = time.time()
        result2 = self.detector.analyze_divergence_optimized(symbol=
            'EURUSD', price_data=self.price_data, related_pairs=self.
            related_pairs)
        execution_time = time.time() - start_time
        self.assertLess(execution_time, 0.01)

    def test_calculate_trend_strength(self):
        """Test trend strength calculation."""
        price_data = MemoryOptimizedDataFrame(self.price_data['EURUSD'])
        strength = self.detector._calculate_trend_strength(price_data,
            'bullish')
        self.assertGreaterEqual(strength, -1.0)
        self.assertLessEqual(strength, 1.0)
        strength = self.detector._calculate_trend_strength(price_data,
            'bearish')
        self.assertGreaterEqual(strength, -1.0)
        self.assertLessEqual(strength, 1.0)

    def test_calculate_reversal_strength(self):
        """Test reversal strength calculation."""
        price_data = MemoryOptimizedDataFrame(self.price_data['EURUSD'])
        strength = self.detector._calculate_reversal_strength(price_data,
            'bullish')
        self.assertGreaterEqual(strength, -1.0)
        self.assertLessEqual(strength, 1.0)
        strength = self.detector._calculate_reversal_strength(price_data,
            'bearish')
        self.assertGreaterEqual(strength, -1.0)
        self.assertLessEqual(strength, 1.0)

    def test_calculate_breakout_strength(self):
        """Test breakout strength calculation."""
        price_data = MemoryOptimizedDataFrame(self.price_data['EURUSD'])
        strength = self.detector._calculate_breakout_strength(price_data,
            'bullish')
        self.assertGreaterEqual(strength, -1.0)
        self.assertLessEqual(strength, 1.0)
        strength = self.detector._calculate_breakout_strength(price_data,
            'bearish')
        self.assertGreaterEqual(strength, -1.0)
        self.assertLessEqual(strength, 1.0)

    def test_calculate_momentum(self):
        """Test momentum calculation."""
        momentum = self.detector._calculate_momentum(self.price_data['EURUSD'])
        self.assertIsNotNone(momentum)
        self.assertGreaterEqual(momentum, -1.0)
        self.assertLessEqual(momentum, 1.0)
        empty_df = pd.DataFrame()
        momentum = self.detector._calculate_momentum(empty_df)
        self.assertIsNone(momentum)

    def test_calculate_rsi_vectorized(self):
        """Test RSI calculation."""
        prices = np.array([1.0, 1.1, 1.05, 1.15, 1.2, 1.15, 1.25, 1.3, 1.35,
            1.4, 1.45, 1.5, 1.45, 1.4, 1.35])
        rsi = self.detector._calculate_rsi_vectorized(prices)
        self.assertIsNotNone(rsi)
        self.assertGreaterEqual(rsi, 0.0)
        self.assertLessEqual(rsi, 100.0)
        short_prices = np.array([1.0, 1.1, 1.05])
        rsi = self.detector._calculate_rsi_vectorized(short_prices)
        self.assertIsNone(rsi)

    def test_calculate_ema(self):
        """Test EMA calculation."""
        prices = np.array([1.0, 1.1, 1.05, 1.15, 1.2, 1.15, 1.25, 1.3, 1.35,
            1.4, 1.45, 1.5, 1.45, 1.4, 1.35])
        ema = self.detector._calculate_ema(prices, 5)
        self.assertIsNotNone(ema)
        short_prices = np.array([1.0, 1.1, 1.05])
        ema = self.detector._calculate_ema(short_prices, 5)
        self.assertIsNone(ema)


if __name__ == '__main__':
    unittest.main()
