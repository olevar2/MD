"""
Integration tests for the feature store client.
"""
import unittest
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta
import os
import json
import time
from unittest.mock import patch, MagicMock, AsyncMock

from strategy_execution_engine.clients.feature_store_client import FeatureStoreClient
from strategy_execution_engine.caching.feature_cache import FeatureCache
from strategy_execution_engine.monitoring.feature_store_metrics import feature_store_metrics


class TestFeatureStoreIntegration(unittest.TestCase):
    """Integration test suite for the feature store client."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test suite."""
        # Check if feature store service is available
        cls.feature_store_available = False
        try:
            # Try to connect to the feature store service
            import requests
            feature_store_url = os.getenv('FEATURE_STORE_URL', 'http://localhost:8001/api/v1')
            response = requests.get(f"{feature_store_url}/health", timeout=2)
            cls.feature_store_available = response.status_code == 200
        except Exception:
            cls.feature_store_available = False
        
        # Reset metrics
        feature_store_metrics.reset_metrics()
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a client
        self.client = FeatureStoreClient(
            base_url=os.getenv('FEATURE_STORE_URL', 'http://localhost:8001/api/v1'),
            use_cache=True,
            cache_ttl=60
        )
        
        # Sample data for tests
        self.symbol = "EUR/USD"
        self.start_date = datetime.now() - timedelta(days=30)
        self.end_date = datetime.now()
        self.timeframe = "1h"
    
    async def async_setUp(self):
        """Async setup for tests."""
        # Initialize the client's session
        await self.client.get_session()
    
    async def async_tearDown(self):
        """Async teardown for tests."""
        # Close the client's session
        await self.client.close()
    
    def test_feature_store_availability(self):
        """Test if the feature store service is available."""
        if not self.feature_store_available:
            self.skipTest("Feature store service is not available")
        
        self.assertTrue(self.feature_store_available)
    
    @unittest.skipIf(not hasattr(TestFeatureStoreIntegration, 'feature_store_available') or 
                    not TestFeatureStoreIntegration.feature_store_available,
                    "Feature store service is not available")
    def test_get_ohlcv_data_integration(self):
        """Test getting OHLCV data from the feature store."""
        async def test_async():
            # Get OHLCV data
            result = await self.client.get_ohlcv_data(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe=self.timeframe
            )
            
            # Check the result
            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)
            self.assertIn('timestamp', result.columns)
            self.assertIn('open', result.columns)
            self.assertIn('high', result.columns)
            self.assertIn('low', result.columns)
            self.assertIn('close', result.columns)
            self.assertIn('volume', result.columns)
            
            # Check metrics
            self.assertGreater(feature_store_metrics.metrics["api_calls"]["total"], 0)
            self.assertGreater(feature_store_metrics.metrics["api_calls"]["get_ohlcv_data"], 0)
        
        # Run the async test
        asyncio.run(test_async())
    
    @unittest.skipIf(not hasattr(TestFeatureStoreIntegration, 'feature_store_available') or 
                    not TestFeatureStoreIntegration.feature_store_available,
                    "Feature store service is not available")
    def test_get_indicators_integration(self):
        """Test getting indicators from the feature store."""
        async def test_async():
            # Get indicators
            result = await self.client.get_indicators(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe=self.timeframe,
                indicators=["sma_50", "sma_200", "rsi_14"]
            )
            
            # Check the result
            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)
            self.assertIn('timestamp', result.columns)
            
            # Check that at least one of the requested indicators is in the result
            indicator_columns = [col for col in result.columns if col in ["sma_50", "sma_200", "rsi_14"]]
            self.assertGreater(len(indicator_columns), 0)
            
            # Check metrics
            self.assertGreater(feature_store_metrics.metrics["api_calls"]["total"], 0)
            self.assertGreater(feature_store_metrics.metrics["api_calls"]["get_indicators"], 0)
        
        # Run the async test
        asyncio.run(test_async())
    
    @unittest.skipIf(not hasattr(TestFeatureStoreIntegration, 'feature_store_available') or 
                    not TestFeatureStoreIntegration.feature_store_available,
                    "Feature store service is not available")
    def test_compute_feature_integration(self):
        """Test computing a feature from the feature store."""
        async def test_async():
            # Compute a feature
            result = await self.client.compute_feature(
                feature_name="volatility",
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe=self.timeframe,
                parameters={"window": 14}
            )
            
            # Check the result
            self.assertIsInstance(result, pd.DataFrame)
            
            # If the feature is not available, the result might be empty
            if not result.empty:
                self.assertIn('timestamp', result.columns)
                self.assertIn('volatility', result.columns)
            
            # Check metrics
            self.assertGreater(feature_store_metrics.metrics["api_calls"]["total"], 0)
            self.assertGreater(feature_store_metrics.metrics["api_calls"]["compute_feature"], 0)
        
        # Run the async test
        asyncio.run(test_async())
    
    @unittest.skipIf(not hasattr(TestFeatureStoreIntegration, 'feature_store_available') or 
                    not TestFeatureStoreIntegration.feature_store_available,
                    "Feature store service is not available")
    def test_caching_integration(self):
        """Test caching in the feature store client."""
        async def test_async():
            # Get indicators twice with the same parameters
            result1 = await self.client.get_indicators(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe=self.timeframe,
                indicators=["sma_50", "sma_200"]
            )
            
            # Check metrics before second call
            api_calls_before = feature_store_metrics.metrics["api_calls"]["total"]
            cache_hits_before = feature_store_metrics.metrics["cache"]["hits"]
            
            # Get indicators again with the same parameters
            result2 = await self.client.get_indicators(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe=self.timeframe,
                indicators=["sma_50", "sma_200"]
            )
            
            # Check metrics after second call
            api_calls_after = feature_store_metrics.metrics["api_calls"]["total"]
            cache_hits_after = feature_store_metrics.metrics["cache"]["hits"]
            
            # Check that the second call used the cache
            self.assertEqual(api_calls_before + 1, api_calls_after)
            self.assertGreater(cache_hits_after, cache_hits_before)
            
            # Check that the results are the same
            pd.testing.assert_frame_equal(result1, result2)
        
        # Run the async test
        asyncio.run(test_async())
    
    @unittest.skipIf(not hasattr(TestFeatureStoreIntegration, 'feature_store_available') or 
                    not TestFeatureStoreIntegration.feature_store_available,
                    "Feature store service is not available")
    def test_error_handling_integration(self):
        """Test error handling in the feature store client."""
        async def test_async():
            # Try to get indicators with an invalid symbol
            with self.assertRaises(Exception):
                await self.client.get_indicators(
                    symbol="INVALID/SYMBOL",
                    start_date=self.start_date,
                    end_date=self.end_date,
                    timeframe=self.timeframe,
                    indicators=["sma_50"]
                )
            
            # Check metrics
            self.assertGreater(feature_store_metrics.metrics["errors"]["total"], 0)
        
        # Run the async test
        asyncio.run(test_async())
    
    def test_fallback_integration(self):
        """Test fallback mechanisms in the feature store client."""
        async def test_async():
            # Force circuit breaker to open
            self.client.circuit_breaker.failure_threshold = 1
            self.client.circuit_breaker.record_failure()
            
            # Try to get indicators with circuit breaker open
            result = await self.client.get_indicators(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe=self.timeframe,
                indicators=["sma_50"]
            )
            
            # Check that fallback data was returned
            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)
            
            # Check metrics
            self.assertGreater(feature_store_metrics.metrics["fallbacks"]["total"], 0)
            self.assertGreater(feature_store_metrics.metrics["fallbacks"]["get_indicators"], 0)
        
        # Run the async test
        asyncio.run(test_async())
    
    def test_strategy_integration(self):
        """Test integration with a strategy."""
        # Import a strategy that uses the feature store client
        from strategy_execution_engine.strategies.ma_crossover_strategy import MACrossoverStrategy
        
        async def test_async():
            # Create a strategy
            strategy = MACrossoverStrategy(
                name="Test MA Crossover",
                parameters={"use_feature_store": True}
            )
            
            # Create sample data
            data = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1h'),
                'open': np.random.rand(100) + 1.0,
                'high': np.random.rand(100) + 1.1,
                'low': np.random.rand(100) + 0.9,
                'close': np.random.rand(100) + 1.0,
                'volume': np.random.randint(1000, 10000, 100)
            })
            data.set_index('timestamp', inplace=True)
            data['symbol'] = self.symbol
            
            # Generate signals
            signals = await strategy.generate_signals(data)
            
            # Check the result
            self.assertIsInstance(signals, pd.DataFrame)
            self.assertEqual(len(signals), len(data))
            self.assertIn('signal', signals.columns)
            
            # Clean up
            await strategy.cleanup()
        
        # Run the async test
        asyncio.run(test_async())


def run_async_test(test_case, test_func):
    """Helper function to run async tests."""
    async def test_wrapper():
        await test_case.async_setUp()
        try:
            await test_func()
        finally:
            await test_case.async_tearDown()
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_wrapper())


if __name__ == '__main__':
    unittest.main()
