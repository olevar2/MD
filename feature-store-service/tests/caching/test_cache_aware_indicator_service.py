"""
Tests for the CacheAwareIndicatorService class in the feature store caching system.
"""
import asyncio
import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from common_lib.caching import AdaptiveCacheManager, cached, get_cache_manager
from feature_store_service.caching.cache_aware_indicator_service import CacheAwareIndicatorService
from feature_store_service.caching.cache_key import CacheKey


class TestCacheAwareIndicatorService(unittest.TestCase):
    """Test cases for CacheAwareIndicatorService class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for disk cache
        self.temp_dir = tempfile.mkdtemp()
        
        # Configure cache
        self.config = {
            'memory_cache_size': 1000000,  # 1MB
            'memory_cache_ttl': 300,  # 5 minutes
            'use_disk_cache': True,
            'disk_cache_path': self.temp_dir,
            'disk_cache_size': 10000000,  # 10MB
            'disk_cache_ttl': 3600  # 1 hour
        }
        
        # Initialize cache manager
        self.cache_manager = CacheManager(self.config)
        
        # Create mock indicator factory
        self.indicator_factory = MagicMock()
        
        # Initialize service
        self.service = CacheAwareIndicatorService(
            cache_manager=self.cache_manager,
            indicator_factory=self.indicator_factory
        )
        
        # Create test data
        self.test_data = pd.DataFrame({
            'open': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'close': np.random.rand(100),
            'volume': np.random.rand(100),
        }, index=pd.date_range(start='2025-01-01', periods=100, freq='H'))

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_calculate_indicator_with_cache_hit(self):
        """Test calculating an indicator with a cache hit."""
        # Define test case function
        async def test_case():
            # Define indicator parameters
            indicator_type = "SMA"
            params = {"window": 20}
            symbol = "EURUSD"
            timeframe = "1h"
            
            # Create expected result
            expected_result = self.test_data.copy()
            expected_result['sma_20'] = expected_result['close'].rolling(window=20).mean()
            
            # Create cache key
            cache_key = CacheKey(
                indicator_type=indicator_type,
                params=params,
                symbol=symbol,
                timeframe=timeframe,
                start_time=self.test_data.index.min(),
                end_time=self.test_data.index.max()
            )
            
            # Pre-populate cache with expected result
            await self.cache_manager.put(cache_key, expected_result)
            
            # Calculate indicator
            result = await self.service.calculate_indicator(
                indicator_type=indicator_type,
                params=params,
                data=self.test_data,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Verify result matches expected
            pd.testing.assert_frame_equal(result, expected_result)
            
            # Verify that the indicator factory was NOT called (cache hit)
            self.indicator_factory.create.assert_not_called()
            
        # Run test case
        asyncio.run(test_case())

    def test_calculate_indicator_with_cache_miss(self):
        """Test calculating an indicator with a cache miss."""
        # Define test case function
        async def test_case():
            # Define indicator parameters
            indicator_type = "SMA"
            params = {"window": 20}
            symbol = "EURUSD"
            timeframe = "1h"
            
            # Create expected result
            expected_result = self.test_data.copy()
            expected_result['sma_20'] = expected_result['close'].rolling(window=20).mean()
            
            # Set up mock indicator
            mock_indicator = MagicMock()
            mock_indicator.calculate.return_value = expected_result
            self.indicator_factory.create.return_value = mock_indicator
            
            # Calculate indicator
            result = await self.service.calculate_indicator(
                indicator_type=indicator_type,
                params=params,
                data=self.test_data,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Verify result matches expected
            pd.testing.assert_frame_equal(result, expected_result)
            
            # Verify that the indicator factory was called (cache miss)
            self.indicator_factory.create.assert_called_once_with(indicator_type, **params)
            mock_indicator.calculate.assert_called_once()
            
            # Verify result was stored in cache
            cache_key = CacheKey(
                indicator_type=indicator_type,
                params=params,
                symbol=symbol,
                timeframe=timeframe,
                start_time=self.test_data.index.min(),
                end_time=self.test_data.index.max()
            )
            cached_result = await self.cache_manager.get(cache_key)
            self.assertIsNotNone(cached_result)
            pd.testing.assert_frame_equal(cached_result, expected_result)
            
        # Run test case
        asyncio.run(test_case())

    def test_calculate_batch(self):
        """Test calculating multiple indicators in batch."""
        # Define test case function
        async def test_case():
            # Define indicator configurations
            indicator_configs = [
                {
                    'type': 'SMA',
                    'params': {'window': 10}
                },
                {
                    'type': 'EMA',
                    'params': {'window': 20}
                }
            ]
            symbol = "EURUSD"
            timeframe = "1h"
            
            # Create expected results for each indicator
            expected_result1 = self.test_data.copy()
            expected_result1['sma_10'] = expected_result1['close'].rolling(window=10).mean()
            
            expected_result2 = self.test_data.copy()
            expected_result2['ema_20'] = expected_result2['close'].ewm(span=20).mean()
            
            # Combined expected result
            expected_combined = self.test_data.copy()
            expected_combined['sma_10'] = expected_combined['close'].rolling(window=10).mean()
            expected_combined['ema_20'] = expected_combined['close'].ewm(span=20).mean()
            
            # Mock indicator factory and indicators
            mock_indicator1 = MagicMock()
            mock_indicator1.calculate.return_value = expected_result1
            
            mock_indicator2 = MagicMock()
            mock_indicator2.calculate.return_value = expected_result2
            
            def mock_create(indicator_type, **params):
    """
    Mock create.
    
    Args:
        indicator_type: Description of indicator_type
        params: Description of params
    
    """

                if indicator_type == 'SMA' and params.get('window') == 10:
                    return mock_indicator1
                elif indicator_type == 'EMA' and params.get('window') == 20:
                    return mock_indicator2
                raise ValueError(f"Unexpected indicator: {indicator_type} with params {params}")
                
            self.indicator_factory.create.side_effect = mock_create
            
            # Calculate batch
            result = await self.service.calculate_batch(
                indicator_configs=indicator_configs,
                data=self.test_data,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Verify combined result has both indicators
            self.assertIn('sma_10', result.columns)
            self.assertIn('ema_20', result.columns)
            
            # Verify both indicators were created
            self.assertEqual(self.indicator_factory.create.call_count, 2)
            
        # Run test case
        asyncio.run(test_case())

    def test_cache_stats_and_clearing(self):
        """Test getting cache stats and clearing cache."""
        # Define test case function
        async def test_case():
            # Define indicator parameters
            indicator_type = "SMA"
            params = {"window": 20}
            symbol = "EURUSD"
            timeframe = "1h"
            
            # Create expected result
            expected_result = self.test_data.copy()
            expected_result['sma_20'] = expected_result['close'].rolling(window=20).mean()
            
            # Set up mock indicator
            mock_indicator = MagicMock()
            mock_indicator.calculate.return_value = expected_result
            self.indicator_factory.create.return_value = mock_indicator
            
            # Calculate indicator
            await self.service.calculate_indicator(
                indicator_type=indicator_type,
                params=params,
                data=self.test_data,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Get cache stats
            stats = await self.service.get_cache_stats()
            
            # Verify stats format
            self.assertIn('memory_cache', stats)
            self.assertIn('disk_cache', stats)
            self.assertIn('performance', stats)
            
            # Clear cache for symbol
            cleared_count = await self.service.clear_cache_for_symbol(symbol)
            
            # Verify cache was cleared
            self.assertGreaterEqual(cleared_count, 1)
            
            # Calculate again to verify cache miss
            self.indicator_factory.create.reset_mock()
            await self.service.calculate_indicator(
                indicator_type=indicator_type,
                params=params,
                data=self.test_data,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Verify indicator factory was called (cache miss after clearing)
            self.indicator_factory.create.assert_called_once()
            
        # Run test case
        asyncio.run(test_case())


if __name__ == "__main__":
    unittest.main()
