"""
Tests for the feature store client.
"""
import unittest
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from adapters.feature_store_client import FeatureStoreClient
from repositories.feature_cache import FeatureCache


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TestFeatureStoreClient(unittest.TestCase):
    """Test suite for the feature store client."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = FeatureStoreClient(base_url=
            'http://test-feature-store:8001/api/v1', use_cache=True)
        self.client._session = AsyncMock()
        self.sample_ohlcv = pd.DataFrame({'timestamp': pd.date_range(start=
            '2023-01-01', periods=10, freq='1h'), 'open': np.random.rand(10
            ) + 1.0, 'high': np.random.rand(10) + 1.1, 'low': np.random.
            rand(10) + 0.9, 'close': np.random.rand(10) + 1.0, 'volume': np
            .random.randint(1000, 10000, 10)})
        self.sample_indicators = pd.DataFrame({'timestamp': pd.date_range(
            start='2023-01-01', periods=10, freq='1h'), 'sma_50': np.random
            .rand(10) + 1.0, 'sma_200': np.random.rand(10) + 1.0, 'rsi_14':
            np.random.rand(10) * 100})

    async def async_setUp(self):
        """Async setup for tests."""
        await self.client.get_session()

    async def async_tearDown(self):
        """Async teardown for tests."""
        await self.client.close()

    def test_init(self):
        """Test client initialization."""
        client = FeatureStoreClient(base_url=
            'http://custom-url:8001/api/v1', api_key='test-key', use_cache=
            True, cache_ttl=600)
        self.assertEqual(client.base_url, 'http://custom-url:8001/api/v1')
        self.assertEqual(client.headers['X-API-Key'], 'test-key')
        self.assertEqual(client.cache_ttl, 600)
        self.assertTrue(client.use_cache)

    @patch('aiohttp.ClientSession.get')
    async def test_get_ohlcv_data(self, mock_get):
        """Test getting OHLCV data."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {'data': self.sample_ohlcv.
            to_dict(orient='records')}
        mock_get.return_value.__aenter__.return_value = mock_response
        result = await self.client.get_ohlcv_data(symbol='EUR/USD',
            start_date='2023-01-01', end_date='2023-01-10', timeframe='1h')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 10)
        self.assertIn('timestamp', result.columns)
        self.assertIn('close', result.columns)
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertIn('params', kwargs)
        self.assertEqual(kwargs['params']['symbol'], 'EUR/USD')

    @patch('aiohttp.ClientSession.get')
    async def test_get_indicators(self, mock_get):
        """Test getting indicators."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {'data': self.sample_indicators.
            to_dict(orient='records')}
        mock_get.return_value.__aenter__.return_value = mock_response
        result = await self.client.get_indicators(symbol='EUR/USD',
            start_date='2023-01-01', end_date='2023-01-10', timeframe='1h',
            indicators=['sma_50', 'sma_200', 'rsi_14'])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 10)
        self.assertIn('sma_50', result.columns)
        self.assertIn('sma_200', result.columns)
        self.assertIn('rsi_14', result.columns)
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertIn('params', kwargs)
        self.assertEqual(kwargs['params']['symbol'], 'EUR/USD')
        self.assertEqual(kwargs['params']['indicators'],
            'sma_50,sma_200,rsi_14')

    @patch('aiohttp.ClientSession.post')
    async def test_compute_feature(self, mock_post):
        """Test computing a feature."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {'data': [{'timestamp':
            '2023-01-01T00:00:00', 'volatility': 0.01}, {'timestamp':
            '2023-01-01T01:00:00', 'volatility': 0.02}]}
        mock_post.return_value.__aenter__.return_value = mock_response
        result = await self.client.compute_feature(feature_name=
            'volatility', symbol='EUR/USD', start_date='2023-01-01',
            end_date='2023-01-02', timeframe='1h', parameters={'window': 14})
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('timestamp', result.columns)
        self.assertIn('volatility', result.columns)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn('json', kwargs)
        self.assertEqual(kwargs['json']['feature_name'], 'volatility')
        self.assertEqual(kwargs['json']['symbol'], 'EUR/USD')
        self.assertEqual(kwargs['json']['parameters'], {'window': 14})

    @patch('aiohttp.ClientSession.get')
    async def test_get_available_indicators(self, mock_get):
        """Test getting available indicators."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {'indicators': [{'id': 'sma',
            'name': 'Simple Moving Average'}, {'id': 'ema', 'name':
            'Exponential Moving Average'}]}
        mock_get.return_value.__aenter__.return_value = mock_response
        result = await self.client.get_available_indicators()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['id'], 'sma')
        self.assertEqual(result[1]['id'], 'ema')
        mock_get.assert_called_once()

    @patch('aiohttp.ClientSession.get')
    async def test_get_indicator_metadata(self, mock_get):
        """Test getting indicator metadata."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {'id': 'sma', 'name':
            'Simple Moving Average', 'description':
            'Calculates the average of prices over a specified period.',
            'parameters': {'period': {'type': 'integer', 'default': 14,
            'min': 1, 'max': 500}}}
        mock_get.return_value.__aenter__.return_value = mock_response
        result = await self.client.get_indicator_metadata('sma')
        self.assertIsInstance(result, dict)
        self.assertEqual(result['id'], 'sma')
        self.assertEqual(result['name'], 'Simple Moving Average')
        self.assertIn('parameters', result)
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], f'{self.client.base_url}/indicators/sma')

    @patch('aiohttp.ClientSession.get')
    async def test_error_handling(self, mock_get):
        """Test error handling."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text.return_value = 'Not found'
        mock_get.return_value.__aenter__.return_value = mock_response
        with self.assertRaises(Exception):
            await self.client.get_ohlcv_data(symbol='INVALID/PAIR',
                start_date='2023-01-01', end_date='2023-01-10', timeframe='1h')

    @patch('aiohttp.ClientSession.get')
    async def test_caching(self, mock_get):
        """Test caching."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {'data': self.sample_ohlcv.
            to_dict(orient='records')}
        mock_get.return_value.__aenter__.return_value = mock_response
        result1 = await self.client.get_ohlcv_data(symbol='EUR/USD',
            start_date='2023-01-01', end_date='2023-01-10', timeframe='1h')
        result2 = await self.client.get_ohlcv_data(symbol='EUR/USD',
            start_date='2023-01-01', end_date='2023-01-10', timeframe='1h')
        pd.testing.assert_frame_equal(result1, result2)
        mock_get.assert_called_once()

    def test_fallback_data(self):
        """Test fallback data generation."""
        ohlcv = self.client._get_fallback_ohlcv_data(symbol='EUR/USD',
            start_date=datetime.now() - timedelta(days=10), end_date=
            datetime.now(), timeframe='1h')
        self.assertIsInstance(ohlcv, pd.DataFrame)
        self.assertGreater(len(ohlcv), 0)
        self.assertIn('timestamp', ohlcv.columns)
        self.assertIn('open', ohlcv.columns)
        self.assertIn('high', ohlcv.columns)
        self.assertIn('low', ohlcv.columns)
        self.assertIn('close', ohlcv.columns)
        self.assertIn('volume', ohlcv.columns)
        indicators = self.client._get_fallback_indicator_data(symbol=
            'EUR/USD', start_date=datetime.now() - timedelta(days=10),
            end_date=datetime.now(), timeframe='1h', indicators=['sma_50',
            'rsi_14'])
        self.assertIsInstance(indicators, pd.DataFrame)
        self.assertGreater(len(indicators), 0)
        self.assertIn('timestamp', indicators.columns)
        self.assertIn('sma_50', indicators.columns)
        self.assertIn('rsi_14', indicators.columns)


@with_exception_handling
def run_async_test(test_case, test_func):
    """Helper function to run async tests."""

    @async_with_exception_handling
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
