"""
End-to-End Tests for the Analysis Engine

This module provides comprehensive end-to-end tests for the Analysis Engine,
testing the entire system from API to database.
"""
import os
import sys
import json
import logging
import unittest
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.utils.test_data_generator import generate_test_data
from tests.utils.test_server import TestServer
from tests.utils.test_client import TestClient

class EndToEndTests(unittest.TestCase):
    """End-to-end tests for the Analysis Engine."""

    @classmethod
    def set_up_class(cls):
        """Set up the test environment."""
        cls.server = TestServer()
        cls.server.start()
        cls.client = TestClient(base_url=cls.server.base_url)
        cls.test_data = generate_test_data(symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'], timeframes=['H1', 'H4', 'D1'], days=30)
        asyncio.run(cls._upload_test_data())

    @classmethod
    def tear_down_class(cls):
        """Tear down the test environment."""
        cls.server.stop()

    @classmethod
    async def _upload_test_data(cls):
        """Upload test data to server."""
        logger.info('Uploading test data to server...')
        for symbol, timeframe_data in cls.test_data.items():
            for timeframe, data in timeframe_data.items():
                await cls.client.upload_test_data(symbol, timeframe, data)

    async def _test_confluence_detection(self, use_ml=False):
        """Test confluence detection."""
        logger.info(f'Testing {('ML-based' if use_ml else 'optimized')} confluence detection...')
        symbol = 'EURUSD'
        timeframe = 'H1'
        signal_type = 'trend'
        signal_direction = 'bullish'
        if use_ml:
            response = await self.client.detect_confluence_ml(symbol=symbol, timeframe=timeframe, signal_type=signal_type, signal_direction=signal_direction)
        else:
            response = await self.client.detect_confluence(symbol=symbol, timeframe=timeframe, signal_type=signal_type, signal_direction=signal_direction)
        self.assertIsNotNone(response)
        self.assertEqual(response['symbol'], symbol)
        self.assertEqual(response['signal_type'], signal_type)
        self.assertEqual(response['signal_direction'], signal_direction)
        self.assertIn('confluence_score', response)
        self.assertIn('confirmation_count', response)
        self.assertIn('contradiction_count', response)
        self.assertIn('confirmations', response)
        self.assertIn('contradictions', response)
        self.assertIn('neutrals', response)
        if use_ml:
            self.assertIn('pattern_score', response)
            self.assertIn('prediction_score', response)
            self.assertIn('price_prediction', response)
            self.assertIn('patterns', response)
        return response

    async def _test_divergence_analysis(self, use_ml=False):
        """Test divergence analysis."""
        logger.info(f'Testing {('ML-based' if use_ml else 'optimized')} divergence analysis...')
        symbol = 'EURUSD'
        timeframe = 'H1'
        if use_ml:
            response = await self.client.analyze_divergence_ml(symbol=symbol, timeframe=timeframe)
        else:
            response = await self.client.analyze_divergence(symbol=symbol, timeframe=timeframe)
        self.assertIsNotNone(response)
        self.assertEqual(response['symbol'], symbol)
        self.assertIn('divergence_score', response)
        self.assertIn('divergences_found', response)
        self.assertIn('divergences', response)
        if use_ml:
            self.assertIn('price_prediction', response)
        return response

    async def _test_pattern_recognition(self, use_ml=False):
        """Test pattern recognition."""
        logger.info(f'Testing {('ML-based' if use_ml else 'traditional')} pattern recognition...')
        symbol = 'EURUSD'
        timeframe = 'H1'
        if use_ml:
            response = await self.client.recognize_patterns_ml(symbol=symbol, timeframe=timeframe)
        else:
            response = await self.client.recognize_patterns(symbol=symbol, timeframe=timeframe)
        self.assertIsNotNone(response)
        self.assertEqual(response['symbol'], symbol)
        self.assertIn('patterns', response)
        return response

    async def _test_currency_strength(self):
        """Test currency strength analysis."""
        logger.info('Testing currency strength analysis...')
        timeframe = 'H1'
        method = 'combined'
        response = await self.client.get_currency_strength(timeframe=timeframe, method=method)
        self.assertIsNotNone(response)
        self.assertEqual(response['timeframe'], timeframe)
        self.assertEqual(response['method'], method)
        self.assertIn('currencies', response)
        self.assertIn('strongest', response)
        self.assertIn('weakest', response)
        return response

    async def _test_related_pairs(self):
        """Test related pairs retrieval."""
        logger.info('Testing related pairs retrieval...')
        symbol = 'EURUSD'
        min_correlation = 0.6
        timeframe = 'H1'
        response = await self.client.get_related_pairs(symbol=symbol, min_correlation=min_correlation, timeframe=timeframe)
        self.assertIsNotNone(response)
        self.assertEqual(response['symbol'], symbol)
        self.assertEqual(response['timeframe'], timeframe)
        self.assertIn('related_pairs', response)
        return response

    async def _test_system_status(self):
        """Test system status retrieval."""
        logger.info('Testing system status retrieval...')
        response = await self.client.get_system_status()
        self.assertIsNotNone(response)
        self.assertIn('status', response)
        self.assertIn('components', response)
        return response

    async def _test_ml_models(self):
        """Test ML models retrieval."""
        logger.info('Testing ML models retrieval...')
        response = await self.client.list_models()
        self.assertIsNotNone(response)
        self.assertIsInstance(response, list)
        if response:
            model_name = response[0]['name']
            model_info = await self.client.get_model_info(model_name)
            self.assertIsNotNone(model_info)
            self.assertEqual(model_info['name'], model_name)
        return response

    async def _run_all_tests(self):
        """Run all tests."""
        await self._test_system_status()
        confluence_result = await self._test_confluence_detection(use_ml=False)
        divergence_result = await self._test_divergence_analysis(use_ml=False)
        pattern_result = await self._test_pattern_recognition(use_ml=False)
        ml_confluence_result = await self._test_confluence_detection(use_ml=True)
        ml_divergence_result = await self._test_divergence_analysis(use_ml=True)
        ml_pattern_result = await self._test_pattern_recognition(use_ml=True)
        currency_strength_result = await self._test_currency_strength()
        related_pairs_result = await self._test_related_pairs()
        ml_models_result = await self._test_ml_models()
        return {'confluence': confluence_result, 'divergence': divergence_result, 'pattern': pattern_result, 'ml_confluence': ml_confluence_result, 'ml_divergence': ml_divergence_result, 'ml_pattern': ml_pattern_result, 'currency_strength': currency_strength_result, 'related_pairs': related_pairs_result, 'ml_models': ml_models_result}

    def test_end_to_end(self):
        """Run end-to-end tests."""
        results = asyncio.run(self._run_all_tests())
        self.assertIsNotNone(results['confluence'])
        self.assertIsNotNone(results['divergence'])
        self.assertIsNotNone(results['pattern'])
        self.assertIsNotNone(results['ml_confluence'])
        self.assertIsNotNone(results['ml_divergence'])
        self.assertIsNotNone(results['ml_pattern'])
        self.assertIsNotNone(results['currency_strength'])
        self.assertIsNotNone(results['related_pairs'])
        self._compare_results(optimized=results['confluence'], ml=results['ml_confluence'], name='confluence')
        self._compare_results(optimized=results['divergence'], ml=results['ml_divergence'], name='divergence')
        self._compare_results(optimized=results['pattern'], ml=results['ml_pattern'], name='pattern')
        logger.info('All end-to-end tests passed!')

    def _compare_results(self, optimized, ml, name):
        """Compare optimized and ML results."""
        logger.info(f'Comparing optimized vs ML {name} results...')
        optimized_time = optimized.get('execution_time', 0)
        ml_time = ml.get('execution_time', 0)
        logger.info(f'Optimized {name} execution time: {optimized_time:.3f}s')
        logger.info(f'ML {name} execution time: {ml_time:.3f}s')
        logger.info(f'Difference: {(ml_time - optimized_time) / optimized_time * 100:.1f}%')
        if name == 'confluence':
            optimized_score = optimized.get('confluence_score', 0)
            ml_score = ml.get('confluence_score', 0)
        elif name == 'divergence':
            optimized_score = optimized.get('divergence_score', 0)
            ml_score = ml.get('divergence_score', 0)
        else:
            optimized_patterns = optimized.get('patterns', {})
            ml_patterns = ml.get('patterns', {})
            optimized_score = max(optimized_patterns.values()) if optimized_patterns else 0
            ml_score = max(ml_patterns.values()) if ml_patterns else 0
        logger.info(f'Optimized {name} score: {optimized_score:.3f}')
        logger.info(f'ML {name} score: {ml_score:.3f}')
        logger.info(f'Difference: {abs(ml_score - optimized_score):.3f}')
if __name__ == '__main__':
    unittest.main()