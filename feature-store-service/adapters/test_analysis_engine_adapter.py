"""
Tests for the Analysis Engine Adapter.

This module contains tests for the Analysis Engine Adapter.
"""

import unittest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from adapters.analysis_engine_adapter import AnalysisEngineAdapter
from common_lib.errors.base_exceptions import ServiceError


class TestAnalysisEngineAdapter(unittest.TestCase):
    """Tests for the Analysis Engine Adapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.adapter = AnalysisEngineAdapter()
        self.adapter.client = MagicMock()
        self.adapter.logger = MagicMock()

    def test_init(self):
        """Test initialization."""
        adapter = AnalysisEngineAdapter()
        self.assertIsNotNone(adapter)
        self.assertIsNotNone(adapter.client)
        self.assertIsNotNone(adapter.logger)

    def test_analyze_market(self):
        """Test analyze_market method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        self.adapter.client.post.return_value = mock_response

        # Call method
        result = asyncio.run(self.adapter.analyze_market(
            symbol="EURUSD",
            timeframe="1h",
            analysis_type="trend",
            start_time=datetime.utcnow() - timedelta(days=7),
            end_time=datetime.utcnow()
        ))

        # Verify result
        self.assertEqual(result, {"result": "success"})
        self.adapter.client.post.assert_called_once()

    def test_get_analysis_types(self):
        """Test get_analysis_types method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"analysis_types": ["trend", "volatility"]}
        self.adapter.client.get.return_value = mock_response

        # Call method
        result = asyncio.run(self.adapter.get_analysis_types())

        # Verify result
        self.assertEqual(result, ["trend", "volatility"])
        self.adapter.client.get.assert_called_once()

    def test_backtest_strategy(self):
        """Test backtest_strategy method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        self.adapter.client.post.return_value = mock_response

        # Call method
        result = asyncio.run(self.adapter.backtest_strategy(
            strategy_id="test_strategy",
            symbol="EURUSD",
            timeframe="1h",
            start_time=datetime.utcnow() - timedelta(days=7),
            end_time=datetime.utcnow()
        ))

        # Verify result
        self.assertEqual(result, {"result": "success"})
        self.adapter.client.post.assert_called_once()

    def test_calculate_indicator(self):
        """Test calculate_indicator method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"value": 1.0}, {"value": 2.0}]}
        self.adapter.client.post.return_value = mock_response

        # Call method
        data = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        result = asyncio.run(self.adapter.calculate_indicator(
            indicator_name="sma",
            data=data,
            parameters={"period": 2}
        ))

        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.adapter.client.post.assert_called_once()

    def test_get_indicator_info(self):
        """Test get_indicator_info method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"name": "sma", "description": "Simple Moving Average"}
        self.adapter.client.get.return_value = mock_response

        # Call method
        result = asyncio.run(self.adapter.get_indicator_info("sma"))

        # Verify result
        self.assertEqual(result, {"name": "sma", "description": "Simple Moving Average"})
        self.adapter.client.get.assert_called_once()

    def test_list_indicators(self):
        """Test list_indicators method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"indicators": ["sma", "ema", "rsi"]}
        self.adapter.client.get.return_value = mock_response

        # Call method
        result = asyncio.run(self.adapter.list_indicators())

        # Verify result
        self.assertEqual(result, ["sma", "ema", "rsi"])
        self.adapter.client.get.assert_called_once()

    def test_recognize_patterns(self):
        """Test recognize_patterns method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "patterns": [
                {"type": "double_top", "start": 10, "end": 20, "confidence": 0.8},
                {"type": "head_and_shoulders", "start": 30, "end": 50, "confidence": 0.7}
            ]
        }
        self.adapter.client.post.return_value = mock_response

        # Call method
        data = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        result = asyncio.run(self.adapter.recognize_patterns(
            data=data,
            pattern_types=["double_top", "head_and_shoulders"]
        ))

        # Verify result
        self.assertEqual(len(result["patterns"]), 2)
        self.assertEqual(result["patterns"][0]["type"], "double_top")
        self.adapter.client.post.assert_called_once()

    def test_get_pattern_types(self):
        """Test get_pattern_types method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "pattern_types": [
                {"id": "double_top", "name": "Double Top", "category": "reversal"},
                {"id": "head_and_shoulders", "name": "Head and Shoulders", "category": "reversal"}
            ]
        }
        self.adapter.client.get.return_value = mock_response

        # Call method
        result = asyncio.run(self.adapter.get_pattern_types())

        # Verify result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "double_top")
        self.adapter.client.get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
