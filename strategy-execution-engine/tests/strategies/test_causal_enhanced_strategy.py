"""
Unit tests for the Causal Enhanced Strategy
"""

import unittest
import asyncio
import pandas as pd
import networkx as nx
import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json

from strategy_execution_engine.strategies.causal_enhanced_strategy import CausalEnhancedStrategy, dataframe_to_json_list


class TestCausalEnhancedStrategy(unittest.TestCase):
    """Test the Causal Enhanced Strategy."""

    def setUp(self):
        """Set up test environment."""
        self.strategy = CausalEnhancedStrategy(
            name="test_causal_strategy",
            parameters={
                "symbols": ["EURUSD", "GBPUSD"],
                "timeframe": "1h",
                "window_size": 7,
                "position_size_pct": 1.0,
                "max_positions": 2,
                "stop_loss_pct": 1.0,
                "take_profit_pct": 2.0,
                "causal_min_confidence": 0.6,
                "lookback_periods": 100,
                "counterfactual_scenarios": 2,
                "effect_threshold": 0.1
            }
        )
        
        # Mock the HTTP client
        self.strategy.analysis_engine_client = AsyncMock()
        
        # Create sample data
        self.market_data = pd.DataFrame({
            'EURUSD_close': [1.1, 1.11, 1.12, 1.13, 1.14],
            'GBPUSD_close': [1.3, 1.31, 1.32, 1.33, 1.34],
            'EURUSD_volume': [1000, 1100, 1200, 1300, 1400],
            'GBPUSD_volume': [2000, 2100, 2200, 2300, 2400]
        }, index=pd.date_range(start='2025-01-01', periods=5, freq='D'))
        
        # Create sample causal graph
        self.sample_graph = nx.DiGraph()
        self.sample_graph.add_edge('EURUSD_close', 'GBPUSD_close', weight=0.7)
        self.sample_graph.add_edge('EURUSD_volume', 'EURUSD_close', weight=0.5)
        
        # Set sample effect estimates
        self.strategy.effect_estimates = {
            ('EURUSD_close', 'GBPUSD_close'): 0.7,
            ('EURUSD_volume', 'EURUSD_close'): 0.5
        }
        self.strategy.causal_graph = self.sample_graph

    @pytest.mark.asyncio
    async def test_call_generate_counterfactuals(self):
        """Test the _call_generate_counterfactuals method."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = AsyncMock()
        
        # Create counterfactuals response
        scenario_name = "scenario_EURUSD"
        mock_data = [
            {"timestamp": "2025-01-01", "EURUSD_close": 1.15, "EURUSD_volume": 1100},
            {"timestamp": "2025-01-02", "EURUSD_close": 1.16, "EURUSD_volume": 1200},
            {"timestamp": "2025-01-03", "EURUSD_close": 1.17, "EURUSD_volume": 1300},
            {"timestamp": "2025-01-04", "EURUSD_close": 1.18, "EURUSD_volume": 1400},
            {"timestamp": "2025-01-05", "EURUSD_close": 1.19, "EURUSD_volume": 1500}
        ]
        
        mock_response.json = AsyncMock(return_value={
            "counterfactuals": {
                scenario_name: mock_data
            }
        })
        
        self.strategy.analysis_engine_client.post = AsyncMock(return_value=mock_response)
        
        # Test the method
        interventions = {"EURUSD_volume": 1500}
        result = await self.strategy._call_generate_counterfactuals(
            data=self.market_data,
            target_var="EURUSD_close",
            interventions=interventions
        )
        
        # Assertions
        self.strategy.analysis_engine_client.post.assert_called_once()
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 5)
        self.assertTrue("EURUSD_close" in result.columns)
        self.assertEqual(result["EURUSD_close"].iloc[-1], 1.19)
        
    @pytest.mark.asyncio
    async def test_call_generate_counterfactuals_error_handling(self):
        """Test error handling in _call_generate_counterfactuals method."""
        # Test HTTP error
        self.strategy.analysis_engine_client.post.side_effect = httpx.HTTPStatusError(
            "HTTP Error", 
            request=httpx.Request("POST", "http://example.com"),
            response=httpx.Response(500)
        )
        
        result = await self.strategy._call_generate_counterfactuals(
            data=self.market_data,
            target_var="EURUSD_close",
            interventions={"EURUSD_volume": 1500}
        )
        
        self.assertIsNone(result)
        
        # Test request error
        self.strategy.analysis_engine_client.post.side_effect = httpx.RequestError(
            "Connection error", request=httpx.Request("POST", "http://example.com")
        )
        
        result = await self.strategy._call_generate_counterfactuals(
            data=self.market_data,
            target_var="EURUSD_close",
            interventions={"EURUSD_volume": 1500}
        )
        
        self.assertIsNone(result)
        
        # Test invalid response format
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "wrong_key": "wrong_value"  # Missing counterfactuals key
        })
        
        self.strategy.analysis_engine_client.post = AsyncMock(return_value=mock_response)
        
        result = await self.strategy._call_generate_counterfactuals(
            data=self.market_data,
            target_var="EURUSD_close",
            interventions={"EURUSD_volume": 1500}
        )
        
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
