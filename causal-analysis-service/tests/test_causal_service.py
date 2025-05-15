"""
Tests for the causal service.
"""
import unittest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import networkx as nx

from causal_analysis_service.services.causal_service import CausalService
from causal_analysis_service.models.causal_models import (
    CausalGraphRequest,
    InterventionEffectRequest,
    CounterfactualRequest,
    CurrencyPairRelationshipRequest,
    RegimeChangeDriverRequest,
    TradingSignalEnhancementRequest,
    CorrelationBreakdownRiskRequest
)

class TestCausalService(unittest.TestCase):
    """
    Tests for the causal service.
    """
    def setUp(self):
        """
        Set up the test case.
        """
        self.causal_service = CausalService()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def test_generate_causal_graph(self):
        """
        Test generating a causal graph.
        """
        request = CausalGraphRequest(
            symbol="EURUSD",
            timeframe="1d",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            algorithm="granger"
        )
        
        response = self.loop.run_until_complete(
            self.causal_service.generate_causal_graph(request)
        )
        
        self.assertIsNotNone(response)
        self.assertIsNotNone(response["graph_id"])
        self.assertGreater(len(response["nodes"]), 0)
        self.assertGreater(len(response["edges"]), 0)
    
    def test_analyze_intervention_effect(self):
        """
        Test analyzing intervention effect.
        """
        request = InterventionEffectRequest(
            symbol="EURUSD",
            timeframe="1d",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            treatment="volatility",
            outcome="close",
            algorithm="dowhy"
        )
        
        response = self.loop.run_until_complete(
            self.causal_service.analyze_intervention_effect(request)
        )
        
        self.assertIsNotNone(response)
        self.assertIsNotNone(response["effect_id"])
        self.assertEqual(response["treatment"], "volatility")
        self.assertEqual(response["outcome"], "close")
        self.assertIsNotNone(response["causal_effect"])
    
    def test_generate_counterfactual_scenario(self):
        """
        Test generating a counterfactual scenario.
        """
        request = CounterfactualRequest(
            symbol="EURUSD",
            timeframe="1d",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            intervention={"volatility": 0.02},
            target_variables=["close", "high", "low"],
            algorithm="counterfactual"
        )
        
        response = self.loop.run_until_complete(
            self.causal_service.generate_counterfactual_scenario(request)
        )
        
        self.assertIsNotNone(response)
        self.assertIsNotNone(response["counterfactual_id"])
        self.assertEqual(response["intervention"], {"volatility": 0.02})
        self.assertEqual(response["target_variables"], ["close", "high", "low"])
        self.assertIsNotNone(response["counterfactual_values"])
    
    def test_discover_currency_pair_relationships(self):
        """
        Test discovering currency pair relationships.
        """
        request = CurrencyPairRelationshipRequest(
            symbols=["EURUSD", "GBPUSD", "USDJPY"],
            timeframe="1d",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            algorithm="granger"
        )
        
        response = self.loop.run_until_complete(
            self.causal_service.discover_currency_pair_relationships(request)
        )
        
        self.assertIsNotNone(response)
        self.assertIsNotNone(response["relationship_id"])
        self.assertEqual(response["symbols"], ["EURUSD", "GBPUSD", "USDJPY"])
        self.assertGreater(len(response["nodes"]), 0)
        self.assertGreater(len(response["edges"]), 0)
    
    def test_discover_regime_change_drivers(self):
        """
        Test discovering regime change drivers.
        """
        request = RegimeChangeDriverRequest(
            symbol="EURUSD",
            timeframe="1d",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            regime_variable="regime",
            algorithm="dowhy"
        )
        
        response = self.loop.run_until_complete(
            self.causal_service.discover_regime_change_drivers(request)
        )
        
        self.assertIsNotNone(response)
        self.assertIsNotNone(response["driver_id"])
        self.assertEqual(response["regime_variable"], "regime")
        self.assertGreater(len(response["drivers"]), 0)
    
    def test_enhance_trading_signals(self):
        """
        Test enhancing trading signals.
        """
        request = TradingSignalEnhancementRequest(
            signals=[
                {
                    "symbol": "EURUSD",
                    "direction": "buy",
                    "confidence": 0.7,
                    "timestamp": datetime.now().isoformat()
                }
            ],
            market_data={
                "close": [1.1, 1.2, 1.3, 1.4, 1.5],
                "high": [1.15, 1.25, 1.35, 1.45, 1.55],
                "low": [1.05, 1.15, 1.25, 1.35, 1.45],
                "volume": [1000, 1100, 1200, 1300, 1400],
                "volatility": [0.01, 0.02, 0.01, 0.03, 0.02]
            }
        )
        
        response = self.loop.run_until_complete(
            self.causal_service.enhance_trading_signals(request)
        )
        
        self.assertIsNotNone(response)
        self.assertEqual(len(response["enhanced_signals"]), 1)
        self.assertEqual(response["count"], 1)
        self.assertGreater(len(response["causal_factors_considered"]), 0)
    
    def test_assess_correlation_breakdown_risk(self):
        """
        Test assessing correlation breakdown risk.
        """
        request = CorrelationBreakdownRiskRequest(
            symbols=["EURUSD", "GBPUSD", "USDJPY"],
            timeframe="1d",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            stress_scenarios=[
                {
                    "name": "volatility_spike",
                    "changes": {
                        "EURUSD": 0.05,
                        "GBPUSD": -0.03,
                        "USDJPY": 0.02
                    }
                }
            ],
            algorithm="counterfactual"
        )
        
        response = self.loop.run_until_complete(
            self.causal_service.assess_correlation_breakdown_risk(request)
        )
        
        self.assertIsNotNone(response)
        self.assertIsNotNone(response["risk_id"])
        self.assertEqual(response["symbols"], ["EURUSD", "GBPUSD", "USDJPY"])
        self.assertIsNotNone(response["baseline_correlations"])
        self.assertIsNotNone(response["stress_correlations"])
        self.assertIsNotNone(response["breakdown_risk_scores"])

if __name__ == "__main__":
    unittest.main()