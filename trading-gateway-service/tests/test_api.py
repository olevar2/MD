"""
Tests for API endpoints in the Trading Gateway Service.

This module contains tests for the API endpoints of the Trading Gateway Service.
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from trading_gateway_service.main import app
from trading_gateway_service.error import MarketDataError

class TestAPI(unittest.TestCase):
    """Tests for API endpoints."""
    
    def setUp(self):
        """Set up test client and mocks."""
        self.client = TestClient(app)
        
        # Mock app state
        app.state.market_data_service = MagicMock()
        app.state.order_reconciliation_service = MagicMock()
        app.state.monitoring = MagicMock()
        app.state.degraded_mode_manager = MagicMock()
        
        # Set up mock return values
        app.state.market_data_service.get_market_data.return_value = [
            {"timestamp": "2023-01-01T00:00:00", "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.1, "volume": 100}
        ]
        
        app.state.order_reconciliation_service.reconcile_orders.return_value = {
            "reconciled_orders": 10,
            "discrepancies": []
        }
        
        app.state.monitoring.get_metrics.return_value = {
            "order_latency_ms": [100, 120, 90],
            "market_data_latency_ms": [30, 40, 20],
            "error_rate": 0.01
        }
        
        app.state.degraded_mode_manager.get_status.return_value = {
            "is_degraded": False,
            "level": "NORMAL",
            "reason": None,
            "message": None,
            "active_fallbacks": []
        }
    
    def test_get_market_data(self):
        """Test the get_market_data endpoint."""
        # Make request
        response = self.client.get("/api/v1/market-data/EURUSD?timeframe=1h")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["symbol"], "EURUSD")
        self.assertEqual(response.json()["timeframe"], "1h")
        self.assertIsInstance(response.json()["data"], list)
        
        # Check that the market data service was called correctly
        app.state.market_data_service.get_market_data.assert_called_once_with("EURUSD", "1h")
    
    def test_get_market_data_error(self):
        """Test the get_market_data endpoint with an error."""
        # Set up mock to raise an error
        app.state.market_data_service.get_market_data.side_effect = MarketDataError(
            message="Failed to fetch market data",
            symbol="EURUSD",
            details={"timeframe": "1h"}
        )
        
        # Make request
        response = self.client.get("/api/v1/market-data/EURUSD?timeframe=1h")
        
        # Check response
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error_type"], "MarketDataError")
        self.assertEqual(response.json()["message"], "Failed to fetch market data")
        self.assertEqual(response.json()["details"]["symbol"], "EURUSD")
    
    def test_reconcile_orders(self):
        """Test the reconcile_orders endpoint."""
        # Make request
        response = self.client.post("/api/v1/reconcile/orders")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self.assertEqual(response.json()["reconciled_orders"], 10)
        self.assertEqual(response.json()["discrepancies"], [])
        
        # Check that the reconciliation service was called correctly
        app.state.order_reconciliation_service.reconcile_orders.assert_called_once()
    
    def test_get_performance_metrics(self):
        """Test the get_performance_metrics endpoint."""
        # Make request
        response = self.client.get("/api/v1/monitoring/performance")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self.assertIsInstance(response.json()["metrics"], dict)
        
        # Check that the monitoring service was called correctly
        app.state.monitoring.get_metrics.assert_called_once()
    
    def test_get_degraded_mode_status(self):
        """Test the get_degraded_mode_status endpoint."""
        # Make request
        response = self.client.get("/api/v1/status/degraded-mode")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["is_degraded"], False)
        self.assertEqual(response.json()["level"], "NORMAL")
        
        # Check that the degraded mode manager was called correctly
        app.state.degraded_mode_manager.get_status.assert_called_once()

if __name__ == "__main__":
    unittest.main()
