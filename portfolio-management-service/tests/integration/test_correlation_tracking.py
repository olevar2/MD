"""
Integration tests for cross-asset correlation tracking functionality.

This module contains tests that verify the correlation tracking module works
correctly with different asset classes and properly integrates with other services.
"""
import pytest
import json
from datetime import datetime, timedelta
import uuid
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from portfolio_management_service.main import app
from portfolio_management_service.models.correlation import CorrelationStrength
from portfolio_management_service.multi_asset.correlation_tracking import CrossAssetCorrelationTracker
from portfolio_management_service.services.correlation_service import CorrelationService


@pytest.fixture
def test_client():
    """Create a test client for API testing"""
    return TestClient(app)


@pytest.fixture
def mock_correlation_service():
    """Create a mock correlation service for testing"""
    with patch('portfolio_management_service.multi_asset.correlation_tracking.CorrelationService') as mock_service:
        service_instance = MagicMock()
        mock_service.return_value = service_instance
        yield service_instance


@pytest.fixture
def mock_market_data_service():
    """Create a mock market data service for testing"""
    with patch('portfolio_management_service.multi_asset.correlation_tracking.MarketDataService') as mock_service:
        service_instance = MagicMock()
        mock_service.return_value = service_instance
        yield service_instance


@pytest.fixture
def correlation_tracker(mock_correlation_service, mock_market_data_service):
    """Create a correlation tracker instance with mocked dependencies"""
    return CrossAssetCorrelationTracker(
        correlation_service=mock_correlation_service,
        market_data_service=mock_market_data_service
    )


@pytest.fixture
def sample_correlation_data():
    """Create sample correlation data for testing"""
    return {
        "base_symbol": "EURUSD",
        "base_asset_class": "forex",
        "correlations": [
            {
                "symbol": "GBPUSD",
                "asset_class": "forex",
                "correlation_value": 0.85,
                "strength": CorrelationStrength.STRONG,
                "direction": "positive"
            },
            {
                "symbol": "GOLD",
                "asset_class": "commodities",
                "correlation_value": -0.62,
                "strength": CorrelationStrength.MODERATE,
                "direction": "negative"
            },
            {
                "symbol": "BTC",
                "asset_class": "crypto",
                "correlation_value": 0.18,
                "strength": CorrelationStrength.WEAK,
                "direction": "positive"
            }
        ]
    }


class TestCrossAssetCorrelation:
    """Test suite for cross-asset correlation tracking"""

    def test_calculate_correlations(self, correlation_tracker, mock_market_data_service, mock_correlation_service):
        """Test calculation of correlations between assets of different classes"""
        # Setup
        symbols = ["EURUSD", "GBPUSD", "GOLD", "BTC"]
        timeframe = "1d"
        lookback_period = 30
        
        # Mock market data service to return price data for each symbol
        price_data = {}
        for symbol in symbols:
            # Create a list of 30 days of mock price data
            price_data[symbol] = [
                {"timestamp": (datetime.now() - timedelta(days=i)).isoformat(), 
                 "close": 100 + i * 0.5}
                for i in range(lookback_period)
            ]
        
        mock_market_data_service.get_historical_prices.side_effect = lambda symbol, tf, period: price_data[symbol]
        
        # Calculate correlations
        result = correlation_tracker.calculate_correlations("EURUSD", symbols[1:], timeframe, lookback_period)
        
        # Verify
        assert len(result) == len(symbols) - 1
        for corr in result:
            assert "symbol" in corr
            assert "correlation_value" in corr
            assert "strength" in corr
            assert "direction" in corr
        
        # Verify service calls
        assert mock_market_data_service.get_historical_prices.call_count == len(symbols)

    def test_update_correlation_matrix(self, correlation_tracker, mock_correlation_service):
        """Test updating the correlation matrix with new correlation data"""
        # Setup
        base_symbol = "EURUSD"
        corr_data = [
            {
                "symbol": "GBPUSD",
                "asset_class": "forex",
                "correlation_value": 0.85,
                "strength": CorrelationStrength.STRONG,
                "direction": "positive"
            },
            {
                "symbol": "GOLD",
                "asset_class": "commodities",
                "correlation_value": -0.62,
                "strength": CorrelationStrength.MODERATE,
                "direction": "negative"
            }
        ]
        
        # Execute
        correlation_tracker.update_correlation_matrix(base_symbol, "forex", corr_data)
        
        # Verify service call
        mock_correlation_service.update_correlations.assert_called_once_with(
            base_symbol, "forex", corr_data
        )

    def test_api_get_correlations(self, test_client, sample_correlation_data):
        """Test API endpoint for retrieving correlations for a symbol"""
        with patch('portfolio_management_service.api.v1.correlation_api.correlation_tracker') as mock_tracker:
            # Setup
            symbol = "EURUSD"
            threshold = 0.5
            limit = 10
            
            mock_tracker.get_correlations.return_value = sample_correlation_data
            
            # Execute
            response = test_client.get(f"/api/v1/correlations/symbol/{symbol}?threshold={threshold}&limit={limit}")
            
            # Verify
            assert response.status_code == 200
            result = response.json()
            
            assert result["base_symbol"] == symbol
            assert len(result["correlations"]) == 3
            
            # Verify the service call
            mock_tracker.get_correlations.assert_called_once_with(symbol, threshold, limit)

    def test_api_get_cross_asset_correlations(self, test_client):
        """Test API endpoint for retrieving cross-asset correlations"""
        with patch('portfolio_management_service.api.v1.correlation_api.correlation_tracker') as mock_tracker:
            # Setup
            mock_matrix = {
                "forex": {
                    "crypto": 0.3,
                    "stocks": 0.15,
                    "commodities": -0.25
                },
                "crypto": {
                    "forex": 0.3,
                    "stocks": 0.42,
                    "commodities": 0.18
                },
                "stocks": {
                    "forex": 0.15,
                    "crypto": 0.42,
                    "commodities": 0.05
                },
                "commodities": {
                    "forex": -0.25,
                    "crypto": 0.18,
                    "stocks": 0.05
                }
            }
            
            mock_tracker.get_cross_asset_correlation_matrix.return_value = mock_matrix
            
            # Execute
            response = test_client.get("/api/v1/correlations/cross-asset")
            
            # Verify
            assert response.status_code == 200
            result = response.json()
            
            assert "forex" in result
            assert "crypto" in result
            assert "stocks" in result
            assert "commodities" in result
            
            # Check that the matrix is symmetric (approximately)
            assert result["forex"]["crypto"] == result["crypto"]["forex"]
            assert result["forex"]["stocks"] == result["stocks"]["forex"]
            
            # Verify service call
            mock_tracker.get_cross_asset_correlation_matrix.assert_called_once()

    def test_calculate_diversification_benefit(self, correlation_tracker):
        """Test calculation of diversification benefit from correlations"""
        # Setup - Create a correlation matrix with different correlation strengths
        correlations = [
            {"asset_class": "forex", "correlation_value": 0.2},
            {"asset_class": "stocks", "correlation_value": -0.3},
            {"asset_class": "crypto", "correlation_value": 0.1}
        ]
        
        # Execute
        result = correlation_tracker.calculate_diversification_benefit(correlations)
        
        # Verify - The benefit should be higher when correlations are lower/more negative
        assert result > 0
        assert result <= 1
        
    def test_get_cross_asset_correlation_matrix(self, correlation_tracker, mock_correlation_service):
        """Test retrieving the complete cross-asset correlation matrix"""
        # Setup
        mock_matrix = {
            "forex": {
                "crypto": 0.3,
                "stocks": 0.15,
                "commodities": -0.25
            },
            "crypto": {
                "forex": 0.3,
                "stocks": 0.42, 
                "commodities": 0.18
            },
            "stocks": {
                "forex": 0.15,
                "crypto": 0.42,
                "commodities": 0.05
            },
            "commodities": {
                "forex": -0.25,
                "crypto": 0.18,
                "stocks": 0.05
            }
        }
        
        mock_correlation_service.get_cross_asset_correlation_matrix.return_value = mock_matrix
        
        # Execute
        result = correlation_tracker.get_cross_asset_correlation_matrix()
        
        # Verify
        assert result == mock_matrix
        mock_correlation_service.get_cross_asset_correlation_matrix.assert_called_once()
        
        # Check matrix symmetry
        assert result["forex"]["crypto"] == result["crypto"]["forex"]
        assert result["forex"]["stocks"] == result["stocks"]["forex"]
        
    def test_find_correlation_clusters(self, correlation_tracker, mock_correlation_service):
        """Test finding clusters of correlated assets"""
        # Setup
        mock_clusters = [
            {
                "cluster_id": 1,
                "assets": ["EURUSD", "GBPUSD", "AUDUSD"],
                "asset_classes": ["forex", "forex", "forex"],
                "avg_correlation": 0.82,
                "strength": CorrelationStrength.STRONG
            },
            {
                "cluster_id": 2,
                "assets": ["GOLD", "SILVER", "OIL"],
                "asset_classes": ["commodities", "commodities", "commodities"],
                "avg_correlation": 0.65,
                "strength": CorrelationStrength.MODERATE
            }
        ]
        
        mock_correlation_service.find_correlation_clusters.return_value = mock_clusters
        threshold = 0.6
        
        # Execute
        result = correlation_tracker.find_correlation_clusters(threshold)
        
        # Verify
        assert result == mock_clusters
        mock_correlation_service.find_correlation_clusters.assert_called_once_with(threshold)
        
    def test_get_correlation_trend(self, correlation_tracker, mock_correlation_service):
        """Test retrieving correlation trend data over time"""
        # Setup
        base_symbol = "EURUSD"
        target_symbol = "GOLD"
        period = "30d"
        
        trend_data = [
            {"date": "2025-03-15", "correlation": -0.58},
            {"date": "2025-03-22", "correlation": -0.62},
            {"date": "2025-03-29", "correlation": -0.67},
            {"date": "2025-04-05", "correlation": -0.70},
            {"date": "2025-04-12", "correlation": -0.65}
        ]
        
        mock_correlation_service.get_correlation_history.return_value = trend_data
        
        # Execute
        result = correlation_tracker.get_correlation_trend(base_symbol, target_symbol, period)
        
        # Verify
        assert result == trend_data
        mock_correlation_service.get_correlation_history.assert_called_once_with(
            base_symbol, target_symbol, period
        )
        
    def test_api_get_correlation_clusters(self, test_client):
        """Test API endpoint for retrieving correlation clusters"""
        with patch('portfolio_management_service.api.v1.correlation_api.correlation_tracker') as mock_tracker:
            # Setup
            threshold = 0.6
            mock_clusters = [
                {
                    "cluster_id": 1,
                    "assets": ["EURUSD", "GBPUSD", "AUDUSD"],
                    "asset_classes": ["forex", "forex", "forex"],
                    "avg_correlation": 0.82,
                    "strength": "strong"
                }
            ]
            
            mock_tracker.find_correlation_clusters.return_value = mock_clusters
            
            # Execute
            response = test_client.get(f"/api/v1/correlations/clusters?threshold={threshold}")
            
            # Verify
            assert response.status_code == 200
            result = response.json()
            
            assert len(result) == 1
            assert result[0]["cluster_id"] == 1
            assert len(result[0]["assets"]) == 3
            
            # Verify service call
            mock_tracker.find_correlation_clusters.assert_called_once_with(threshold)
            
    def test_api_get_correlation_trend(self, test_client):
        """Test API endpoint for retrieving correlation trend data"""
        with patch('portfolio_management_service.api.v1.correlation_api.correlation_tracker') as mock_tracker:
            # Setup
            base_symbol = "EURUSD"
            target_symbol = "GOLD"
            period = "30d"
            
            trend_data = [
                {"date": "2025-03-15", "correlation": -0.58},
                {"date": "2025-03-22", "correlation": -0.62},
                {"date": "2025-03-29", "correlation": -0.67},
                {"date": "2025-04-05", "correlation": -0.70},
                {"date": "2025-04-12", "correlation": -0.65}
            ]
            
            mock_tracker.get_correlation_trend.return_value = trend_data
            
            # Execute
            response = test_client.get(f"/api/v1/correlations/trend?base_symbol={base_symbol}&target_symbol={target_symbol}&period={period}")
            
            # Verify
            assert response.status_code == 200
            result = response.json()
            assert len(result) == 5
            assert "date" in result[0]
            assert "correlation" in result[0]            
            # Verify service call
            mock_tracker.get_correlation_trend.assert_called_once_with(base_symbol, target_symbol, period)
