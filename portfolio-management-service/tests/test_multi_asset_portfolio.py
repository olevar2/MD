"""
Integration tests for multi-asset portfolio management functionality.

This module contains tests that verify the multi-asset portfolio manager works
correctly with different asset classes and properly integrates with other services.
"""
import pytest
import json
from datetime import datetime, timedelta
import uuid
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from core.main_1 import app
from core.position import Position, PositionCreate, PositionStatus, AssetClass
from core.multi_asset_portfolio_manager import MultiAssetPortfolioManager
from services.portfolio_service import PortfolioService
from analysis_engine.services.multi_asset_service import MultiAssetService


@pytest.fixture
def test_client():
    """Create a test client for API testing"""
    return TestClient(app)


@pytest.fixture
def mock_portfolio_service():
    """Create a mock portfolio service for testing"""
    with patch('portfolio_management_service.multi_asset.multi_asset_portfolio_manager.PortfolioService') as mock_service:
        service_instance = MagicMock()
        mock_service.return_value = service_instance
        yield service_instance


@pytest.fixture
def mock_multi_asset_service():
    """Create a mock multi-asset service for testing"""
    with patch('portfolio_management_service.multi_asset.multi_asset_portfolio_manager.MultiAssetService') as mock_service:
        service_instance = MagicMock()
        mock_service.return_value = service_instance
        yield service_instance


@pytest.fixture
def portfolio_manager(mock_portfolio_service, mock_multi_asset_service):
    """Create a portfolio manager instance with mocked dependencies"""
    return MultiAssetPortfolioManager(
        portfolio_service=mock_portfolio_service,
        multi_asset_service=mock_multi_asset_service
    )


@pytest.fixture
def sample_account_id():
    """Generate a sample account ID for testing"""
    return f"test-account-{uuid.uuid4()}"


@pytest.fixture
def sample_position_forex():
    """Create a sample forex position for testing"""
    return {
        "id": str(uuid.uuid4()),
        "symbol": "EURUSD",
        "direction": "long",
        "quantity": 10000,
        "entry_price": 1.1850,
        "current_price": 1.1900,
        "account_id": "test-account",
        "asset_class": AssetClass.FOREX,
        "pip_value": 0.0001,
        "margin_rate": 0.03,
        "status": PositionStatus.OPEN,
        "entry_date": datetime.now().isoformat(),
        "current_value": 11900.0,
        "unrealized_pl": 50.0,
        "margin_used": 357.0
    }


@pytest.fixture
def sample_position_crypto():
    """Create a sample crypto position for testing"""
    return {
        "id": str(uuid.uuid4()),
        "symbol": "BTCUSDT",
        "direction": "long",
        "quantity": 0.5,
        "entry_price": 45000.0,
        "current_price": 48000.0,
        "account_id": "test-account",
        "asset_class": AssetClass.CRYPTO,
        "margin_rate": 0.5,
        "status": PositionStatus.OPEN,
        "entry_date": datetime.now().isoformat(),
        "current_value": 24000.0,
        "unrealized_pl": 1500.0,
        "margin_used": 12000.0
    }


@pytest.fixture
def sample_position_stock():
    """Create a sample stock position for testing"""
    return {
        "id": str(uuid.uuid4()),
        "symbol": "AAPL",
        "direction": "long",
        "quantity": 10,
        "entry_price": 170.0,
        "current_price": 175.0,
        "account_id": "test-account",
        "asset_class": AssetClass.STOCKS,
        "status": PositionStatus.OPEN,
        "entry_date": datetime.now().isoformat(),
        "current_value": 1750.0,
        "unrealized_pl": 50.0,
        "margin_used": 1750.0
    }


class TestMultiAssetPositionCreation:
    """Test suite for multi-asset position creation"""

    def test_create_forex_position(self, portfolio_manager, mock_multi_asset_service, mock_portfolio_service):
        """Test creation of forex position with asset-specific parameters"""
        # Setup
        position_data = {
            "symbol": "EURUSD",
            "direction": "long",
            "quantity": 10000, 
            "entry_price": 1.1850,
            "account_id": "test-account"
        }
        
        mock_multi_asset_service.get_asset_info.return_value = {
            "asset_class": "forex",
            "trading_parameters": {
                "pip_value": 0.0001,
                "margin_rate": 0.03
            }
        }
        
        mock_portfolio_service.create_position.return_value = Position(
            id=str(uuid.uuid4()),
            **position_data,
            asset_class="forex",
            pip_value=0.0001,
            margin_rate=0.03,
            status=PositionStatus.OPEN
        )
        
        # Execute
        result = portfolio_manager.create_position(position_data)
        
        # Verify
        assert result is not None
        assert result.asset_class == "forex"
        assert result.pip_value == 0.0001
        assert result.margin_rate == 0.03
        
        # Verify service calls
        mock_multi_asset_service.get_asset_info.assert_called_once_with("EURUSD")
        mock_portfolio_service.create_position.assert_called_once()

    def test_create_crypto_position(self, portfolio_manager, mock_multi_asset_service, mock_portfolio_service):
        """Test creation of crypto position with asset-specific parameters"""
        # Setup
        position_data = {
            "symbol": "BTCUSDT",
            "direction": "long",
            "quantity": 0.5,
            "entry_price": 45000.0,
            "account_id": "test-account"
        }
        
        mock_multi_asset_service.get_asset_info.return_value = {
            "asset_class": "crypto",
            "trading_parameters": {
                "margin_rate": 0.5,
                "trading_fee": 0.001
            }
        }
        
        mock_portfolio_service.create_position.return_value = Position(
            id=str(uuid.uuid4()),
            **position_data,
            asset_class="crypto",
            margin_rate=0.5,
            fee=0.001,
            status=PositionStatus.OPEN
        )
        
        # Execute
        result = portfolio_manager.create_position(position_data)
        
        # Verify
        assert result is not None
        assert result.asset_class == "crypto"
        assert result.margin_rate == 0.5
        assert result.fee == 0.001
        
        # Verify service calls
        mock_multi_asset_service.get_asset_info.assert_called_once_with("BTCUSDT")
        mock_portfolio_service.create_position.assert_called_once()

    def test_create_stock_position(self, portfolio_manager, mock_multi_asset_service, mock_portfolio_service):
        """Test creation of stock position with asset-specific parameters"""
        # Setup
        position_data = {
            "symbol": "AAPL",
            "direction": "long",
            "quantity": 10,
            "entry_price": 170.0,
            "account_id": "test-account"
        }
        
        mock_multi_asset_service.get_asset_info.return_value = {
            "asset_class": "stocks",
            "trading_parameters": {
                "trading_fee": 0.0025
            }
        }
        
        mock_portfolio_service.create_position.return_value = Position(
            id=str(uuid.uuid4()),
            **position_data,
            asset_class="stocks",
            fee=0.0025,
            status=PositionStatus.OPEN
        )
        
        # Execute
        result = portfolio_manager.create_position(position_data)
        
        # Verify
        assert result is not None
        assert result.asset_class == "stocks"
        assert result.fee == 0.0025
        
        # Verify service calls
        mock_multi_asset_service.get_asset_info.assert_called_once_with("AAPL")
        mock_portfolio_service.create_position.assert_called_once()

    def test_api_create_forex_position(self, test_client, mock_portfolio_service, mock_multi_asset_service):
        """Test API endpoint for creating a forex position"""
        with patch('portfolio_management_service.api.v1.multi_asset_portfolio_api.portfolio_manager') as mock_manager:
            # Setup
            position_data = {
                "symbol": "EURUSD",
                "direction": "long",
                "quantity": 10000, 
                "entry_price": 1.1850,
                "account_id": "test-account"
            }
            
            position_id = str(uuid.uuid4())
            mock_manager.create_position.return_value = Position(
                id=position_id,
                **position_data,
                asset_class="forex",
                pip_value=0.0001,
                margin_rate=0.03,
                status=PositionStatus.OPEN
            )
            
            # Execute
            response = test_client.post("/api/v1/multi-asset/positions", json=position_data)
            
            # Verify
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == position_id
            assert result["asset_class"] == "forex"
            assert result["pip_value"] == 0.0001
            
            # Verify service call
            mock_manager.create_position.assert_called_once_with(position_data)


class TestPortfolioSummary:
    """Test suite for portfolio summary functionality"""

    def test_portfolio_summary_aggregation(self, portfolio_manager, mock_portfolio_service, sample_account_id):
        """Test that portfolio summary correctly aggregates positions by asset class"""
        # Setup
        forex_position = {
            "id": str(uuid.uuid4()),
            "symbol": "EURUSD",
            "asset_class": "forex",
            "current_value": 10000,
            "unrealized_pl": 100,
            "margin_used": 300
        }
        
        crypto_position = {
            "id": str(uuid.uuid4()),
            "symbol": "BTCUSDT",
            "asset_class": "crypto",
            "current_value": 20000,
            "unrealized_pl": -500,
            "margin_used": 10000
        }
        
        stock_position = {
            "id": str(uuid.uuid4()),
            "symbol": "AAPL",
            "asset_class": "stocks",
            "current_value": 5000,
            "unrealized_pl": 200,
            "margin_used": 5000
        }
        
        mock_portfolio_service.get_portfolio_summary.return_value = {
            "total_value": 35000,
            "unrealized_pl": -200,
            "positions": [forex_position, crypto_position, stock_position]
        }
        
        # Execute
        result = portfolio_manager.get_portfolio_summary(sample_account_id)
        
        # Verify
        assert "by_asset_class" in result
        by_asset_class = result["by_asset_class"]
        
        # Check forex aggregation
        assert "forex" in by_asset_class
        assert by_asset_class["forex"]["count"] == 1
        assert by_asset_class["forex"]["value"] == 10000
        assert by_asset_class["forex"]["profit_loss"] == 100
        assert by_asset_class["forex"]["margin_used"] == 300
        assert by_asset_class["forex"]["allocation_pct"] == (10000 / 35000) * 100
        
        # Check crypto aggregation
        assert "crypto" in by_asset_class
        assert by_asset_class["crypto"]["count"] == 1
        assert by_asset_class["crypto"]["value"] == 20000
        assert by_asset_class["crypto"]["profit_loss"] == -500
        assert by_asset_class["crypto"]["margin_used"] == 10000
        assert by_asset_class["crypto"]["allocation_pct"] == (20000 / 35000) * 100
        
        # Check stocks aggregation
        assert "stocks" in by_asset_class
        assert by_asset_class["stocks"]["count"] == 1
        assert by_asset_class["stocks"]["value"] == 5000
        assert by_asset_class["stocks"]["profit_loss"] == 200
        assert by_asset_class["stocks"]["margin_used"] == 5000
        assert by_asset_class["stocks"]["allocation_pct"] == (5000 / 35000) * 100

    def test_api_get_portfolio_summary(self, test_client, sample_account_id):
        """Test API endpoint for getting portfolio summary with asset class breakdown"""
        with patch('portfolio_management_service.api.v1.multi_asset_portfolio_api.portfolio_manager') as mock_manager:
            # Setup
            portfolio_summary = {
                "total_value": 35000,
                "unrealized_pl": -200,
                "by_asset_class": {
                    "forex": {
                        "count": 2,
                        "value": 15000,
                        "profit_loss": 150,
                        "margin_used": 450,
                        "allocation_pct": 42.86
                    },
                    "crypto": {
                        "count": 1,
                        "value": 20000,
                        "profit_loss": -500,
                        "margin_used": 10000,
                        "allocation_pct": 57.14
                    }
                },
                "cross_asset_risk": {
                    "cross_correlation": 0.2,
                    "diversification_score": 0.7
                }
            }
            
            mock_manager.get_portfolio_summary.return_value = portfolio_summary
            
            # Execute
            response = test_client.get(f"/api/v1/multi-asset/portfolio/{sample_account_id}/summary")
            
            # Verify
            assert response.status_code == 200
            result = response.json()
            
            assert result["total_value"] == 35000
            assert "by_asset_class" in result
            assert "forex" in result["by_asset_class"]
            assert "crypto" in result["by_asset_class"]
            assert "cross_asset_risk" in result
            
            # Verify service call
            mock_manager.get_portfolio_summary.assert_called_once_with(sample_account_id)


class TestRiskCalculation:
    """Test suite for risk calculation functionality"""

    def test_unified_risk_calculation(self, portfolio_manager, mock_portfolio_service, sample_account_id):
        """Test calculation of unified risk metrics across asset classes"""
        # Setup
        mock_positions = [
            Position(**{
                "id": "pos1",
                "symbol": "EURUSD",
                "direction": "long",
                "quantity": 10000,
                "entry_price": 1.1850,
                "current_price": 1.1900,
                "account_id": sample_account_id,
                "asset_class": AssetClass.FOREX,
                "status": PositionStatus.OPEN
            }),
            Position(**{
                "id": "pos2",
                "symbol": "BTCUSDT",
                "direction": "long",
                "quantity": 0.5,
                "entry_price": 45000,
                "current_price": 48000,
                "account_id": sample_account_id,
                "asset_class": AssetClass.CRYPTO,
                "status": PositionStatus.OPEN
            })
        ]
        
        mock_portfolio_service.get_positions.return_value = mock_positions
        
        # Create patches for the risk calculation methods
        with patch.object(portfolio_manager, '_calculate_value_at_risk') as mock_var:
            with patch.object(portfolio_manager, '_calculate_correlation_adjusted_risk') as mock_corr:
                with patch.object(portfolio_manager, '_calculate_concentration_risk') as mock_conc:
                    # Setup mock returns
                    mock_var.return_value = {"var_95": 1200.0, "var_99": 1800.0}
                    mock_corr.return_value = {"adjusted_portfolio_risk": 0.8, "diversification_benefit": 0.15}
                    mock_conc.return_value = {"concentration_score": 0.7, "max_concentrated_class": "crypto"}
                    
                    # Execute
                    result = portfolio_manager.calculate_unified_risk(sample_account_id)
                    
                    # Verify
                    assert "value_at_risk" in result
                    assert result["value_at_risk"]["var_95"] == 1200.0
                    assert result["value_at_risk"]["var_99"] == 1800.0
                    
                    assert "correlation_risk" in result
                    assert result["correlation_risk"]["adjusted_portfolio_risk"] == 0.8
                    assert result["correlation_risk"]["diversification_benefit"] == 0.15
                    
                    assert "concentration_risk" in result
                    assert result["concentration_risk"]["concentration_score"] == 0.7
                    assert result["concentration_risk"]["max_concentrated_class"] == "crypto"
                    
                    # Verify that the calculation methods were called with the positions
                    mock_var.assert_called_once_with(mock_positions)
                    mock_corr.assert_called_once_with(mock_positions)
                    mock_conc.assert_called_once_with(mock_positions)
                    
                    # Verify that the get_positions method was called
                    mock_portfolio_service.get_positions.assert_called_once_with(sample_account_id)

    def test_api_get_unified_risk(self, test_client, sample_account_id):
        """Test API endpoint for getting unified risk metrics"""
        with patch('portfolio_management_service.api.v1.multi_asset_portfolio_api.portfolio_manager') as mock_manager:
            # Setup
            risk_metrics = {
                "value_at_risk": {
                    "var_95": 1200.0,
                    "var_99": 1800.0
                },
                "correlation_risk": {
                    "adjusted_portfolio_risk": 0.8,
                    "diversification_benefit": 0.15
                },
                "concentration_risk": {
                    "concentration_score": 0.7,
                    "max_concentrated_class": "crypto"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            mock_manager.calculate_unified_risk.return_value = risk_metrics
            
            # Execute
            response = test_client.get(f"/api/v1/multi-asset/portfolio/{sample_account_id}/risk")
            
            # Verify
            assert response.status_code == 200
            result = response.json()
            
            assert "value_at_risk" in result
            assert "correlation_risk" in result
            assert "concentration_risk" in result
            
            # Verify service call
            mock_manager.calculate_unified_risk.assert_called_once_with(sample_account_id)


class TestAssetAllocationRecommendations:
    """Test suite for asset allocation recommendation functionality"""

    def test_get_asset_allocation_recommendations(self, portfolio_manager, mock_portfolio_service, sample_account_id):
        """Test generation of asset allocation recommendations"""
        # Setup
        portfolio_summary = {
            "by_asset_class": {
                "forex": {
                    "count": 2,
                    "value": 15000,
                    "allocation_pct": 60
                },
                "crypto": {
                    "count": 1,
                    "value": 10000,
                    "allocation_pct": 40
                }
            }
        }
        
        # Mock the get_portfolio_summary method
        with patch.object(portfolio_manager, 'get_portfolio_summary') as mock_summary:
            mock_summary.return_value = portfolio_summary
            
            # Execute
            result = portfolio_manager.get_asset_allocation_recommendations(sample_account_id)
            
            # Verify
            assert "current_allocation" in result
            assert result["current_allocation"] == portfolio_summary["by_asset_class"]
            
            assert "recommended_allocation" in result
            recommended = result["recommended_allocation"]
            assert "forex" in recommended
            assert "crypto" in recommended
            assert "stocks" in recommended
            
            # Verify that get_portfolio_summary was called
            mock_summary.assert_called_once_with(sample_account_id)

    def test_api_get_allocation_recommendations(self, test_client, sample_account_id):
        """Test API endpoint for getting asset allocation recommendations"""
        with patch('portfolio_management_service.api.v1.multi_asset_portfolio_api.portfolio_manager') as mock_manager:
            # Setup
            recommendations = {
                "current_allocation": {
                    "forex": {
                        "count": 2,
                        "value": 15000,
                        "allocation_pct": 60
                    },
                    "crypto": {
                        "count": 1,
                        "value": 10000,
                        "allocation_pct": 40
                    }
                },
                "recommended_allocation": {
                    "forex": 40,
                    "crypto": 20,
                    "stocks": 30,
                    "commodities": 10
                },
                "explanation": "Recommended allocation balances risk across asset classes"
            }
            
            mock_manager.get_asset_allocation_recommendations.return_value = recommendations
            
            # Execute
            response = test_client.get(f"/api/v1/multi-asset/portfolio/{sample_account_id}/allocation-recommendations")
            
            # Verify
            assert response.status_code == 200
            result = response.json()
            
            assert "current_allocation" in result
            assert "recommended_allocation" in result
            assert "explanation" in result
            
            # Verify service call
            mock_manager.get_asset_allocation_recommendations.assert_called_once_with(sample_account_id)


class TestCrossAssetCorrelation:
    """Test suite for cross-asset correlation functionality"""

    def test_calculate_cross_asset_risk(self, portfolio_manager):
        """Test calculation of cross-asset risk metrics"""
        # This is a placeholder test for the _calculate_cross_asset_risk method
        # In a real implementation, we would test the actual correlation calculation logic
        positions = [
            {
                "id": "pos1",
                "symbol": "EURUSD",
                "asset_class": "forex"
            },
            {
                "id": "pos2",
                "symbol": "BTCUSDT",
                "asset_class": "crypto"
            }
        ]
        
        result = portfolio_manager._calculate_cross_asset_risk(positions)
        
        # Verify the structure of the result
        assert "cross_correlation" in result
        assert "diversification_score" in result
