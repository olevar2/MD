"""
Unit tests for the Trading Gateway Client.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

from common_lib.clients.exceptions import (
    ClientError,
    ClientConnectionError,
    ClientTimeoutError,
    ClientValidationError,
    ClientAuthenticationError
)
from common_lib.clients.base_client import ClientConfig

from core.error import (
    ServiceError,
    DataFetchError
)
from adapters.trading_gateway_client import TradingGatewayClient
from core.client_factory import get_trading_gateway_client


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    with patch('common_lib.clients.get_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.get = AsyncMock()
        mock_client.post = AsyncMock()
        mock_client.put = AsyncMock()
        mock_client.delete = AsyncMock()
        mock_get_client.return_value = mock_client
        yield mock_client


@pytest.fixture
def client_config():
    """Create a client configuration for testing."""
    return ClientConfig(
        base_url="http://trading-gateway-service:8000/api/v1",
        service_name="trading-gateway-service",
        api_key="test-api-key",
        timeout_seconds=5.0,
        max_retries=3,
        retry_base_delay=0.1,
        retry_backoff_factor=2.0,
        circuit_breaker_failure_threshold=5,
        circuit_breaker_reset_timeout_seconds=30
    )


@pytest.fixture
def client(client_config):
    """Create a client instance for testing."""
    return TradingGatewayClient(client_config)


@pytest.mark.asyncio
async def test_execute_order_success(mock_client):
    """Test successful order execution."""
    # Arrange
    mock_client.post.return_value = {
        "order_id": "test-order-id",
        "status": "executed",
        "execution_price": 1.2345
    }
    
    # Get client from factory (which will use our mocked client)
    with patch('strategy_execution_engine.clients.client_factory.get_client', return_value=mock_client):
        client = get_trading_gateway_client()
        
        # Act
        order = {
            "symbol": "EURUSD",
            "side": "buy",
            "quantity": 1.0,
            "order_type": "market"
        }
        result = await client.execute_order(order)
        
        # Assert
        assert result["order_id"] == "test-order-id"
        assert result["status"] == "executed"
        assert result["execution_price"] == 1.2345
        mock_client.post.assert_called_once_with("orders", data=order)


@pytest.mark.asyncio
async def test_execute_order_error(mock_client):
    """Test error handling during order execution."""
    # Arrange
    mock_client.post.side_effect = ClientError("Failed to execute order", "trading-gateway-service")
    
    # Get client from factory (which will use our mocked client)
    with patch('strategy_execution_engine.clients.client_factory.get_client', return_value=mock_client):
        client = get_trading_gateway_client()
        
        # Act/Assert
        order = {
            "symbol": "EURUSD",
            "side": "buy",
            "quantity": 1.0,
            "order_type": "market"
        }
        with pytest.raises(ServiceError) as excinfo:
            await client.execute_order(order)
        
        assert "Failed to execute order" in str(excinfo.value)
        mock_client.post.assert_called_once_with("orders", data=order)


@pytest.mark.asyncio
async def test_get_order_status_success(mock_client):
    """Test successful order status retrieval."""
    # Arrange
    mock_client.get.return_value = {
        "order_id": "test-order-id",
        "status": "filled",
        "execution_price": 1.2345
    }
    
    # Get client from factory (which will use our mocked client)
    with patch('strategy_execution_engine.clients.client_factory.get_client', return_value=mock_client):
        client = get_trading_gateway_client()
        
        # Act
        result = await client.get_order_status("test-order-id")
        
        # Assert
        assert result["order_id"] == "test-order-id"
        assert result["status"] == "filled"
        assert result["execution_price"] == 1.2345
        mock_client.get.assert_called_once_with("orders/test-order-id")


@pytest.mark.asyncio
async def test_get_order_status_not_found(mock_client):
    """Test order status retrieval for non-existent order."""
    # Arrange
    error = ClientError("Order not found", "trading-gateway-service")
    error.status_code = 404
    mock_client.get.side_effect = error
    
    # Get client from factory (which will use our mocked client)
    with patch('strategy_execution_engine.clients.client_factory.get_client', return_value=mock_client):
        client = get_trading_gateway_client()
        
        # Act/Assert
        with pytest.raises(DataFetchError) as excinfo:
            await client.get_order_status("non-existent-order-id")
        
        assert "Order not found" in str(excinfo.value)
        mock_client.get.assert_called_once_with("orders/non-existent-order-id")


@pytest.mark.asyncio
async def test_get_positions_success(mock_client):
    """Test successful positions retrieval."""
    # Arrange
    mock_client.get.return_value = {
        "positions": [
            {
                "position_id": "pos-1",
                "symbol": "EURUSD",
                "side": "buy",
                "quantity": 1.0,
                "entry_price": 1.2345
            },
            {
                "position_id": "pos-2",
                "symbol": "GBPUSD",
                "side": "sell",
                "quantity": 0.5,
                "entry_price": 1.3456
            }
        ]
    }
    
    # Get client from factory (which will use our mocked client)
    with patch('strategy_execution_engine.clients.client_factory.get_client', return_value=mock_client):
        client = get_trading_gateway_client()
        
        # Act
        result = await client.get_positions()
        
        # Assert
        assert len(result) == 2
        assert result[0]["position_id"] == "pos-1"
        assert result[1]["position_id"] == "pos-2"
        mock_client.get.assert_called_once_with("positions")


@pytest.mark.asyncio
async def test_get_market_data_success(mock_client):
    """Test successful market data retrieval."""
    # Arrange
    mock_client.get.return_value = {
        "symbol": "EURUSD",
        "timeframe": "1m",
        "data": [
            {
                "timestamp": "2023-01-01T00:00:00Z",
                "open": 1.2345,
                "high": 1.2355,
                "low": 1.2335,
                "close": 1.2350,
                "volume": 100
            }
        ]
    }
    
    # Get client from factory (which will use our mocked client)
    with patch('strategy_execution_engine.clients.client_factory.get_client', return_value=mock_client):
        client = get_trading_gateway_client()
        
        # Act
        result = await client.get_market_data("EURUSD", "1m", 100)
        
        # Assert
        assert result["symbol"] == "EURUSD"
        assert result["timeframe"] == "1m"
        assert len(result["data"]) == 1
        mock_client.get.assert_called_once_with("market-data", params={
            "instrument": "EURUSD",
            "timeframe": "1m",
            "count": 100
        })


@pytest.mark.asyncio
async def test_check_health_success(mock_client):
    """Test successful health check."""
    # Arrange
    mock_client.get.return_value = {
        "status": "ok",
        "version": "1.0.0"
    }
    
    # Get client from factory (which will use our mocked client)
    with patch('strategy_execution_engine.clients.client_factory.get_client', return_value=mock_client):
        client = get_trading_gateway_client()
        
        # Act
        result = await client.check_health()
        
        # Assert
        assert result["status"] == "healthy"
        assert result["message"] == "Connection successful"
        assert result["details"]["status"] == "ok"
        assert result["details"]["version"] == "1.0.0"
        mock_client.get.assert_called_once_with("health")


@pytest.mark.asyncio
async def test_backward_compatibility_methods(mock_client):
    """Test backward compatibility methods."""
    # Arrange
    mock_client.post.return_value = {
        "order_id": "test-order-id",
        "status": "executed"
    }
    mock_client.get.return_value = {
        "positions": [
            {"position_id": "pos-1", "symbol": "EURUSD"}
        ]
    }
    
    # Get client from factory (which will use our mocked client)
    with patch('strategy_execution_engine.clients.client_factory.get_client', return_value=mock_client):
        client = get_trading_gateway_client()
        
        # Act - Test submit_order (alias for execute_order)
        order = {"symbol": "EURUSD", "side": "buy", "quantity": 1.0}
        submit_result = await client.submit_order(order)
        
        # Assert
        assert submit_result["order_id"] == "test-order-id"
        mock_client.post.assert_called_once_with("orders", data=order)
        
        # Act - Test get_open_positions (alias for get_positions)
        positions_result = await client.get_open_positions()
        
        # Assert
        assert len(positions_result) == 1
        assert positions_result[0]["position_id"] == "pos-1"
        mock_client.get.assert_called_once_with("positions")
