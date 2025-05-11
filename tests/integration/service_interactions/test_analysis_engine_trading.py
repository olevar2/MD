"""
Integration tests for Analysis Engine Service and Trading Gateway Service interactions.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from analysis_engine.service_clients import ServiceClients as AnalysisEngineServiceClients
from trading_gateway_service.service_clients import ServiceClients as TradingGatewayServiceClients


@pytest.mark.asyncio
@patch("analysis_engine.service_clients.get_service_client_config")
@patch("common_lib.service_client.ResilientServiceClient")
async def test_analysis_engine_calls_trading(
    mock_resilient_service_client,
    mock_get_service_client_config,
    mock_logger
):
    """Test Analysis Engine Service calling Trading Gateway Service."""
    # Mock the service client config
    mock_service_client_config = MagicMock()
    mock_service_client_config.base_url = "http://localhost:8005"
    mock_service_client_config.timeout = 30
    mock_service_client_config.retry = MagicMock()
    mock_service_client_config.retry.max_retries = 3
    mock_service_client_config.retry.initial_backoff = 1.0
    mock_service_client_config.circuit_breaker = MagicMock()
    mock_service_client_config.circuit_breaker.failure_threshold = 5
    mock_service_client_config.circuit_breaker.recovery_timeout = 30.0
    mock_get_service_client_config.return_value = mock_service_client_config
    
    # Mock the resilient service client
    mock_client = AsyncMock()
    mock_client.post.return_value = {
        "order_id": "123456",
        "status": "submitted",
        "symbol": "EURUSD",
        "side": "buy",
        "type": "market",
        "quantity": 10000,
        "price": 1.1,
        "timestamp": "2023-01-01T00:00:00Z"
    }
    mock_resilient_service_client.return_value = mock_client
    
    # Create service clients
    service_clients = AnalysisEngineServiceClients(logger=mock_logger)
    
    # Get trading client
    trading_client = service_clients.get_trading_client()
    
    # Call the trading service
    response = await trading_client.post(
        "/api/v1/orders",
        json={
            "symbol": "EURUSD",
            "side": "buy",
            "type": "market",
            "quantity": 10000,
            "price": 1.1
        }
    )
    
    # Verify the response
    assert response["order_id"] == "123456"
    assert response["status"] == "submitted"
    assert response["symbol"] == "EURUSD"
    assert response["side"] == "buy"
    assert response["type"] == "market"
    assert response["quantity"] == 10000
    assert response["price"] == 1.1
    
    # Verify the mock was called
    mock_client.post.assert_called_once_with(
        "/api/v1/orders",
        json={
            "symbol": "EURUSD",
            "side": "buy",
            "type": "market",
            "quantity": 10000,
            "price": 1.1
        }
    )


@pytest.mark.asyncio
@patch("trading_gateway_service.service_clients.get_service_client_config")
@patch("common_lib.service_client.ResilientServiceClient")
async def test_trading_calls_analysis_engine(
    mock_resilient_service_client,
    mock_get_service_client_config,
    mock_logger
):
    """Test Trading Gateway Service calling Analysis Engine Service."""
    # Mock the service client config
    mock_service_client_config = MagicMock()
    mock_service_client_config.base_url = "http://localhost:8004"
    mock_service_client_config.timeout = 30
    mock_service_client_config.retry = MagicMock()
    mock_service_client_config.retry.max_retries = 3
    mock_service_client_config.retry.initial_backoff = 1.0
    mock_service_client_config.circuit_breaker = MagicMock()
    mock_service_client_config.circuit_breaker.failure_threshold = 5
    mock_service_client_config.circuit_breaker.recovery_timeout = 30.0
    mock_get_service_client_config.return_value = mock_service_client_config
    
    # Mock the resilient service client
    mock_client = AsyncMock()
    mock_client.get.return_value = {
        "symbol": "EURUSD",
        "timeframe": "1h",
        "strategy": "trend_following",
        "signal": "buy",
        "confidence": 0.85,
        "timestamp": "2023-01-01T00:00:00Z",
        "parameters": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        }
    }
    mock_resilient_service_client.return_value = mock_client
    
    # Create service clients
    service_clients = TradingGatewayServiceClients(logger=mock_logger)
    
    # Get analysis engine client
    analysis_engine_client = service_clients.get_analysis_engine_client()
    
    # Call the analysis engine service
    response = await analysis_engine_client.get(
        "/api/v1/strategies/signals",
        params={
            "symbol": "EURUSD",
            "timeframe": "1h",
            "strategy": "trend_following"
        }
    )
    
    # Verify the response
    assert response["symbol"] == "EURUSD"
    assert response["timeframe"] == "1h"
    assert response["strategy"] == "trend_following"
    assert response["signal"] == "buy"
    assert response["confidence"] == 0.85
    
    # Verify the mock was called
    mock_client.get.assert_called_once_with(
        "/api/v1/strategies/signals",
        params={
            "symbol": "EURUSD",
            "timeframe": "1h",
            "strategy": "trend_following"
        }
    )
