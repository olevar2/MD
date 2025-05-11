"""
Integration tests for Market Data Service and Feature Store Service interactions.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from feature_store_service.service_clients import ServiceClients as FeatureStoreServiceClients
from market_data_service.service_clients import ServiceClients as MarketDataServiceClients


@pytest.mark.asyncio
@patch("feature_store_service.service_clients.get_service_client_config")
@patch("common_lib.service_client.ResilientServiceClient")
async def test_feature_store_calls_market_data(
    mock_resilient_service_client,
    mock_get_service_client_config,
    mock_logger
):
    """Test Feature Store Service calling Market Data Service."""
    # Mock the service client config
    mock_service_client_config = MagicMock()
    mock_service_client_config.base_url = "http://localhost:8001"
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
        "data": [
            {"timestamp": "2023-01-01T00:00:00Z", "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.1, "volume": 1000},
            {"timestamp": "2023-01-01T01:00:00Z", "open": 1.1, "high": 1.3, "low": 1.1, "close": 1.2, "volume": 1200}
        ]
    }
    mock_resilient_service_client.return_value = mock_client
    
    # Create service clients
    service_clients = FeatureStoreServiceClients(logger=mock_logger)
    
    # Get market data client
    market_data_client = service_clients.get_market_data_client()
    
    # Call the market data service
    response = await market_data_client.get(
        "/api/v1/market-data/ohlcv",
        params={
            "symbol": "EURUSD",
            "timeframe": "1h",
            "start": "2023-01-01T00:00:00Z",
            "end": "2023-01-01T01:00:00Z"
        }
    )
    
    # Verify the response
    assert response["symbol"] == "EURUSD"
    assert response["timeframe"] == "1h"
    assert len(response["data"]) == 2
    
    # Verify the mock was called
    mock_client.get.assert_called_once_with(
        "/api/v1/market-data/ohlcv",
        params={
            "symbol": "EURUSD",
            "timeframe": "1h",
            "start": "2023-01-01T00:00:00Z",
            "end": "2023-01-01T01:00:00Z"
        }
    )


@pytest.mark.asyncio
@patch("market_data_service.service_clients.get_service_client_config")
@patch("common_lib.service_client.ResilientServiceClient")
async def test_market_data_calls_feature_store(
    mock_resilient_service_client,
    mock_get_service_client_config,
    mock_logger
):
    """Test Market Data Service calling Feature Store Service."""
    # Mock the service client config
    mock_service_client_config = MagicMock()
    mock_service_client_config.base_url = "http://localhost:8003"
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
        "indicator": "sma",
        "parameters": {"period": 14},
        "data": [
            {"timestamp": "2023-01-01T00:00:00Z", "value": 1.15},
            {"timestamp": "2023-01-01T01:00:00Z", "value": 1.16}
        ]
    }
    mock_resilient_service_client.return_value = mock_client
    
    # Create service clients
    service_clients = MarketDataServiceClients(logger=mock_logger)
    
    # Get feature store client
    feature_store_client = service_clients.get_feature_store_client()
    
    # Call the feature store service
    response = await feature_store_client.get(
        "/api/v1/indicators/technical",
        params={
            "symbol": "EURUSD",
            "timeframe": "1h",
            "indicator": "sma",
            "parameters": {"period": 14},
            "start": "2023-01-01T00:00:00Z",
            "end": "2023-01-01T01:00:00Z"
        }
    )
    
    # Verify the response
    assert response["symbol"] == "EURUSD"
    assert response["timeframe"] == "1h"
    assert response["indicator"] == "sma"
    assert response["parameters"] == {"period": 14}
    assert len(response["data"]) == 2
    
    # Verify the mock was called
    mock_client.get.assert_called_once_with(
        "/api/v1/indicators/technical",
        params={
            "symbol": "EURUSD",
            "timeframe": "1h",
            "indicator": "sma",
            "parameters": {"period": 14},
            "start": "2023-01-01T00:00:00Z",
            "end": "2023-01-01T01:00:00Z"
        }
    )
