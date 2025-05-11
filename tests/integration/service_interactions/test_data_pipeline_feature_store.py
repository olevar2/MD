"""
Integration tests for Data Pipeline Service and Feature Store Service interactions.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from data_pipeline_service.service_clients import ServiceClients as DataPipelineServiceClients
from feature_store_service.service_clients import ServiceClients as FeatureStoreServiceClients


@pytest.mark.asyncio
@patch("data_pipeline_service.service_clients.get_service_client_config")
@patch("common_lib.service_client.ResilientServiceClient")
async def test_data_pipeline_calls_feature_store(
    mock_resilient_service_client,
    mock_get_service_client_config,
    mock_logger
):
    """Test Data Pipeline Service calling Feature Store Service."""
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
    mock_client.post.return_value = {
        "batch_id": "123456",
        "status": "success",
        "processed_count": 100,
        "timestamp": "2023-01-01T00:00:00Z"
    }
    mock_resilient_service_client.return_value = mock_client
    
    # Create service clients
    service_clients = DataPipelineServiceClients(logger=mock_logger)
    
    # Get feature store client
    feature_store_client = service_clients.get_feature_store_client()
    
    # Call the feature store service
    response = await feature_store_client.post(
        "/api/v1/features/batch",
        json={
            "symbol": "EURUSD",
            "timeframe": "1h",
            "features": [
                {"name": "sma_14", "values": [1.1, 1.2, 1.3]},
                {"name": "rsi_14", "values": [60, 65, 70]}
            ],
            "timestamps": [
                "2023-01-01T00:00:00Z",
                "2023-01-01T01:00:00Z",
                "2023-01-01T02:00:00Z"
            ]
        }
    )
    
    # Verify the response
    assert response["batch_id"] == "123456"
    assert response["status"] == "success"
    assert response["processed_count"] == 100
    
    # Verify the mock was called
    mock_client.post.assert_called_once_with(
        "/api/v1/features/batch",
        json={
            "symbol": "EURUSD",
            "timeframe": "1h",
            "features": [
                {"name": "sma_14", "values": [1.1, 1.2, 1.3]},
                {"name": "rsi_14", "values": [60, 65, 70]}
            ],
            "timestamps": [
                "2023-01-01T00:00:00Z",
                "2023-01-01T01:00:00Z",
                "2023-01-01T02:00:00Z"
            ]
        }
    )


@pytest.mark.asyncio
@patch("feature_store_service.service_clients.get_service_client_config")
@patch("common_lib.service_client.ResilientServiceClient")
async def test_feature_store_calls_data_pipeline(
    mock_resilient_service_client,
    mock_get_service_client_config,
    mock_logger
):
    """Test Feature Store Service calling Data Pipeline Service."""
    # Mock the service client config
    mock_service_client_config = MagicMock()
    mock_service_client_config.base_url = "http://localhost:8002"
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
        "pipeline_id": "123456",
        "status": "running",
        "progress": 0.75,
        "start_time": "2023-01-01T00:00:00Z",
        "estimated_completion_time": "2023-01-01T01:00:00Z"
    }
    mock_resilient_service_client.return_value = mock_client
    
    # Create service clients
    service_clients = FeatureStoreServiceClients(logger=mock_logger)
    
    # Get data pipeline client
    data_pipeline_client = service_clients.get_data_pipeline_client()
    
    # Call the data pipeline service
    response = await data_pipeline_client.get(
        "/api/v1/pipelines/status",
        params={"pipeline_id": "123456"}
    )
    
    # Verify the response
    assert response["pipeline_id"] == "123456"
    assert response["status"] == "running"
    assert response["progress"] == 0.75
    
    # Verify the mock was called
    mock_client.get.assert_called_once_with(
        "/api/v1/pipelines/status",
        params={"pipeline_id": "123456"}
    )
