"""
Test feedback router module.

This module provides functionality for...
"""

\
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock # Added MagicMock
from datetime import datetime, timezone # Added timezone
from typing import Dict, Any

from core.feedback_router import FeedbackRouter, FeedbackTarget
from core.feedback_collector import FeedbackCollector
import httpx # Added httpx import

# Mock services needed by FeedbackRouter if they are used in tested methods
@pytest.fixture
def mock_feedback_collector():
    """
    Mock feedback collector.
    
    """

    return AsyncMock()

@pytest.fixture
def mock_timeframe_feedback_service():
    """
    Mock timeframe feedback service.
    
    """

    return AsyncMock()

@pytest.fixture
def mock_statistical_validator():
    return AsyncMock()

@pytest.fixture
def mock_strategy_mutator():
    return AsyncMock()

@pytest.fixture
def feedback_router(feedback_collector, strategy_mutator):
    """Create a FeedbackRouter instance with mock dependencies and mock httpx client."""
    # Patch the logger to avoid actual logging during tests
    with patch('strategy_execution_engine.trading.feedback_router.get_logger') as mock_get_logger, \
         patch('httpx.AsyncClient') as MockAsyncClient: # Patch httpx.AsyncClient
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Configure the mock AsyncClient instance that will be created in __init__
        mock_http_client_instance = AsyncMock(spec=httpx.AsyncClient)
        MockAsyncClient.return_value = mock_http_client_instance

        router = FeedbackRouter(
            feedback_collector=feedback_collector,
            timeframe_feedback_service=None, # Keep mocks/None as needed for other tests
            statistical_validator=None,
            strategy_mutator=strategy_mutator,
            ml_service_url="http://mock-ml-service/retrain", # Use mock URLs
            risk_service_url="http://mock-risk-service/feedback",
            market_regime_service_url="http://mock-market-regime-service/feedback"
        )
        # Assign mocks for assertion
        router.logger = mock_logger 
        router.http_client = mock_http_client_instance # Ensure the instance uses the mock client

        yield router # Use yield for setup/teardown if needed

# === Test for _route_model_retraining_feedback ===

@pytest.mark.asyncio
async def test_route_model_retraining_feedback_success(feedback_router):
    """Test successful routing of model retraining feedback."""
    feedback_data = {
        "routing_id": "route-123",
        "model_id": "model-abc",
        "performance": 0.95,
        "timestamp": datetime.now(timezone.utc).isoformat() # Use timezone-aware time
    }
    
    # Mock the response from the ML service
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "retraining_scheduled", "details": "ok"}
    mock_response.raise_for_status = MagicMock() # Mock this method to do nothing on 200 OK
    
    # Configure the mock client's post method
    feedback_router.http_client.post = AsyncMock(return_value=mock_response)

    result = await feedback_router._route_model_retraining_feedback(feedback_data)

    # Assertions
    feedback_router.http_client.post.assert_awaited_once_with(
        "http://mock-ml-service/retrain", 
        json=feedback_data, 
        timeout=10.0
    )
    mock_response.raise_for_status.assert_called_once()
    assert result["status"] == "routed_successfully"
    assert result["target_service"] == "ml_integration"
    assert result["service_response"] == {"status": "retraining_scheduled", "details": "ok"}
    feedback_router.logger.info.assert_any_call(f"Routing model retraining feedback: {feedback_data.get('routing_id')}")
    feedback_router.logger.info.assert_any_call(f"Successfully routed feedback {feedback_data.get('routing_id')} to ML service. Response: {result['service_response']}")


@pytest.mark.asyncio
async def test_route_model_retraining_feedback_http_request_error(feedback_router):
    """Test HTTP request error during routing of model retraining feedback."""
    feedback_data = {"routing_id": "route-456", "model_id": "model-xyz"}
    
    # Configure the mock client's post method to raise an HTTP request error
    feedback_router.http_client.post = AsyncMock(side_effect=httpx.RequestError("Connection failed", request=MagicMock()))

    result = await feedback_router._route_model_retraining_feedback(feedback_data)

    # Assertions
    feedback_router.http_client.post.assert_awaited_once_with(
        "http://mock-ml-service/retrain", 
        json=feedback_data, 
        timeout=10.0
    )
    assert result["status"] == "error"
    assert "HTTP request failed: Connection failed" in result["reason"]
    feedback_router.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_route_model_retraining_feedback_http_status_error(feedback_router):
    """Test handling of non-2xx HTTP response (HTTPStatusError) from ML service."""
    feedback_data = {"routing_id": "route-789", "model_id": "model-123"}

    # Mock a non-2xx response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.request = MagicMock() # Request object is needed for raise_for_status
    # Configure raise_for_status to raise the specific error
    mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        "Server Error", request=mock_response.request, response=mock_response
    ))
    
    feedback_router.http_client.post = AsyncMock(return_value=mock_response)

    result = await feedback_router._route_model_retraining_feedback(feedback_data)

    # Assertions
    feedback_router.http_client.post.assert_awaited_once_with(
        "http://mock-ml-service/retrain", 
        json=feedback_data, 
        timeout=10.0
    )
    mock_response.raise_for_status.assert_called_once() # raise_for_status should be called
    assert result["status"] == "error"
    assert result["reason"] == "HTTP status error: 500"
    assert result["details"] == "Internal Server Error"
    feedback_router.logger.error.assert_called_once()

@pytest.mark.asyncio
async def test_route_model_retraining_feedback_unexpected_error(feedback_router):
    """Test handling of unexpected errors during routing."""
    feedback_data = {"routing_id": "route-unexpected", "model_id": "model-err"}

    # Configure post to raise a generic exception
    feedback_router.http_client.post = AsyncMock(side_effect=Exception("Something broke"))

    result = await feedback_router._route_model_retraining_feedback(feedback_data)

    # Assertions
    assert result["status"] == "error"
    assert "Unexpected error: Something broke" in result["reason"]
    feedback_router.logger.error.assert_called_once()


# === Tests for market regime feedback routing ===

@pytest.mark.asyncio
async def test_route_market_regime_feedback_success(feedback_router):
    """Test successful routing of market regime feedback."""
    feedback_data = {
        "routing_id": "mr-123",
        "instrument": "EUR/USD",
        "market_regime": {
            "type": "trending",
            "confidence": 0.85
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Mock the response from the market regime service
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "received", "id": "mr-analysis-123"}
    mock_response.raise_for_status = MagicMock()
    
    # Configure the mock client's post method
    feedback_router.http_client.post = AsyncMock(return_value=mock_response)
    
    # Call the method
    result = await feedback_router._route_market_regime_feedback(feedback_data)
    
    # Assertions
    feedback_router.http_client.post.assert_awaited_once()
    assert result["status"] == "routed_successfully"
    assert result["target_service"] == "market_regime"


@pytest.mark.asyncio
async def test_route_market_regime_feedback_error(feedback_router):
    """Test error handling in market regime feedback routing."""
    feedback_data = {
        "routing_id": "mr-456",
        "instrument": "EUR/USD"
    }
    
    # Configure mock to raise an error
    feedback_router.http_client.post = AsyncMock(
        side_effect=httpx.RequestError("Connection failed", request=MagicMock())
    )
    
    # Call the method
    result = await feedback_router._route_market_regime_feedback(feedback_data)
    
    # Assertions
    assert result["status"] == "error"
    assert "HTTP request failed" in result["reason"]


# === Tests for risk management feedback routing ===

@pytest.mark.asyncio
async def test_route_risk_management_feedback_success(feedback_router):
    """Test successful routing of risk management feedback."""
    feedback_data = {
        "routing_id": "risk-123",
        "strategy_id": "strategy-abc",
        "risk_metrics": {
            "var": 0.05,
            "max_drawdown": 0.15
        },
        "drawdown": 0.08,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Mock the response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "processed", "recommendation": "no_change"}
    mock_response.raise_for_status = MagicMock()
    
    # Configure the mock client
    feedback_router.http_client.post = AsyncMock(return_value=mock_response)
    
    # Call the method
    result = await feedback_router._route_risk_management_feedback(feedback_data)
    
    # Assertions
    feedback_router.http_client.post.assert_awaited_once()
    assert result["status"] == "routed_successfully"
    assert result["target_service"] == "risk_management"


@pytest.mark.asyncio
async def test_route_risk_management_feedback_missing_data(feedback_router):
    """Test handling of missing data in risk management feedback."""
    feedback_data = {
        "routing_id": "risk-456",
        "strategy_id": "strategy-xyz"
        # Missing risk metrics and drawdown
    }
    
    # Call the method
    result = await feedback_router._route_risk_management_feedback(feedback_data)
    
    # Assertions
    assert result["status"] == "error"
    assert "Missing risk metrics" in result["reason"]
    # HTTP client should not be called
    feedback_router.http_client.post.assert_not_awaited()


if __name__ == "__main__":
    pytest.main()
