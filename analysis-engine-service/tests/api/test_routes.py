"""
Tests for the API routes module.

This module contains tests for the API routes and their functionality.
"""

import pytest
from unittest.mock import MagicMock, patch
from analysis_engine.core.errors import AnalysisEngineError, ValidationError
# Fixtures like 'client' and 'mock_analysis_service' are now imported from conftest.py

@pytest.fixture
def mock_market_data():
    """Create mock market data."""
    return {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "data": [
            {"timestamp": "2024-01-01T00:00:00", "open": 1.1000, "high": 1.1100, "low": 1.0900, "close": 1.1050, "volume": 1000},
            {"timestamp": "2024-01-01T01:00:00", "open": 1.1050, "high": 1.1150, "low": 1.1000, "close": 1.1100, "volume": 1200}
        ]
    }

def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()
    assert "status" in response.json()

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_version_endpoint(client):
    """Test the version endpoint."""
    response = client.get("/api/version")
    assert response.status_code == 200
    assert "version" in response.json()
    assert "timestamp" in response.json()

def test_metrics_endpoint(client):
    """Test the metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "python_gc_collections_total" in response.text

def test_confluence_analysis(client, mock_analysis_service: MagicMock, mock_market_data): # Added type hint for clarity
    """Test the confluence analysis endpoint."""
    with patch("analysis_engine.api.v1.analysis_results_api.get_analysis_service", return_value=mock_analysis_service):
        response = client.post(
            "/api/v1/analysis/confluence",
            json=mock_market_data
        )
        assert response.status_code == 200
        assert "result" in response.json()
        assert "confidence" in response.json()

def test_multi_timeframe_analysis(client, mock_analysis_service: MagicMock, mock_market_data): # Added type hint
    """Test the multi-timeframe analysis endpoint."""
    with patch("analysis_engine.api.v1.analysis_results_api.get_analysis_service", return_value=mock_analysis_service):
        response = client.post(
            "/api/v1/analysis/multi-timeframe",
            json=mock_market_data
        )
        assert response.status_code == 200
        assert "result" in response.json()
        assert "confidence" in response.json()

def test_analysis_error_handling(client, mock_analysis_service: MagicMock, mock_market_data): # Added type hint
    """Test error handling in analysis endpoints."""
    mock_analysis_service.analyze.side_effect = AnalysisEngineError("Test error")
    
    with patch("analysis_engine.api.v1.analysis_results_api.get_analysis_service", return_value=mock_analysis_service):
        response = client.post(
            "/api/v1/analysis/confluence",
            json=mock_market_data
        )
        assert response.status_code == 500
        assert "error" in response.json()

def test_validation_error_handling(client, mock_market_data):
    """Test validation error handling."""
    invalid_data = mock_market_data.copy()
    invalid_data["symbol"] = ""  # Invalid symbol
    
    response = client.post(
        "/api/v1/analysis/confluence",
        json=invalid_data
    )
    assert response.status_code == 422
    assert "error" in response.json()

def test_cors_headers(client):
    """Test CORS headers."""
    response = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:3000"}
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers
    assert "access-control-allow-headers" in response.headers

def test_rate_limiting(client, mock_analysis_service: MagicMock, mock_market_data): # Added type hint
    """Test rate limiting."""
    with patch("analysis_engine.api.v1.analysis_results_api.get_analysis_service", return_value=mock_analysis_service):
        # Make multiple requests in quick succession
        for _ in range(11):  # Assuming rate limit is 10 requests per minute
            response = client.post(
                "/api/v1/analysis/confluence",
                json=mock_market_data
            )
        
        # The last request should be rate limited
        assert response.status_code == 429
        assert "error" in response.json()

def test_websocket_connection(client):
    """Test WebSocket connection."""
    with client.websocket_connect("/ws/analysis") as websocket:
        data = websocket.receive_json()
        assert "type" in data
        assert "message" in data

def test_websocket_analysis_updates(client, mock_analysis_service: MagicMock): # Added type hint
    """Test WebSocket analysis updates."""
    with patch("analysis_engine.api.v1.analysis_results_api.get_analysis_service", return_value=mock_analysis_service):
        with client.websocket_connect("/ws/analysis") as websocket:
            # Send analysis request
            websocket.send_json({
                "type": "analysis_request",
                "data": {
                    "symbol": "EURUSD",
                    "timeframe": "H1"
                }
            })
            
            # Receive analysis update
            data = websocket.receive_json()
            assert data["type"] == "analysis_update"
            assert "result" in data

def test_authentication_required(client, mock_market_data):
    """Test authentication requirement."""
    response = client.post(
        "/api/v1/analysis/confluence",
        json=mock_market_data,
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401
    assert "error" in response.json()

def test_authorization_required(client, mock_market_data):
    """Test authorization requirement."""
    with patch("analysis_engine.api.v1.analysis_results_api.verify_token", return_value={"sub": "test_user", "role": "user"}):
        response = client.post(
            "/api/v1/analysis/confluence",
            json=mock_market_data,
            headers={"Authorization": "Bearer valid_token"}
        )
        assert response.status_code == 403
        assert "error" in response.json()