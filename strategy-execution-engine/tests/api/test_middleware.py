"""
Tests for middleware in the Strategy Execution Engine.
"""

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import json
import time

from strategy_execution_engine.api.middleware import (
    RequestLoggingMiddleware,
    MetricsMiddleware,
    setup_middleware
)


@pytest.fixture
def app():
    """Create a FastAPI app for testing."""
    return FastAPI()


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


def test_request_logging_middleware(app, client):
    """Test RequestLoggingMiddleware."""
    # Add middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Add test endpoint
    @app.get("/test")
    def test_endpoint():
        return {"message": "Test"}
    
    # Make request with logging middleware
    with patch("strategy_execution_engine.api.middleware.logger") as mock_logger:
        response = client.get("/test", headers={"X-Request-ID": "test-id"})
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == {"message": "Test"}
        assert response.headers["X-Request-ID"] == "test-id"
        
        # Verify logging
        mock_logger.info.assert_any_call(
            "Request started: GET /test (ID: test-id, Client: testclient)"
        )
        mock_logger.info.assert_any_call(
            "Request completed: GET /test (ID: test-id, Status: 200, Time: 0.0000s)"
        )


def test_request_logging_middleware_auto_id(app, client):
    """Test RequestLoggingMiddleware with auto-generated request ID."""
    # Add middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Add test endpoint
    @app.get("/test")
    def test_endpoint():
        return {"message": "Test"}
    
    # Make request without request ID
    response = client.get("/test")
    
    # Verify response has request ID
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] != ""


def test_request_logging_middleware_error(app, client):
    """Test RequestLoggingMiddleware with error."""
    # Add middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Add test endpoint that raises an error
    @app.get("/error")
    def error_endpoint():
        raise ValueError("Test error")
    
    # Make request with logging middleware
    with patch("strategy_execution_engine.api.middleware.logger") as mock_logger:
        # Request should raise an error
        with pytest.raises(Exception):
            response = client.get("/error", headers={"X-Request-ID": "test-id"})
        
        # Verify logging
        mock_logger.info.assert_any_call(
            "Request started: GET /error (ID: test-id, Client: testclient)"
        )
        mock_logger.error.assert_called_once()
        # Check that the error log contains the expected information
        error_log = mock_logger.error.call_args[0][0]
        assert "Request failed: GET /error" in error_log
        assert "ID: test-id" in error_log
        assert "Error: Test error" in error_log


def test_metrics_middleware(app, client):
    """Test MetricsMiddleware."""
    # Add middleware
    app.add_middleware(MetricsMiddleware)
    
    # Add test endpoint
    @app.get("/test")
    def test_endpoint():
        return {"message": "Test"}
    
    # Make request with metrics middleware
    response = client.get("/test")
    
    # Verify response
    assert response.status_code == 200
    assert response.json() == {"message": "Test"}
    
    # Note: In a real test, we would verify that metrics were recorded
    # but that would require mocking the Prometheus client


def test_setup_middleware(app):
    """Test setup_middleware function."""
    # Setup middleware
    with patch("strategy_execution_engine.api.middleware.logger") as mock_logger:
        setup_middleware(app)
        
        # Verify middleware was added
        assert any(isinstance(m, RequestLoggingMiddleware) for m in app.user_middleware)
        assert any(isinstance(m, MetricsMiddleware) for m in app.user_middleware)
        
        # Verify logging
        mock_logger.info.assert_called_with("Middleware configured")
