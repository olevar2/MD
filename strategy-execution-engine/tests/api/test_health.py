"""
Tests for health check endpoints in the Strategy Execution Engine.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import json
from datetime import datetime

from strategy_execution_engine.main import create_app
from strategy_execution_engine.core.container import ServiceContainer


@pytest.fixture
def mock_service_container():
    """Create a mock service container."""
    container = MagicMock(spec=ServiceContainer)
    container.is_initialized = True
    return container


@pytest.fixture
def client(mock_service_container):
    """Create a test client with mocked dependencies."""
    app = create_app()
    
    # Override dependencies
    app.state.service_container = mock_service_container
    
    return TestClient(app)


def test_health_check(client):
    """Test the basic health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    # Verify response data
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_detailed_health_check(client, mock_service_container):
    """Test the detailed health check endpoint."""
    # Configure mock service container for health checks
    with patch("strategy_execution_engine.api.health.check_analysis_engine_connection", 
               new=AsyncMock(return_value={"status": "healthy", "message": "Connection successful"})), \
         patch("strategy_execution_engine.api.health.check_feature_store_connection", 
               new=AsyncMock(return_value={"status": "healthy", "message": "Connection successful"})), \
         patch("strategy_execution_engine.api.health.check_trading_gateway_connection", 
               new=AsyncMock(return_value={"status": "healthy", "message": "Connection successful"})), \
         patch("strategy_execution_engine.api.health.check_strategy_loader", 
               return_value={"status": "healthy", "message": "Strategy loader operational", "strategies_loaded": 5}), \
         patch("strategy_execution_engine.api.health.check_backtester", 
               return_value={"status": "healthy", "message": "Backtester operational"}):
        
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        
        # Verify response data
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "database" in data
        assert "services" in data
        
        # Verify services data
        services = data["services"]
        assert services["analysis_engine"]["status"] == "healthy"
        assert services["feature_store"]["status"] == "healthy"
        assert services["trading_gateway"]["status"] == "healthy"
        assert services["strategy_loader"]["status"] == "healthy"
        assert services["backtester"]["status"] == "healthy"


def test_detailed_health_check_degraded(client, mock_service_container):
    """Test the detailed health check endpoint with degraded service."""
    # Configure mock service container for health checks with one degraded service
    with patch("strategy_execution_engine.api.health.check_analysis_engine_connection", 
               new=AsyncMock(return_value={"status": "healthy", "message": "Connection successful"})), \
         patch("strategy_execution_engine.api.health.check_feature_store_connection", 
               new=AsyncMock(return_value={"status": "unhealthy", "message": "Connection failed"})), \
         patch("strategy_execution_engine.api.health.check_trading_gateway_connection", 
               new=AsyncMock(return_value={"status": "healthy", "message": "Connection successful"})), \
         patch("strategy_execution_engine.api.health.check_strategy_loader", 
               return_value={"status": "healthy", "message": "Strategy loader operational", "strategies_loaded": 5}), \
         patch("strategy_execution_engine.api.health.check_backtester", 
               return_value={"status": "healthy", "message": "Backtester operational"}):
        
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        
        # Verify response data shows degraded status
        assert data["status"] == "degraded"
        
        # Verify services data
        services = data["services"]
        assert services["analysis_engine"]["status"] == "healthy"
        assert services["feature_store"]["status"] == "unhealthy"
        assert services["trading_gateway"]["status"] == "healthy"


def test_detailed_health_check_error(client, mock_service_container):
    """Test the detailed health check endpoint with an error during check."""
    # Configure mock service container to raise an exception during health check
    with patch("strategy_execution_engine.api.health.check_analysis_engine_connection", 
               new=AsyncMock(side_effect=Exception("Test exception"))):
        
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        
        # Verify response data shows unhealthy status
        assert data["status"] == "unhealthy"
        assert "services" in data
        assert "error" in data["services"]
