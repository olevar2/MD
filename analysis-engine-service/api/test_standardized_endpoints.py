"""
Tests for standardized API endpoints.

This module contains tests for the standardized API endpoints in the Analysis Engine Service.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Add the mocks directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../mocks')))

# Mock common_lib
sys.modules['common_lib'] = __import__('common_lib')
sys.modules['common_lib.config'] = __import__('common_lib').config

from analysis_engine.main import app
from analysis_engine.api.v1.standardized.health import setup_health_routes
from analysis_engine.api.v1.standardized.adaptive_layer import setup_adaptive_layer_routes
from analysis_engine.api.v1.standardized.market_regime import setup_market_regime_routes
from analysis_engine.core.container import ServiceContainer

# Create a test client
client = TestClient(app)

# Mock service container
@pytest.fixture
def mock_service_container():
    """Create a mock service container."""
    container = MagicMock(spec=ServiceContainer)

    # Mock health check
    container.health_check = AsyncMock()
    container.health_check.check_health.return_value = {
        "status": "healthy",
        "services": {
            "database": {"status": "healthy"},
            "redis": {"status": "healthy"},
            "market_data_service": {"status": "healthy"}
        }
    }

    # Mock adaptive layer services
    container.parameter_adjustment_service = AsyncMock()
    container.parameter_adjustment_service.get_parameter_history.return_value = {
        "history": [
            {
                "timestamp": "2023-01-01T00:00:00Z",
                "parameters": {"param1": 1.0, "param2": 2.0},
                "reason": "Market regime change"
            }
        ]
    }

    container.feedback_loop = AsyncMock()
    container.feedback_loop.get_adaptation_insights.return_value = {
        "insights": [
            {
                "timestamp": "2023-01-01T00:00:00Z",
                "insight": "Parameter adjustment improved performance by 10%"
            }
        ]
    }

    container.feedback_loop.get_performance_by_regime.return_value = {
        "performance": {
            "trending": {"win_rate": 65.0, "profit_factor": 1.8},
            "ranging": {"win_rate": 55.0, "profit_factor": 1.2}
        }
    }

    # Mock market regime services
    container.market_regime_service = AsyncMock()
    container.market_regime_service.get_regime_history.return_value = {
        "history": [
            {
                "timestamp": "2023-01-01T00:00:00Z",
                "regime": "trending",
                "confidence": 0.85
            }
        ]
    }

    container.market_regime_service.generate_performance_report.return_value = {
        "overall_performance": {
            "win_rate": 60.0,
            "profit_factor": 1.5
        },
        "regime_performance": {
            "trending": {"win_rate": 65.0, "profit_factor": 1.8},
            "ranging": {"win_rate": 55.0, "profit_factor": 1.2}
        }
    }

    return container

# Health check endpoint tests
def test_health_check(mock_service_container):
    """Test the health check endpoint."""
    with patch('analysis_engine.api.v1.standardized.health.get_service_container', return_value=mock_service_container):
        response = client.get("/api/v1/analysis/health-checks")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_liveness_probe():
    """Test the liveness probe endpoint."""
    response = client.get("/api/v1/analysis/health-checks/liveness")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"

def test_readiness_probe(mock_service_container):
    """Test the readiness probe endpoint."""
    with patch('analysis_engine.api.v1.standardized.health.get_service_container', return_value=mock_service_container):
        response = client.get("/api/v1/analysis/health-checks/readiness")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

# Adaptive layer endpoint tests
def test_get_parameter_history(mock_service_container):
    """Test the get parameter history endpoint."""
    with patch('analysis_engine.api.v1.standardized.adaptive_layer.get_parameter_adjustment_service', return_value=mock_service_container.parameter_adjustment_service):
        response = client.get("/api/v1/analysis/adaptations/parameters/history/strategy-123/EURUSD/H1")
        assert response.status_code == 200
        assert "history" in response.json()
        assert len(response.json()["history"]) > 0

def test_get_adaptation_insights(mock_service_container):
    """Test the get adaptation insights endpoint."""
    with patch('analysis_engine.api.v1.standardized.adaptive_layer.get_feedback_loop', return_value=mock_service_container.feedback_loop):
        response = client.get("/api/v1/analysis/adaptations/feedback/insights/strategy-123")
        assert response.status_code == 200
        assert "insights" in response.json()
        assert len(response.json()["insights"]) > 0

def test_get_performance_by_regime(mock_service_container):
    """Test the get performance by regime endpoint."""
    with patch('analysis_engine.api.v1.standardized.adaptive_layer.get_feedback_loop', return_value=mock_service_container.feedback_loop):
        response = client.get("/api/v1/analysis/adaptations/feedback/performance/strategy-123")
        assert response.status_code == 200
        assert "performance" in response.json()
        assert "trending" in response.json()["performance"]
        assert "ranging" in response.json()["performance"]

# Market regime endpoint tests
def test_get_regime_history(mock_service_container):
    """Test the get regime history endpoint."""
    with patch('analysis_engine.api.v1.standardized.market_regime.get_market_regime_service', return_value=mock_service_container.market_regime_service):
        response = client.get("/api/v1/analysis/market-regimes/history?symbol=EURUSD&timeframe=H1")
        assert response.status_code == 200
        assert "history" in response.json()
        assert len(response.json()["history"]) > 0

def test_get_performance_report(mock_service_container):
    """Test the get performance report endpoint."""
    with patch('analysis_engine.api.v1.standardized.market_regime.get_market_regime_service', return_value=mock_service_container.market_regime_service):
        response = client.get("/api/v1/analysis/market-regimes/performance-report?instrument=EURUSD&timeframe=H1")
        assert response.status_code == 200
        assert "overall_performance" in response.json()
        assert "regime_performance" in response.json()
