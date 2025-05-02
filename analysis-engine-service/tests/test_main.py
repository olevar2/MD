"""
Tests for the main application module.

This module contains tests for the FastAPI application initialization,
lifespan management, and error handling.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import asyncio
from unittest.mock import MagicMock, patch

from analysis_engine.main import create_app, lifespan, handle_shutdown, setup_signal_handlers
from analysis_engine.core.config import get_settings
from analysis_engine.core.errors import AnalysisEngineError, ValidationError

# Fixtures 'app' and 'client' are now imported from conftest.py

@pytest.fixture
def mock_service_container():
    """Create a mock service container."""
    container = MagicMock()
    container.initialize = MagicMock(return_value=None)
    container.cleanup = MagicMock(return_value=None)
    return container

@pytest.fixture
def mock_memory_monitor():
    """Create a mock memory monitor."""
    monitor = MagicMock()
    monitor.start_monitoring = MagicMock(return_value=None)
    monitor.stop_monitoring = MagicMock(return_value=None)
    return monitor

def test_app_creation():
    """Test FastAPI application creation."""
    app = create_app()
    assert isinstance(app, FastAPI)
    assert app.title == "Analysis Engine Service"
    assert app.version == "1.0.0"

def test_app_routes(client):
    """Test that all routes are properly registered."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    response = client.get("/api/version")
    assert response.status_code == 200
    assert "version" in response.json()

def test_cors_middleware(client):
    """Test CORS middleware configuration."""
    response = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:3000"}
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers

@pytest.mark.asyncio
async def test_lifespan_startup(app, mock_service_container, mock_memory_monitor):
    """Test application startup in lifespan context."""
    app.state.service_container = mock_service_container
    
    with patch("analysis_engine.main.get_memory_monitor", return_value=mock_memory_monitor):
        async with lifespan(app):
            mock_service_container.initialize.assert_called_once()
            mock_memory_monitor.start_monitoring.assert_called_once()

@pytest.mark.asyncio
async def test_lifespan_shutdown(app, mock_service_container, mock_memory_monitor):
    """Test application shutdown in lifespan context."""
    app.state.service_container = mock_service_container
    
    with patch("analysis_engine.main.get_memory_monitor", return_value=mock_memory_monitor):
        async with lifespan(app):
            pass  # Let the context manager handle shutdown
        
        mock_service_container.cleanup.assert_called_once()
        mock_memory_monitor.stop_monitoring.assert_called_once()

@pytest.mark.asyncio
async def test_lifespan_error_handling(app, mock_service_container):
    """Test error handling during lifespan."""
    mock_service_container.initialize.side_effect = Exception("Test error")
    
    with pytest.raises(Exception) as exc_info:
        async with lifespan(app):
            pass
    
    assert str(exc_info.value) == "Test error"

def test_error_handlers(client):
    """Test error handlers for different types of errors."""
    # Test AnalysisEngineError
    with patch("analysis_engine.main.AnalysisEngineError") as mock_error:
        mock_error.side_effect = AnalysisEngineError("Test error")
        response = client.get("/api/health")
        assert response.status_code == 500
        assert "error" in response.json()

    # Test ValidationError
    with patch("analysis_engine.main.ValidationError") as mock_error:
        mock_error.side_effect = ValidationError("Invalid input")
        response = client.get("/api/health")
        assert response.status_code == 400
        assert "error" in response.json()

@pytest.mark.asyncio
async def test_handle_shutdown():
    """Test shutdown signal handling."""
    shutdown_event = asyncio.Event()
    await handle_shutdown("SIGTERM")
    assert shutdown_event.is_set()

def test_setup_signal_handlers():
    """Test signal handler setup."""
    loop = asyncio.get_event_loop()
    with patch.object(loop, "add_signal_handler") as mock_handler:
        setup_signal_handlers()
        assert mock_handler.call_count == 2  # SIGINT and SIGTERM

def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "python_gc_collections_total" in response.text

def test_optional_services_initialization(app, mock_service_container):
    """Test initialization of optional services."""
    app.state.service_container = mock_service_container
    
    # Test with CAUSAL_INFERENCE_AVAILABLE = True
    with patch("analysis_engine.main.CAUSAL_INFERENCE_AVAILABLE", True), \
         patch("analysis_engine.main.CausalInferenceService") as mock_causal:
        async with lifespan(app):
            mock_causal.assert_called_once()
    
    # Test with MULTI_TIMEFRAME_AVAILABLE = True
    with patch("analysis_engine.main.MULTI_TIMEFRAME_AVAILABLE", True), \
         patch("analysis_engine.main.MultiTimeframeAnalyzer") as mock_mtf:
        async with lifespan(app):
            mock_mtf.assert_called_once()