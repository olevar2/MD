"""
Tests for FastAPI application memory monitoring integration.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from analysis_engine.core.memory_monitor import MemoryMonitor
from analysis_engine.core.config import Settings
from unittest.mock import patch

@pytest.fixture
def app():
    """Create test FastAPI application."""
    app = FastAPI()
    return app

@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)

@pytest.fixture
def memory_monitor():
    """Create memory monitor instance."""
    return MemoryMonitor()

@pytest.mark.asyncio
async def test_app_memory_monitoring_startup(app, memory_monitor):
    """Test memory monitoring startup in FastAPI application."""
    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        await memory_monitor.start()
    
    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        await memory_monitor.stop()
    
    # Test startup
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 404  # Default 404 for root
        assert memory_monitor.is_monitoring

@pytest.mark.asyncio
async def test_app_memory_monitoring_shutdown(app, memory_monitor):
    """Test memory monitoring shutdown in FastAPI application."""
    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        await memory_monitor.start()
    
    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        await memory_monitor.stop()
    
    # Test shutdown
    with TestClient(app) as client:
        # Start monitoring
        await memory_monitor.start()
        assert memory_monitor.is_monitoring
        
        # Simulate shutdown
        await memory_monitor.stop()
        assert not memory_monitor.is_monitoring

@pytest.mark.asyncio
async def test_app_memory_monitoring_endpoint(app, memory_monitor):
    """Test memory monitoring endpoint in FastAPI application."""
    # Add memory monitoring endpoint
    @app.get("/api/v1/monitoring/memory")
    async def get_memory_stats():
        return memory_monitor.get_memory_stats()
    
    # Test endpoint
    with TestClient(app) as client:
        response = client.get("/api/v1/monitoring/memory")
        assert response.status_code == 200
        data = response.json()
        assert "process_memory_percent" in data
        assert "system_memory_percent" in data

@pytest.mark.asyncio
async def test_app_memory_monitoring_error_handling(app, memory_monitor):
    """Test memory monitoring error handling in FastAPI application."""
    # Add memory monitoring endpoint with error
    @app.get("/api/v1/monitoring/memory")
    async def get_memory_stats():
        try:
            return memory_monitor.get_memory_stats()
        except Exception as e:
            return {"error": str(e)}
    
    # Test error handling
    with TestClient(app) as client:
        # Mock memory monitor to raise error
        with patch.object(memory_monitor, 'get_memory_stats') as mock_stats:
            mock_stats.side_effect = Exception("Test error")
            
            response = client.get("/api/v1/monitoring/memory")
            assert response.status_code == 200
            data = response.json()
            assert "error" in data
            assert data["error"] == "Test error" 