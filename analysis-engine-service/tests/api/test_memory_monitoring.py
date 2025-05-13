"""
Tests for FastAPI application memory monitoring integration.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from analysis_engine.core.memory_monitor import MemoryMonitor
from analysis_engine.core.config import Settings
from unittest.mock import patch


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

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

    @app.on_event('startup')
    async def startup_event():
    """
    Startup event.
    
    """

        await memory_monitor.start()

    @app.on_event('shutdown')
    async def shutdown_event():
    """
    Shutdown event.
    
    """

        await memory_monitor.stop()
    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == 404
        assert memory_monitor.is_monitoring


@pytest.mark.asyncio
async def test_app_memory_monitoring_shutdown(app, memory_monitor):
    """Test memory monitoring shutdown in FastAPI application."""

    @app.on_event('startup')
    async def startup_event():
    """
    Startup event.
    
    """

        await memory_monitor.start()

    @app.on_event('shutdown')
    async def shutdown_event():
    """
    Shutdown event.
    
    """

        await memory_monitor.stop()
    with TestClient(app) as client:
        await memory_monitor.start()
        assert memory_monitor.is_monitoring
        await memory_monitor.stop()
        assert not memory_monitor.is_monitoring


@pytest.mark.asyncio
async def test_app_memory_monitoring_endpoint(app, memory_monitor):
    """Test memory monitoring endpoint in FastAPI application."""

    @app.get('/api/v1/monitoring/memory')
    async def get_memory_stats():
    """
    Get memory stats.
    
    """

        return memory_monitor.get_memory_stats()
    with TestClient(app) as client:
        response = client.get('/api/v1/monitoring/memory')
        assert response.status_code == 200
        data = response.json()
        assert 'process_memory_percent' in data
        assert 'system_memory_percent' in data


@pytest.mark.asyncio
@async_with_exception_handling
async def test_app_memory_monitoring_error_handling(app, memory_monitor):
    """Test memory monitoring error handling in FastAPI application."""

    @app.get('/api/v1/monitoring/memory')
    @async_with_exception_handling
    async def get_memory_stats():
    """
    Get memory stats.
    
    """

        try:
            return memory_monitor.get_memory_stats()
        except Exception as e:
            return {'error': str(e)}
    with TestClient(app) as client:
        with patch.object(memory_monitor, 'get_memory_stats') as mock_stats:
            mock_stats.side_effect = Exception('Test error')
            response = client.get('/api/v1/monitoring/memory')
            assert response.status_code == 200
            data = response.json()
            assert 'error' in data
            assert data['error'] == 'Test error'
