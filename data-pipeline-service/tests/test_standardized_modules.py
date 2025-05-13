"""
Tests for standardized modules in the Data Pipeline Service.
"""
import os
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from data_pipeline_service.config.standardized_config import settings, get_service_settings
from data_pipeline_service.logging_setup_standardized import get_service_logger
from data_pipeline_service.database_standardized import database
from data_pipeline_service.service_clients_standardized import service_clients
from data_pipeline_service.error_handling_standardized import (
    handle_error,
    handle_exception,
    handle_async_exception,
    get_status_code
)
from data_pipeline_service.main import app


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def mock_db_session():
    """Create a mock database session."""
    mock_session = AsyncMock(spec=AsyncSession)
    return mock_session


def test_settings():
    """Test that settings are loaded correctly."""
    # Test that settings are loaded
    assert settings is not None
    assert settings.SERVICE_NAME == "data-pipeline-service"
    
    # Test that settings can be accessed
    assert settings.API_HOST == "0.0.0.0"
    assert isinstance(settings.API_PORT, int)
    
    # Test that settings can be reloaded
    service_settings = get_service_settings()
    assert service_settings is settings


def test_logger():
    """Test that logger is created correctly."""
    # Test that logger can be created
    logger = get_service_logger("test")
    assert logger is not None
    
    # Test that logger has the correct name
    assert logger.name == "test"


@pytest.mark.asyncio
async def test_database_mock():
    """Test database connectivity with mocks."""
    # Mock the database connection
    with patch.object(database, 'connect', new_callable=AsyncMock) as mock_connect:
        with patch.object(database, 'close', new_callable=AsyncMock) as mock_close:
            with patch.object(database, 'get_session', new_callable=AsyncMock) as mock_get_session:
                # Test connect
                await database.connect()
                mock_connect.assert_called_once()
                
                # Test get_session
                await database.get_session()
                mock_get_session.assert_called_once()
                
                # Test close
                await database.close()
                mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_service_clients_mock():
    """Test service clients with mocks."""
    # Mock the service clients
    with patch.object(service_clients, 'get_client', new_callable=MagicMock) as mock_get_client:
        with patch.object(service_clients, 'connect_all', new_callable=AsyncMock) as mock_connect_all:
            with patch.object(service_clients, 'close_all', new_callable=AsyncMock) as mock_close_all:
                # Test get_client
                service_clients.get_client("market_data_service")
                mock_get_client.assert_called_once_with("market_data_service")
                
                # Test connect_all
                await service_clients.connect_all()
                mock_connect_all.assert_called_once()
                
                # Test close_all
                await service_clients.close_all()
                mock_close_all.assert_called_once()


def test_error_handling():
    """Test error handling."""
    # Test handle_error
    error = ValueError("Test error")
    error_response = handle_error(error, operation="test_operation")
    assert error_response["error"] == "ValueError"
    assert error_response["message"] == "Test error"
    assert error_response["operation"] == "test_operation"
    
    # Test get_status_code
    status_code = get_status_code(error)
    assert status_code == 500


@pytest.mark.asyncio
async def test_handle_async_exception():
    """Test async exception handling decorator."""
    # Define a test function with the decorator
    @handle_async_exception(operation="test_operation")
    async def test_function(should_raise=False):
        if should_raise:
            raise ValueError("Test error")
        return "Success"
    
    # Test successful execution
    result = await test_function()
    assert result == "Success"
    
    # Test exception handling
    with pytest.raises(ValueError):
        await test_function(should_raise=True)


@pytest.mark.asyncio
async def test_api_endpoints_mock(test_client, mock_db_session):
    """Test API endpoints with mocks."""
    # Mock the get_db_session dependency
    app.dependency_overrides = {}
    
    # Mock the database session
    with patch('data_pipeline_service.api.v1.ohlcv.get_db_session', return_value=mock_db_session):
        # Mock the OHLCVRepository and OHLCVService
        with patch('data_pipeline_service.api.v1.ohlcv.OHLCVRepository') as mock_repo:
            with patch('data_pipeline_service.api.v1.ohlcv.OHLCVService') as mock_service:
                # Configure mocks
                mock_repo_instance = mock_repo.return_value
                mock_service_instance = mock_service.return_value
                mock_service_instance.get_ohlcv_data = AsyncMock(return_value={
                    "data": [],
                    "page": 1,
                    "page_size": 100,
                    "total": 0
                })
                mock_service_instance.get_timeframe_delta = MagicMock(return_value=timedelta(days=1))
                
                # Test get_ohlcv_data endpoint
                response = test_client.get("/v1/ohlcv?symbol=EUR/USD&timeframe=M1")
                assert response.status_code == 200
                assert response.json() == {
                    "data": [],
                    "page": 1,
                    "page_size": 100,
                    "total": 0
                }
                
                # Verify mocks were called
                mock_repo.assert_called_once_with(mock_db_session)
                mock_service.assert_called_once_with(mock_repo_instance)
                mock_service_instance.get_ohlcv_data.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
