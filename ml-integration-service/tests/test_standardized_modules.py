"""
Tests for standardized modules in the ML Integration Service.
"""
import os
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient
from pydantic import ValidationError

from ml_integration_service.config.standardized_config import settings, get_service_settings
from ml_integration_service.logging_setup_standardized import get_service_logger
from ml_integration_service.service_clients_standardized import service_clients
from ml_integration_service.error_handling_standardized import (
    handle_error,
    handle_exception,
    handle_async_exception,
    get_status_code,
    MLIntegrationError,
    ModelNotFoundError,
    ModelTrainingError
)
from ml_integration_service.main import app


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_settings():
    """Test that settings are loaded correctly."""
    # Test that settings are loaded
    assert settings is not None
    assert settings.SERVICE_NAME == "ml-integration-service"
    
    # Test that settings can be accessed
    assert settings.HOST == "0.0.0.0"
    assert isinstance(settings.PORT, int)
    
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
async def test_service_clients_mock():
    """Test service clients with mocks."""
    # Mock the service clients
    with patch.object(service_clients, 'get_client', new_callable=MagicMock) as mock_get_client:
        with patch.object(service_clients, 'connect_all', new_callable=AsyncMock) as mock_connect_all:
            with patch.object(service_clients, 'close_all', new_callable=AsyncMock) as mock_close_all:
                # Test get_client
                service_clients.get_client("ml_workbench_service")
                mock_get_client.assert_called_once_with("ml_workbench_service")
                
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
    
    # Test custom error types
    model_error = ModelNotFoundError("Model not found")
    error_response = handle_error(model_error, operation="test_operation")
    assert error_response["error"] == "ModelNotFoundError"
    assert error_response["message"] == "Model not found"
    assert error_response["operation"] == "test_operation"
    
    # Test status code for custom error types
    status_code = get_status_code(model_error)
    assert status_code == 404


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
async def test_health_endpoint_mock(test_client):
    """Test health endpoint with mocks."""
    # Mock the dependencies
    with patch('ml_integration_service.api.v1.health_api.get_model_repository') as mock_get_model_repository:
        with patch('ml_integration_service.api.v1.health_api.get_feature_service') as mock_get_feature_service:
            with patch('ml_integration_service.api.v1.health_api.get_data_validator') as mock_get_data_validator:
                with patch('ml_integration_service.api.v1.health_api.get_reconciliation_service') as mock_get_reconciliation_service:
                    with patch('ml_integration_service.api.v1.health_api.check_model_repository_health', new_callable=AsyncMock) as mock_check_model_repository:
                        with patch('ml_integration_service.api.v1.health_api.check_feature_service_health', new_callable=AsyncMock) as mock_check_feature_service:
                            with patch('ml_integration_service.api.v1.health_api.check_data_validator_health', new_callable=AsyncMock) as mock_check_data_validator:
                                with patch('ml_integration_service.api.v1.health_api.check_reconciliation_service_health', new_callable=AsyncMock) as mock_check_reconciliation:
                                    # Configure mocks
                                    mock_check_model_repository.return_value = {"status": "healthy", "details": {}}
                                    mock_check_feature_service.return_value = {"status": "healthy", "details": {}}
                                    mock_check_data_validator.return_value = {"status": "healthy", "details": {}}
                                    mock_check_reconciliation.return_value = {"status": "healthy", "details": {}}
                                    
                                    # Test health endpoint
                                    response = test_client.get("/health")
                                    assert response.status_code == 200
                                    assert response.json()["status"] == "healthy"
                                    assert response.json()["version"] == settings.SERVICE_VERSION
                                    
                                    # Verify mocks were called
                                    mock_get_model_repository.assert_called_once()
                                    mock_get_feature_service.assert_called_once()
                                    mock_get_data_validator.assert_called_once()
                                    mock_get_reconciliation_service.assert_called_once()
                                    mock_check_model_repository.assert_called_once()
                                    mock_check_feature_service.assert_called_once()
                                    mock_check_data_validator.assert_called_once()
                                    mock_check_reconciliation.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
