"""
Error Handling Tests

This module contains tests for the error handling functionality.
"""

import pytest
from fastapi import Request, status
from fastapi.responses import JSONResponse

from analysis_engine.core.exceptions_bridge import ForexTradingPlatformError
from analysis_engine.core.errors import (
    AnalysisEngineError,
    ValidationError,
    DataFetchError,
    AnalysisError,
    ConfigurationError,
    ServiceUnavailableError,
    forex_platform_exception_handler,
    analysis_engine_exception_handler,
    validation_exception_handler,
    data_fetch_exception_handler,
    analysis_exception_handler,
    configuration_exception_handler,
    service_unavailable_exception_handler,
    generic_exception_handler
)

@pytest.fixture
def mock_request():
    """Create a mock request."""
    return Request({"type": "http", "method": "GET", "path": "/test"})

@pytest.mark.asyncio
async def test_forex_platform_exception_handler(mock_request):
    """Test the ForexTradingPlatformError handler."""
    error = ForexTradingPlatformError("Platform error", "TEST_ERROR", test_detail="test_value")
    response = await forex_platform_exception_handler(mock_request, error)

    assert isinstance(response, JSONResponse)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    # Check that the response contains the expected data
    content = response.body.decode()
    assert "Platform error" in content
    assert "TEST_ERROR" in content
    assert "test_detail" in content
    assert "test_value" in content

@pytest.mark.asyncio
async def test_analysis_engine_error(mock_request):
    """Test AnalysisEngineError handler."""
    error = AnalysisEngineError("Test error", status_code=418)
    response = await analysis_engine_exception_handler(mock_request, error)
    assert response.status_code == 418

    content = response.body.decode()
    assert "Test error" in content
    assert "AnalysisEngineError" in content

@pytest.mark.asyncio
async def test_validation_error(mock_request):
    """Test ValidationError handler."""
    error = ValidationError("Invalid input")
    response = await validation_exception_handler(mock_request, error)
    assert response.status_code == 400

    content = response.body.decode()
    assert "Invalid input" in content
    assert "ValidationError" in content

@pytest.mark.asyncio
async def test_data_fetch_error(mock_request):
    """Test DataFetchError handler."""
    error = DataFetchError("Failed to fetch data", source="test_source")
    response = await data_fetch_exception_handler(mock_request, error)
    assert response.status_code == 503

    content = response.body.decode()
    assert "Failed to fetch data" in content
    assert "DataFetchError" in content
    assert "test_source" in content

@pytest.mark.asyncio
async def test_analysis_error(mock_request):
    """Test AnalysisError handler."""
    error = AnalysisError("Analysis failed", details={"analyzer": "test_analyzer"})
    response = await analysis_exception_handler(mock_request, error)
    assert response.status_code == 500

    content = response.body.decode()
    assert "Analysis failed" in content
    assert "AnalysisError" in content
    assert "test_analyzer" in content

@pytest.mark.asyncio
async def test_configuration_error(mock_request):
    """Test ConfigurationError handler."""
    error = ConfigurationError("Invalid configuration")
    response = await configuration_exception_handler(mock_request, error)
    assert response.status_code == 500

    content = response.body.decode()
    assert "Invalid configuration" in content
    assert "ConfigurationError" in content

@pytest.mark.asyncio
async def test_service_unavailable_error(mock_request):
    """Test ServiceUnavailableError handler."""
    error = ServiceUnavailableError("test_service")
    response = await service_unavailable_exception_handler(mock_request, error)
    assert response.status_code == 503

    content = response.body.decode()
    assert "Service unavailable: test_service" in content
    assert "ServiceUnavailableError" in content

@pytest.mark.asyncio
async def test_generic_exception_handler(mock_request):
    """Test generic exception handler."""
    error = Exception("Unexpected error")
    response = await generic_exception_handler(mock_request, error)
    assert response.status_code == 500

    content = response.body.decode()
    assert "An unexpected error occurred" in content
    assert "InternalServerError" in content
    assert "Unexpected error" in content