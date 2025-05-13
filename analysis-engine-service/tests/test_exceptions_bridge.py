"""
Tests for the exceptions bridge module.

This module tests the integration between common-lib exceptions and service-specific exceptions.
"""

import pytest
from fastapi import Request, status
from fastapi.responses import JSONResponse

from analysis_engine.core.exceptions_bridge import (
    ForexTradingPlatformError,
    DataValidationError,
    DataFetchError,
    ConfigurationError,
    ServiceError,
    ServiceUnavailableError
)
from analysis_engine.core.errors import (
    AnalysisEngineError,
    ValidationError,
    DataFetchError as ServiceDataFetchError,
    AnalysisError,
    ConfigurationError as ServiceConfigurationError,
    ServiceUnavailableError as ServiceServiceUnavailableError,
    forex_platform_exception_handler
)

@pytest.fixture
def mock_request():
    """Create a mock request."""
    return Request({"type": "http", "method": "GET", "path": "/test"})

class TestExceptionsBridge:
    """Tests for the exceptions bridge module."""

    def test_analysis_engine_error_inheritance(self):
        """Test that AnalysisEngineError inherits from ForexTradingPlatformError."""
        error = AnalysisEngineError("Test error")
        assert isinstance(error, ForexTradingPlatformError)
        assert error.message == "Test error"
        assert error.error_code == "ANALYSIS_ENGINE_ERROR"
        assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_validation_error_inheritance(self):
        """Test that ValidationError inherits from DataValidationError and AnalysisEngineError."""
        error = ValidationError("Validation failed")
        assert isinstance(error, DataValidationError)
        assert isinstance(error, AnalysisEngineError)
        assert error.message == "Validation failed"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.status_code == status.HTTP_400_BAD_REQUEST

    def test_data_fetch_error_inheritance(self):
        """Test that DataFetchError inherits from common-lib DataFetchError and AnalysisEngineError."""
        error = ServiceDataFetchError("Failed to fetch data", source="test_source")
        assert isinstance(error, DataFetchError)
        assert isinstance(error, AnalysisEngineError)
        assert error.message == "Failed to fetch data"
        assert error.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "source" in error.details
        assert error.details["source"] == "test_source"

    def test_analysis_error(self):
        """Test AnalysisError."""
        error = AnalysisError("Analysis failed", details={"analyzer": "test_analyzer"})
        assert isinstance(error, AnalysisEngineError)
        assert error.message == "Analysis failed"
        assert error.error_code == "ANALYSIS_ERROR"
        assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "analyzer" in error.details
        assert error.details["analyzer"] == "test_analyzer"

    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from common-lib ConfigurationError and AnalysisEngineError."""
        error = ServiceConfigurationError("Configuration error")
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, AnalysisEngineError)
        assert error.message == "Configuration error"
        assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_service_unavailable_error_inheritance(self):
        """Test that ServiceUnavailableError inherits from common-lib ServiceUnavailableError and AnalysisEngineError."""
        error = ServiceServiceUnavailableError("test_service")
        assert isinstance(error, ServiceUnavailableError)
        assert isinstance(error, AnalysisEngineError)
        assert "Service unavailable: test_service" in error.message
        assert error.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "service_name" in error.details
        assert error.details["service_name"] == "test_service"

    @pytest.mark.asyncio
    async def test_forex_platform_exception_handler(self, mock_request):
        """Test the forex_platform_exception_handler."""
        error = ForexTradingPlatformError("Platform error", "TEST_ERROR", test_detail="test_value")
        response = await forex_platform_exception_handler(mock_request, error)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        content = response.body.decode()
        assert "Platform error" in content
        assert "TEST_ERROR" in content
        assert "test_detail" in content
        assert "test_value" in content

    @pytest.mark.asyncio
    async def test_service_specific_exception_handler(self, mock_request):
        """Test the forex_platform_exception_handler with a service-specific exception."""
        error = AnalysisError("Analysis failed", details={"analyzer": "test_analyzer"})
        response = await forex_platform_exception_handler(mock_request, error)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        content = response.body.decode()
        assert "Analysis failed" in content
        assert "ANALYSIS_ERROR" in content
        assert "analyzer" in content
        assert "test_analyzer" in content
