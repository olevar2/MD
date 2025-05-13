"""
Tests for error handlers in Portfolio Management Service.
"""
import pytest
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from portfolio_management_service.error import (
    register_exception_handlers,
    PortfolioManagementError,
    PortfolioNotFoundError,
    PositionNotFoundError,
    InsufficientBalanceError,
    PortfolioOperationError
)
from api.error_handlers import (
    get_correlation_id,
    format_error_response,
    format_validation_errors
)


def test_get_correlation_id():
    """Test get_correlation_id function."""
    # Test with correlation ID in request state
    mock_request = MagicMock()
    mock_request.state.correlation_id = "test-correlation-id"
    assert get_correlation_id(mock_request) == "test-correlation-id"
    
    # Test with correlation ID in headers
    mock_request = MagicMock()
    mock_request.state = MagicMock(spec=[])  # No correlation_id attribute
    mock_request.headers = {"X-Correlation-ID": "test-correlation-id"}
    assert get_correlation_id(mock_request) == "test-correlation-id"
    
    # Test with no correlation ID
    mock_request = MagicMock()
    mock_request.state = MagicMock(spec=[])  # No correlation_id attribute
    mock_request.headers = {}
    correlation_id = get_correlation_id(mock_request)
    assert isinstance(correlation_id, str)
    assert len(correlation_id) > 0  # Should generate a new UUID


def test_format_error_response():
    """Test format_error_response function."""
    response = format_error_response(
        error_code="TEST_ERROR",
        message="Test error",
        details={"test": "value"},
        correlation_id="test-correlation-id",
        service="test-service"
    )
    
    assert response["error"]["code"] == "TEST_ERROR"
    assert response["error"]["message"] == "Test error"
    assert response["error"]["details"] == {"test": "value"}
    assert response["error"]["correlation_id"] == "test-correlation-id"
    assert response["error"]["service"] == "test-service"
    assert "timestamp" in response["error"]
    
    # Test with default values
    response = format_error_response(
        error_code="TEST_ERROR",
        message="Test error"
    )
    
    assert response["error"]["code"] == "TEST_ERROR"
    assert response["error"]["message"] == "Test error"
    assert response["error"]["details"] == {}
    assert "correlation_id" in response["error"]
    assert response["error"]["service"] == "portfolio-management-service"
    assert "timestamp" in response["error"]


def test_format_validation_errors():
    """Test format_validation_errors function."""
    errors = [
        {"loc": ["body", "name"], "msg": "field required", "type": "value_error.missing"},
        {"loc": ["body", "age"], "msg": "value is not a valid integer", "type": "type_error.integer"}
    ]
    
    formatted = format_validation_errors(errors)
    
    assert formatted["body.name"] == "field required"
    assert formatted["body.age"] == "value is not a valid integer"
    
    # Test with empty errors
    formatted = format_validation_errors([])
    assert formatted == {}


def create_test_app():
    """Create a test FastAPI app with error handlers."""
    app = FastAPI()
    register_exception_handlers(app)
    
    @app.get("/test-portfolio-not-found")
    async def test_portfolio_not_found():
        raise PortfolioNotFoundError(
            message="Portfolio not found",
            portfolio_id="test-portfolio"
        )
    
    @app.get("/test-position-not-found")
    async def test_position_not_found():
        raise PositionNotFoundError(
            message="Position not found",
            position_id="test-position"
        )
    
    @app.get("/test-insufficient-balance")
    async def test_insufficient_balance():
        raise InsufficientBalanceError(
            message="Insufficient balance",
            required_amount=100.0,
            available_amount=50.0,
            currency="USD"
        )
    
    @app.get("/test-portfolio-operation")
    async def test_portfolio_operation():
        raise PortfolioOperationError(
            message="Operation failed",
            operation="test-operation"
        )
    
    @app.get("/test-generic-error")
    async def test_generic_error():
        raise Exception("Test error")
    
    return app


def test_error_handlers():
    """Test error handlers with a FastAPI test client."""
    app = create_test_app()
    client = TestClient(app)
    
    # Test PortfolioNotFoundError (404)
    response = client.get("/test-portfolio-not-found")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "PORTFOLIO_NOT_FOUND_ERROR"
    assert data["error"]["message"] == "Portfolio not found"
    assert data["error"]["details"]["portfolio_id"] == "test-portfolio"
    
    # Test PositionNotFoundError (404)
    response = client.get("/test-position-not-found")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "POSITION_NOT_FOUND_ERROR"
    assert data["error"]["message"] == "Position not found"
    assert data["error"]["details"]["position_id"] == "test-position"
    
    # Test InsufficientBalanceError (403)
    response = client.get("/test-insufficient-balance")
    assert response.status_code == 403
    data = response.json()
    assert data["error"]["code"] == "INSUFFICIENT_BALANCE_ERROR"
    assert data["error"]["message"] == "Insufficient balance"
    assert data["error"]["details"]["required_amount"] == 100.0
    assert data["error"]["details"]["available_amount"] == 50.0
    assert data["error"]["details"]["currency"] == "USD"
    
    # Test PortfolioOperationError (400)
    response = client.get("/test-portfolio-operation")
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == "PORTFOLIO_OPERATION_ERROR"
    assert data["error"]["message"] == "Operation failed"
    assert data["error"]["details"]["operation"] == "test-operation"
    
    # Test generic error (500)
    response = client.get("/test-generic-error")
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert data["error"]["message"] == "An unexpected error occurred"