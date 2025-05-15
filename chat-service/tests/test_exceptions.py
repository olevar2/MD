"""
Tests for custom exceptions and exception handler (app.exceptions)
"""
import pytest
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from unittest.mock import MagicMock
import logging

from app.exceptions import (
    ChatServiceError,
    ValidationError,
    AuthenticationError,
    ResourceNotFoundError,
    DatabaseError,
    EventBusError,
    chat_service_exception_handler,
    setup_exception_handlers
)

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# --- Test Custom Exception Classes --- #

def test_chat_service_error_base():
    error = ChatServiceError("Base error message", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    assert error.message == "Base error message"
    assert error.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert error.detail is None
    assert str(error) == "Base error message"

def test_validation_error():
    error = ValidationError("Invalid input", detail={"field": "name", "error": "too short"})
    assert error.message == "Invalid input"
    assert error.status_code == status.HTTP_400_BAD_REQUEST
    assert error.detail == {"field": "name", "error": "too short"}
    assert str(error) == "Invalid input - Detail: {'field': 'name', 'error': 'too short'}"

def test_authentication_error():
    error = AuthenticationError("Auth failed")
    assert error.message == "Auth failed"
    assert error.status_code == status.HTTP_401_UNAUTHORIZED
    assert error.detail is None

def test_resource_not_found_error():
    error = ResourceNotFoundError("Item not found", resource_id="item123")
    assert error.message == "Item not found"
    assert error.status_code == status.HTTP_404_NOT_FOUND
    assert error.detail == {"resource_id": "item123"}
    assert str(error) == "Item not found - Detail: {'resource_id': 'item123'}"

def test_database_error():
    original_exc = ValueError("DB connection failed")
    error = DatabaseError("DB operation failed", original_exception=original_exc)
    assert error.message == "DB operation failed"
    assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert error.detail == {"original_exception": str(original_exc)}
    assert str(error) == f"DB operation failed - Detail: {{'original_exception': '{str(original_exc)}'}}"

def test_event_bus_error():
    error = EventBusError("Event publish failed", event_type="user_created")
    assert error.message == "Event publish failed"
    assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert error.detail == {"event_type": "user_created"}

# --- Test chat_service_exception_handler --- #

@pytest.fixture
def mock_request_with_correlation_id() -> Request:
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.correlation_id = "test-correlation-id-123"
    return request

@pytest.fixture
def mock_request_without_correlation_id() -> Request:
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    # Simulate correlation_id not being set or attribute error
    delattr(request.state, 'correlation_id') # Ensure it's not there
    return request

async def test_handler_with_validation_error(mock_request_with_correlation_id, caplog):
    caplog.set_level(logging.ERROR)
    exc = ValidationError("Invalid data", detail={"field": "email"})
    response = await chat_service_exception_handler(mock_request_with_correlation_id, exc)
    
    assert isinstance(response, JSONResponse)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    content = response.body.decode()
    assert '"error_code": "VALIDATION_ERROR"' in content
    assert '"message": "Invalid data"' in content
    assert '"detail": {"field": "email"}' in content
    assert '"correlation_id": "test-correlation-id-123"' in content
    assert "ChatService Error: VALIDATION_ERROR - Message: Invalid data - Detail: {'field': 'email'} - Correlation ID: test-correlation-id-123" in caplog.text

async def test_handler_with_authentication_error(mock_request_with_correlation_id, caplog):
    exc = AuthenticationError("Token expired")
    response = await chat_service_exception_handler(mock_request_with_correlation_id, exc)
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    content = response.body.decode()
    assert '"error_code": "AUTHENTICATION_ERROR"' in content
    assert '"message": "Token expired"' in content
    assert '"correlation_id": "test-correlation-id-123"' in content
    assert "ChatService Error: AUTHENTICATION_ERROR - Message: Token expired - Correlation ID: test-correlation-id-123" in caplog.text

async def test_handler_with_resource_not_found_error(mock_request_without_correlation_id, caplog):
    exc = ResourceNotFoundError("User not found", resource_id="user-xyz")
    response = await chat_service_exception_handler(mock_request_without_correlation_id, exc)
    
    assert response.status_code == status.HTTP_404_NOT_FOUND
    content = response.body.decode()
    assert '"error_code": "RESOURCE_NOT_FOUND"' in content
    assert '"message": "User not found"' in content
    assert '"detail": {"resource_id": "user-xyz"}' in content
    assert '"correlation_id": null' in content # or check it's not present if that's the behavior
    assert "ChatService Error: RESOURCE_NOT_FOUND - Message: User not found - Detail: {'resource_id': 'user-xyz'} - Correlation ID: None" in caplog.text

async def test_handler_with_generic_chat_service_error(mock_request_with_correlation_id, caplog):
    exc = ChatServiceError("A generic service error occurred", status_code=status.HTTP_418_IM_A_TEAPOT)
    response = await chat_service_exception_handler(mock_request_with_correlation_id, exc)
    
    assert response.status_code == status.HTTP_418_IM_A_TEAPOT
    content = response.body.decode()
    assert '"error_code": "CHAT_SERVICE_ERROR"' in content # Default error code for base class
    assert '"message": "A generic service error occurred"' in content
    assert '"correlation_id": "test-correlation-id-123"' in content
    assert "ChatService Error: CHAT_SERVICE_ERROR - Message: A generic service error occurred - Correlation ID: test-correlation-id-123" in caplog.text

# --- Test setup_exception_handlers --- #

def test_setup_exception_handlers():
    app = FastAPI()
    setup_exception_handlers(app)
    
    # Check if the handler is registered for ChatServiceError
    # FastAPI stores exception handlers in app.exception_handlers
    # The key can be the status code or the exception class itself.
    # We expect it to be registered for the ChatServiceError class.
    assert ChatServiceError in app.exception_handlers
    assert app.exception_handlers[ChatServiceError] == chat_service_exception_handler

    # Optionally, check for other specific exceptions if they were meant to be individually registered
    # (though the current setup_exception_handlers only registers the base ChatServiceError)
    # For example, if ValidationError was also registered directly:
    # assert ValidationError in app.exception_handlers
    # assert app.exception_handlers[ValidationError] == chat_service_exception_handler