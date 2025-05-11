"""
Unit tests for the service template error handling module.
"""

import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from common_lib.templates.service_template.error_handling import (
    handle_error,
    handle_exception,
    handle_async_exception,
    get_status_code
)
from common_lib.errors.base_exceptions import (
    BaseError,
    ValidationError,
    DataError,
    ServiceError,
    SecurityError,
    BusinessError
)


@patch("common_lib.templates.service_template.error_handling.error_handler")
def test_handle_error(mock_error_handler):
    """Test handle_error function."""
    # Mock the error_handler
    mock_error_response = {"error": "test error"}
    mock_error_handler.handle_error.return_value = mock_error_response

    # Create an error
    error = ValueError("Test error")

    # Call the function
    response = handle_error(
        error=error,
        operation="test_operation",
        correlation_id="test-correlation-id",
        include_traceback=True
    )

    # Verify the result
    assert response == mock_error_response

    # Verify the mock was called
    mock_error_handler.handle_error.assert_called_once_with(
        error=error,
        operation="test_operation",
        correlation_id="test-correlation-id",
        include_traceback=True
    )


@patch("common_lib.templates.service_template.error_handling.error_handler")
def test_handle_exception(mock_error_handler):
    """Test handle_exception function."""
    # Mock the error_handler
    mock_decorator = MagicMock()
    mock_error_handler.handle_exception.return_value = mock_decorator

    # Call the function
    decorator = handle_exception(
        operation="test_operation",
        correlation_id="test-correlation-id",
        include_traceback=True
    )

    # Verify the result
    assert decorator == mock_decorator

    # Verify the mock was called
    mock_error_handler.handle_exception.assert_called_once_with(
        operation="test_operation",
        correlation_id="test-correlation-id",
        include_traceback=True
    )


@patch("common_lib.templates.service_template.error_handling.error_handler")
def test_handle_async_exception(mock_error_handler):
    """Test handle_async_exception function."""
    # Mock the error_handler
    mock_decorator = MagicMock()
    mock_error_handler.handle_async_exception.return_value = mock_decorator

    # Call the function
    decorator = handle_async_exception(
        operation="test_operation",
        correlation_id="test-correlation-id",
        include_traceback=True
    )

    # Verify the result
    assert decorator == mock_decorator

    # Verify the mock was called
    mock_error_handler.handle_async_exception.assert_called_once_with(
        operation="test_operation",
        correlation_id="test-correlation-id",
        include_traceback=True
    )


@patch("common_lib.templates.service_template.error_handling.error_handler")
def test_get_status_code(mock_error_handler):
    """Test get_status_code function."""
    # Mock the error_handler
    mock_error_handler.get_status_code.return_value = 400

    # Create an error
    error = ValidationError("Test error")

    # Call the function
    status_code = get_status_code(error)

    # Verify the result
    assert status_code == 400

    # Verify the mock was called
    mock_error_handler.get_status_code.assert_called_once_with(error)
