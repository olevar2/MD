"""
Simple test script for the error handling module.
"""

import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the error handling module
from common_lib.errors import (
    ForexTradingError,
    ValidationError,
    DatabaseError,
    APIError,
    ResourceNotFoundError,
    log_exception,
    format_exception,
    create_error_response,
    handle_exceptions,
    ErrorHandler
)


def test_error_classes():
    """Test the error classes."""
    print("Testing error classes...")
    
    # Test ForexTradingError
    error = ForexTradingError(
        message="Test error",
        code="TEST_ERROR",
        http_status=400,
        details={"test": "value"}
    )
    assert error.message == "Test error"
    assert error.code == "TEST_ERROR"
    assert error.http_status == 400
    assert error.details == {"test": "value"}
    
    # Test ValidationError
    error = ValidationError(
        message="Invalid value",
        field="username",
        value="invalid",
        code="INVALID_USERNAME"
    )
    assert error.message == "Invalid value"
    assert error.code == "INVALID_USERNAME"
    assert error.http_status == 400
    assert error.details["field"] == "username"
    assert error.details["value"] == "invalid"
    
    # Test DatabaseError
    error = DatabaseError(
        message="Database connection failed",
        operation="connect"
    )
    assert error.message == "Database connection failed"
    assert error.code == "DATABASE_ERROR"
    assert error.http_status == 500
    assert error.details["operation"] == "connect"
    
    # Test ResourceNotFoundError
    error = ResourceNotFoundError(
        message="User not found",
        resource_type="user",
        resource_id="123"
    )
    assert error.message == "User not found"
    assert error.code == "RESOURCE_NOT_FOUND"
    assert error.http_status == 404
    assert error.details["resource_type"] == "user"
    assert error.details["resource_id"] == "123"
    
    print("Error classes test passed!")
    return True


def test_error_handling_utilities():
    """Test the error handling utilities."""
    print("Testing error handling utilities...")
    
    # Set up a logger for testing
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.ERROR)
    
    # Test log_exception
    error = ValidationError(
        message="Invalid value",
        field="username",
        value="invalid"
    )
    log_exception(error, logger)
    
    # Test format_exception
    formatted = format_exception(error)
    assert "error" in formatted
    assert formatted["error"]["type"] == "ValidationError"
    assert formatted["error"]["message"] == "Invalid value"
    assert "correlation_id" in formatted["error"]
    
    # Test create_error_response
    response = create_error_response(error)
    assert "content" in response
    assert "status_code" in response
    assert response["status_code"] == 400
    
    print("Error handling utilities test passed!")
    return True


def test_error_handler_decorator():
    """Test the error handler decorator."""
    print("Testing error handler decorator...")
    
    # Define a function that raises an error
    @handle_exceptions()
    def function_with_error():
    """
    Function with error.
    
    """

        raise ValidationError(
            message="Invalid value",
            field="username",
            value="invalid"
        )
    
    # Call the function and check the response
    response = function_with_error()
    assert "content" in response
    assert "status_code" in response
    assert response["status_code"] == 400
    
    print("Error handler decorator test passed!")
    return True


def test_error_handler_class():
    """Test the ErrorHandler class."""
    print("Testing ErrorHandler class...")
    
    # Create an error handler
    handler = ErrorHandler()
    
    # Define a function that raises an error
    @handler.handle
    def function_with_error():
    """
    Function with error.
    
    """

        raise ValidationError(
            message="Invalid value",
            field="username",
            value="invalid"
        )
    
    # Call the function and check the response
    response = function_with_error()
    assert "content" in response
    assert "status_code" in response
    assert response["status_code"] == 400
    
    print("ErrorHandler class test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("Running all error handling tests...")
    
    test_error_classes()
    test_error_handling_utilities()
    test_error_handler_decorator()
    test_error_handler_class()
    
    print("All error handling tests passed!")
    return True


if __name__ == "__main__":
    run_all_tests()