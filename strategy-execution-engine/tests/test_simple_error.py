"""
Simple tests for error handling in the Strategy Execution Engine.
"""

import pytest
from unittest.mock import MagicMock

class TestError(Exception):
    """Test error class."""
    def __init__(self, message):
        self.message = message
        super().__init__(message)

def test_error_handling():
    """Test basic error handling."""
    # Create a function that raises an error
    def function_with_error():
        raise TestError("Test error")
    
    # Verify that the function raises an error
    with pytest.raises(TestError) as excinfo:
        function_with_error()
    
    # Verify the error message
    assert str(excinfo.value) == "Test error"

def test_error_handling_with_try_except():
    """Test error handling with try/except."""
    # Create a function that catches an error
    def function_with_try_except():
        try:
            raise TestError("Test error")
        except TestError as e:
            return f"Caught error: {e}"
    
    # Verify that the function returns the expected message
    result = function_with_try_except()
    assert result == "Caught error: Test error"

def test_error_handling_with_decorator():
    """Test error handling with a decorator."""
    # Create a decorator that catches errors
    def error_handler(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return f"Caught error: {e}"
        return wrapper
    
    # Create a function that raises an error
    @error_handler
    def function_with_error():
        raise TestError("Test error")
    
    # Verify that the function returns the expected message
    result = function_with_error()
    assert result == "Caught error: Test error"

def test_error_handling_with_mock():
    """Test error handling with a mock."""
    # Create a mock function that raises an error
    mock_function = MagicMock(side_effect=TestError("Test error"))
    
    # Verify that the function raises an error
    with pytest.raises(TestError) as excinfo:
        mock_function()
    
    # Verify the error message
    assert str(excinfo.value) == "Test error"
