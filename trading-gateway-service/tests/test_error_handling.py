"""
Tests for error handling in the Trading Gateway Service.

This module contains tests for the error handling functionality of the
Trading Gateway Service.
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock

from trading_gateway_service.error import (
    ForexTradingPlatformError,
    BrokerConnectionError,
    OrderValidationError,
    MarketDataError,
    ServiceError,
    handle_exception,
    with_exception_handling,
    async_with_exception_handling,
    convert_js_error,
    convert_to_js_error
)

class TestErrorHandling(unittest.TestCase):
    """Tests for error handling functionality."""
    
    def test_handle_exception_with_platform_error(self):
        """Test handling a ForexTradingPlatformError."""
        # Create a ForexTradingPlatformError
        error = ServiceError(
            message="Test service error",
            error_code="TEST_ERROR",
            details={"test": "value"}
        )
        
        # Handle the error without reraising
        with patch("logging.Logger.error") as mock_log:
            handled_error = handle_exception(error, reraise=False)
            
            # Check that the error was logged
            mock_log.assert_called_once()
            
            # Check that the original error was returned
            self.assertEqual(handled_error, error)
    
    def test_handle_exception_with_standard_error(self):
        """Test handling a standard Python error."""
        # Create a standard Python error
        error = ValueError("Test value error")
        
        # Handle the error without reraising
        with patch("logging.Logger.error") as mock_log:
            handled_error = handle_exception(error, reraise=False)
            
            # Check that the error was logged
            mock_log.assert_called_once()
            
            # Check that the error was converted to a ForexTradingPlatformError
            self.assertIsInstance(handled_error, ForexTradingPlatformError)
            self.assertEqual(handled_error.message, "ValueError: Test value error")
    
    def test_with_exception_handling_decorator(self):
        """Test the with_exception_handling decorator."""
        # Create a function that raises an error
        @with_exception_handling
        def test_function():
            raise ValueError("Test value error")
        
        # Call the function and check that it raises a ForexTradingPlatformError
        with self.assertRaises(ForexTradingPlatformError):
            test_function()
    
    def test_async_with_exception_handling_decorator(self):
        """Test the async_with_exception_handling decorator."""
        # Create an async function that raises an error
        @async_with_exception_handling
        async def test_async_function():
            raise ValueError("Test value error")
        
        # Call the function and check that it raises a ForexTradingPlatformError
        with self.assertRaises(ForexTradingPlatformError):
            asyncio.run(test_async_function())
    
    def test_convert_js_error(self):
        """Test converting a JavaScript error to a Python error."""
        # Create a JavaScript error as JSON
        js_error = {
            "error_type": "OrderExecutionError",
            "error_code": "ORDER_EXECUTION_ERROR",
            "message": "Failed to execute order",
            "details": {"order_id": "123456"}
        }
        
        # Convert the error
        py_error = convert_js_error(js_error)
        
        # Check that the error was converted correctly
        self.assertIsInstance(py_error, ForexTradingPlatformError)
        self.assertEqual(py_error.message, "Failed to execute order")
        self.assertEqual(py_error.error_code, "ORDER_EXECUTION_ERROR")
        self.assertEqual(py_error.details, {"order_id": "123456"})
    
    def test_convert_to_js_error(self):
        """Test converting a Python error to a JavaScript error."""
        # Create a ForexTradingPlatformError
        py_error = OrderValidationError(
            message="Order validation failed",
            order_id="123456",
            validation_errors=["Invalid price"]
        )
        
        # Convert the error
        js_error = convert_to_js_error(py_error)
        
        # Check that the error was converted correctly
        self.assertEqual(js_error["error_type"], "OrderValidationError")
        self.assertEqual(js_error["message"], "Order validation failed")
        self.assertEqual(js_error["details"]["order_id"], "123456")
        self.assertEqual(js_error["details"]["validation_errors"], ["Invalid price"])

if __name__ == "__main__":
    unittest.main()
