"""
Tests for the API error handling utilities.
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from common_lib.errors import (
    BaseError,
    ValidationError,
    ErrorCode,
    create_error_response,
    fastapi_exception_handler
)


class TestAPIErrorHandling(unittest.TestCase):
    """Test case for API error handling utilities."""
    
    def setUp(self):
        """Set up test case."""
        # Create mock logger
        self.mock_logger = MagicMock()
        self.patcher = patch('common_lib.errors.api.logger', self.mock_logger)
        self.patcher.start()
    
    def tearDown(self):
        """Tear down test case."""
        self.patcher.stop()
    
    def test_create_error_response_base_error(self):
        """Test create_error_response with BaseError."""
        # Create error
        error = ValidationError(
            message="Invalid input",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"field": "name", "value": ""}
        )
        
        # Create error response
        response, status_code = create_error_response(error)
        
        # Check response
        self.assertIn("error", response)
        self.assertEqual(response["error"]["message"], "Invalid input")
        self.assertEqual(response["error"]["error_code"], ErrorCode.VALIDATION_ERROR.name)
        self.assertIn("correlation_id", response["error"])
        self.assertIn("details", response["error"])
        self.assertEqual(response["error"]["details"]["field"], "name")
        self.assertEqual(response["error"]["details"]["value"], "")
        
        # Check status code
        self.assertEqual(status_code, 400)
        
        # Check logger
        self.mock_logger.log.assert_called_once()
        args, kwargs = self.mock_logger.log.call_args
        self.assertEqual(args[0], logging.ERROR)
        self.assertIn("ValidationError", args[1])
        self.assertIn("Invalid input", args[1])
    
    def test_create_error_response_http_exception(self):
        """Test create_error_response with HTTPException."""
        # Create error
        error = HTTPException(status_code=404, detail="Item not found")
        
        # Create error response
        response, status_code = create_error_response(error)
        
        # Check response
        self.assertIn("error", response)
        self.assertEqual(response["error"]["message"], "Item not found")
        self.assertEqual(response["error"]["error_code"], "HTTP_404")
        self.assertIn("correlation_id", response["error"])
        self.assertIn("details", response["error"])
        
        # Check status code
        self.assertEqual(status_code, 404)
        
        # Check logger
        self.mock_logger.log.assert_called_once()
        args, kwargs = self.mock_logger.log.call_args
        self.assertEqual(args[0], logging.ERROR)
        self.assertIn("HTTPException", args[1])
        self.assertIn("Item not found", args[1])
    
    def test_create_error_response_generic_error(self):
        """Test create_error_response with generic error."""
        # Create error
        error = ValueError("Invalid value")
        
        # Create error response
        response, status_code = create_error_response(error)
        
        # Check response
        self.assertIn("error", response)
        self.assertEqual(response["error"]["message"], "Internal server error")
        self.assertEqual(response["error"]["error_code"], ErrorCode.UNKNOWN_ERROR.name)
        self.assertIn("correlation_id", response["error"])
        self.assertIn("details", response["error"])
        self.assertEqual(response["error"]["details"]["original_error"], "Invalid value")
        
        # Check status code
        self.assertEqual(status_code, 500)
        
        # Check logger
        self.mock_logger.log.assert_called_once()
        args, kwargs = self.mock_logger.log.call_args
        self.assertEqual(args[0], logging.ERROR)
        self.assertIn("Unhandled exception", args[1])
        self.assertIn("Invalid value", args[1])
    
    def test_create_error_response_with_correlation_id(self):
        """Test create_error_response with correlation ID."""
        # Create error
        error = ValueError("Invalid value")
        
        # Create error response
        response, status_code = create_error_response(error, correlation_id="test-correlation-id")
        
        # Check response
        self.assertIn("error", response)
        self.assertEqual(response["error"]["correlation_id"], "test-correlation-id")
        
        # Check logger
        self.mock_logger.log.assert_called_once()
        args, kwargs = self.mock_logger.log.call_args
        self.assertEqual(kwargs["extra"]["correlation_id"], "test-correlation-id")
    
    def test_create_error_response_with_status_code(self):
        """Test create_error_response with status code."""
        # Create error
        error = ValueError("Invalid value")
        
        # Create error response
        response, status_code = create_error_response(error, status_code=418)
        
        # Check status code
        self.assertEqual(status_code, 418)
    
    def test_create_error_response_with_include_traceback(self):
        """Test create_error_response with include_traceback."""
        # Create error
        error = ValueError("Invalid value")
        
        # Create error response
        response, status_code = create_error_response(error, include_traceback=True)
        
        # Check response
        self.assertIn("error", response)
        self.assertIn("traceback", response["error"]["details"])
    
    def test_fastapi_exception_handler(self):
        """Test fastapi_exception_handler."""
        # Create mock request
        request = MagicMock(spec=Request)
        request.state.correlation_id = "test-correlation-id"
        
        # Create error
        error = ValidationError(
            message="Invalid input",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"field": "name", "value": ""}
        )
        
        # Call exception handler
        response = fastapi_exception_handler(request, error)
        
        # Check response
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.body.decode())
        self.assertIn("Invalid input", response.body.decode())
        self.assertIn("test-correlation-id", response.body.decode())
        self.assertEqual(response.headers["X-Correlation-ID"], "test-correlation-id")


if __name__ == '__main__':
    unittest.main()
"""
