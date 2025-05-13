"""
Tests for error handlers in the Trading Gateway Service.

This module contains tests for the error handlers of the Trading Gateway Service.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import Request, status
from fastapi.responses import JSONResponse

from api.exception_handlers import (
    forex_platform_exception_handler,
    data_validation_exception_handler,
    service_exception_handler,
    trading_exception_handler,
    order_execution_exception_handler,
    broker_connection_exception_handler,
    market_data_exception_handler,
    validation_exception_handler,
    generic_exception_handler
)
from trading_gateway_service.error import (
    ForexTradingPlatformError,
    DataValidationError,
    ServiceError,
    TradingError,
    OrderExecutionError,
    BrokerConnectionError,
    MarketDataError
)

class TestErrorHandlers(unittest.IsolatedAsyncioTestCase):
    """Tests for error handlers."""
    
    async def test_forex_platform_exception_handler(self):
        """Test the forex_platform_exception_handler."""
        # Create a request and exception
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        request.method = "GET"
        
        exception = ForexTradingPlatformError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"test": "value"}
        )
        
        # Call the handler
        with patch("logging.Logger.error") as mock_log:
            response = await forex_platform_exception_handler(request, exception)
            
            # Check that the error was logged
            mock_log.assert_called_once()
            
            # Check the response
            self.assertIsInstance(response, JSONResponse)
            self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
            self.assertEqual(response.body.decode(), '{"error_type":"ForexTradingPlatformError","error_code":"TEST_ERROR","message":"Test error","details":{"test":"value"}}')
    
    async def test_data_validation_exception_handler(self):
        """Test the data_validation_exception_handler."""
        # Create a request and exception
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        request.method = "GET"
        
        exception = DataValidationError(
            message="Validation error",
            error_code="VALIDATION_ERROR",
            details={"field": "value", "error": "Invalid value"}
        )
        
        # Call the handler
        with patch("logging.Logger.warning") as mock_log:
            response = await data_validation_exception_handler(request, exception)
            
            # Check that the error was logged
            mock_log.assert_called_once()
            
            # Check the response
            self.assertIsInstance(response, JSONResponse)
            self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
            self.assertEqual(response.body.decode(), '{"error_type":"DataValidationError","error_code":"VALIDATION_ERROR","message":"Validation error","details":{"field":"value","error":"Invalid value"}}')
    
    async def test_service_exception_handler(self):
        """Test the service_exception_handler."""
        # Create a request and exception
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        request.method = "GET"
        
        exception = ServiceError(
            message="Service error",
            error_code="SERVICE_ERROR",
            details={"service": "test_service"}
        )
        
        # Call the handler
        with patch("logging.Logger.error") as mock_log:
            response = await service_exception_handler(request, exception)
            
            # Check that the error was logged
            mock_log.assert_called_once()
            
            # Check the response
            self.assertIsInstance(response, JSONResponse)
            self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
            self.assertEqual(response.body.decode(), '{"error_type":"ServiceError","error_code":"SERVICE_ERROR","message":"Service error","details":{"service":"test_service"}}')
    
    async def test_trading_exception_handler(self):
        """Test the trading_exception_handler."""
        # Create a request and exception
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        request.method = "GET"
        
        exception = TradingError(
            message="Trading error",
            error_code="TRADING_ERROR",
            details={"order_id": "123456"}
        )
        
        # Call the handler
        with patch("logging.Logger.error") as mock_log:
            response = await trading_exception_handler(request, exception)
            
            # Check that the error was logged
            mock_log.assert_called_once()
            
            # Check the response
            self.assertIsInstance(response, JSONResponse)
            self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
            self.assertEqual(response.body.decode(), '{"error_type":"TradingError","error_code":"TRADING_ERROR","message":"Trading error","details":{"order_id":"123456"}}')
    
    async def test_generic_exception_handler(self):
        """Test the generic_exception_handler."""
        # Create a request and exception
        request = MagicMock(spec=Request)
        request.url.path = "/api/test"
        request.method = "GET"
        
        exception = ValueError("Test value error")
        
        # Call the handler
        with patch("logging.Logger.error") as mock_log:
            response = await generic_exception_handler(request, exception)
            
            # Check that the error was logged
            mock_log.assert_called_once()
            
            # Check the response
            self.assertIsInstance(response, JSONResponse)
            self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
            self.assertEqual(response.body.decode(), '{"error_type":"ServiceError","error_code":"INTERNAL_SERVER_ERROR","message":"An unexpected error occurred","details":{"exception_type":"ValueError"}}')

if __name__ == "__main__":
    unittest.main()
