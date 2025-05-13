"""
Tests for the error handling decorators.
"""

import asyncio
import logging
import unittest
from unittest.mock import MagicMock, patch

from common_lib.errors import (
    BaseError,
    ServiceError,
    ValidationError,
    with_exception_handling,
    async_with_exception_handling
)


class TestErrorHandlingDecorators(unittest.TestCase):
    """Test case for error handling decorators."""
    
    def setUp(self):
        """Set up test case."""
        # Create mock logger
        self.mock_logger = MagicMock()
        self.patcher = patch('common_lib.errors.decorators.logger', self.mock_logger)
        self.patcher.start()
    
    def tearDown(self):
        """Tear down test case."""
        self.patcher.stop()
    
    def test_with_exception_handling_no_error(self):
        """Test with_exception_handling decorator with no error."""
        # Define test function
        @with_exception_handling()
        def test_func():
            return "success"
        
        # Call function
        result = test_func()
        
        # Check result
        self.assertEqual(result, "success")
        
        # Check logger
        self.mock_logger.log.assert_not_called()
    
    def test_with_exception_handling_base_error(self):
        """Test with_exception_handling decorator with BaseError."""
        # Define test function
        @with_exception_handling()
        def test_func():
            raise ValidationError("Invalid input")
        
        # Call function and check exception
        with self.assertRaises(ValidationError):
            test_func()
        
        # Check logger
        self.mock_logger.log.assert_called_once()
        args, kwargs = self.mock_logger.log.call_args
        self.assertEqual(args[0], logging.ERROR)
        self.assertIn("ValidationError", args[1])
        self.assertIn("Invalid input", args[1])
    
    def test_with_exception_handling_generic_error(self):
        """Test with_exception_handling decorator with generic error."""
        # Define test function
        @with_exception_handling(error_class=ServiceError)
        def test_func():
            raise ValueError("Invalid value")
        
        # Call function and check exception
        with self.assertRaises(ServiceError):
            test_func()
        
        # Check logger
        self.mock_logger.log.assert_called_once()
        args, kwargs = self.mock_logger.log.call_args
        self.assertEqual(args[0], logging.ERROR)
        self.assertIn("ServiceError", args[1])
        self.assertIn("Invalid value", args[1])
    
    def test_with_exception_handling_no_reraise(self):
        """Test with_exception_handling decorator with no reraise."""
        # Define test function
        @with_exception_handling(reraise=False)
        def test_func():
            raise ValueError("Invalid value")
        
        # Call function
        result = test_func()
        
        # Check result
        self.assertIsNone(result)
        
        # Check logger
        self.mock_logger.log.assert_called_once()
    
    def test_with_exception_handling_cleanup(self):
        """Test with_exception_handling decorator with cleanup."""
        # Define cleanup function
        cleanup_func = MagicMock()
        
        # Define test function
        @with_exception_handling(cleanup_func=cleanup_func)
        def test_func():
            raise ValueError("Invalid value")
        
        # Call function and check exception
        with self.assertRaises(ServiceError):
            test_func()
        
        # Check cleanup function
        cleanup_func.assert_called_once()
    
    def test_with_exception_handling_context(self):
        """Test with_exception_handling decorator with context."""
        # Define test function
        @with_exception_handling(context={"test": "value"})
        def test_func():
            raise ValueError("Invalid value")
        
        # Call function and check exception
        with self.assertRaises(ServiceError):
            test_func()
        
        # Check logger
        self.mock_logger.log.assert_called_once()
        args, kwargs = self.mock_logger.log.call_args
        self.assertEqual(kwargs["extra"]["test"], "value")
    
    def test_async_with_exception_handling_no_error(self):
        """Test async_with_exception_handling decorator with no error."""
        # Define test function
        @async_with_exception_handling()
        async def test_func():
            return "success"
        
        # Call function
        result = asyncio.run(test_func())
        
        # Check result
        self.assertEqual(result, "success")
        
        # Check logger
        self.mock_logger.log.assert_not_called()
    
    def test_async_with_exception_handling_base_error(self):
        """Test async_with_exception_handling decorator with BaseError."""
        # Define test function
        @async_with_exception_handling()
        async def test_func():
            raise ValidationError("Invalid input")
        
        # Call function and check exception
        with self.assertRaises(ValidationError):
            asyncio.run(test_func())
        
        # Check logger
        self.mock_logger.log.assert_called_once()
        args, kwargs = self.mock_logger.log.call_args
        self.assertEqual(args[0], logging.ERROR)
        self.assertIn("ValidationError", args[1])
        self.assertIn("Invalid input", args[1])
    
    def test_async_with_exception_handling_generic_error(self):
        """Test async_with_exception_handling decorator with generic error."""
        # Define test function
        @async_with_exception_handling(error_class=ServiceError)
        async def test_func():
            raise ValueError("Invalid value")
        
        # Call function and check exception
        with self.assertRaises(ServiceError):
            asyncio.run(test_func())
        
        # Check logger
        self.mock_logger.log.assert_called_once()
        args, kwargs = self.mock_logger.log.call_args
        self.assertEqual(args[0], logging.ERROR)
        self.assertIn("ServiceError", args[1])
        self.assertIn("Invalid value", args[1])
    
    def test_async_with_exception_handling_no_reraise(self):
        """Test async_with_exception_handling decorator with no reraise."""
        # Define test function
        @async_with_exception_handling(reraise=False)
        async def test_func():
            raise ValueError("Invalid value")
        
        # Call function
        result = asyncio.run(test_func())
        
        # Check result
        self.assertIsNone(result)
        
        # Check logger
        self.mock_logger.log.assert_called_once()
    
    def test_async_with_exception_handling_cleanup(self):
        """Test async_with_exception_handling decorator with cleanup."""
        # Define cleanup function
        cleanup_func = MagicMock()
        
        # Define test function
        @async_with_exception_handling(cleanup_func=cleanup_func)
        async def test_func():
            raise ValueError("Invalid value")
        
        # Call function and check exception
        with self.assertRaises(ServiceError):
            asyncio.run(test_func())
        
        # Check cleanup function
        cleanup_func.assert_called_once()
    
    def test_async_with_exception_handling_context(self):
        """Test async_with_exception_handling decorator with context."""
        # Define test function
        @async_with_exception_handling(context={"test": "value"})
        async def test_func():
            raise ValueError("Invalid value")
        
        # Call function and check exception
        with self.assertRaises(ServiceError):
            asyncio.run(test_func())
        
        # Check logger
        self.mock_logger.log.assert_called_once()
        args, kwargs = self.mock_logger.log.call_args
        self.assertEqual(kwargs["extra"]["test"], "value")


if __name__ == '__main__':
    unittest.main()
"""
