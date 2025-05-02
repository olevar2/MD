"""
Unit tests for the structured logger module.
"""

import json
import logging
import sys
from datetime import datetime
from io import StringIO
from unittest import TestCase, mock

from core_foundations.utils.logger import StructuredLogger, get_logger


class TestStructuredLogger(TestCase):
    """Tests for the StructuredLogger class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a string buffer to capture log output
        self.log_output = StringIO()
        self.handler = logging.StreamHandler(self.log_output)
        
        # Configure logger to use our handler
        self.logger = StructuredLogger("test-service", log_level="DEBUG")
        self.logger.logger.handlers = [self.handler]
    
    def test_basic_logging(self):
        """Test basic logging functionality."""
        self.logger.info("Test message")
        
        # Get the logged output
        log_data = json.loads(self.log_output.getvalue())
        
        # Check structure and content
        self.assertEqual(log_data["service"], "test-service")
        self.assertEqual(log_data["level"], "INFO")
        self.assertEqual(log_data["message"], "Test message")
        self.assertIn("timestamp", log_data)
    
    def test_context_logging(self):
        """Test logging with context."""
        self.logger.info("Test with context", request_id="123", user_id="user456")
        
        # Get the logged output
        log_data = json.loads(self.log_output.getvalue())
        
        # Check context values
        self.assertEqual(log_data["extra"]["request_id"], "123")
        self.assertEqual(log_data["extra"]["user_id"], "user456")
    
    def test_with_context(self):
        """Test the with_context method."""
        context_logger = self.logger.with_context(request_id="789", session_id="sess123")
        context_logger.logger.handlers = [self.handler]
        
        context_logger.warning("Test with persistent context")
        
        # Get the logged output
        log_data = json.loads(self.log_output.getvalue())
        
        # Check context values are present
        self.assertEqual(log_data["level"], "WARNING")
        self.assertEqual(log_data["request_id"], "789")
        self.assertEqual(log_data["session_id"], "sess123")
    
    def test_error_logging(self):
        """Test error logging with exception information."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            self.logger.error("An error occurred", exc_info=True)
        
        # Get the logged output
        log_data = json.loads(self.log_output.getvalue())
        
        # Check exception data
        self.assertEqual(log_data["level"], "ERROR")
        self.assertIn("exception", log_data)
        self.assertEqual(log_data["exception"]["type"], "ValueError")
        self.assertEqual(log_data["exception"]["message"], "Test exception")
        self.assertIn("traceback", log_data["exception"])
    
    def test_get_logger_helper(self):
        """Test the get_logger helper function."""
        helper_logger = get_logger("helper-service")
        self.assertIsInstance(helper_logger, StructuredLogger)
        self.assertEqual(helper_logger.service_name, "helper-service")


if __name__ == "__main__":
    import unittest
    unittest.main()