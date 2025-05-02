"""
Unit tests for exception hierarchy.
"""

import unittest
import json
from typing import Dict, Any

from common_lib.exceptions import (
    ForexTradingPlatformError,
    ConfigurationError,
    ConfigNotFoundError,
    ConfigValidationError,
    DataError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    DataTransformationError,
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    AuthenticationError,
    AuthorizationError
)


class TestExceptionHierarchy(unittest.TestCase):
    """Tests for exception hierarchy classes."""
    
    def test_base_exception(self):
        """Test base exception class."""
        # Test with default message
        ex = ForexTradingPlatformError()
        self.assertEqual(ex.message, "An error occurred in the Forex Trading Platform")
        self.assertEqual(ex.error_code, "FOREX_PLATFORM_ERROR")
        self.assertEqual(ex.details, {})
        
        # Test with custom message and details
        ex = ForexTradingPlatformError("Custom error message", "CUSTOM_ERROR", param1="value1")
        self.assertEqual(ex.message, "Custom error message")
        self.assertEqual(ex.error_code, "CUSTOM_ERROR")
        self.assertEqual(ex.details, {"param1": "value1"})
        
        # Test to_dict method
        result = ex.to_dict()
        self.assertEqual(result["error_type"], "ForexTradingPlatformError")
        self.assertEqual(result["error_code"], "CUSTOM_ERROR")
        self.assertEqual(result["message"], "Custom error message")
        self.assertEqual(result["details"], {"param1": "value1"})
    
    def test_configuration_errors(self):
        """Test configuration error classes."""
        # Test ConfigurationError
        ex = ConfigurationError("Config error")
        self.assertEqual(ex.message, "Config error")
        self.assertEqual(ex.error_code, "CONFIG_ERROR")
        
        # Test ConfigNotFoundError
        ex = ConfigNotFoundError("settings.yaml")
        self.assertEqual(ex.message, "Configuration not found: settings.yaml")
        self.assertEqual(ex.error_code, "CONFIG_NOT_FOUND")
        self.assertEqual(ex.details["config_name"], "settings.yaml")
        
        # Test ConfigValidationError
        validation_errors = {"api_key": "Required field missing"}
        ex = ConfigValidationError(validation_errors)
        self.assertEqual(ex.message, "Configuration validation failed")
        self.assertEqual(ex.error_code, "CONFIG_VALIDATION_ERROR")
        self.assertEqual(ex.details["validation_errors"], validation_errors)
    
    def test_data_errors(self):
        """Test data error classes."""
        # Test DataError
        ex = DataError("Generic data error")
        self.assertEqual(ex.message, "Generic data error")
        self.assertEqual(ex.error_code, "DATA_ERROR")
        
        # Test DataValidationError
        validation_errors = {"price": "Must be positive"}
        ex = DataValidationError("Invalid data", validation_errors)
        self.assertEqual(ex.message, "Invalid data")
        self.assertEqual(ex.error_code, "DATA_VALIDATION_ERROR")
        self.assertEqual(ex.details["validation_errors"], validation_errors)
        
        # Test DataFetchError
        ex = DataFetchError(source="api.example.com", status_code=404)
        self.assertEqual(ex.message, "Failed to fetch data from api.example.com")
        self.assertEqual(ex.error_code, "DATA_FETCH_ERROR")
        self.assertEqual(ex.details["source"], "api.example.com")
        self.assertEqual(ex.details["status_code"], 404)
        
        # Test DataStorageError
        ex = DataStorageError(storage_type="database")
        self.assertEqual(ex.message, "Failed to store data in database")
        self.assertEqual(ex.error_code, "DATA_STORAGE_ERROR")
        self.assertEqual(ex.details["storage_type"], "database")
        
        # Test DataTransformationError
        ex = DataTransformationError(transformation="JSON parsing")
        self.assertEqual(ex.message, "Failed to transform data with JSON parsing")
        self.assertEqual(ex.error_code, "DATA_TRANSFORMATION_ERROR")
        self.assertEqual(ex.details["transformation"], "JSON parsing")
    
    def test_service_errors(self):
        """Test service error classes."""
        # Test ServiceError
        ex = ServiceError(service_name="auth-service")
        self.assertEqual(ex.message, "Error in service: auth-service")
        self.assertEqual(ex.error_code, "SERVICE_ERROR")
        self.assertEqual(ex.details["service_name"], "auth-service")
        
        # Test ServiceUnavailableError
        ex = ServiceUnavailableError("data-service")
        self.assertEqual(ex.message, "Service unavailable: data-service")
        self.assertEqual(ex.details["error_code"], "SERVICE_UNAVAILABLE")
        self.assertEqual(ex.details["service_name"], "data-service")
        
        # Test ServiceTimeoutError
        ex = ServiceTimeoutError("analytics-service", 30.5)
        self.assertEqual(ex.message, "Service timeout: analytics-service after 30.5 seconds")
        self.assertEqual(ex.details["error_code"], "SERVICE_TIMEOUT")
        self.assertEqual(ex.details["service_name"], "analytics-service")
        self.assertEqual(ex.details["timeout_seconds"], 30.5)
    
    def test_auth_errors(self):
        """Test authentication and authorization error classes."""
        # Test AuthenticationError
        ex = AuthenticationError("Invalid credentials")
        self.assertEqual(ex.message, "Invalid credentials")
        self.assertEqual(ex.error_code, "AUTHENTICATION_ERROR")
        
        # Test AuthorizationError
        ex = AuthorizationError(resource="user_data", action="write")
        self.assertEqual(ex.message, "Not authorized to write on user_data")
        self.assertEqual(ex.error_code, "AUTHORIZATION_ERROR")
        self.assertEqual(ex.details["resource"], "user_data")
        self.assertEqual(ex.details["action"], "write")
    
    def test_json_serialization(self):
        """Test that exceptions can be serialized to JSON properly."""
        ex = DataFetchError(
            message="API request failed",
            source="market-data-api",
            status_code=500
        )
        
        # Convert exception to dictionary
        ex_dict = ex.to_dict()
        
        # Serialize to JSON and back
        json_str = json.dumps(ex_dict)
        parsed_dict = json.loads(json_str)
        
        # Verify fields
        self.assertEqual(parsed_dict["error_type"], "DataFetchError")
        self.assertEqual(parsed_dict["error_code"], "DATA_FETCH_ERROR")
        self.assertEqual(parsed_dict["message"], "API request failed")
        self.assertEqual(parsed_dict["details"]["source"], "market-data-api")
        self.assertEqual(parsed_dict["details"]["status_code"], 500)


if __name__ == "__main__":
    unittest.main()
