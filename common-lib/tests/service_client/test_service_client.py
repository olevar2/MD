"""
Simple test script for the service client module.
"""

import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the service client module
from common_lib.service_client import (
    BaseServiceClient,
    AsyncBaseServiceClient,
    ServiceClientConfig,
    RetryConfig,
    CircuitBreakerConfig,
    TimeoutConfig,
    HTTPServiceClient,
    AsyncHTTPServiceClient,
)


def test_service_client_config():
    """Test the service client configuration classes."""
    print("Testing service client configuration classes...")
    
    # Test RetryConfig
    retry_config = RetryConfig(
        max_retries=5,
        initial_backoff_ms=200,
        max_backoff_ms=20000,
        backoff_multiplier=3.0,
        retry_on_exceptions=[ValueError, TypeError],
        retry_on_status_codes=[429, 500, 503]
    )
    assert retry_config.max_retries == 5
    assert retry_config.initial_backoff_ms == 200
    assert retry_config.max_backoff_ms == 20000
    assert retry_config.backoff_multiplier == 3.0
    assert retry_config.retry_on_exceptions == [ValueError, TypeError]
    assert retry_config.retry_on_status_codes == [429, 500, 503]
    
    # Test CircuitBreakerConfig
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=10,
        recovery_timeout_ms=60000,
        half_open_success_threshold=5
    )
    assert circuit_breaker_config.failure_threshold == 10
    assert circuit_breaker_config.recovery_timeout_ms == 60000
    assert circuit_breaker_config.half_open_success_threshold == 5
    
    # Test TimeoutConfig
    timeout_config = TimeoutConfig(
        connect_timeout_ms=10000,
        read_timeout_ms=60000,
        total_timeout_ms=120000
    )
    assert timeout_config.connect_timeout_ms == 10000
    assert timeout_config.read_timeout_ms == 60000
    assert timeout_config.total_timeout_ms == 120000
    
    # Test ServiceClientConfig
    service_client_config = ServiceClientConfig(
        service_name="test-service",
        base_url="http://test-service.example.com",
        retry_config=retry_config,
        circuit_breaker_config=circuit_breaker_config,
        timeout_config=timeout_config,
        headers={"X-API-Key": "test-api-key"}
    )
    assert service_client_config.service_name == "test-service"
    assert service_client_config.base_url == "http://test-service.example.com"
    assert service_client_config.retry_config == retry_config
    assert service_client_config.circuit_breaker_config == circuit_breaker_config
    assert service_client_config.timeout_config == timeout_config
    assert service_client_config.headers == {"X-API-Key": "test-api-key"}
    
    print("Service client configuration classes test passed!")
    return True


def test_http_service_client():
    """Test the HTTP service client class."""
    print("Testing HTTP service client class...")
    
    # Create a service client config
    service_client_config = ServiceClientConfig(
        service_name="test-service",
        base_url="http://test-service.example.com",
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Create an HTTP service client
    http_client = HTTPServiceClient(service_client_config)
    
    # Check that the client was created successfully
    assert http_client.config == service_client_config
    assert http_client.logger.name == "HTTPServiceClient.test-service"
    assert http_client._circuit_open is False
    assert http_client._failure_count == 0
    
    print("HTTP service client class test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("Running all service client tests...")
    
    test_service_client_config()
    test_http_service_client()
    
    print("All service client tests passed!")
    return True


if __name__ == "__main__":
    run_all_tests()