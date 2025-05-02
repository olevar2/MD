"""
Resilient HTTP Client Module

This module provides resilient HTTP operations with:
1. Retry mechanisms for transient HTTP errors
2. Circuit breakers to prevent cascading failures
3. Timeout handling for HTTP operations
4. Bulkheads to isolate critical operations
"""

import logging
import functools
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, Coroutine
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from analysis_engine.config.settings import Settings
from analysis_engine.resilience import (
    retry_with_policy, timeout_handler, create_circuit_breaker, bulkhead
)
from analysis_engine.resilience.config import (
    get_circuit_breaker_config,
    get_retry_config,
    get_timeout_config,
    get_bulkhead_config
)

# Type variables for function return types
T = TypeVar('T')
R = TypeVar('R')

# Setup logger
logger = logging.getLogger(__name__)

class ResilientHTTPClient:
    """
    HTTP client with resilience patterns.
    
    This class provides:
    1. Retry mechanisms for transient HTTP errors
    2. Circuit breakers to prevent cascading failures
    3. Timeout handling for HTTP operations
    4. Bulkheads to isolate critical operations
    """
    
    def __init__(self, service_type: str = "default", settings: Settings = None):
        """
        Initialize the resilient HTTP client.
        
        Args:
            service_type: Type of service (feature_store, data_pipeline, etc.)
            settings: Application settings containing HTTP configuration
        """
        self.settings = settings or Settings()
        self.service_type = service_type
        self.timeout = get_timeout_config(service_type)
        
        # Initialize circuit breaker
        self.circuit_breaker = create_circuit_breaker(
            service_name="analysis_engine",
            resource_name=f"http_{service_type}",
            config=get_circuit_breaker_config(service_type)
        )
        
        # Initialize bulkhead
        bulkhead_config = get_bulkhead_config(service_type)
        self.bulkhead_name = f"http_{service_type}"
        self.max_concurrent = bulkhead_config["max_concurrent"]
        self.max_waiting = bulkhead_config["max_waiting"]
        self.wait_timeout = bulkhead_config["wait_timeout"]
        
        # Initialize retry config
        retry_config = get_retry_config(service_type)
        self.max_attempts = retry_config["max_attempts"]
        self.base_delay = retry_config["base_delay"]
        self.max_delay = retry_config["max_delay"]
        self.backoff_factor = retry_config["backoff_factor"]
        self.jitter = retry_config["jitter"]
        
        self._is_initialized = True
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """
        Make a GET request with resilience patterns.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments for requests.get
            
        Returns:
            Response object
        """
        return self._request_with_resilience("GET", url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """
        Make a POST request with resilience patterns.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments for requests.post
            
        Returns:
            Response object
        """
        return self._request_with_resilience("POST", url, **kwargs)
    
    def put(self, url: str, **kwargs) -> requests.Response:
        """
        Make a PUT request with resilience patterns.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments for requests.put
            
        Returns:
            Response object
        """
        return self._request_with_resilience("PUT", url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> requests.Response:
        """
        Make a DELETE request with resilience patterns.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments for requests.delete
            
        Returns:
            Response object
        """
        return self._request_with_resilience("DELETE", url, **kwargs)
    
    def _request_with_resilience(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make a request with resilience patterns.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
        """
        # Set timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        
        # Define the request function
        def request_func():
            return requests.request(method, url, **kwargs)
        
        # Apply resilience patterns
        return self._execute_with_resilience(request_func)
    
    def _execute_with_resilience(self, request_func: Callable[[], T]) -> T:
        """
        Execute a function with resilience patterns.
        
        Args:
            request_func: Function to execute
            
        Returns:
            Result of the function
        """
        # Apply retry
        @retry_with_policy(
            max_attempts=self.max_attempts,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            backoff_factor=self.backoff_factor,
            jitter=self.jitter,
            exceptions=[RequestException, Timeout, ConnectionError],
            service_name="analysis_engine",
            operation_name=f"http_{self.service_type}"
        )
        # Apply timeout
        @timeout_handler(timeout_seconds=self.timeout)
        # Apply bulkhead
        @bulkhead(
            name=self.bulkhead_name,
            max_concurrent=self.max_concurrent,
            max_waiting=self.max_waiting,
            wait_timeout=self.wait_timeout
        )
        def resilient_func():
            return self.circuit_breaker.execute(request_func)
        
        return resilient_func()
    
    @property
    def is_initialized(self) -> bool:
        """Check if HTTP client is initialized."""
        return self._is_initialized


# Client cache
_http_clients: Dict[str, ResilientHTTPClient] = {}

def get_http_client(service_type: str = "default") -> ResilientHTTPClient:
    """
    Get a resilient HTTP client for a specific service type.
    
    Args:
        service_type: Type of service (feature_store, data_pipeline, etc.)
        
    Returns:
        ResilientHTTPClient instance
    """
    if service_type not in _http_clients:
        _http_clients[service_type] = ResilientHTTPClient(service_type)
    return _http_clients[service_type]
