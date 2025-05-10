"""
Base Service Client Module

This module provides a base client class for service communication with built-in
resilience patterns, error handling, metrics collection, and structured logging.

Key features:
1. Resilience patterns (retry, circuit breaker, timeout, bulkhead)
2. Standardized error handling
3. Metrics collection
4. Structured logging with correlation IDs
5. Configurable client behavior
"""

import logging
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, Type, TypeVar, cast
from urllib.parse import urljoin

import aiohttp
import requests
from pydantic import BaseModel

from common_lib.correlation import (
    get_correlation_id,
    generate_correlation_id,
    CORRELATION_ID_HEADER,
    add_correlation_id_to_headers,
    ClientCorrelationMixin
)

from common_lib.resilience import (
    retry_with_policy,
    create_circuit_breaker,
    CircuitBreakerConfig,
    timeout_handler,
    bulkhead,
    CircuitBreakerOpen,
    RetryExhaustedException,
    TimeoutError,
    BulkheadFullException
)
from common_lib.exceptions import (
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    DataValidationError,
    AuthenticationError
)

# Type variables for return type hints
T = TypeVar('T')
ResponseType = Union[Dict[str, Any], List[Dict[str, Any]], List[Any], Any]

logger = logging.getLogger(__name__)


class ClientConfig(BaseModel):
    """Configuration for service clients."""

    # Service connection
    base_url: str
    timeout_seconds: float = 10.0
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    default_headers: Optional[Dict[str, str]] = None

    # Resilience settings
    max_retries: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 30.0
    retry_backoff_factor: float = 2.0
    retry_jitter: bool = True

    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_timeout_seconds: int = 60

    # Bulkhead settings
    bulkhead_max_concurrent: int = 10
    bulkhead_max_waiting: int = 20

    # Metrics and logging
    enable_metrics: bool = True
    enable_request_logging: bool = True
    log_request_body: bool = False  # Set to False by default for security
    log_response_body: bool = False  # Set to False by default for performance

    # Service name for metrics and logging
    service_name: str = "unknown-service"


class BaseServiceClient(ClientCorrelationMixin):
    """
    Base class for service clients with built-in resilience patterns.

    This class provides:
    1. Standard HTTP methods with resilience patterns
    2. Error handling and mapping
    3. Metrics collection
    4. Structured logging
    5. Correlation ID propagation

    Subclasses should implement service-specific methods using the base HTTP methods.
    """

    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """
        Initialize the base service client.

        Args:
            config: Client configuration
        """
        # Convert dict to ClientConfig if needed
        if isinstance(config, dict):
            self.config = ClientConfig(**config)
        else:
            self.config = config

        self.base_url = self.config.base_url.rstrip('/')
        self.timeout = self.config.timeout_seconds

        # Initialize session
        self.session: Optional[aiohttp.ClientSession] = None

        # Initialize circuit breaker
        self.circuit_breaker = create_circuit_breaker(
            service_name=self.config.service_name,
            resource_name="http",
            config=CircuitBreakerConfig(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                reset_timeout_seconds=self.config.circuit_breaker_reset_timeout_seconds
            )
        )

        logger.info(f"{self.__class__.__name__} initialized with base URL: {self.base_url}")

    async def _ensure_session(self) -> None:
        """Ensure that an aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def _get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get request headers with API key and correlation ID if available.

        Args:
            additional_headers: Additional headers to include

        Returns:
            Headers dictionary
        """
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": str(uuid.uuid4())
        }

        # Add API key if available
        if self.config.api_key:
            headers[self.config.api_key_header] = self.config.api_key

        # Add correlation ID from context or generate a new one
        correlation_id = get_correlation_id()
        if correlation_id:
            headers[CORRELATION_ID_HEADER] = correlation_id
        elif self.config.default_headers and CORRELATION_ID_HEADER in self.config.default_headers:
            headers[CORRELATION_ID_HEADER] = self.config.default_headers[CORRELATION_ID_HEADER]

        # Add default headers from config
        if self.config.default_headers:
            for key, value in self.config.default_headers.items():
                if key not in headers:  # Don't override existing headers
                    headers[key] = value

        # Add additional headers
        if additional_headers:
            headers.update(additional_headers)

        return headers

    def _build_url(self, endpoint: str) -> str:
        """
        Build a full URL from the endpoint.

        Args:
            endpoint: API endpoint

        Returns:
            Full URL
        """
        # Handle both relative and absolute URLs
        if endpoint.startswith(('http://', 'https://')):
            return endpoint

        # Remove leading slash if present
        endpoint = endpoint.lstrip('/')

        return f"{self.base_url}/{endpoint}"

    def _map_exception(self, exception: Exception) -> Exception:
        """
        Map HTTP exceptions to service-specific exceptions.

        Args:
            exception: Original exception

        Returns:
            Mapped exception
        """
        if isinstance(exception, aiohttp.ClientResponseError):
            status = exception.status
            message = str(exception)

            if status == 401 or status == 403:
                return AuthenticationError(f"Authentication failed: {message}")
            elif status == 404:
                return ServiceError(f"Resource not found: {message}")
            elif status == 422:
                return DataValidationError(f"Validation error: {message}")
            elif status >= 500:
                return ServiceUnavailableError(f"Service error: {message}")

        if isinstance(exception, aiohttp.ClientConnectorError):
            return ServiceUnavailableError(f"Service connection error: {str(exception)}")

        if isinstance(exception, asyncio.TimeoutError) or isinstance(exception, TimeoutError):
            return ServiceTimeoutError(f"Service timeout: {str(exception)}")

        if isinstance(exception, CircuitBreakerOpen):
            return ServiceUnavailableError(f"Circuit breaker open: {str(exception)}")

        if isinstance(exception, RetryExhaustedException):
            return ServiceUnavailableError(f"Retry exhausted: {str(exception)}")

        if isinstance(exception, BulkheadFullException):
            return ServiceUnavailableError(f"Service overloaded: {str(exception)}")

        # Return the original exception if no mapping is found
        return exception

    def _log_request(self, method: str, url: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> None:
        """
        Log request details.

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            data: Request body
        """
        if not self.config.enable_request_logging:
            return

        log_data = {
            "method": method,
            "url": url,
            "service": self.config.service_name,
        }

        if params:
            log_data["params"] = params

        if data and self.config.log_request_body:
            log_data["body"] = data

        logger.info(f"Service request: {json.dumps(log_data)}")

    def _log_response(self, method: str, url: str, status: int, response_time: float, response_data: Any = None) -> None:
        """
        Log response details.

        Args:
            method: HTTP method
            url: Request URL
            status: Response status code
            response_time: Response time in milliseconds
            response_data: Response body
        """
        if not self.config.enable_request_logging:
            return

        log_data = {
            "method": method,
            "url": url,
            "status": status,
            "response_time_ms": response_time,
            "service": self.config.service_name,
        }

        if response_data is not None and self.config.log_response_body:
            try:
                log_data["body"] = response_data
            except Exception:
                log_data["body"] = "<non-serializable>"

        logger.info(f"Service response: {json.dumps(log_data)}")

    def _record_metrics(self, method: str, endpoint: str, status: int, response_time: float) -> None:
        """
        Record metrics for the request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            status: Response status code
            response_time: Response time in milliseconds
        """
        if not self.config.enable_metrics:
            return

        # This is a placeholder for actual metrics implementation
        # In a real implementation, this would send metrics to a metrics system
        # like Prometheus, StatsD, or a custom metrics collector
        logger.debug(
            f"METRIC: service_request "
            f"method={method} "
            f"endpoint={endpoint} "
            f"service={self.config.service_name} "
            f"status={status} "
            f"response_time_ms={response_time:.2f}"
        )

    @retry_with_policy(
        exceptions=[
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ConnectionError,
            TimeoutError
        ]
    )
    @timeout_handler(timeout_seconds=10.0)  # Default timeout, will be overridden by instance timeout
    @bulkhead(name="service_client", max_concurrent=10, max_waiting=20)  # Default values, will be overridden
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseType:
        """
        Make an HTTP request with resilience patterns.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            headers: Additional headers
            timeout: Request timeout (overrides instance timeout)

        Returns:
            Response data

        Raises:
            ServiceError: For service-specific errors
            ServiceUnavailableError: For service availability issues
            ServiceTimeoutError: For service timeout issues
            DataValidationError: For data validation errors
            AuthenticationError: For authentication issues
        """
        await self._ensure_session()
        url = self._build_url(endpoint)
        request_headers = self._get_headers(headers)
        timeout_value = timeout or self.timeout

        # Log request
        self._log_request(method, url, params, data)

        start_time = time.time()
        status_code = None

        try:
            # Execute request with circuit breaker
            async def execute_request():
                async with self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=request_headers,
                    timeout=aiohttp.ClientTimeout(total=timeout_value)
                ) as response:
                    nonlocal status_code
                    status_code = response.status
                    response.raise_for_status()

                    # For empty responses, return None
                    if response.content_length == 0:
                        return None

                    # Try to parse as JSON, fall back to text
                    try:
                        return await response.json()
                    except ValueError:
                        return await response.text()

            # Execute with circuit breaker
            result = await self.circuit_breaker.execute(execute_request)

            # Calculate response time
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log response
            self._log_response(method, url, status_code or 200, response_time, result)

            # Record metrics
            self._record_metrics(method, endpoint, status_code or 200, response_time)

            return result

        except Exception as e:
            # Calculate response time even for errors
            response_time = (time.time() - start_time) * 1000

            # Log error
            logger.error(f"Error in {method} request to {url}: {str(e)}")

            # Record error metrics
            self._record_metrics(method, endpoint, status_code or 500, response_time)

            # Map and raise exception
            mapped_exception = self._map_exception(e)
            raise mapped_exception from e

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseType:
        """
        Make a GET request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout

        Returns:
            Response data
        """
        return await self._make_request("GET", endpoint, params=params, headers=headers, timeout=timeout)

    async def post(
        self,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseType:
        """
        Make a POST request.

        Args:
            endpoint: API endpoint
            data: Request body
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout

        Returns:
            Response data
        """
        return await self._make_request("POST", endpoint, params=params, data=data, headers=headers, timeout=timeout)

    async def put(
        self,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseType:
        """
        Make a PUT request.

        Args:
            endpoint: API endpoint
            data: Request body
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout

        Returns:
            Response data
        """
        return await self._make_request("PUT", endpoint, params=params, data=data, headers=headers, timeout=timeout)

    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseType:
        """
        Make a DELETE request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout

        Returns:
            Response data
        """
        return await self._make_request("DELETE", endpoint, params=params, headers=headers, timeout=timeout)

    # Synchronous versions of the HTTP methods for services that don't use async

    def sync_get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseType:
        """
        Make a synchronous GET request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout

        Returns:
            Response data
        """
        url = self._build_url(endpoint)
        request_headers = self._get_headers(headers)
        timeout_value = timeout or self.timeout

        # Log request
        self._log_request("GET", url, params)

        start_time = time.time()

        try:
            response = requests.get(
                url,
                params=params,
                headers=request_headers,
                timeout=timeout_value
            )
            response.raise_for_status()

            # Calculate response time
            response_time = (time.time() - start_time) * 1000

            # Parse response
            if response.content:
                result = response.json()
            else:
                result = None

            # Log response
            self._log_response("GET", url, response.status_code, response_time, result)

            # Record metrics
            self._record_metrics("GET", endpoint, response.status_code, response_time)

            return result

        except Exception as e:
            # Calculate response time even for errors
            response_time = (time.time() - start_time) * 1000

            # Log error
            logger.error(f"Error in GET request to {url}: {str(e)}")

            # Record error metrics
            status_code = getattr(e, 'response', {}).get('status_code', 500)
            self._record_metrics("GET", endpoint, status_code, response_time)

            # Map and raise exception
            mapped_exception = self._map_exception(e)
            raise mapped_exception from e

    def sync_post(
        self,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseType:
        """
        Make a synchronous POST request.

        Args:
            endpoint: API endpoint
            data: Request body
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout

        Returns:
            Response data
        """
        url = self._build_url(endpoint)
        request_headers = self._get_headers(headers)
        timeout_value = timeout or self.timeout

        # Log request
        self._log_request("POST", url, params, data)

        start_time = time.time()

        try:
            response = requests.post(
                url,
                json=data,
                params=params,
                headers=request_headers,
                timeout=timeout_value
            )
            response.raise_for_status()

            # Calculate response time
            response_time = (time.time() - start_time) * 1000

            # Parse response
            if response.content:
                result = response.json()
            else:
                result = None

            # Log response
            self._log_response("POST", url, response.status_code, response_time, result)

            # Record metrics
            self._record_metrics("POST", endpoint, response.status_code, response_time)

            return result

        except Exception as e:
            # Calculate response time even for errors
            response_time = (time.time() - start_time) * 1000

            # Log error
            logger.error(f"Error in POST request to {url}: {str(e)}")

            # Record error metrics
            status_code = getattr(e, 'response', {}).get('status_code', 500)
            self._record_metrics("POST", endpoint, status_code, response_time)

            # Map and raise exception
            mapped_exception = self._map_exception(e)
            raise mapped_exception from e

    def with_correlation_id(self, correlation_id: Optional[str] = None) -> 'BaseServiceClient':
        """
        Create a new client instance with the specified correlation ID.

        This method allows for easy propagation of correlation IDs across service calls.

        Args:
            correlation_id: Correlation ID to use (defaults to current context)

        Returns:
            New client instance with correlation ID set
        """
        # Use provided correlation ID or get from context
        if correlation_id is None:
            correlation_id = get_correlation_id()

        # Generate a new correlation ID if still not available
        if correlation_id is None:
            correlation_id = generate_correlation_id()

        # Create a copy of the configuration
        if hasattr(self.config, "model_dump"):
            # For Pydantic v2
            config_dict = self.config.model_dump()
        else:
            # For Pydantic v1 or dict
            config_dict = self.config.dict() if hasattr(self.config, "dict") else dict(self.config)

        # Ensure default_headers exists
        if "default_headers" not in config_dict:
            config_dict["default_headers"] = {}

        # Update headers with correlation ID
        config_dict["default_headers"][CORRELATION_ID_HEADER] = correlation_id

        # Create new client with updated configuration
        return self.__class__(config_dict)