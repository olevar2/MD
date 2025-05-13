"""
Resilient Service Client Module

This module provides a resilient service client for making HTTP requests to services.
It includes resilience patterns like circuit breaker, retry, bulkhead, and timeout.
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple, Type
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientResponse, ClientSession, ClientTimeout

from common_lib.errors.base_exceptions import (
    ServiceError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ConflictError,
    RateLimitError,
    ErrorCode
)
from common_lib.resilience import (
    ResilienceConfig,
    Resilience,
    circuit_breaker,
    retry_with_backoff,
    timeout,
    bulkhead,
    with_resilience
)


class ServiceClientConfig:
    """Configuration for a service client."""

    def __init__(
        self,
        service_name: str,
        base_url: str,
        timeout: float = 10.0,
        retry_count: int = 3,
        retry_backoff: float = 0.5,
        max_concurrent_requests: int = 20,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_recovery_time: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the service client configuration.

        Args:
            service_name: Name of the service
            base_url: Base URL of the service
            timeout: Timeout for requests in seconds
            retry_count: Number of retries for failed requests
            retry_backoff: Backoff factor for retries
            max_concurrent_requests: Maximum number of concurrent requests
            circuit_breaker_threshold: Number of failures before opening the circuit
            circuit_breaker_recovery_time: Time in seconds before trying to close the circuit
            headers: Default headers to include in all requests
        """
        self.service_name = service_name
        self.base_url = base_url
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_backoff = retry_backoff
        self.max_concurrent_requests = max_concurrent_requests
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_recovery_time = circuit_breaker_recovery_time
        self.headers = headers or {}


class ResilientServiceClient:
    """
    Resilient service client for making HTTP requests to services.

    This client includes resilience patterns like circuit breaker, retry,
    bulkhead, and timeout to provide robust service communication.
    """

    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the resilient service client.

        Args:
            config: Configuration for the service client
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Create session
        self.session = None

        # Initialize logger
        self.logger.info(f"Initialized ResilientServiceClient for {config.service_name} at {config.base_url}")

    async def __aenter__(self):
        """Enter async context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        await self.close()

    async def connect(self):
        """Connect to the service."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers=self.config.headers,
                timeout=ClientTimeout(total=self.config.timeout)
            )

    async def close(self):
        """Close the connection to the service."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None



    async def _handle_response(self, response: ClientResponse) -> Dict[str, Any]:
        """
        Handle a response from the service.

        Args:
            response: Response from the service

        Returns:
            Response data

        Raises:
            ServiceError: If the response indicates an error
        """
        # Check if the response is successful
        if 200 <= response.status < 300:
            # Parse the response
            try:
                if response.content_type == 'application/json':
                    return await response.json()
                else:
                    return {'content': await response.text()}
            except Exception as e:
                raise ServiceError(
                    f"Failed to parse response: {str(e)}",
                    error_code=ErrorCode.API_RESPONSE_ERROR,
                    cause=e
                )
        else:
            # Handle error response
            error_message = f"Service returned error: {response.status}"
            error_code = ErrorCode.API_RESPONSE_ERROR

            # Try to parse error details
            try:
                error_data = await response.json()
                if isinstance(error_data, dict) and 'error' in error_data:
                    error_message = error_data['error'].get('message', error_message)
                    error_code_str = error_data['error'].get('code', None)
                    if error_code_str:
                        try:
                            error_code = ErrorCode(error_code_str)
                        except ValueError:
                            pass
            except Exception:
                # If we can't parse the error, just use the status code
                pass

            # Map HTTP status code to error type
            if response.status == 400:
                raise ValidationError(error_message, error_code=error_code)
            elif response.status == 401:
                raise AuthenticationError(error_message, error_code=error_code)
            elif response.status == 403:
                raise AuthorizationError(error_message, error_code=error_code)
            elif response.status == 404:
                raise NotFoundError(error_message, error_code=error_code)
            elif response.status == 409:
                raise ConflictError(error_message, error_code=error_code)
            elif response.status == 429:
                raise RateLimitError(error_message, error_code=error_code)
            else:
                raise ServiceError(error_message, error_code=error_code)

    @with_resilience(
        # Circuit breaker config
        enable_circuit_breaker=True,
        failure_threshold=5,
        recovery_timeout=30.0,
        # Retry config
        enable_retry=True,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        backoff_factor=2.0,
        jitter=True,
        # Bulkhead config
        enable_bulkhead=True,
        max_concurrent=20,
        max_queue=40,
        # Timeout config
        enable_timeout=True,
        timeout_seconds=30.0,
        # General config
        expected_exceptions=[aiohttp.ClientError, asyncio.TimeoutError, ServiceError]
    )
    async def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the service.

        Args:
            method: HTTP method
            path: Path to request
            params: Query parameters
            json: JSON body
            data: Form data
            headers: Request headers
            operation: Name of the operation (for resilience)

        Returns:
            Response data

        Raises:
            ServiceError: If the request fails
        """
        # Ensure we have a session
        if self.session is None or self.session.closed:
            await self.connect()

        # Build the URL
        url = urljoin(self.config.base_url, path)

        # Determine the operation name for logging
        if operation is None:
            operation = f"{method.lower()}_{path.replace('/', '_')}"

        # Make the request
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                data=data,
                headers=headers
            ) as response:
                return await self._handle_response(response)
        except aiohttp.ClientError as e:
            raise ServiceError(
                f"Request failed: {str(e)}",
                error_code=ErrorCode.API_REQUEST_ERROR,
                cause=e
            )
        except asyncio.TimeoutError as e:
            raise ServiceError(
                f"Request timed out",
                error_code=ErrorCode.API_TIMEOUT_ERROR,
                cause=e
            )

    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make a GET request to the service.

        Args:
            path: Path to request
            params: Query parameters
            headers: Request headers
            operation: Name of the operation (for resilience)

        Returns:
            Response data

        Raises:
            ServiceError: If the request fails
        """
        return await self.request(
            method="GET",
            path=path,
            params=params,
            headers=headers,
            operation=operation
        )

    async def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make a POST request to the service.

        Args:
            path: Path to request
            json: JSON body
            data: Form data
            params: Query parameters
            headers: Request headers
            operation: Name of the operation (for resilience)

        Returns:
            Response data

        Raises:
            ServiceError: If the request fails
        """
        return await self.request(
            method="POST",
            path=path,
            json=json,
            data=data,
            params=params,
            headers=headers,
            operation=operation
        )

    async def put(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make a PUT request to the service.

        Args:
            path: Path to request
            json: JSON body
            data: Form data
            params: Query parameters
            headers: Request headers
            operation: Name of the operation (for resilience)

        Returns:
            Response data

        Raises:
            ServiceError: If the request fails
        """
        return await self.request(
            method="PUT",
            path=path,
            json=json,
            data=data,
            params=params,
            headers=headers,
            operation=operation
        )

    async def delete(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make a DELETE request to the service.

        Args:
            path: Path to request
            params: Query parameters
            headers: Request headers
            operation: Name of the operation (for resilience)

        Returns:
            Response data

        Raises:
            ServiceError: If the request fails
        """
        return await self.request(
            method="DELETE",
            path=path,
            params=params,
            headers=headers,
            operation=operation
        )

    async def patch(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make a PATCH request to the service.

        Args:
            path: Path to request
            json: JSON body
            data: Form data
            params: Query parameters
            headers: Request headers
            operation: Name of the operation (for resilience)

        Returns:
            Response data

        Raises:
            ServiceError: If the request fails
        """
        return await self.request(
            method="PATCH",
            path=path,
            json=json,
            data=data,
            params=params,
            headers=headers,
            operation=operation
        )
