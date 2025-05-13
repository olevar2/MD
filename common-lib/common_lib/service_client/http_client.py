"""
HTTP service client classes for the forex trading platform.

This module provides HTTP service client implementations used across the platform.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import aiohttp
import requests
from requests.exceptions import RequestException

from common_lib.errors import (
    APIError,
    ForexTradingError,
    ServiceUnavailableError,
    ThirdPartyServiceError,
    TimeoutError,
)
from common_lib.service_client.base_client import (
    AsyncBaseServiceClient,
    BaseServiceClient,
    ServiceClientConfig,
)


class HTTPServiceClient(BaseServiceClient[Dict[str, Any], Dict[str, Any]]):
    """
    HTTP service client for synchronous requests.
    
    This class provides a standardized way to make HTTP requests to other services
    with built-in retry, circuit breaker, and timeout functionality.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the HTTP service client.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        super().__init__(config, logger)
        
        # Set default retry configuration for HTTP requests
        if not self.config.retry_config.retry_on_exceptions:
            self.config.retry_config.retry_on_exceptions = [RequestException]
        
        if not self.config.retry_config.retry_on_status_codes:
            self.config.retry_config.retry_on_status_codes = [429, 500, 502, 503, 504]
    
    def send_request(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send an HTTP request to the service.
        
        Args:
            request: Request parameters
                - method: HTTP method (GET, POST, etc.)
                - path: URL path (will be appended to base_url)
                - params: Query parameters
                - headers: HTTP headers
                - json: JSON body
                - data: Form data
                - files: Files to upload
                - auth: Authentication tuple or object
                - timeout: Request timeout in seconds
                
        Returns:
            Response from the service
            
        Raises:
            ForexTradingError: If the request fails
        """
        # Check circuit breaker
        self._check_circuit_breaker()
        
        # Extract request parameters
        method = request.get("method", "GET")
        path = request.get("path", "")
        params = request.get("params")
        headers = {**self.config.headers, **(request.get("headers") or {})}
        json_data = request.get("json")
        data = request.get("data")
        files = request.get("files")
        auth = request.get("auth")
        timeout = request.get("timeout")
        
        # Build URL
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        
        # Set timeout
        if timeout is None:
            timeout = (
                self.config.timeout_config.connect_timeout_ms / 1000,
                self.config.timeout_config.read_timeout_ms / 1000
            )
        
        # Initialize retry counter
        retry_count = 0
        
        while True:
            try:
                # Log request
                self.logger.debug(
                    f"Sending {method} request to {url}",
                    extra={
                        "service": self.config.service_name,
                        "method": method,
                        "url": url,
                        "retry_count": retry_count
                    }
                )
                
                # Send request
                start_time = time.time()
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    json=json_data,
                    data=data,
                    files=files,
                    auth=auth,
                    timeout=timeout
                )
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Log response
                self.logger.debug(
                    f"Received response from {url}: {response.status_code}",
                    extra={
                        "service": self.config.service_name,
                        "method": method,
                        "url": url,
                        "status_code": response.status_code,
                        "elapsed_ms": elapsed_ms
                    }
                )
                
                # Check if we should retry based on status code
                if self._should_retry(Exception(), response.status_code, retry_count):
                    retry_count += 1
                    backoff_time = self._get_backoff_time(retry_count)
                    self.logger.warning(
                        f"Retrying request to {url} after {backoff_time:.2f}s (status code: {response.status_code})",
                        extra={
                            "service": self.config.service_name,
                            "method": method,
                            "url": url,
                            "status_code": response.status_code,
                            "retry_count": retry_count,
                            "backoff_time": backoff_time
                        }
                    )
                    time.sleep(backoff_time)
                    continue
                
                # Raise exception for error status codes
                response.raise_for_status()
                
                # Parse response
                if response.headers.get("content-type") == "application/json":
                    result = response.json()
                else:
                    result = {"content": response.text, "status_code": response.status_code}
                
                # Update circuit breaker
                self._update_circuit_breaker(True)
                
                return result
            
            except Exception as e:
                # Log error
                self.logger.error(
                    f"Error sending request to {url}: {str(e)}",
                    extra={
                        "service": self.config.service_name,
                        "method": method,
                        "url": url,
                        "error": str(e),
                        "retry_count": retry_count
                    },
                    exc_info=True
                )
                
                # Check if we should retry
                status_code = response.status_code if "response" in locals() else None
                if self._should_retry(e, status_code, retry_count):
                    retry_count += 1
                    backoff_time = self._get_backoff_time(retry_count)
                    self.logger.warning(
                        f"Retrying request to {url} after {backoff_time:.2f}s",
                        extra={
                            "service": self.config.service_name,
                            "method": method,
                            "url": url,
                            "error": str(e),
                            "retry_count": retry_count,
                            "backoff_time": backoff_time
                        }
                    )
                    time.sleep(backoff_time)
                    continue
                
                # Update circuit breaker
                self._update_circuit_breaker(False)
                
                # Convert exception to ForexTradingError
                if isinstance(e, requests.Timeout):
                    raise TimeoutError(
                        message=f"Request to {self.config.service_name} timed out",
                        operation=f"{method} {url}",
                        timeout_seconds=timeout[1] if isinstance(timeout, tuple) else timeout
                    ) from e
                elif isinstance(e, requests.HTTPError):
                    status_code = e.response.status_code
                    try:
                        error_data = e.response.json()
                    except (ValueError, json.JSONDecodeError):
                        error_data = {"message": e.response.text}
                    
                    raise APIError(
                        message=error_data.get("message", str(e)),
                        endpoint=url,
                        method=method,
                        http_status=status_code,
                        details=error_data
                    ) from e
                elif isinstance(e, RequestException):
                    raise ThirdPartyServiceError(
                        message=f"Error communicating with {self.config.service_name}: {str(e)}",
                        service_name=self.config.service_name
                    ) from e
                else:
                    raise ThirdPartyServiceError(
                        message=f"Unexpected error communicating with {self.config.service_name}: {str(e)}",
                        service_name=self.config.service_name
                    ) from e


class AsyncHTTPServiceClient(AsyncBaseServiceClient[Dict[str, Any], Dict[str, Any]]):
    """
    HTTP service client for asynchronous requests.
    
    This class provides a standardized way to make asynchronous HTTP requests to other services
    with built-in retry, circuit breaker, and timeout functionality.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the async HTTP service client.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        super().__init__(config, logger)
        
        # Set default retry configuration for HTTP requests
        if not self.config.retry_config.retry_on_exceptions:
            self.config.retry_config.retry_on_exceptions = [aiohttp.ClientError]
        
        if not self.config.retry_config.retry_on_status_codes:
            self.config.retry_config.retry_on_status_codes = [429, 500, 502, 503, 504]
        
        # Create session
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session.
        
        Returns:
            aiohttp.ClientSession: Session to use for requests
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.config.headers,
                timeout=aiohttp.ClientTimeout(
                    total=self.config.timeout_config.total_timeout_ms / 1000,
                    connect=self.config.timeout_config.connect_timeout_ms / 1000,
                    sock_read=self.config.timeout_config.read_timeout_ms / 1000
                )
            )
        return self._session
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def send_request(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send an HTTP request to the service.
        
        Args:
            request: Request parameters
                - method: HTTP method (GET, POST, etc.)
                - path: URL path (will be appended to base_url)
                - params: Query parameters
                - headers: HTTP headers
                - json: JSON body
                - data: Form data
                - timeout: Request timeout in seconds
                
        Returns:
            Response from the service
            
        Raises:
            ForexTradingError: If the request fails
        """
        # Check circuit breaker
        self._check_circuit_breaker()
        
        # Extract request parameters
        method = request.get("method", "GET")
        path = request.get("path", "")
        params = request.get("params")
        headers = {**self.config.headers, **(request.get("headers") or {})}
        json_data = request.get("json")
        data = request.get("data")
        timeout = request.get("timeout")
        
        # Build URL
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        
        # Get session
        session = await self._get_session()
        
        # Initialize retry counter
        retry_count = 0
        
        while True:
            try:
                # Log request
                self.logger.debug(
                    f"Sending {method} request to {url}",
                    extra={
                        "service": self.config.service_name,
                        "method": method,
                        "url": url,
                        "retry_count": retry_count
                    }
                )
                
                # Send request
                start_time = time.time()
                async with session.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    json=json_data,
                    data=data,
                    timeout=timeout
                ) as response:
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Log response
                    self.logger.debug(
                        f"Received response from {url}: {response.status}",
                        extra={
                            "service": self.config.service_name,
                            "method": method,
                            "url": url,
                            "status_code": response.status,
                            "elapsed_ms": elapsed_ms
                        }
                    )
                    
                    # Check if we should retry based on status code
                    if self._should_retry(Exception(), response.status, retry_count):
                        retry_count += 1
                        backoff_time = self._get_backoff_time(retry_count)
                        self.logger.warning(
                            f"Retrying request to {url} after {backoff_time:.2f}s (status code: {response.status})",
                            extra={
                                "service": self.config.service_name,
                                "method": method,
                                "url": url,
                                "status_code": response.status,
                                "retry_count": retry_count,
                                "backoff_time": backoff_time
                            }
                        )
                        await asyncio.sleep(backoff_time)
                        continue
                    
                    # Raise exception for error status codes
                    response.raise_for_status()
                    
                    # Parse response
                    if response.headers.get("content-type") == "application/json":
                        result = await response.json()
                    else:
                        result = {"content": await response.text(), "status_code": response.status}
                    
                    # Update circuit breaker
                    self._update_circuit_breaker(True)
                    
                    return result
            
            except Exception as e:
                # Log error
                self.logger.error(
                    f"Error sending request to {url}: {str(e)}",
                    extra={
                        "service": self.config.service_name,
                        "method": method,
                        "url": url,
                        "error": str(e),
                        "retry_count": retry_count
                    },
                    exc_info=True
                )
                
                # Check if we should retry
                status_code = response.status if "response" in locals() else None
                if self._should_retry(e, status_code, retry_count):
                    retry_count += 1
                    backoff_time = self._get_backoff_time(retry_count)
                    self.logger.warning(
                        f"Retrying request to {url} after {backoff_time:.2f}s",
                        extra={
                            "service": self.config.service_name,
                            "method": method,
                            "url": url,
                            "error": str(e),
                            "retry_count": retry_count,
                            "backoff_time": backoff_time
                        }
                    )
                    await asyncio.sleep(backoff_time)
                    continue
                
                # Update circuit breaker
                self._update_circuit_breaker(False)
                
                # Convert exception to ForexTradingError
                if isinstance(e, asyncio.TimeoutError):
                    raise TimeoutError(
                        message=f"Request to {self.config.service_name} timed out",
                        operation=f"{method} {url}",
                        timeout_seconds=timeout
                    ) from e
                elif isinstance(e, aiohttp.ClientResponseError):
                    status_code = e.status
                    message = e.message
                    
                    raise APIError(
                        message=message,
                        endpoint=url,
                        method=method,
                        http_status=status_code
                    ) from e
                elif isinstance(e, aiohttp.ClientError):
                    raise ThirdPartyServiceError(
                        message=f"Error communicating with {self.config.service_name}: {str(e)}",
                        service_name=self.config.service_name
                    ) from e
                else:
                    raise ThirdPartyServiceError(
                        message=f"Unexpected error communicating with {self.config.service_name}: {str(e)}",
                        service_name=self.config.service_name
                    ) from e