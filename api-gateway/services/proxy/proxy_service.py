"""
Proxy Service

This module provides a service for proxying requests to backend services.
"""

import logging
import httpx
import asyncio
from typing import Dict, Any, Optional, List, Union

from fastapi import Request, Response
from starlette.responses import JSONResponse, StreamingResponse

from common_lib.config.config_manager import ConfigManager
from common_lib.exceptions import ServiceUnavailableError, ServiceTimeoutError
from common_lib.resilience.circuit_breaker import CircuitBreaker
from common_lib.resilience.retry import retry
from common_lib.resilience.timeout import timeout

from ...core.response.standard_response import create_error_response


class ProxyService:
    """
    Service for proxying requests to backend services.
    """

    def __init__(self):
        """
        Initialize the proxy service.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config_manager = ConfigManager()

        # Get configuration
        try:
            service_specific = self.config_manager.get_service_specific_config()
            if hasattr(service_specific, "services"):
                self.services_config = getattr(service_specific, "services")
            else:
                self.services_config = {}
        except Exception as e:
            self.logger.warning(f"Error getting services configuration: {str(e)}")
            self.services_config = {}

        # Create HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)

        # Create circuit breakers for each service
        self.circuit_breakers = {}
        for service_name, service_config in self.services_config.items():
            self.circuit_breakers[service_name] = CircuitBreaker(
                name=service_name,
                failure_threshold=service_config.get("failure_threshold", 5),
                recovery_timeout=service_config.get("recovery_timeout", 30),
                expected_exceptions=[ServiceUnavailableError, ServiceTimeoutError]
            )

    async def close(self):
        """
        Close the proxy service.
        """
        await self.client.aclose()

    def _get_service_url(self, service_name: str) -> str:
        """
        Get the URL for a service.

        Args:
            service_name: Service name

        Returns:
            Service URL

        Raises:
            ValueError: If the service is not configured
        """
        if service_name not in self.services_config:
            raise ValueError(f"Service {service_name} is not configured")

        return self.services_config[service_name].get("url", "")

    def _get_service_timeout(self, service_name: str) -> float:
        """
        Get the timeout for a service.

        Args:
            service_name: Service name

        Returns:
            Service timeout in seconds
        """
        if service_name not in self.services_config:
            return 30.0

        return self.services_config[service_name].get("timeout", 30.0)

    def _get_service_retry_config(self, service_name: str) -> Dict[str, Any]:
        """
        Get the retry configuration for a service.

        Args:
            service_name: Service name

        Returns:
            Retry configuration
        """
        if service_name not in self.services_config:
            return {
                "retries": 3,
                "delay": 1.0,
                "backoff": 2.0
            }

        retry_config = self.services_config[service_name].get("retry", {})
        return {
            "retries": retry_config.get("retries", 3),
            "delay": retry_config.get("delay", 1.0),
            "backoff": retry_config.get("backoff", 2.0)
        }

    def _get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """
        Get the circuit breaker for a service.

        Args:
            service_name: Service name

        Returns:
            Circuit breaker
        """
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                name=service_name,
                failure_threshold=5,
                recovery_timeout=30,
                expected_exceptions=[ServiceUnavailableError, ServiceTimeoutError]
            )

        return self.circuit_breakers[service_name]

    def _prepare_headers(self, request: Request) -> Dict[str, str]:
        """
        Prepare headers for the proxied request.

        Args:
            request: Original request

        Returns:
            Headers for the proxied request
        """
        # Copy headers from the original request
        headers = dict(request.headers)

        # Remove headers that should not be forwarded
        headers.pop("host", None)
        headers.pop("connection", None)
        headers.pop("content-length", None)

        # Add X-Forwarded headers
        headers["X-Forwarded-For"] = request.client.host if request.client else "unknown"
        headers["X-Forwarded-Proto"] = request.url.scheme
        headers["X-Forwarded-Host"] = request.headers.get("host", "")

        return headers

    async def proxy_request(
        self,
        request: Request,
        service_name: str,
        path: str,
        correlation_id: str,
        request_id: str
    ) -> Response:
        """
        Proxy a request to a backend service.

        Args:
            request: Original request
            service_name: Service name
            path: Request path
            correlation_id: Correlation ID
            request_id: Request ID

        Returns:
            Response from the backend service

        Raises:
            ServiceUnavailableError: If the service is unavailable
            ServiceTimeoutError: If the service times out
        """
        # Get service URL
        try:
            service_url = self._get_service_url(service_name)
        except ValueError as e:
            self.logger.error(f"Error getting service URL: {str(e)}")
            return JSONResponse(
                status_code=404,
                content=create_error_response(
                    code="SERVICE_NOT_FOUND",
                    message=f"Service {service_name} not found",
                    correlation_id=correlation_id,
                    request_id=request_id
                ).dict()
            )

        # Get service timeout
        service_timeout = self._get_service_timeout(service_name)

        # Get retry configuration
        retry_config = self._get_service_retry_config(service_name)

        # Get circuit breaker
        circuit_breaker = self._get_circuit_breaker(service_name)

        # Prepare URL
        url = f"{service_url}{path}"

        # Prepare headers
        headers = self._prepare_headers(request)

        # Prepare request
        method = request.method
        params = dict(request.query_params)
        body = await request.body()

        # Define request function
        @circuit_breaker
        @retry(
            retries=retry_config["retries"],
            delay=retry_config["delay"],
            backoff=retry_config["backoff"],
            exceptions=[httpx.HTTPError, httpx.TimeoutException]
        )
        @timeout(service_timeout)
        async def send_request():
            """
            Send the request to the backend service.

            Returns:
                Response from the backend service

            Raises:
                ServiceUnavailableError: If the service is unavailable
                ServiceTimeoutError: If the service times out
            """
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    content=body,
                    headers=headers
                )
                return response
            except httpx.TimeoutException as e:
                raise ServiceTimeoutError(f"Service {service_name} timed out") from e
            except httpx.HTTPError as e:
                raise ServiceUnavailableError(f"Service {service_name} is unavailable") from e

        try:
            # Send the request
            response = await send_request()

            # Create response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        except ServiceTimeoutError as e:
            self.logger.error(f"Service timeout: {str(e)}")
            return JSONResponse(
                status_code=504,
                content=create_error_response(
                    code="SERVICE_TIMEOUT",
                    message=f"Service {service_name} timed out",
                    correlation_id=correlation_id,
                    request_id=request_id
                ).dict()
            )
        except ServiceUnavailableError as e:
            self.logger.error(f"Service unavailable: {str(e)}")
            return JSONResponse(
                status_code=503,
                content=create_error_response(
                    code="SERVICE_UNAVAILABLE",
                    message=f"Service {service_name} is unavailable",
                    correlation_id=correlation_id,
                    request_id=request_id
                ).dict()
            )
        except Exception as e:
            self.logger.error(f"Error proxying request: {str(e)}")
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    code="INTERNAL_SERVER_ERROR",
                    message="Internal server error",
                    correlation_id=correlation_id,
                    request_id=request_id
                ).dict()
            )