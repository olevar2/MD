#!/usr/bin/env python3
"""
Forex Trading Platform Error Handling Standardization

This script standardizes error handling across the forex trading platform by:
1. Creating custom exception classes in common-lib
2. Implementing consistent error handlers
3. Replacing standard exceptions with custom exceptions
4. Adding correlation IDs to error responses

Usage:
python standardize_error_handling.py [--project-root <project_root>] [--service <service_name>]
"""

import os
import sys
import re
import ast
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"

# Custom exception template
CUSTOM_EXCEPTIONS_TEMPLATE = """\"\"\"
Custom exceptions for the forex trading platform.
\"\"\"

from typing import Dict, Any, Optional


class ForexBaseException(Exception):
    \"\"\"Base exception for all forex platform exceptions.\"\"\"

    def __init__(self, message: str, correlation_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        \"\"\"
        Initialize the exception.

        Args:
            message: Error message
            correlation_id: Correlation ID for tracking the error
            details: Additional error details
        \"\"\"
        self.message = message
        self.correlation_id = correlation_id
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        \"\"\"
        Convert the exception to a dictionary.

        Returns:
            Dictionary representation of the exception
        \"\"\"
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'correlation_id': self.correlation_id,
            'details': self.details
        }


# Service exceptions
class ServiceException(ForexBaseException):
    \"\"\"Exception raised for service-related errors.\"\"\"
    pass


class ServiceUnavailableException(ServiceException):
    \"\"\"Exception raised when a service is unavailable.\"\"\"
    pass


class ServiceTimeoutException(ServiceException):
    \"\"\"Exception raised when a service request times out.\"\"\"
    pass


class ServiceAuthenticationException(ServiceException):
    \"\"\"Exception raised when service authentication fails.\"\"\"
    pass


class ServiceAuthorizationException(ServiceException):
    \"\"\"Exception raised when service authorization fails.\"\"\"
    pass


# Data exceptions
class DataException(ForexBaseException):
    \"\"\"Exception raised for data-related errors.\"\"\"
    pass


class DataValidationException(DataException):
    \"\"\"Exception raised when data validation fails.\"\"\"
    pass


class DataNotFoundException(DataException):
    \"\"\"Exception raised when data is not found.\"\"\"
    pass


class DataDuplicateException(DataException):
    \"\"\"Exception raised when duplicate data is detected.\"\"\"
    pass


class DataCorruptionException(DataException):
    \"\"\"Exception raised when data corruption is detected.\"\"\"
    pass


# Trading exceptions
class TradingException(ForexBaseException):
    \"\"\"Exception raised for trading-related errors.\"\"\"
    pass


class OrderExecutionException(TradingException):
    \"\"\"Exception raised when order execution fails.\"\"\"
    pass


class OrderValidationException(TradingException):
    \"\"\"Exception raised when order validation fails.\"\"\"
    pass


class InsufficientFundsException(TradingException):
    \"\"\"Exception raised when there are insufficient funds for a trade.\"\"\"
    pass


class MarketClosedException(TradingException):
    \"\"\"Exception raised when the market is closed.\"\"\"
    pass


# Configuration exceptions
class ConfigurationException(ForexBaseException):
    \"\"\"Exception raised for configuration-related errors.\"\"\"
    pass


class ConfigurationNotFoundException(ConfigurationException):
    \"\"\"Exception raised when a configuration is not found.\"\"\"
    pass


class ConfigurationValidationException(ConfigurationException):
    \"\"\"Exception raised when configuration validation fails.\"\"\"
    pass


# Security exceptions
class SecurityException(ForexBaseException):
    \"\"\"Exception raised for security-related errors.\"\"\"
    pass


class AuthenticationException(SecurityException):
    \"\"\"Exception raised when authentication fails.\"\"\"
    pass


class AuthorizationException(SecurityException):
    \"\"\"Exception raised when authorization fails.\"\"\"
    pass


class RateLimitException(SecurityException):
    \"\"\"Exception raised when rate limit is exceeded.\"\"\"
    pass


# Infrastructure exceptions
class InfrastructureException(ForexBaseException):
    \"\"\"Exception raised for infrastructure-related errors.\"\"\"
    pass


class DatabaseException(InfrastructureException):
    \"\"\"Exception raised for database-related errors.\"\"\"
    pass


class NetworkException(InfrastructureException):
    \"\"\"Exception raised for network-related errors.\"\"\"
    pass


class ResourceExhaustionException(InfrastructureException):
    \"\"\"Exception raised when a resource is exhausted.\"\"\"
    pass
"""

# Error handler template for FastAPI
FASTAPI_ERROR_HANDLER_TEMPLATE = """\"\"\"
Error handlers for FastAPI.
\"\"\"

from typing import Dict, Any, Callable
import uuid
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from common_lib.exceptions import (
    ForexBaseException,
    ServiceException,
    DataException,
    TradingException,
    ConfigurationException,
    SecurityException,
    InfrastructureException
)

logger = logging.getLogger(__name__)


def generate_correlation_id() -> str:
    \"\"\"
    Generate a correlation ID for error tracking.

    Returns:
        Correlation ID
    \"\"\"
    return str(uuid.uuid4())


def get_correlation_id(request: Request) -> str:
    \"\"\"
    Get the correlation ID from the request or generate a new one.

    Args:
        request: FastAPI request

    Returns:
        Correlation ID
    \"\"\"
    if hasattr(request.state, 'correlation_id') and request.state.correlation_id:
        return request.state.correlation_id

    # Check if it's in the headers
    correlation_id = request.headers.get('X-Correlation-ID')
    if correlation_id:
        return correlation_id

    # Generate a new one
    return generate_correlation_id()


def setup_error_handlers(app):
    \"\"\"
    Set up error handlers for the FastAPI application.

    Args:
        app: FastAPI application
    \"\"\"

    @app.exception_handler(ForexBaseException)
    async def forex_exception_handler(request: Request, exc: ForexBaseException):
        \"\"\"
        Handle ForexBaseException and its subclasses.

        Args:
            request: FastAPI request
            exc: Exception

        Returns:
            JSON response with error details
        \"\"\"
        correlation_id = exc.correlation_id or get_correlation_id(request)

        # Log the error
        logger.error(
            f"ForexBaseException: {exc.message}",
            extra={
                'correlation_id': correlation_id,
                'exception_type': exc.__class__.__name__,
                'details': exc.details
            }
        )

        # Determine status code based on exception type
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        if isinstance(exc, ServiceException):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(exc, DataException):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, TradingException):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, ConfigurationException):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif isinstance(exc, SecurityException):
            status_code = status.HTTP_401_UNAUTHORIZED
        elif isinstance(exc, InfrastructureException):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        # Create response
        response = {
            'error': exc.__class__.__name__,
            'message': exc.message,
            'correlation_id': correlation_id
        }

        if exc.details:
            response['details'] = exc.details

        return JSONResponse(
            status_code=status_code,
            content=response,
            headers={'X-Correlation-ID': correlation_id}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        \"\"\"
        Handle FastAPI validation errors.

        Args:
            request: FastAPI request
            exc: Exception

        Returns:
            JSON response with error details
        \"\"\"
        correlation_id = get_correlation_id(request)

        # Log the error
        logger.error(
            f"RequestValidationError: {str(exc)}",
            extra={
                'correlation_id': correlation_id,
                'exception_type': 'RequestValidationError',
                'details': exc.errors()
            }
        )

        # Create response
        response = {
            'error': 'RequestValidationError',
            'message': 'Request validation failed',
            'correlation_id': correlation_id,
            'details': exc.errors()
        }

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response,
            headers={'X-Correlation-ID': correlation_id}
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        \"\"\"
        Handle Starlette HTTP exceptions.

        Args:
            request: FastAPI request
            exc: Exception

        Returns:
            JSON response with error details
        \"\"\"
        correlation_id = get_correlation_id(request)

        # Log the error
        logger.error(
            f"HTTPException: {exc.detail}",
            extra={
                'correlation_id': correlation_id,
                'exception_type': 'HTTPException',
                'status_code': exc.status_code
            }
        )

        # Create response
        response = {
            'error': 'HTTPException',
            'message': exc.detail,
            'correlation_id': correlation_id
        }

        return JSONResponse(
            status_code=exc.status_code,
            content=response,
            headers={'X-Correlation-ID': correlation_id}
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        \"\"\"
        Handle all other exceptions.

        Args:
            request: FastAPI request
            exc: Exception

        Returns:
            JSON response with error details
        \"\"\"
        correlation_id = get_correlation_id(request)

        # Log the error
        logger.error(
            f"Unhandled exception: {str(exc)}",
            extra={
                'correlation_id': correlation_id,
                'exception_type': exc.__class__.__name__
            },
            exc_info=True
        )

        # Create response
        response = {
            'error': 'InternalServerError',
            'message': 'An unexpected error occurred',
            'correlation_id': correlation_id
        }

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response,
            headers={'X-Correlation-ID': correlation_id}
        )
"""

# Middleware template for correlation ID
CORRELATION_ID_MIDDLEWARE_TEMPLATE = """\"\"\"
Middleware for correlation ID handling.
\"\"\"

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    \"\"\"
    Middleware for handling correlation IDs.

    This middleware ensures that every request has a correlation ID for tracing.
    If the request already has a correlation ID in the headers, it will be used.
    Otherwise, a new correlation ID will be generated.
    \"\"\"

    async def dispatch(self, request: Request, call_next):
        \"\"\"
        Process the request and add correlation ID.

        Args:
            request: FastAPI request
            call_next: Next middleware or endpoint

        Returns:
            Response
        \"\"\"
        # Get or generate correlation ID
        correlation_id = request.headers.get('X-Correlation-ID')
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Store in request state
        request.state.correlation_id = correlation_id

        # Process the request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers['X-Correlation-ID'] = correlation_id

        return response
"""

class ErrorHandlingStandardizer:
    """Standardizes error handling across the forex trading platform."""

    def __init__(self, project_root: str, service_name: Optional[str] = None):
        """
        Initialize the standardizer.

        Args:
            project_root: Root directory of the project
            service_name: Name of the service to standardize (None for all services)
        """
        self.project_root = project_root
        self.service_name = service_name
        self.services = []
        self.implementations = []

    def identify_services(self) -> None:
        """Identify services in the project based on directory structure."""
        logger.info("Identifying services...")

        # Look for service directories
        for item in os.listdir(self.project_root):
            item_path = os.path.join(self.project_root, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it's likely a service
                if (
                    item.endswith('-service') or
                    item.endswith('_service') or
                    item.endswith('-api') or
                    item.endswith('-engine') or
                    'service' in item.lower() or
                    'api' in item.lower()
                ):
                    self.services.append(item)

        logger.info(f"Identified {len(self.services)} services")

    def create_common_lib_exceptions(self) -> None:
        """Create custom exception classes in common-lib."""
        logger.info("Creating custom exception classes in common-lib...")

        # Create common-lib directory if it doesn't exist
        common_lib_path = os.path.join(self.project_root, 'common-lib')
        os.makedirs(common_lib_path, exist_ok=True)

        # Create exceptions.py file
        exceptions_path = os.path.join(common_lib_path, 'exceptions.py')
        with open(exceptions_path, 'w', encoding='utf-8') as f:
            f.write(CUSTOM_EXCEPTIONS_TEMPLATE)

        self.implementations.append(f"Created custom exceptions: {exceptions_path}")

    def implement_error_handlers(self, service_name: str) -> None:
        """
        Implement error handlers for a service.

        Args:
            service_name: Name of the service
        """
        logger.info(f"Implementing error handlers for {service_name}...")

        service_path = os.path.join(self.project_root, service_name)

        # Create api directory if it doesn't exist
        api_path = os.path.join(service_path, 'api')
        os.makedirs(api_path, exist_ok=True)

        # Create error_handlers.py file
        error_handlers_path = os.path.join(api_path, 'error_handlers.py')
        with open(error_handlers_path, 'w', encoding='utf-8') as f:
            f.write(FASTAPI_ERROR_HANDLER_TEMPLATE)

        self.implementations.append(f"Created error handlers: {error_handlers_path}")

        # Create middleware directory if it doesn't exist
        middleware_path = os.path.join(service_path, 'api', 'middleware')
        os.makedirs(middleware_path, exist_ok=True)

        # Create correlation_id.py file
        correlation_id_path = os.path.join(middleware_path, 'correlation_id.py')
        with open(correlation_id_path, 'w', encoding='utf-8') as f:
            f.write(CORRELATION_ID_MIDDLEWARE_TEMPLATE)

        self.implementations.append(f"Created correlation ID middleware: {correlation_id_path}")

        # Create __init__.py file if it doesn't exist
        init_file = os.path.join(middleware_path, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write('"""\nMiddleware for the service.\n"""\n')
            self.implementations.append(f"Created file: {init_file}")

    def standardize_error_handling(self) -> List[str]:
        """
        Standardize error handling across the forex trading platform.

        Returns:
            List of implementations
        """
        logger.info("Starting error handling standardization...")

        # Identify services
        self.identify_services()

        if not self.services:
            logger.info("No services found")
            return []

        # Filter services if a specific service was specified
        if self.service_name:
            if self.service_name in self.services:
                services_to_standardize = [self.service_name]
            else:
                logger.error(f"Service not found: {self.service_name}")
                return []
        else:
            services_to_standardize = self.services

        # Create custom exception classes in common-lib
        self.create_common_lib_exceptions()

        # Implement error handlers for each service
        for service in services_to_standardize:
            self.implement_error_handlers(service)

        logger.info("Error handling standardization complete")
        return self.implementations

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Standardize error handling")
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--service",
        help="Name of the service to standardize (default: all services)"
    )
    args = parser.parse_args()

    # Standardize error handling
    standardizer = ErrorHandlingStandardizer(args.project_root, args.service)
    implementations = standardizer.standardize_error_handling()

    # Print summary
    print("\nError Handling Standardization Summary:")
    print(f"- Applied {len(implementations)} implementations")

    if implementations:
        print("\nImplementations:")
        for i, implementation in enumerate(implementations):
            print(f"  {i+1}. {implementation}")

    # Save results to file
    output_dir = os.path.join(args.project_root, 'tools', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'error_handling_changes.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_implementations': len(implementations),
            'implementations': implementations
        }, f, indent=2)

    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
