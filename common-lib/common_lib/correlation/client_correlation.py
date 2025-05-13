"""
Client Correlation Module

This module provides utilities for propagating correlation IDs in service clients.
"""

import logging
import functools
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, Union

from common_lib.correlation.correlation_id import (
    get_correlation_id,
    generate_correlation_id,
    CORRELATION_ID_HEADER
)

logger = logging.getLogger(__name__)

# Type variable for client methods
T = TypeVar('T')


def add_correlation_id_to_headers(
    headers: Dict[str, str],
    correlation_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Add correlation ID to request headers.

    Args:
        headers: Headers dictionary
        correlation_id: Optional correlation ID to use (defaults to current context)

    Returns:
        Updated headers dictionary
    """
    # Make a copy of the headers
    updated_headers = headers.copy() if headers else {}

    # Use provided correlation ID or get from context
    if correlation_id is None:
        correlation_id = get_correlation_id()

    # Generate a new correlation ID if still not available
    if correlation_id is None:
        correlation_id = generate_correlation_id()
        logger.debug(f"Generated new correlation ID for client request: {correlation_id}")

    # Add to headers
    updated_headers[CORRELATION_ID_HEADER] = correlation_id

    return updated_headers


def with_correlation_headers(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for adding correlation ID to client method headers.

    This decorator ensures that the correlation ID from the current context
    is added to the request headers.

    Args:
        func: The client method to decorate

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        # Check if headers are in kwargs
        if "headers" in kwargs:
            kwargs["headers"] = add_correlation_id_to_headers(kwargs["headers"])
        else:
            # Add headers with correlation ID
            kwargs["headers"] = add_correlation_id_to_headers({})

        # Call the original function
        return func(*args, **kwargs)

    return wrapper


def with_async_correlation_headers(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for adding correlation ID to async client method headers.

    This decorator ensures that the correlation ID from the current context
    is added to the request headers for async client methods.

    Args:
        func: The async client method to decorate

    Returns:
        Decorated async function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        # Check if headers are in kwargs
        if "headers" in kwargs:
            kwargs["headers"] = add_correlation_id_to_headers(kwargs["headers"])
        else:
            # Add headers with correlation ID
            kwargs["headers"] = add_correlation_id_to_headers({})

        # Call the original function
        return await func(*args, **kwargs)

    return wrapper


class ClientCorrelationMixin:
    """
    Mixin for adding correlation ID support to service clients.

    This mixin provides methods for creating client instances with
    correlation ID headers and decorators for client methods.
    """

    def with_correlation_id(self, correlation_id: Optional[str] = None) -> 'ClientCorrelationMixin':
        """
        Create a new client instance with the specified correlation ID.

        This method should be implemented by subclasses to create a new
        client instance with the correlation ID added to default headers.

        Args:
            correlation_id: Correlation ID to use (defaults to current context)

        Returns:
            New client instance with correlation ID set
        """
        raise NotImplementedError("Subclasses must implement with_correlation_id")

    @classmethod
    def add_correlation_headers(cls, func: Callable) -> Callable:
        """
        Class method decorator for adding correlation headers to client methods.

        Args:
            func: The client method to decorate

        Returns:
            Decorated function
        """
        return with_correlation_headers(func)

    @classmethod
    def add_async_correlation_headers(cls, func: Callable) -> Callable:
        """
        Class method decorator for adding correlation headers to async client methods.

        Args:
            func: The async client method to decorate

        Returns:
            Decorated async function
        """
        return with_async_correlation_headers(func)
