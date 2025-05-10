"""
Correlation ID Utility

This module provides a standardized implementation for correlation ID generation,
propagation, and retrieval across different communication patterns.

Features:
1. Consistent correlation ID generation
2. Thread-local and async-context storage for correlation IDs
3. Automatic propagation between services
4. Support for HTTP, messaging, and event-based communication
"""

import uuid
import logging
import contextvars
import threading
from typing import Optional, Dict, Any, Union, TypeVar, Generic, Callable, AsyncGenerator
from contextlib import contextmanager, asynccontextmanager

# Thread-local storage for synchronous code
_thread_local = threading.local()

# Context variable for asynchronous code
_correlation_id_var = contextvars.ContextVar("correlation_id", default=None)

# Logger
logger = logging.getLogger(__name__)

# Header name constant
CORRELATION_ID_HEADER = "X-Correlation-ID"


def generate_correlation_id() -> str:
    """
    Generate a new correlation ID.

    Returns:
        A unique correlation ID string
    """
    return str(uuid.uuid4())


def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID from thread-local or async context.

    This function checks both thread-local storage and async context variables,
    making it usable in both synchronous and asynchronous code.

    Returns:
        The current correlation ID or None if not set
    """
    # Try async context first
    correlation_id = _correlation_id_var.get()

    # If not found, try thread-local
    if correlation_id is None:
        correlation_id = getattr(_thread_local, "correlation_id", None)

    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """
    Set the correlation ID in both thread-local and async context.

    Args:
        correlation_id: The correlation ID to set
    """
    # Set in thread-local for synchronous code
    _thread_local.correlation_id = correlation_id

    # Set in context var for async code
    _correlation_id_var.set(correlation_id)

    logger.debug(f"Set correlation ID: {correlation_id}")


def clear_correlation_id() -> None:
    """
    Clear the correlation ID from both thread-local and async context.
    """
    # Clear from thread-local
    if hasattr(_thread_local, "correlation_id"):
        delattr(_thread_local, "correlation_id")

    # Reset context var
    _correlation_id_var.set(None)

    logger.debug("Cleared correlation ID")


@contextmanager
def correlation_id_context(correlation_id: Optional[str] = None) -> None:
    """
    Context manager for setting a correlation ID in synchronous code.

    Args:
        correlation_id: The correlation ID to use, or None to generate a new one

    Yields:
        None
    """
    # Save the previous correlation ID
    previous_id = get_correlation_id()

    # Set the new correlation ID
    new_id = correlation_id or generate_correlation_id()
    set_correlation_id(new_id)

    try:
        yield
    finally:
        # Restore the previous correlation ID
        if previous_id is not None:
            set_correlation_id(previous_id)
        else:
            clear_correlation_id()


@asynccontextmanager
async def async_correlation_id_context(correlation_id: Optional[str] = None) -> AsyncGenerator[None, None]:
    """
    Async context manager for setting a correlation ID in asynchronous code.

    Args:
        correlation_id: The correlation ID to use, or None to generate a new one

    Yields:
        None
    """
    # Save the previous correlation ID
    previous_id = get_correlation_id()

    # Set the new correlation ID
    new_id = correlation_id or generate_correlation_id()
    set_correlation_id(new_id)

    try:
        yield
    finally:
        # Restore the previous correlation ID
        if previous_id is not None:
            set_correlation_id(previous_id)
        else:
            clear_correlation_id()


def with_correlation_id(func: Callable) -> Callable:
    """
    Decorator for propagating correlation IDs in synchronous functions.

    This decorator ensures that the function runs with a correlation ID,
    either using an existing one or generating a new one.

    Args:
        func: The function to decorate

    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        # Check if we already have a correlation ID
        current_id = get_correlation_id()

        if current_id is not None:
            # Use existing correlation ID
            return func(*args, **kwargs)
        else:
            # Generate a new correlation ID for this call
            with correlation_id_context():
                return func(*args, **kwargs)

    return wrapper


def with_async_correlation_id(func: Callable) -> Callable:
    """
    Decorator for propagating correlation IDs in asynchronous functions.

    This decorator ensures that the function runs with a correlation ID,
    either using an existing one or generating a new one.

    Args:
        func: The async function to decorate

    Returns:
        Decorated async function
    """
    async def wrapper(*args, **kwargs):
        # Check if we already have a correlation ID
        current_id = get_correlation_id()

        if current_id is not None:
            # Use existing correlation ID
            return await func(*args, **kwargs)
        else:
            # Generate a new correlation ID for this call
            async with async_correlation_id_context():
                return await func(*args, **kwargs)

    return wrapper


def get_correlation_id_from_request(request: Any) -> str:
    """
    Extract correlation ID from request or generate a new one.

    This function works with different request objects (FastAPI, Flask, etc.)
    by trying different attributes and falling back to generating a new ID.

    Args:
        request: The request object

    Returns:
        Correlation ID string
    """
    correlation_id = None

    # Try to get from request state (FastAPI)
    if hasattr(request, "state") and hasattr(request.state, "correlation_id"):
        correlation_id = request.state.correlation_id

    # If not in state, try to get from headers
    if not correlation_id and hasattr(request, "headers"):
        correlation_id = request.headers.get(CORRELATION_ID_HEADER)

    # If still not found, generate a new one
    if not correlation_id:
        correlation_id = generate_correlation_id()

    return correlation_id


def add_correlation_id_to_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Add the current correlation ID to headers.

    If no correlation ID is set, a new one will be generated.

    Args:
        headers: Headers dictionary to update

    Returns:
        Updated headers dictionary
    """
    # Make a copy of the headers
    updated_headers = headers.copy()

    # Get current correlation ID or generate a new one
    correlation_id = get_correlation_id()
    if correlation_id is None:
        correlation_id = generate_correlation_id()

    # Add to headers
    updated_headers[CORRELATION_ID_HEADER] = correlation_id

    return updated_headers
