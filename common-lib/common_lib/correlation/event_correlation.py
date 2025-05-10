"""
Event Correlation Module

This module provides utilities for propagating correlation IDs in event-based communication.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Callable

from common_lib.correlation.correlation_id import (
    get_correlation_id,
    generate_correlation_id,
    with_correlation_id,
    with_async_correlation_id
)

logger = logging.getLogger(__name__)


def add_correlation_to_event_metadata(
    metadata: Dict[str, Any],
    correlation_id: Optional[str] = None,
    causation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add correlation and causation IDs to event metadata.

    Args:
        metadata: Event metadata dictionary
        correlation_id: Optional correlation ID to use (defaults to current context)
        causation_id: Optional causation ID (event that caused this event)

    Returns:
        Updated metadata dictionary
    """
    # Make a copy of the metadata
    updated_metadata = metadata.copy() if metadata else {}

    # Get correlation ID from context if not provided
    if correlation_id is None:
        correlation_id = get_correlation_id()

    # Generate a new correlation ID if still not available
    if correlation_id is None:
        correlation_id = generate_correlation_id()
        logger.debug(f"Generated new correlation ID for event: {correlation_id}")

    # Add correlation ID to metadata
    updated_metadata["correlation_id"] = correlation_id

    # Add causation ID if provided
    if causation_id:
        updated_metadata["causation_id"] = causation_id

    return updated_metadata


def extract_correlation_id_from_event(event: Dict[str, Any]) -> Optional[str]:
    """
    Extract correlation ID from an event.

    Args:
        event: Event dictionary

    Returns:
        Correlation ID if present, None otherwise
    """
    # Check in metadata
    if "metadata" in event and isinstance(event["metadata"], dict):
        if "correlation_id" in event["metadata"]:
            return event["metadata"]["correlation_id"]

    # Check in top-level properties
    if "correlation_id" in event:
        return event["correlation_id"]

    return None


def with_event_correlation(func: Callable) -> Callable:
    """
    Decorator for propagating correlation IDs in event handlers.

    This decorator extracts the correlation ID from the event and sets it
    in the current context before calling the handler function.

    Args:
        func: The event handler function to decorate

    Returns:
        Decorated function
    """
    @with_correlation_id
    def wrapper(event, *args, **kwargs):
        # Extract correlation ID from event
        correlation_id = extract_correlation_id_from_event(event)

        if correlation_id:
            # Set correlation ID in context
            from common_lib.correlation.correlation_id import set_correlation_id
            set_correlation_id(correlation_id)

        # Call the handler
        return func(event, *args, **kwargs)

    return wrapper


def with_async_event_correlation(func: Callable) -> Callable:
    """
    Decorator for propagating correlation IDs in async event handlers.

    This decorator extracts the correlation ID from the event and sets it
    in the current context before calling the async handler function.

    Args:
        func: The async event handler function to decorate

    Returns:
        Decorated async function
    """
    @with_async_correlation_id
    async def wrapper(event, *args, **kwargs):
        # Extract correlation ID from event
        correlation_id = extract_correlation_id_from_event(event)

        if correlation_id:
            # Set correlation ID in context
            from common_lib.correlation.correlation_id import set_correlation_id
            set_correlation_id(correlation_id)

        # Call the handler
        return await func(event, *args, **kwargs)

    return wrapper
