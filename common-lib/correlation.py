"""
Correlation ID utilities for the forex trading platform.
"""

import uuid
import threading
from typing import Optional

# Thread-local storage for correlation ID
_thread_local = threading.local()


def set_correlation_id(correlation_id: str) -> None:
    """
    Set the correlation ID for the current thread.
    
    Args:
        correlation_id: Correlation ID to set
    """
    _thread_local.correlation_id = correlation_id


def get_correlation_id() -> Optional[str]:
    """
    Get the correlation ID for the current thread.
    
    Returns:
        Correlation ID or None if not set
    """
    return getattr(_thread_local, 'correlation_id', None)


def generate_correlation_id() -> str:
    """
    Generate a new correlation ID.
    
    Returns:
        New correlation ID
    """
    return str(uuid.uuid4())


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current thread."""
    if hasattr(_thread_local, 'correlation_id'):
        delattr(_thread_local, 'correlation_id')
