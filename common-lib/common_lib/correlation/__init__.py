"""
Correlation ID Module

This module provides a standardized implementation for correlation ID generation,
propagation, and retrieval across different communication patterns.

Features:
1. Consistent correlation ID generation
2. Thread-local and async-context storage for correlation IDs
3. Automatic propagation between services
4. Support for HTTP, messaging, and event-based communication
"""

# Core correlation ID functionality
from common_lib.correlation.correlation_id import (
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    correlation_id_context,
    async_correlation_id_context,
    with_correlation_id,
    with_async_correlation_id,
    get_correlation_id_from_request,
    add_correlation_id_to_headers,
    CORRELATION_ID_HEADER
)

# Middleware for web frameworks
from common_lib.correlation.middleware import (
    FastAPICorrelationIdMiddleware,
    create_correlation_id_middleware
)

# Event-based communication
from common_lib.correlation.event_correlation import (
    add_correlation_to_event_metadata,
    extract_correlation_id_from_event,
    with_event_correlation,
    with_async_event_correlation
)

# Service client integration
from common_lib.correlation.client_correlation import (
    add_correlation_id_to_headers,
    with_correlation_headers,
    with_async_correlation_headers,
    ClientCorrelationMixin
)

__all__ = [
    # Core functionality
    'generate_correlation_id',
    'get_correlation_id',
    'set_correlation_id',
    'clear_correlation_id',
    'correlation_id_context',
    'async_correlation_id_context',
    'with_correlation_id',
    'with_async_correlation_id',
    'get_correlation_id_from_request',
    'add_correlation_id_to_headers',
    'CORRELATION_ID_HEADER',

    # Middleware
    'FastAPICorrelationIdMiddleware',
    'create_correlation_id_middleware',

    # Event correlation
    'add_correlation_to_event_metadata',
    'extract_correlation_id_from_event',
    'with_event_correlation',
    'with_async_event_correlation',

    # Client correlation
    'add_correlation_id_to_headers',
    'with_correlation_headers',
    'with_async_correlation_headers',
    'ClientCorrelationMixin'
]
