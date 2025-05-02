"""
Service Client Module

This module provides standardized client implementations for service communication.
It includes base classes, factories, and utilities for creating resilient service clients.
"""

from common_lib.clients.base_client import BaseServiceClient, ClientConfig
from common_lib.clients.client_factory import (
    create_client,
    get_client,
    register_client_config,
    get_client_config
)
from common_lib.clients.exceptions import (
    ClientError,
    ClientConnectionError,
    ClientTimeoutError,
    ClientValidationError,
    ClientAuthenticationError,
    CircuitBreakerOpenError,
    BulkheadFullError,
    RetryExhaustedError
)

__all__ = [
    # Base client
    "BaseServiceClient",
    "ClientConfig",
    
    # Client factory
    "create_client",
    "get_client",
    "register_client_config",
    "get_client_config",
    
    # Exceptions
    "ClientError",
    "ClientConnectionError",
    "ClientTimeoutError",
    "ClientValidationError",
    "ClientAuthenticationError",
    "CircuitBreakerOpenError",
    "BulkheadFullError",
    "RetryExhaustedError"
]
