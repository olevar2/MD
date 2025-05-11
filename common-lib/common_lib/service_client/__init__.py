"""
Service client package for the forex trading platform.

This package provides standardized service client functionality used across multiple services.
"""

from common_lib.service_client.base_client import (
    BaseServiceClient,
    AsyncBaseServiceClient,
    ServiceClientConfig as BaseServiceClientConfig,
    RetryConfig,
    CircuitBreakerConfig,
    TimeoutConfig,
)

from common_lib.service_client.http_client import (
    HTTPServiceClient,
    AsyncHTTPServiceClient,
)

from common_lib.service_client.resilient_client import (
    ServiceClientConfig as ResilientServiceClientConfig,
    ResilientServiceClient,
)

__all__ = [
    # Base client classes
    'BaseServiceClient',
    'AsyncBaseServiceClient',
    'BaseServiceClientConfig',
    'RetryConfig',
    'CircuitBreakerConfig',
    'TimeoutConfig',

    # HTTP client classes
    'HTTPServiceClient',
    'AsyncHTTPServiceClient',

    # Resilient client classes
    'ResilientServiceClientConfig',
    'ResilientServiceClient',
]