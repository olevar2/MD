"""
Resilience package for the forex trading platform.

This package provides resilience patterns for the platform.
"""

from common_lib.resilience.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerConfig
from common_lib.resilience.retry import retry, RetryPolicy, retry_with_policy
from common_lib.resilience.bulkhead import Bulkhead
from common_lib.resilience.timeout import with_timeout, Timeout, TimeoutError
from common_lib.resilience.fallback import with_fallback, Fallback
from common_lib.resilience.resilience import (
    Resilience,
    ResilienceConfig,
    resilient,
    get_resilience
)

__all__ = [
    'CircuitBreaker',
    'CircuitState',
    'CircuitBreakerConfig',
    'retry',
    'RetryPolicy',
    'retry_with_policy',
    'Bulkhead',
    'with_timeout',
    'Timeout',
    'TimeoutError',
    'with_fallback',
    'Fallback',
    'Resilience',
    'ResilienceConfig',
    'resilient',
    'get_resilience'
]