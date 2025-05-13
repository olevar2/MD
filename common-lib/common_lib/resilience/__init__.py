"""
Resilience package for the forex trading platform.

This package provides resilience patterns for the platform, including circuit breaker,
retry, bulkhead, timeout, and fallback patterns. It also includes decorators for
applying these patterns to functions.
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
from common_lib.resilience.decorators import (
    circuit_breaker,
    retry_with_backoff,
    timeout,
    bulkhead,
    with_resilience
)

__all__ = [
    # Core resilience components
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
    'get_resilience',

    # Standardized decorators
    'circuit_breaker',
    'retry_with_backoff',
    'timeout',
    'bulkhead',
    'with_resilience'
]