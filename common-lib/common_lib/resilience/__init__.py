"""
Resilience package for the forex trading platform.

This package provides resilience patterns for the platform, including circuit breaker,
retry, bulkhead, timeout, and fallback patterns. It also includes decorators for
applying these patterns to functions.
"""

from common_lib.resilience.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerConfig, CircuitBreakerOpen
from common_lib.resilience.retry import retry, RetryPolicy, retry_with_policy, RetryExhaustedException
from common_lib.resilience.retry_policy import register_common_retryable_exceptions
from common_lib.resilience.bulkhead import Bulkhead
from common_lib.resilience.timeout import with_timeout, Timeout, TimeoutError
from common_lib.resilience.fallback import with_fallback, Fallback
from common_lib.resilience.resilience import (
    Resilience,
    ResilienceConfig,
    resilient,
    get_resilience as get_resilience_instance
)
from common_lib.resilience.decorators import (
    circuit_breaker,
    retry_with_backoff,
    timeout,
    bulkhead,
    with_resilience
)

# Import standardized configuration
from common_lib.resilience.config import (
    CircuitBreakerConfig as StandardCircuitBreakerConfig,
    RetryConfig,
    BulkheadConfig,
    TimeoutConfig,
    ResilienceConfig as StandardResilienceConfig,
    ResilienceProfiles,
    get_resilience_config
)

# Import factory functions
from common_lib.resilience.factory import (
    create_circuit_breaker,
    create_retry_policy,
    create_bulkhead,
    create_timeout,
    create_resilience,
    get_circuit_breaker,
    get_retry_policy,
    get_bulkhead,
    get_timeout,
    get_resilience
)

# Import enhanced decorators
from common_lib.resilience.enhanced_decorators import (
    with_standard_circuit_breaker,
    with_standard_retry,
    with_standard_bulkhead,
    with_standard_timeout,
    with_standard_resilience,
    with_database_resilience,
    with_broker_api_resilience,
    with_market_data_resilience,
    with_external_api_resilience,
    with_critical_resilience,
    with_high_throughput_resilience
)

__all__ = [
    # Core resilience components
    'CircuitBreaker',
    'CircuitState',
    'CircuitBreakerConfig',
    'CircuitBreakerOpen',
    'retry',
    'RetryPolicy',
    'retry_with_policy',
    'RetryExhaustedException',
    'register_common_retryable_exceptions',
    'Bulkhead',
    'with_timeout',
    'Timeout',
    'TimeoutError',
    'with_fallback',
    'Fallback',
    'Resilience',
    'ResilienceConfig',
    'resilient',
    'get_resilience_instance',

    # Standardized decorators
    'circuit_breaker',
    'retry_with_backoff',
    'timeout',
    'bulkhead',
    'with_resilience',

    # Standardized configuration
    'StandardCircuitBreakerConfig',
    'RetryConfig',
    'BulkheadConfig',
    'TimeoutConfig',
    'StandardResilienceConfig',
    'ResilienceProfiles',
    'get_resilience_config',
    
    # Factory functions
    'create_circuit_breaker',
    'create_retry_policy',
    'create_bulkhead',
    'create_timeout',
    'create_resilience',
    'get_circuit_breaker',
    'get_retry_policy',
    'get_bulkhead',
    'get_timeout',
    'get_resilience',
    
    # Enhanced decorators
    'with_standard_circuit_breaker',
    'with_standard_retry',
    'with_standard_bulkhead',
    'with_standard_timeout',
    'with_standard_resilience',
    'with_database_resilience',
    'with_broker_api_resilience',
    'with_market_data_resilience',
    'with_external_api_resilience',
    'with_critical_resilience',
    'with_high_throughput_resilience'
]