"""
Resilience Module for Analysis Engine Service

This module provides resilience patterns for the Analysis Engine Service,
building on the common-lib resilience module.

It includes:
1. Circuit breakers for external service calls
2. Retry mechanisms with backoff for transient failures
3. Bulkhead patterns to isolate critical operations
4. Timeout handling for external operations
"""

# Import resilience patterns from common-lib
from common_lib.resilience import (
    # Circuit breaker components
    CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitBreakerOpen,
    create_circuit_breaker,
    
    # Retry components
    retry_with_policy, RetryExhaustedException,
    register_common_retryable_exceptions, register_database_retryable_exceptions,
    
    # Timeout handler components
    timeout_handler, async_timeout, sync_timeout, TimeoutError,
    
    # Bulkhead components
    Bulkhead, bulkhead, BulkheadFullException
)

# Re-export all components
__all__ = [
    # Circuit breaker components
    "CircuitBreaker", "CircuitBreakerConfig", "CircuitState", "CircuitBreakerOpen",
    "create_circuit_breaker",
    
    # Retry components
    "retry_with_policy", "RetryExhaustedException",
    "register_common_retryable_exceptions", "register_database_retryable_exceptions",
    
    # Timeout handler components
    "timeout_handler", "async_timeout", "sync_timeout", "TimeoutError",
    
    # Bulkhead components
    "Bulkhead", "bulkhead", "BulkheadFullException",
]
