"""
Resilience Module for Forex Trading Platform

This module centralizes resilience patterns for robust service communication
in the Forex trading platform. It re-exports resilience components from core-foundations
and provides additional utilities for integration with common_lib components.

Key components:
1. Circuit Breaker - Prevents cascading failures by stopping calls to failing services
2. Retry Policy - Automatically retries temporary failures with exponential backoff
3. Timeout Handler - Ensures operations complete within specific time constraints
4. Bulkhead Pattern - Isolates failures by partitioning resources

Usage:
    from common_lib.resilience import circuit_breaker, retry_with_policy, timeout_handler, bulkhead
"""

# --- BEGIN ADDED PATH MANIPULATION ---
import sys
import os
import logging

# Ensure core-foundations is in the path when this module is imported
core_foundations_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../core-foundations'))
if core_foundations_dir not in sys.path:
    sys.path.insert(0, core_foundations_dir)
    # Optional: Log that the path was added, useful for debugging
    # logging.info(f"Added {core_foundations_dir} to sys.path from resilience/__init__.py")
# --- END ADDED PATH MANIPULATION ---

# Try to import from core_foundations, fall back to stubs if not available
import importlib.util

# Check if core_foundations is available
if importlib.util.find_spec('core_foundations') is not None:
    from core_foundations.resilience import (
        DegradedModeManager, DegradedModeStrategy, DependencyStatus, with_degraded_mode,
        CRITICAL_SERVICE_CONFIG, STANDARD_SERVICE_CONFIG
    )
else:
    from .core_fallback import (
        DegradedModeManager, DegradedModeStrategy, DependencyStatus, with_degraded_mode,
        CRITICAL_SERVICE_CONFIG, STANDARD_SERVICE_CONFIG
    )
    logging.warning("core_foundations not found, using fallback implementations.")

# Import our enhanced implementations
from .circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitBreakerOpen,
    create_circuit_breaker
)

from .retry_policy import (
    RetryPolicy, retry_with_policy, RetryExhaustedException,
    register_common_retryable_exceptions, register_database_retryable_exceptions
)

from .timeout_handler import (
    timeout_handler, async_timeout, sync_timeout, TimeoutError
)

from .bulkhead import (
    Bulkhead, bulkhead, BulkheadFullException
)

# Export all components
__all__ = [
    # Circuit breaker components
    "CircuitBreaker", "CircuitBreakerConfig", "CircuitState", "CircuitBreakerOpen",
    "create_circuit_breaker",
    
    # Retry components
    "RetryPolicy", "retry_with_policy", "RetryExhaustedException",
    "register_common_retryable_exceptions", "register_database_retryable_exceptions",
    
    # Timeout handler components
    "timeout_handler", "async_timeout", "sync_timeout", "TimeoutError",
    
    # Bulkhead components
    "Bulkhead", "bulkhead", "BulkheadFullException",
    
    # Degraded mode components from core
    "DegradedModeManager", "DegradedModeStrategy", "DependencyStatus", "with_degraded_mode",
    
    # Standard configurations
    "CRITICAL_SERVICE_CONFIG", "STANDARD_SERVICE_CONFIG"
]
