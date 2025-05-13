"""
Circuit Breaker System for Risk Management Service.

Re-exports the standardized circuit breaker from core-foundations.
"""

from core_foundations.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpen
)

# Expose the core circuit breaker classes for risk management use
__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitBreakerOpen"
]
