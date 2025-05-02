"""
Circuit Breaker Pattern Implementation

This module provides enhanced circuit breaker functionality by extending
the core implementation from core-foundations with integration to common-lib
monitoring, logging, and configuration.
"""

import logging
from typing import Any, Callable, Dict, Optional, Type, Union, List

# Import the base circuit breaker implementation
from core_foundations.resilience.circuit_breaker import (
    CircuitBreaker as CoreCircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpen
)

# Setup logger
logger = logging.getLogger(__name__)

# Re-export CircuitBreakerConfig, CircuitState, and CircuitBreakerOpen
__all__ = [
    "CircuitBreaker", "CircuitBreakerConfig", "CircuitState", 
    "CircuitBreakerOpen", "create_circuit_breaker"
]


class CircuitBreaker(CoreCircuitBreaker):
    """
    Enhanced CircuitBreaker that integrates with common_lib monitoring
    and provides additional utilities specific to the Forex platform.
    """
    
    @property
    def current_state(self):
        """Alias for state property to maintain backward compatibility."""
        return self.state
    
    def get_extended_metrics(self) -> Dict[str, Any]:
        """
        Get extended metrics for monitoring the circuit breaker.
        
        Returns:
            Dictionary with extended metrics including error percentages and state duration
        """
        metrics = super().get_metrics()
        
        # Add additional metrics
        if metrics["total_calls"] > 0:
            metrics["error_rate"] = metrics["failed_calls"] / metrics["total_calls"]
        else:
            metrics["error_rate"] = 0.0
            
        return metrics
        
    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function with circuit breaker protection.
        Wraps the original call method with improved error handling.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The function's result
            
        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Any exception from the function if circuit is closed
        """
        return await self.call(func, *args, **kwargs)


def create_circuit_breaker(
    service_name: str, 
    resource_name: str, 
    config: Optional[CircuitBreakerConfig] = None,
    fallback_function: Optional[Callable[..., Any]] = None,
    **kwargs
) -> CircuitBreaker:
    """
    Factory function to create a properly configured CircuitBreaker.
    
    Args:
        service_name: Name of the service using the circuit breaker
        resource_name: Name of the resource being protected
        config: Optional circuit breaker configuration
        fallback_function: Optional fallback function
        **kwargs: Additional parameters for the CircuitBreaker constructor
        
    Returns:
        Configured CircuitBreaker instance
    """
    name = f"{service_name}.{resource_name}"
    return CircuitBreaker(name=name, config=config, fallback_function=fallback_function, **kwargs)
