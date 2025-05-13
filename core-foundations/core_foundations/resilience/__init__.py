"""
Resilience Module for Forex Trading Platform

This module provides resilience patterns for robust service communication
in the Forex trading platform. It combines circuit breakers, retry policies,
and degraded mode operation to ensure the platform remains operational
even when components experience failures.
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerOpen # Removed non-existent import
from .retry_policy import RetryPolicy, RetryExhaustedException, retry, register_common_retryable_exceptions
from .degraded_mode import DegradedModeManager, DegradedModeStrategy, DependencyStatus, with_degraded_mode # Added

import logging
from typing import Dict, Any, Callable, Optional, Type

# Configure logger
logger = logging.getLogger(__name__)

# Standard configurations for different service criticality levels

# For critical services that require very robust handling
CRITICAL_SERVICE_CONFIG = {
    'circuit_breaker': {
        'failure_threshold': 3,       # Fewer failures before opening
        'recovery_timeout': 30,       # Quicker recovery attempts
        'reset_timeout': 300,         # 5 minutes to fully reset after recovery
    },
    'retry': {
        'max_attempts': 5,            # More retry attempts
        'base_delay': 0.5,            # Start with a short delay
        'max_delay': 10.0,            # But cap maximum delay
        'backoff_factor': 2.0,
        'jitter': True,
    }
}

# For standard services
STANDARD_SERVICE_CONFIG = {
    'circuit_breaker': {
        'failure_threshold': 5,
        'recovery_timeout': 60,
        'reset_timeout': 300,
    },
    'retry': {
        'max_attempts': 3,
        'base_delay': 1.0,
        'max_delay': 30.0,
        'backoff_factor': 2.0,
        'jitter': True,
    }
}

# For non-critical background services
BACKGROUND_SERVICE_CONFIG = {
    'circuit_breaker': {
        'failure_threshold': 10,      # More failures allowed before breaking
        'recovery_timeout': 300,      # Longer recovery timeout
        'reset_timeout': 600,         # 10 minutes to fully reset
    },
    'retry': {
        'max_attempts': 10,           # Many retry attempts for background tasks
        'base_delay': 2.0,            # Start with longer delays
        'max_delay': 300.0,           # Up to 5 minutes between retries
        'backoff_factor': 2.0,
        'jitter': True,
    }
}


def create_resilient_client(
    name: str,
    criticality: str = 'standard',
    metric_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    custom_circuit_breaker_config: Optional[Dict[str, Any]] = None,
    custom_retry_config: Optional[Dict[str, Any]] = None,
    retryable_exceptions: Optional[list[Type[Exception]]] = None,
) -> Dict[str, Any]:
    """
    Create resilience components configured for a specific service client.
    
    Args:
        name: Name of the client/service for identification
        criticality: Criticality level ('critical', 'standard', 'background')
        metric_handler: Optional callback for reporting metrics
        custom_circuit_breaker_config: Custom circuit breaker configuration
        custom_retry_config: Custom retry configuration
        retryable_exceptions: Custom list of exceptions that should trigger retries
        
    Returns:
        Dictionary containing configured CircuitBreaker and RetryPolicy instances
    """
    # Select base configuration based on criticality
    if criticality == 'critical':
        base_config = CRITICAL_SERVICE_CONFIG
    elif criticality == 'standard':
        base_config = STANDARD_SERVICE_CONFIG
    elif criticality == 'background':
        base_config = BACKGROUND_SERVICE_CONFIG
    else:
        raise ValueError(f"Unknown criticality level: {criticality}")
    
    # Merge custom configurations with base configurations
    cb_config = {**base_config['circuit_breaker']}
    if custom_circuit_breaker_config:
        cb_config.update(custom_circuit_breaker_config)
        
    retry_config = {**base_config['retry']}
    if custom_retry_config:
        retry_config.update(custom_retry_config)
    
    # Create the circuit breaker
    circuit_breaker = CircuitBreaker(
        name=f"{name}_circuit_breaker",
        failure_threshold=cb_config['failure_threshold'],
        recovery_timeout=cb_config['recovery_timeout'],
        reset_timeout=cb_config['reset_timeout'],
        metric_handler=metric_handler
    )
    
    # If no exceptions specified, use common network exceptions
    if not retryable_exceptions:
        retryable_exceptions = list(register_common_retryable_exceptions())
    
    # Create the retry policy
    retry_policy = RetryPolicy(
        max_attempts=retry_config['max_attempts'],
        base_delay=retry_config['base_delay'],
        max_delay=retry_config['max_delay'],
        backoff_factor=retry_config['backoff_factor'],
        jitter=retry_config['jitter'],
        exceptions=retryable_exceptions,
        metric_handler=metric_handler
    )
    
    return {
        'circuit_breaker': circuit_breaker,
        'retry_policy': retry_policy
    }


def resilient_function(
    func: Callable,
    circuit_breaker_instance: CircuitBreaker,
    retry_policy_instance: RetryPolicy
) -> Callable:
    """
    Create a resilient version of a function with circuit breaker and retry policy.
    
    Args:
        func: The function to make resilient
        circuit_breaker_instance: CircuitBreaker instance
        retry_policy_instance: RetryPolicy instance
        
    Returns:
        A wrapped function with both circuit breaker and retry policy applied
    """
    def circuit_breaker_wrapper(*args, **kwargs):
    """
    Circuit breaker wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        if not circuit_breaker_instance.allow_request():
            raise CircuitBreakerException(
                f"Circuit {circuit_breaker_instance.name} is OPEN"
            )
        
        try:
            result = retry_policy_instance.execute(lambda: func(*args, **kwargs))
            circuit_breaker_instance.record_success()
            return result
        except Exception as e:
            circuit_breaker_instance.record_failure(e)
            raise
    
    return circuit_breaker_wrapper


def get_resilience_stats() -> Dict[str, Any]:
    """
    Collects statistics from all circuit breakers and retry policies.
    
    Returns:
        Dictionary with resilience statistics
    """
    # This is a placeholder - in a real implementation, this would track 
    # all circuit breaker and retry policy instances to provide stats
    return {
        'message': 'Implement this to collect real-time statistics from all resilience components'
    }


__all__ = [
    # Circuit Breaker
    'CircuitBreaker',
    'CircuitBreakerException',
    'circuit_breaker',

    # Retry Policy
    'RetryPolicy',
    'RetryExhaustedException',
    'retry',
    'register_common_retryable_exceptions',

    # Degraded Mode
    'DegradedModeManager',
    'DegradedModeStrategy',
    'DependencyStatus',
    'with_degraded_mode',

    # Configs
    'CRITICAL_SERVICE_CONFIG',
    'STANDARD_SERVICE_CONFIG',
    'BACKGROUND_SERVICE_CONFIG',
]
