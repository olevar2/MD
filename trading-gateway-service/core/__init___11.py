"""
Resilience package for the Trading Gateway Service.

This package contains resilience components for the Trading Gateway Service,
including circuit breakers, retry mechanisms, bulkheads, and timeout handling.
"""

import logging
from typing import Dict, Any, Optional

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

# Import degraded mode components
from .degraded_mode import (
    DegradationLevel,
    DegradedModeManager
)
from .degraded_mode_strategies import configure_trading_gateway_degraded_mode

# Initialize logger
logger = logging.getLogger(__name__)

# Circuit breaker configurations for different service types
CIRCUIT_BREAKER_CONFIGS = {
    # Broker API
    "broker_api": CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=60,
    ),

    # Market data provider
    "market_data": CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=30,
    ),

    # Order execution
    "order_execution": CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=120,
    ),

    # Risk management
    "risk_management": CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=60,
    ),

    # Analysis engine
    "analysis_engine": CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=60,
    ),

    # Database operations
    "database": CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=30,
    ),

    # Redis operations
    "redis": CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=30,
    ),

    # Default configuration
    "default": CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=60,
    ),
}

# Retry configurations for different service types
RETRY_CONFIGS = {
    # Broker API
    "broker_api": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 30.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },

    # Market data provider
    "market_data": {
        "max_attempts": 5,
        "base_delay": 0.5,
        "max_delay": 10.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },

    # Order execution
    "order_execution": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 30.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },

    # Risk management
    "risk_management": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 30.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },

    # Analysis engine
    "analysis_engine": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 30.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },

    # Database operations
    "database": {
        "max_attempts": 5,
        "base_delay": 0.5,
        "max_delay": 10.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },

    # Redis operations
    "redis": {
        "max_attempts": 3,
        "base_delay": 0.5,
        "max_delay": 5.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },

    # Default configuration
    "default": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 30.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },
}

# Bulkhead configurations for different service types
BULKHEAD_CONFIGS = {
    # Broker API
    "broker_api": {
        "max_concurrent": 10,
        "max_waiting": 20,
    },

    # Market data provider
    "market_data": {
        "max_concurrent": 20,
        "max_waiting": 50,
    },

    # Order execution
    "order_execution": {
        "max_concurrent": 5,
        "max_waiting": 10,
    },

    # Risk management
    "risk_management": {
        "max_concurrent": 10,
        "max_waiting": 20,
    },

    # Analysis engine
    "analysis_engine": {
        "max_concurrent": 10,
        "max_waiting": 20,
    },

    # Database operations
    "database": {
        "max_concurrent": 20,
        "max_waiting": 50,
    },

    # Redis operations
    "redis": {
        "max_concurrent": 20,
        "max_waiting": 50,
    },

    # Default configuration
    "default": {
        "max_concurrent": 10,
        "max_waiting": 20,
    },
}

# Timeout configurations for different service types
TIMEOUT_CONFIGS = {
    # Broker API
    "broker_api": 10.0,  # 10 seconds

    # Market data provider
    "market_data": 5.0,  # 5 seconds

    # Order execution
    "order_execution": 30.0,  # 30 seconds

    # Risk management
    "risk_management": 10.0,  # 10 seconds

    # Analysis engine
    "analysis_engine": 30.0,  # 30 seconds

    # Database operations
    "database": 5.0,  # 5 seconds

    # Redis operations
    "redis": 2.0,  # 2 seconds

    # Default configuration
    "default": 10.0,  # 10 seconds
}


def get_circuit_breaker_config(service_type: str = "default") -> CircuitBreakerConfig:
    """
    Get circuit breaker configuration for a service type.

    Args:
        service_type: Type of service (broker_api, market_data, etc.)

    Returns:
        CircuitBreakerConfig for the service type
    """
    return CIRCUIT_BREAKER_CONFIGS.get(service_type, CIRCUIT_BREAKER_CONFIGS["default"])


def get_retry_config(service_type: str = "default") -> Dict[str, Any]:
    """
    Get retry configuration for a service type.

    Args:
        service_type: Type of service (broker_api, market_data, etc.)

    Returns:
        Retry configuration for the service type
    """
    return RETRY_CONFIGS.get(service_type, RETRY_CONFIGS["default"])


def get_bulkhead_config(service_type: str = "default") -> Dict[str, Any]:
    """
    Get bulkhead configuration for a service type.

    Args:
        service_type: Type of service (broker_api, market_data, etc.)

    Returns:
        Bulkhead configuration for the service type
    """
    return BULKHEAD_CONFIGS.get(service_type, BULKHEAD_CONFIGS["default"])


def get_timeout_config(service_type: str = "default") -> float:
    """
    Get timeout configuration for a service type.

    Args:
        service_type: Type of service (broker_api, market_data, etc.)

    Returns:
        Timeout in seconds for the service type
    """
    return TIMEOUT_CONFIGS.get(service_type, TIMEOUT_CONFIGS["default"])


# Re-export all components
__all__ = [
    # Degraded mode components
    "DegradationLevel",
    "DegradedModeManager",
    "configure_trading_gateway_degraded_mode",

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

    # Utility functions
    "get_circuit_breaker_config", "get_retry_config", "get_bulkhead_config", "get_timeout_config",
]
