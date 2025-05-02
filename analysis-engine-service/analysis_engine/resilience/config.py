"""
Resilience Configuration for Analysis Engine Service

This module provides configuration settings for resilience patterns
used throughout the Analysis Engine Service.
"""

from typing import Dict, Any
from common_lib.resilience import CircuitBreakerConfig

# Circuit breaker configurations for different service types
CIRCUIT_BREAKER_CONFIGS = {
    # Feature Store Service
    "feature_store": CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=60,
    ),
    
    # Data Pipeline Service
    "data_pipeline": CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=60,
    ),
    
    # ML Integration Service
    "ml_integration": CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=120,
    ),
    
    # Strategy Execution Engine
    "strategy_execution": CircuitBreakerConfig(
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
        failure_threshold=3,
        reset_timeout_seconds=60,
    ),
}

# Retry configurations for different operation types
RETRY_CONFIGS = {
    # Feature Store Service
    "feature_store": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 10.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },
    
    # Data Pipeline Service
    "data_pipeline": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 10.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },
    
    # ML Integration Service
    "ml_integration": {
        "max_attempts": 3,
        "base_delay": 2.0,
        "max_delay": 20.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },
    
    # Strategy Execution Engine
    "strategy_execution": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 10.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },
    
    # Database operations
    "database": {
        "max_attempts": 3,
        "base_delay": 0.5,
        "max_delay": 5.0,
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
        "max_delay": 10.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },
}

# Bulkhead configurations for different operation types
BULKHEAD_CONFIGS = {
    # Feature Store Service
    "feature_store": {
        "max_concurrent": 20,
        "max_waiting": 50,
        "wait_timeout": 10.0,
    },
    
    # Data Pipeline Service
    "data_pipeline": {
        "max_concurrent": 20,
        "max_waiting": 50,
        "wait_timeout": 10.0,
    },
    
    # ML Integration Service
    "ml_integration": {
        "max_concurrent": 10,
        "max_waiting": 20,
        "wait_timeout": 15.0,
    },
    
    # Strategy Execution Engine
    "strategy_execution": {
        "max_concurrent": 15,
        "max_waiting": 30,
        "wait_timeout": 10.0,
    },
    
    # Database operations
    "database": {
        "max_concurrent": 30,
        "max_waiting": 100,
        "wait_timeout": 5.0,
    },
    
    # Redis operations
    "redis": {
        "max_concurrent": 50,
        "max_waiting": 200,
        "wait_timeout": 3.0,
    },
    
    # Default configuration
    "default": {
        "max_concurrent": 20,
        "max_waiting": 50,
        "wait_timeout": 10.0,
    },
}

# Timeout configurations for different operation types (in seconds)
TIMEOUT_CONFIGS = {
    # Feature Store Service
    "feature_store": 10.0,
    
    # Data Pipeline Service
    "data_pipeline": 10.0,
    
    # ML Integration Service
    "ml_integration": 15.0,
    
    # Strategy Execution Engine
    "strategy_execution": 10.0,
    
    # Database operations
    "database": 5.0,
    
    # Redis operations
    "redis": 3.0,
    
    # Default configuration
    "default": 10.0,
}

def get_circuit_breaker_config(service_type: str) -> CircuitBreakerConfig:
    """
    Get circuit breaker configuration for a specific service type.
    
    Args:
        service_type: Type of service (feature_store, data_pipeline, etc.)
        
    Returns:
        CircuitBreakerConfig for the specified service type
    """
    return CIRCUIT_BREAKER_CONFIGS.get(service_type, CIRCUIT_BREAKER_CONFIGS["default"])

def get_retry_config(operation_type: str) -> Dict[str, Any]:
    """
    Get retry configuration for a specific operation type.
    
    Args:
        operation_type: Type of operation (feature_store, data_pipeline, etc.)
        
    Returns:
        Retry configuration for the specified operation type
    """
    return RETRY_CONFIGS.get(operation_type, RETRY_CONFIGS["default"])

def get_bulkhead_config(operation_type: str) -> Dict[str, Any]:
    """
    Get bulkhead configuration for a specific operation type.
    
    Args:
        operation_type: Type of operation (feature_store, data_pipeline, etc.)
        
    Returns:
        Bulkhead configuration for the specified operation type
    """
    return BULKHEAD_CONFIGS.get(operation_type, BULKHEAD_CONFIGS["default"])

def get_timeout_config(operation_type: str) -> float:
    """
    Get timeout configuration for a specific operation type.
    
    Args:
        operation_type: Type of operation (feature_store, data_pipeline, etc.)
        
    Returns:
        Timeout in seconds for the specified operation type
    """
    return TIMEOUT_CONFIGS.get(operation_type, TIMEOUT_CONFIGS["default"])
