"""
Resilience Configuration Module

This module provides standardized configuration for resilience patterns across the platform.
It includes configurations for circuit breaker, retry, bulkhead, and timeout patterns.
"""

from typing import Dict, Any, Optional, List, Type, Union
from pydantic import BaseModel, Field
import logging

# Default logger
logger = logging.getLogger(__name__)


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker pattern."""
    
    failure_threshold: int = Field(
        default=5,
        description="Number of failures before opening the circuit"
    )
    reset_timeout_seconds: float = Field(
        default=30.0,
        description="Time in seconds before attempting to close the circuit"
    )
    half_open_max_calls: int = Field(
        default=1,
        description="Maximum number of calls allowed in half-open state"
    )


class RetryConfig(BaseModel):
    """Configuration for retry pattern."""
    
    max_attempts: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    base_delay: float = Field(
        default=1.0,
        description="Initial delay between retries in seconds"
    )
    max_delay: float = Field(
        default=60.0,
        description="Maximum delay between retries in seconds"
    )
    backoff_factor: float = Field(
        default=2.0,
        description="Backoff factor for exponential delay"
    )
    jitter: bool = Field(
        default=True,
        description="Whether to add jitter to delay"
    )


class BulkheadConfig(BaseModel):
    """Configuration for bulkhead pattern."""
    
    max_concurrent: int = Field(
        default=10,
        description="Maximum number of concurrent executions"
    )
    max_queue: int = Field(
        default=20,
        description="Maximum size of the waiting queue"
    )


class TimeoutConfig(BaseModel):
    """Configuration for timeout pattern."""
    
    timeout_seconds: float = Field(
        default=30.0,
        description="Timeout in seconds"
    )


class ResilienceConfig(BaseModel):
    """Combined configuration for all resilience patterns."""
    
    # General configuration
    service_name: str = Field(
        ...,
        description="Name of the service"
    )
    operation_name: str = Field(
        ...,
        description="Name of the operation"
    )
    
    # Circuit breaker configuration
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Whether to enable circuit breaker"
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration"
    )
    
    # Retry configuration
    enable_retry: bool = Field(
        default=True,
        description="Whether to enable retry"
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Retry configuration"
    )
    
    # Bulkhead configuration
    enable_bulkhead: bool = Field(
        default=True,
        description="Whether to enable bulkhead"
    )
    bulkhead: BulkheadConfig = Field(
        default_factory=BulkheadConfig,
        description="Bulkhead configuration"
    )
    
    # Timeout configuration
    enable_timeout: bool = Field(
        default=True,
        description="Whether to enable timeout"
    )
    timeout: TimeoutConfig = Field(
        default_factory=TimeoutConfig,
        description="Timeout configuration"
    )
    
    # Exception handling
    expected_exceptions: List[Type[Exception]] = Field(
        default_factory=lambda: [Exception],
        description="Exceptions that should be handled by resilience patterns"
    )


# Predefined configurations for different service types
class ResilienceProfiles:
    """Predefined resilience profiles for different service types."""
    
    # Critical services that require very robust handling
    CRITICAL_SERVICE = ResilienceConfig(
        service_name="critical-service",
        operation_name="critical-operation",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            reset_timeout_seconds=60.0,
            half_open_max_calls=1
        ),
        retry=RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            backoff_factor=2.0,
            jitter=True
        ),
        bulkhead=BulkheadConfig(
            max_concurrent=5,
            max_queue=10
        ),
        timeout=TimeoutConfig(
            timeout_seconds=10.0
        )
    )
    
    # Standard services with balanced resilience
    STANDARD_SERVICE = ResilienceConfig(
        service_name="standard-service",
        operation_name="standard-operation",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=5,
            reset_timeout_seconds=30.0,
            half_open_max_calls=2
        ),
        retry=RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=60.0,
            backoff_factor=2.0,
            jitter=True
        ),
        bulkhead=BulkheadConfig(
            max_concurrent=10,
            max_queue=20
        ),
        timeout=TimeoutConfig(
            timeout_seconds=30.0
        )
    )
    
    # High-throughput services that need to handle many requests
    HIGH_THROUGHPUT_SERVICE = ResilienceConfig(
        service_name="high-throughput-service",
        operation_name="high-throughput-operation",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=10,
            reset_timeout_seconds=15.0,
            half_open_max_calls=5
        ),
        retry=RetryConfig(
            max_attempts=2,
            base_delay=0.1,
            max_delay=5.0,
            backoff_factor=2.0,
            jitter=True
        ),
        bulkhead=BulkheadConfig(
            max_concurrent=50,
            max_queue=100
        ),
        timeout=TimeoutConfig(
            timeout_seconds=5.0
        )
    )
    
    # Database operations that need careful handling
    DATABASE_OPERATION = ResilienceConfig(
        service_name="database-service",
        operation_name="database-operation",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            reset_timeout_seconds=60.0,
            half_open_max_calls=1
        ),
        retry=RetryConfig(
            max_attempts=3,
            base_delay=0.5,
            max_delay=10.0,
            backoff_factor=2.0,
            jitter=True
        ),
        bulkhead=BulkheadConfig(
            max_concurrent=20,
            max_queue=30
        ),
        timeout=TimeoutConfig(
            timeout_seconds=15.0
        )
    )
    
    # External API calls that might be unreliable
    EXTERNAL_API = ResilienceConfig(
        service_name="external-api",
        operation_name="api-call",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            reset_timeout_seconds=60.0,
            half_open_max_calls=1
        ),
        retry=RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            backoff_factor=2.0,
            jitter=True
        ),
        bulkhead=BulkheadConfig(
            max_concurrent=10,
            max_queue=20
        ),
        timeout=TimeoutConfig(
            timeout_seconds=10.0
        )
    )
    
    # Broker API calls that need special handling
    BROKER_API = ResilienceConfig(
        service_name="broker-api",
        operation_name="broker-call",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            reset_timeout_seconds=120.0,
            half_open_max_calls=1
        ),
        retry=RetryConfig(
            max_attempts=3,
            base_delay=2.0,
            max_delay=60.0,
            backoff_factor=2.0,
            jitter=True
        ),
        bulkhead=BulkheadConfig(
            max_concurrent=5,
            max_queue=10
        ),
        timeout=TimeoutConfig(
            timeout_seconds=30.0
        )
    )
    
    # Market data operations that need to be fast
    MARKET_DATA = ResilienceConfig(
        service_name="market-data",
        operation_name="data-fetch",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=5,
            reset_timeout_seconds=30.0,
            half_open_max_calls=2
        ),
        retry=RetryConfig(
            max_attempts=3,
            base_delay=0.5,
            max_delay=5.0,
            backoff_factor=2.0,
            jitter=True
        ),
        bulkhead=BulkheadConfig(
            max_concurrent=20,
            max_queue=50
        ),
        timeout=TimeoutConfig(
            timeout_seconds=5.0
        )
    )


# Function to get a resilience configuration for a specific service and operation
def get_resilience_config(
    service_name: str,
    operation_name: str,
    service_type: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> ResilienceConfig:
    """
    Get a resilience configuration for a specific service and operation.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        custom_config: Custom configuration to override defaults
        
    Returns:
        ResilienceConfig: Resilience configuration
    """
    # Start with the standard configuration
    config = ResilienceProfiles.STANDARD_SERVICE.model_copy(deep=True)
    config.service_name = service_name
    config.operation_name = operation_name
    
    # Apply service type specific configuration
    if service_type:
        if service_type.lower() == "critical":
            config = ResilienceProfiles.CRITICAL_SERVICE.model_copy(deep=True)
            config.service_name = service_name
            config.operation_name = operation_name
        elif service_type.lower() == "high-throughput":
            config = ResilienceProfiles.HIGH_THROUGHPUT_SERVICE.model_copy(deep=True)
            config.service_name = service_name
            config.operation_name = operation_name
        elif service_type.lower() == "database":
            config = ResilienceProfiles.DATABASE_OPERATION.model_copy(deep=True)
            config.service_name = service_name
            config.operation_name = operation_name
        elif service_type.lower() == "external-api":
            config = ResilienceProfiles.EXTERNAL_API.model_copy(deep=True)
            config.service_name = service_name
            config.operation_name = operation_name
        elif service_type.lower() == "broker-api":
            config = ResilienceProfiles.BROKER_API.model_copy(deep=True)
            config.service_name = service_name
            config.operation_name = operation_name
        elif service_type.lower() == "market-data":
            config = ResilienceProfiles.MARKET_DATA.model_copy(deep=True)
            config.service_name = service_name
            config.operation_name = operation_name
    
    # Apply custom configuration
    if custom_config:
        # Update top-level fields
        for key, value in custom_config.items():
            if hasattr(config, key) and not isinstance(value, dict):
                setattr(config, key, value)
        
        # Update nested configurations
        if "circuit_breaker" in custom_config and isinstance(custom_config["circuit_breaker"], dict):
            for key, value in custom_config["circuit_breaker"].items():
                if hasattr(config.circuit_breaker, key):
                    setattr(config.circuit_breaker, key, value)
        
        if "retry" in custom_config and isinstance(custom_config["retry"], dict):
            for key, value in custom_config["retry"].items():
                if hasattr(config.retry, key):
                    setattr(config.retry, key, value)
        
        if "bulkhead" in custom_config and isinstance(custom_config["bulkhead"], dict):
            for key, value in custom_config["bulkhead"].items():
                if hasattr(config.bulkhead, key):
                    setattr(config.bulkhead, key, value)
        
        if "timeout" in custom_config and isinstance(custom_config["timeout"], dict):
            for key, value in custom_config["timeout"].items():
                if hasattr(config.timeout, key):
                    setattr(config.timeout, key, value)
    
    return config