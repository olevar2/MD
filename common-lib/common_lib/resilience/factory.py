"""
Resilience Factory Module

This module provides factory functions for creating resilience components with standardized configurations.
"""

import logging
from typing import Optional, Dict, Any, List, Type, Callable, TypeVar, Union, Awaitable

from common_lib.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from common_lib.resilience.retry import RetryPolicy
from common_lib.resilience.bulkhead import Bulkhead
from common_lib.resilience.timeout import Timeout
import common_lib.resilience.resilience
from common_lib.resilience.resilience import Resilience
from common_lib.resilience.config import (
    get_resilience_config,
    CircuitBreakerConfig as ConfigCircuitBreakerConfig,
    RetryConfig,
    BulkheadConfig,
    TimeoutConfig,
    ResilienceConfig
)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Default logger
logger = logging.getLogger(__name__)


def create_circuit_breaker(
    service_name: str,
    resource_name: str,
    config: Optional[Union[CircuitBreakerConfig, Dict[str, Any]]] = None,
    service_type: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None
) -> CircuitBreaker:
    """
    Create a circuit breaker with standardized configuration.
    
    Args:
        service_name: Name of the service
        resource_name: Name of the resource
        config: Circuit breaker configuration (optional)
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        logger_instance: Logger instance (optional)
        
    Returns:
        CircuitBreaker: Circuit breaker instance
    """
    # Get logger
    log = logger_instance or logger.getChild(f"circuit_breaker.{service_name}.{resource_name}")
    
    # Get configuration
    if config is None:
        resilience_config = get_resilience_config(
            service_name=service_name,
            operation_name=resource_name,
            service_type=service_type
        )
        cb_config = CircuitBreakerConfig(
            name=f"{service_name}.{resource_name}",
            failure_threshold=resilience_config.circuit_breaker.failure_threshold,
            reset_timeout_seconds=resilience_config.circuit_breaker.reset_timeout_seconds,
            half_open_max_calls=resilience_config.circuit_breaker.half_open_max_calls
        )
    elif isinstance(config, dict):
        cb_config = CircuitBreakerConfig(**config)
    else:
        cb_config = config
    
    # Create circuit breaker
    return CircuitBreaker(
        config=cb_config,
        logger=log
    )


def create_retry_policy(
    service_name: str,
    operation_name: str,
    config: Optional[Union[RetryConfig, Dict[str, Any]]] = None,
    service_type: Optional[str] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> RetryPolicy:
    """
    Create a retry policy with standardized configuration.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Retry policy configuration (optional)
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        exceptions: List of exceptions to retry on (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        RetryPolicy: Retry policy instance
    """
    # Get logger
    log = logger_instance or logger.getChild(f"retry_policy.{service_name}.{operation_name}")
    
    # Get configuration
    if config is None:
        resilience_config = get_resilience_config(
            service_name=service_name,
            operation_name=operation_name,
            service_type=service_type
        )
        retry_config = resilience_config.retry
    elif isinstance(config, dict):
        retry_config = RetryConfig(**config)
    else:
        retry_config = config
    
    # Create retry policy
    return RetryPolicy(
        retries=retry_config.max_attempts,
        delay=retry_config.base_delay,
        max_delay=retry_config.max_delay,
        backoff=retry_config.backoff_factor,
        exceptions=exceptions,
        logger=log
    )


def create_bulkhead(
    service_name: str,
    operation_name: str,
    config: Optional[Union[BulkheadConfig, Dict[str, Any]]] = None,
    service_type: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Bulkhead:
    """
    Create a bulkhead with standardized configuration.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Bulkhead configuration (optional)
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        logger_instance: Logger instance (optional)
        
    Returns:
        Bulkhead: Bulkhead instance
    """
    # Get logger
    log = logger_instance or logger.getChild(f"bulkhead.{service_name}.{operation_name}")
    
    # Get configuration
    if config is None:
        resilience_config = get_resilience_config(
            service_name=service_name,
            operation_name=operation_name,
            service_type=service_type
        )
        bulkhead_config = resilience_config.bulkhead
    elif isinstance(config, dict):
        bulkhead_config = BulkheadConfig(**config)
    else:
        bulkhead_config = config
    
    # Create bulkhead
    return Bulkhead(
        name=f"{service_name}.{operation_name}",
        max_concurrent_calls=bulkhead_config.max_concurrent,
        max_queue_size=bulkhead_config.max_queue,
        logger=log
    )


def create_timeout(
    service_name: str,
    operation_name: str,
    config: Optional[Union[TimeoutConfig, Dict[str, Any], float]] = None,
    service_type: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Timeout:
    """
    Create a timeout with standardized configuration.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Timeout configuration (optional)
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        logger_instance: Logger instance (optional)
        
    Returns:
        Timeout: Timeout instance
    """
    # Get logger
    log = logger_instance or logger.getChild(f"timeout.{service_name}.{operation_name}")
    
    # Get configuration
    if config is None:
        resilience_config = get_resilience_config(
            service_name=service_name,
            operation_name=operation_name,
            service_type=service_type
        )
        timeout_seconds = resilience_config.timeout.timeout_seconds
    elif isinstance(config, (int, float)):
        timeout_seconds = float(config)
    elif isinstance(config, dict):
        timeout_config = TimeoutConfig(**config)
        timeout_seconds = timeout_config.timeout_seconds
    else:
        timeout_seconds = config.timeout_seconds
    
    # Create timeout
    return Timeout(
        timeout=timeout_seconds,
        operation=f"{service_name}.{operation_name}",
        logger=log
    )


def create_resilience(
    service_name: str,
    operation_name: str,
    config: Optional[Union[Dict[str, Any]]] = None,
    service_type: Optional[str] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Resilience:
    """
    Create a resilience instance with standardized configuration.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Resilience configuration (optional)
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        exceptions: List of exceptions to handle (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Resilience: Resilience instance
    """
    # Get logger
    log = logger_instance or logger.getChild(f"resilience.{service_name}.{operation_name}")
    
    # Get configuration
    if config is None:
        resilience_config = get_resilience_config(
            service_name=service_name,
            operation_name=operation_name,
            service_type=service_type
        )
    elif isinstance(config, dict):
        # Create a base config and update it with the provided values
        base_config = get_resilience_config(
            service_name=service_name,
            operation_name=operation_name,
            service_type=service_type
        )
        resilience_config = base_config.model_copy(update=config)
    else:
        resilience_config = config
        
    # Convert to old-style ResilienceConfig
    old_config = common_lib.resilience.resilience.ResilienceConfig(
        service_name=resilience_config.service_name,
        operation_name=resilience_config.operation_name,
        enable_circuit_breaker=resilience_config.enable_circuit_breaker,
        failure_threshold=resilience_config.circuit_breaker.failure_threshold,
        recovery_timeout=resilience_config.circuit_breaker.reset_timeout_seconds,
        enable_retry=resilience_config.enable_retry,
        max_retries=resilience_config.retry.max_attempts,
        retry_delay=resilience_config.retry.base_delay,
        max_delay=resilience_config.retry.max_delay,
        backoff=resilience_config.retry.backoff_factor,
        enable_bulkhead=resilience_config.enable_bulkhead,
        max_concurrent_calls=resilience_config.bulkhead.max_concurrent,
        max_queue_size=resilience_config.bulkhead.max_queue,
        enable_timeout=resilience_config.enable_timeout,
        timeout=resilience_config.timeout.timeout_seconds,
        expected_exceptions=resilience_config.expected_exceptions
    )
    
    # Update exceptions if provided
    if exceptions:
        old_config.expected_exceptions = exceptions
    
    # Create resilience instance
    return Resilience(
        config=old_config,
        logger=log
    )


# Registry for resilience components
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_retry_policies: Dict[str, RetryPolicy] = {}
_bulkheads: Dict[str, Bulkhead] = {}
_timeouts: Dict[str, Timeout] = {}
_resilience_instances: Dict[str, Resilience] = {}


def get_circuit_breaker(
    service_name: str,
    resource_name: str,
    config: Optional[Union[CircuitBreakerConfig, Dict[str, Any]]] = None,
    service_type: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None
) -> CircuitBreaker:
    """
    Get or create a circuit breaker with standardized configuration.
    
    Args:
        service_name: Name of the service
        resource_name: Name of the resource
        config: Circuit breaker configuration (optional)
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        logger_instance: Logger instance (optional)
        
    Returns:
        CircuitBreaker: Circuit breaker instance
    """
    key = f"{service_name}.{resource_name}"
    if key not in _circuit_breakers:
        _circuit_breakers[key] = create_circuit_breaker(
            service_name=service_name,
            resource_name=resource_name,
            config=config,
            service_type=service_type,
            logger_instance=logger_instance
        )
    return _circuit_breakers[key]


def get_retry_policy(
    service_name: str,
    operation_name: str,
    config: Optional[Union[RetryConfig, Dict[str, Any]]] = None,
    service_type: Optional[str] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> RetryPolicy:
    """
    Get or create a retry policy with standardized configuration.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Retry policy configuration (optional)
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        exceptions: List of exceptions to retry on (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        RetryPolicy: Retry policy instance
    """
    key = f"{service_name}.{operation_name}"
    if key not in _retry_policies:
        _retry_policies[key] = create_retry_policy(
            service_name=service_name,
            operation_name=operation_name,
            config=config,
            service_type=service_type,
            exceptions=exceptions,
            logger_instance=logger_instance
        )
    return _retry_policies[key]


def get_bulkhead(
    service_name: str,
    operation_name: str,
    config: Optional[Union[BulkheadConfig, Dict[str, Any]]] = None,
    service_type: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Bulkhead:
    """
    Get or create a bulkhead with standardized configuration.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Bulkhead configuration (optional)
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        logger_instance: Logger instance (optional)
        
    Returns:
        Bulkhead: Bulkhead instance
    """
    key = f"{service_name}.{operation_name}"
    if key not in _bulkheads:
        _bulkheads[key] = create_bulkhead(
            service_name=service_name,
            operation_name=operation_name,
            config=config,
            service_type=service_type,
            logger_instance=logger_instance
        )
    return _bulkheads[key]


def get_timeout(
    service_name: str,
    operation_name: str,
    config: Optional[Union[TimeoutConfig, Dict[str, Any], float]] = None,
    service_type: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Timeout:
    """
    Get or create a timeout with standardized configuration.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Timeout configuration (optional)
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        logger_instance: Logger instance (optional)
        
    Returns:
        Timeout: Timeout instance
    """
    key = f"{service_name}.{operation_name}"
    if key not in _timeouts:
        _timeouts[key] = create_timeout(
            service_name=service_name,
            operation_name=operation_name,
            config=config,
            service_type=service_type,
            logger_instance=logger_instance
        )
    return _timeouts[key]


def get_resilience(
    service_name: str,
    operation_name: str,
    config: Optional[Union[Dict[str, Any]]] = None,
    service_type: Optional[str] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Resilience:
    """
    Get or create a resilience instance with standardized configuration.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Resilience configuration (optional)
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        exceptions: List of exceptions to handle (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Resilience: Resilience instance
    """
    key = f"{service_name}.{operation_name}"
    if key not in _resilience_instances:
        _resilience_instances[key] = create_resilience(
            service_name=service_name,
            operation_name=operation_name,
            config=config,
            service_type=service_type,
            exceptions=exceptions,
            logger_instance=logger_instance
        )
    return _resilience_instances[key]