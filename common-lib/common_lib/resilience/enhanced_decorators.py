"""
Enhanced Decorators Module

This module provides enhanced decorators for resilience patterns that use standardized configurations.
"""

import logging
import functools
from typing import Callable, Any, Optional, Dict, List, Type, TypeVar, Union, Awaitable

from common_lib.resilience.factory import (
    get_circuit_breaker,
    get_retry_policy,
    get_bulkhead,
    get_timeout,
    get_resilience
)
from common_lib.resilience.config import (
    CircuitBreakerConfig,
    RetryConfig,
    BulkheadConfig,
    TimeoutConfig,
    ResilienceConfig as StandardResilienceConfig
)

# Type variables for generic functions
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')
R = TypeVar('R')

# Default logger
logger = logging.getLogger(__name__)


def with_standard_circuit_breaker(
    service_name: str,
    resource_name: str,
    service_type: Optional[str] = None,
    config: Optional[Union[CircuitBreakerConfig, Dict[str, Any]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to apply a standardized circuit breaker to a function.
    
    Args:
        service_name: Name of the service
        resource_name: Name of the resource
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        config: Circuit breaker configuration (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get circuit breaker
            cb = get_circuit_breaker(
                service_name=service_name,
                resource_name=resource_name,
                config=config,
                service_type=service_type,
                logger_instance=logger_instance
            )
            
            # Execute function with circuit breaker
            return await cb.execute(lambda: func(*args, **kwargs))
        
        return wrapper  # type: ignore
    
    return decorator


def with_standard_retry(
    service_name: str,
    operation_name: str,
    service_type: Optional[str] = None,
    config: Optional[Union[RetryConfig, Dict[str, Any]]] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to apply a standardized retry policy to a function.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        config: Retry policy configuration (optional)
        exceptions: List of exceptions to retry on (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get retry policy
            retry = get_retry_policy(
                service_name=service_name,
                operation_name=operation_name,
                config=config,
                service_type=service_type,
                exceptions=exceptions,
                logger_instance=logger_instance
            )
            
            # Execute function with retry policy
            return await retry.execute_async(lambda: func(*args, **kwargs))
        
        return wrapper  # type: ignore
    
    return decorator


def with_standard_bulkhead(
    service_name: str,
    operation_name: str,
    service_type: Optional[str] = None,
    config: Optional[Union[BulkheadConfig, Dict[str, Any]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to apply a standardized bulkhead to a function.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        config: Bulkhead configuration (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get bulkhead
            bh = get_bulkhead(
                service_name=service_name,
                operation_name=operation_name,
                config=config,
                service_type=service_type,
                logger_instance=logger_instance
            )
            
            # Execute function with bulkhead
            return await bh.execute(lambda: func(*args, **kwargs))
        
        return wrapper  # type: ignore
    
    return decorator


def with_standard_timeout(
    service_name: str,
    operation_name: str,
    service_type: Optional[str] = None,
    config: Optional[Union[TimeoutConfig, Dict[str, Any], float]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to apply a standardized timeout to a function.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        config: Timeout configuration (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get timeout
            to = get_timeout(
                service_name=service_name,
                operation_name=operation_name,
                config=config,
                service_type=service_type,
                logger_instance=logger_instance
            )
            
            # Execute function with timeout
            return await to.execute(lambda: func(*args, **kwargs))
        
        return wrapper  # type: ignore
    
    return decorator


def with_standard_resilience(
    service_name: str,
    operation_name: str,
    service_type: Optional[str] = None,
    config: Optional[Union[StandardResilienceConfig, Dict[str, Any]]] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to apply standardized resilience patterns to a function.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        service_type: Type of service (critical, standard, high-throughput, database, external-api, broker-api, market-data)
        config: Resilience configuration (optional)
        exceptions: List of exceptions to handle (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get resilience
            res = get_resilience(
                service_name=service_name,
                operation_name=operation_name,
                config=config,
                service_type=service_type,
                exceptions=exceptions,
                logger_instance=logger_instance
            )
            
            # Execute function with resilience
            return await res.execute_async(lambda: func(*args, **kwargs))
        
        return wrapper  # type: ignore
    
    return decorator


# Specialized decorators for common service types

def with_database_resilience(
    service_name: str,
    operation_name: str,
    config: Optional[Union[StandardResilienceConfig, Dict[str, Any]]] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to apply database-specific resilience patterns to a function.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Resilience configuration (optional)
        exceptions: List of exceptions to handle (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    return with_standard_resilience(
        service_name=service_name,
        operation_name=operation_name,
        service_type="database",
        config=config,
        exceptions=exceptions,
        logger_instance=logger_instance
    )


def with_broker_api_resilience(
    service_name: str,
    operation_name: str,
    config: Optional[Union[StandardResilienceConfig, Dict[str, Any]]] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to apply broker API-specific resilience patterns to a function.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Resilience configuration (optional)
        exceptions: List of exceptions to handle (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    return with_standard_resilience(
        service_name=service_name,
        operation_name=operation_name,
        service_type="broker-api",
        config=config,
        exceptions=exceptions,
        logger_instance=logger_instance
    )


def with_market_data_resilience(
    service_name: str,
    operation_name: str,
    config: Optional[Union[StandardResilienceConfig, Dict[str, Any]]] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to apply market data-specific resilience patterns to a function.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Resilience configuration (optional)
        exceptions: List of exceptions to handle (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    return with_standard_resilience(
        service_name=service_name,
        operation_name=operation_name,
        service_type="market-data",
        config=config,
        exceptions=exceptions,
        logger_instance=logger_instance
    )


def with_external_api_resilience(
    service_name: str,
    operation_name: str,
    config: Optional[Union[StandardResilienceConfig, Dict[str, Any]]] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to apply external API-specific resilience patterns to a function.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Resilience configuration (optional)
        exceptions: List of exceptions to handle (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    return with_standard_resilience(
        service_name=service_name,
        operation_name=operation_name,
        service_type="external-api",
        config=config,
        exceptions=exceptions,
        logger_instance=logger_instance
    )


def with_critical_resilience(
    service_name: str,
    operation_name: str,
    config: Optional[Union[StandardResilienceConfig, Dict[str, Any]]] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to apply critical service-specific resilience patterns to a function.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Resilience configuration (optional)
        exceptions: List of exceptions to handle (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    return with_standard_resilience(
        service_name=service_name,
        operation_name=operation_name,
        service_type="critical",
        config=config,
        exceptions=exceptions,
        logger_instance=logger_instance
    )


def with_high_throughput_resilience(
    service_name: str,
    operation_name: str,
    config: Optional[Union[StandardResilienceConfig, Dict[str, Any]]] = None,
    exceptions: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to apply high-throughput service-specific resilience patterns to a function.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Resilience configuration (optional)
        exceptions: List of exceptions to handle (optional)
        logger_instance: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    return with_standard_resilience(
        service_name=service_name,
        operation_name=operation_name,
        service_type="high-throughput",
        config=config,
        exceptions=exceptions,
        logger_instance=logger_instance
    )