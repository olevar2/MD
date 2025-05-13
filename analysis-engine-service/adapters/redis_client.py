"""
Resilient Redis Client Module

This module provides resilient Redis operations with:
1. Connection pooling with proper configuration
2. Retry mechanisms for transient Redis errors
3. Circuit breakers to prevent cascading failures
4. Timeout handling for Redis operations
"""
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, Coroutine
import redis
from redis import Redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from analysis_engine.config.settings import Settings
from analysis_engine.resilience import retry_with_policy, timeout_handler, create_circuit_breaker
from analysis_engine.resilience.config import get_circuit_breaker_config, get_retry_config, get_timeout_config
T = TypeVar('T')
R = TypeVar('R')
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


class ResilientRedisClient:
    """
    Redis client with resilience patterns.
    
    This class provides:
    """
    provides class.
    
    Attributes:
        Add attributes here
    """

    1. Connection pooling with proper configuration
    2. Retry mechanisms for transient Redis errors
    3. Circuit breakers to prevent cascading failures
    4. Timeout handling for Redis operations
    """

    def __init__(self, settings: Settings=None):
        """
        Initialize the resilient Redis client.
        
        Args:
            settings: Application settings containing Redis configuration
        """
        self.settings = settings or Settings()
        self.redis_url = self.settings.redis.url
        self.redis_password = self.settings.redis.password
        self.redis_db = self.settings.redis.db
        self.redis_socket_timeout = self.settings.redis.socket_timeout
        self.redis_socket_connect_timeout = (self.settings.redis.
            socket_connect_timeout)
        self.redis_pool = redis.ConnectionPool.from_url(self.redis_url,
            password=self.redis_password, db=self.redis_db, socket_timeout=
            self.redis_socket_timeout, socket_connect_timeout=self.
            redis_socket_connect_timeout, max_connections=50)
        self.circuit_breaker = create_circuit_breaker(service_name=
            'analysis_engine', resource_name='redis', config=
            get_circuit_breaker_config('redis'))
        self._is_initialized = True

    @with_resilience('get_redis_client')
    def get_redis_client(self) ->Redis:
        """
        Get a Redis client from the connection pool.
        
        Returns:
            Redis client
        """
        return Redis(connection_pool=self.redis_pool)

    @retry_with_policy(max_attempts=3, base_delay=0.5, max_delay=3.0,
        backoff_factor=2.0, jitter=True, exceptions=[RedisError,
        ConnectionError, TimeoutError], service_name='analysis_engine',
        operation_name='redis_operation')
    @timeout_handler(timeout_seconds=3.0)
    def execute_redis_operation(self, operation_func: Callable[[Redis], T]
        ) ->T:
        """
        Execute a Redis operation with resilience patterns.
        
        Args:
            operation_func: Function that takes a Redis client and returns a result
            
        Returns:
            Result of the operation function
        """
        return self.circuit_breaker.execute(self._execute_redis_operation,
            operation_func)

    @with_exception_handling
    def _execute_redis_operation(self, operation_func: Callable[[Redis], T]
        ) ->T:
        """
        Execute a Redis operation.
        
        Args:
            operation_func: Function that takes a Redis client and returns a result
            
        Returns:
            Result of the operation function
        """
        redis_client = self.get_redis_client()
        try:
            return operation_func(redis_client)
        except Exception as e:
            logger.error(f'Redis operation error: {str(e)}')
            raise

    def close(self) ->None:
        """Close Redis connections."""
        self.redis_pool.disconnect()

    @property
    def is_initialized(self) ->bool:
        """Check if Redis client is initialized."""
        return self._is_initialized


_redis_client = None


def get_redis_client() ->ResilientRedisClient:
    """
    Get the singleton Redis client instance.
    
    Returns:
        ResilientRedisClient instance
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = ResilientRedisClient()
    return _redis_client


def execute_redis_operation(operation_func: Callable[[Redis], T]) ->T:
    """
    Execute a Redis operation with resilience patterns.
    
    Args:
        operation_func: Function that takes a Redis client and returns a result
        
    Returns:
        Result of the operation function
    """
    return get_redis_client().execute_redis_operation(operation_func)
