"""
Decorators for market-analysis-service.
"""

import logging
import functools
import json
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast
import asyncio

from common_lib.caching.cache_interface import CacheInterface

logger = logging.getLogger(__name__)

T = TypeVar("T")


def cached(cache: CacheInterface, prefix: str, ttl: Optional[int] = None) -> Callable:
    """
    Decorator for caching function results.
    
    Args:
        cache: Cache instance
        prefix: Cache key prefix
        ttl: TTL in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator function.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """
            Wrapper function.
            
            Args:
                args: Positional arguments
                kwargs: Keyword arguments
                
            Returns:
                Result of the decorated function
            """
            # Generate cache key
            key = _generate_cache_key(func, args, kwargs)
            cache_key = f"{prefix}:{key}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__} with key {cache_key}")
                return cached_value
            
            # Call function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl)
            logger.debug(f"Cache miss for {func.__name__} with key {cache_key}")
            
            return result
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            """
            Async wrapper function.
            
            Args:
                args: Positional arguments
                kwargs: Keyword arguments
                
            Returns:
                Result of the decorated function
            """
            # Generate cache key
            key = _generate_cache_key(func, args, kwargs)
            cache_key = f"{prefix}:{key}"
            
            # Try to get from cache
            cached_value = await cache.async_get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__} with key {cache_key}")
                return cached_value
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.async_set(cache_key, result, ttl)
            logger.debug(f"Cache miss for {func.__name__} with key {cache_key}")
            
            return result
        
        # Choose the appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def _generate_cache_key(func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
    """
    Generate a cache key for a function call.
    
    Args:
        func: Function
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Cache key
    """
    # Convert args and kwargs to a string
    args_str = json.dumps(args, sort_keys=True, default=str)
    kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
    
    # Generate a hash
    key = f"{func.__module__}.{func.__name__}:{args_str}:{kwargs_str}"
    return hashlib.md5(key.encode()).hexdigest()