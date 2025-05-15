"""
Cache decorators.

This module provides decorators for caching function results.
"""
import functools
import inspect
import logging
from typing import Any, Callable, Optional, TypeVar, cast

from common_lib.caching.cache import Cache, CacheKey

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def cached(
    cache: Cache,
    prefix: str,
    ttl: Optional[int] = None,
    key_generator: Optional[Callable[..., str]] = None
) -> Callable[[F], F]:
    """
    Decorator for caching function results.
    
    Args:
        cache: The cache to use
        prefix: The prefix for the cache key
        ttl: Time to live in seconds (optional)
        key_generator: Function to generate the cache key (optional)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate cache key
                if key_generator:
                    key = key_generator(*args, **kwargs)
                else:
                    key = CacheKey.generate(prefix, *args, kwargs)
                
                # Try to get from cache
                cached_value = await cache.get(key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {key}")
                    return cached_value
                
                # Call the function
                logger.debug(f"Cache miss for {key}")
                result = await func(*args, **kwargs)
                
                # Cache the result
                await cache.set(key, result, ttl)
                
                return result
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate cache key
                if key_generator:
                    key = key_generator(*args, **kwargs)
                else:
                    key = CacheKey.generate(prefix, *args, kwargs)
                
                # Try to get from cache
                cached_value = cache.get(key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {key}")
                    return cached_value
                
                # Call the function
                logger.debug(f"Cache miss for {key}")
                result = func(*args, **kwargs)
                
                # Cache the result
                cache.set(key, result, ttl)
                
                return result
            
            return cast(F, sync_wrapper)
    
    return decorator


def invalidate_cache(
    cache: Cache,
    prefix: str,
    key_generator: Optional[Callable[..., str]] = None
) -> Callable[[F], F]:
    """
    Decorator for invalidating cache entries.
    
    Args:
        cache: The cache to use
        prefix: The prefix for the cache key
        key_generator: Function to generate the cache key (optional)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate cache key
                if key_generator:
                    key = key_generator(*args, **kwargs)
                else:
                    key = CacheKey.generate(prefix, *args, kwargs)
                
                # Call the function
                result = await func(*args, **kwargs)
                
                # Invalidate the cache
                await cache.delete(key)
                logger.debug(f"Invalidated cache for {key}")
                
                return result
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate cache key
                if key_generator:
                    key = key_generator(*args, **kwargs)
                else:
                    key = CacheKey.generate(prefix, *args, kwargs)
                
                # Call the function
                result = func(*args, **kwargs)
                
                # Invalidate the cache
                cache.delete(key)
                logger.debug(f"Invalidated cache for {key}")
                
                return result
            
            return cast(F, sync_wrapper)
    
    return decorator