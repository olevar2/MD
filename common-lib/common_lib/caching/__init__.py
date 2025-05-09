"""
Caching package for common library.

This package provides caching utilities for the forex trading platform,
including Redis-based distributed caching, local memory caching, and
cache invalidation strategies.
"""

from common_lib.caching.cache_service import (
    CacheService,
    get_cache_service,
    cache_result,
    invalidate_cache,
    clear_cache
)

from common_lib.caching.cache_key import (
    CacheKey,
    generate_cache_key
)

from common_lib.caching.invalidation import (
    CacheInvalidationStrategy,
    TimestampInvalidationStrategy,
    VersionInvalidationStrategy,
    DependencyInvalidationStrategy,
    EventInvalidationStrategy
)

__all__ = [
    # Cache Service
    "CacheService",
    "get_cache_service",
    "cache_result",
    "invalidate_cache",
    "clear_cache",
    
    # Cache Key
    "CacheKey",
    "generate_cache_key",
    
    # Invalidation Strategies
    "CacheInvalidationStrategy",
    "TimestampInvalidationStrategy",
    "VersionInvalidationStrategy",
    "DependencyInvalidationStrategy",
    "EventInvalidationStrategy"
]
