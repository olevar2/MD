"""
Caching package for common library.

This package provides caching utilities for the forex trading platform,
including Redis-based distributed caching, local memory caching, and
cache invalidation strategies.

Features:
- Adaptive caching with TTL management
- Predictive caching with access pattern analysis
- Redis integration for distributed caching
- Cache key generation and management
- Cache invalidation strategies
- Decorators for easy caching
"""

# Legacy cache service
from common_lib.caching.cache_service import (
    CacheService,
    get_cache_service,
    cache_result,
    invalidate_cache,
    clear_cache
)

# Cache key utilities
from common_lib.caching.cache_key import (
    CacheKey,
    generate_cache_key
)

# Cache invalidation strategies
from common_lib.caching.invalidation import (
    CacheInvalidationStrategy,
    TimestampInvalidationStrategy,
    VersionInvalidationStrategy,
    DependencyInvalidationStrategy,
    EventInvalidationStrategy
)

# Adaptive cache manager
from common_lib.caching.adaptive_cache_manager import (
    AdaptiveCacheManager,
    CacheEntry,
    cached,
    get_cache_manager
)

# Predictive cache manager
from common_lib.caching.predictive_cache_manager import (
    PredictiveCacheManager,
    AccessPattern,
    PrecomputationTask,
    get_predictive_cache_manager
)

__all__ = [
    # Legacy Cache Service
    "CacheService",
    "get_cache_service",
    "cache_result",
    "invalidate_cache",
    "clear_cache",

    # Cache Key Utilities
    "CacheKey",
    "generate_cache_key",

    # Cache Invalidation Strategies
    "CacheInvalidationStrategy",
    "TimestampInvalidationStrategy",
    "VersionInvalidationStrategy",
    "DependencyInvalidationStrategy",
    "EventInvalidationStrategy",

    # Adaptive Cache Manager
    "AdaptiveCacheManager",
    "CacheEntry",
    "cached",
    "get_cache_manager",

    # Predictive Cache Manager
    "PredictiveCacheManager",
    "AccessPattern",
    "PrecomputationTask",
    "get_predictive_cache_manager"
]
