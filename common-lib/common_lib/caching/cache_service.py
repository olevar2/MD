"""
Cache Service for Forex Trading Platform.

This module provides a standardized caching service with Redis support,
including TTL-based expiration, serialization, and monitoring.
"""

import os
import time
import logging
import functools
import threading
import hashlib
import json
from typing import Dict, Any, Optional, Union, Callable, TypeVar, cast, List, Tuple
from datetime import datetime, timedelta
import inspect

# Import secure serialization instead of pickle
from common_lib.caching.secure_serialization import SecureSerializer

# Redis imports
try:
    import redis
    from redis.exceptions import RedisError
except ImportError:
    # Use fakeredis for testing if redis is not available
    import fakeredis
    redis = fakeredis
    from fakeredis.exceptions import RedisError

# Prometheus metrics
from prometheus_client import Counter, Histogram, Summary

# Local imports
from common_lib.config.settings import AppSettings
from common_lib.monitoring.metrics import get_counter, get_histogram

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for function decorators
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

# Cache metrics
CACHE_HIT_COUNTER = get_counter(
    name="cache_hits_total",
    description="Total number of cache hits",
    labels=["cache_type", "service", "function"]
)

CACHE_MISS_COUNTER = get_counter(
    name="cache_misses_total",
    description="Total number of cache misses",
    labels=["cache_type", "service", "function"]
)

CACHE_ERROR_COUNTER = get_counter(
    name="cache_errors_total",
    description="Total number of cache errors",
    labels=["cache_type", "service", "function", "error_type"]
)

CACHE_LATENCY_HISTOGRAM = get_histogram(
    name="cache_operation_duration_seconds",
    description="Duration of cache operations",
    labels=["cache_type", "service", "operation"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

class CacheService:
    """
    Standardized caching service with Redis support.

    This service provides:
    - Redis-based distributed caching
    - Local memory caching fallback
    - TTL-based expiration
    - Serialization of complex objects
    - Monitoring and metrics
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        service_name: str = "forex-service",
        default_ttl: Optional[Union[int, timedelta]] = 300,
        local_cache_size: int = 1000,
        enable_local_cache: bool = True
    ):
        """
        Initialize the cache service.

        Args:
            redis_url: Redis connection URL (if None, uses environment variable)
            service_name: Name of the service (for metrics)
            default_ttl: Default time-to-live for cached items (in seconds or timedelta)
            local_cache_size: Maximum size of the local cache
            enable_local_cache: Whether to enable local memory caching
        """
        self.service_name = service_name
        self.default_ttl = default_ttl
        self.enable_local_cache = enable_local_cache

        # Initialize Redis client
        settings = AppSettings()
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_client = None

        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            self.redis_client.ping()  # Test connection
            logger.info(f"Successfully connected to Redis at {self.redis_url}")
        except (RedisError, AttributeError) as e:
            logger.warning(f"Failed to connect to Redis at {self.redis_url}: {e}")
            logger.warning("Falling back to local memory cache only")

        # Initialize local cache
        if enable_local_cache:
            self.local_cache: Dict[str, Tuple[Any, Optional[float]]] = {}
            self.local_cache_lock = threading.RLock()
            self.local_cache_size = local_cache_size
            logger.info(f"Local memory cache initialized with size {local_cache_size}")

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        start_time = time.time()

        # Try local cache first if enabled
        if self.enable_local_cache:
            with self.local_cache_lock:
                if key in self.local_cache:
                    value, expiry = self.local_cache[key]

                    # Check if expired
                    if expiry is None or expiry > time.time():
                        CACHE_HIT_COUNTER.labels(
                            cache_type="local",
                            service=self.service_name,
                            function="get"
                        ).inc()

                        CACHE_LATENCY_HISTOGRAM.labels(
                            cache_type="local",
                            service=self.service_name,
                            operation="get"
                        ).observe(time.time() - start_time)

                        return value
                    else:
                        # Expired, remove from local cache
                        del self.local_cache[key]

        # Try Redis if available
        if self.redis_client:
            try:
                redis_key = f"cache:{key}"
                serialized_value = self.redis_client.get(redis_key)

                if serialized_value is not None:
                    # Use secure deserialization instead of pickle
                    value = SecureSerializer.deserialize(serialized_value)

                    # Update local cache if enabled
                    if self.enable_local_cache:
                        self._update_local_cache(key, value, self._get_ttl(redis_key))

                    CACHE_HIT_COUNTER.labels(
                        cache_type="redis",
                        service=self.service_name,
                        function="get"
                    ).inc()

                    CACHE_LATENCY_HISTOGRAM.labels(
                        cache_type="redis",
                        service=self.service_name,
                        operation="get"
                    ).observe(time.time() - start_time)

                    return value
            except (RedisError, json.JSONDecodeError) as e:
                logger.warning(f"Error getting value from Redis for key '{key}': {e}")

                CACHE_ERROR_COUNTER.labels(
                    cache_type="redis",
                    service=self.service_name,
                    function="get",
                    error_type=type(e).__name__
                ).inc()

        # Cache miss
        CACHE_MISS_COUNTER.labels(
            cache_type="all",
            service=self.service_name,
            function="get"
        ).inc()

        CACHE_LATENCY_HISTOGRAM.labels(
            cache_type="all",
            service=self.service_name,
            operation="get"
        ).observe(time.time() - start_time)

        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live for the cached item (in seconds or timedelta)

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        ttl_seconds = self._normalize_ttl(ttl)
        success = False

        # Set in Redis if available
        if self.redis_client:
            try:
                redis_key = f"cache:{key}"
                # Use secure serialization instead of pickle
                serialized_value = SecureSerializer.serialize(value)

                if ttl_seconds is None:
                    success = self.redis_client.set(redis_key, serialized_value)
                else:
                    success = self.redis_client.setex(redis_key, ttl_seconds, serialized_value)

                CACHE_LATENCY_HISTOGRAM.labels(
                    cache_type="redis",
                    service=self.service_name,
                    operation="set"
                ).observe(time.time() - start_time)
            except (RedisError, json.JSONDecodeError) as e:
                logger.warning(f"Error setting value in Redis for key '{key}': {e}")

                CACHE_ERROR_COUNTER.labels(
                    cache_type="redis",
                    service=self.service_name,
                    function="set",
                    error_type=type(e).__name__
                ).inc()

        # Set in local cache if enabled
        if self.enable_local_cache:
            expiry = None if ttl_seconds is None else time.time() + ttl_seconds
            self._update_local_cache(key, value, ttl_seconds)
            success = True

            CACHE_LATENCY_HISTOGRAM.labels(
                cache_type="local",
                service=self.service_name,
                operation="set"
            ).observe(time.time() - start_time)

        return success

    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        success = False

        # Delete from Redis if available
        if self.redis_client:
            try:
                redis_key = f"cache:{key}"
                deleted = self.redis_client.delete(redis_key)
                success = deleted > 0

                CACHE_LATENCY_HISTOGRAM.labels(
                    cache_type="redis",
                    service=self.service_name,
                    operation="delete"
                ).observe(time.time() - start_time)
            except RedisError as e:
                logger.warning(f"Error deleting value from Redis for key '{key}': {e}")

                CACHE_ERROR_COUNTER.labels(
                    cache_type="redis",
                    service=self.service_name,
                    function="delete",
                    error_type=type(e).__name__
                ).inc()

        # Delete from local cache if enabled
        if self.enable_local_cache:
            with self.local_cache_lock:
                if key in self.local_cache:
                    del self.local_cache[key]
                    success = True

            CACHE_LATENCY_HISTOGRAM.labels(
                cache_type="local",
                service=self.service_name,
                operation="delete"
            ).observe(time.time() - start_time)

        return success

    def clear(self, pattern: str = "*") -> bool:
        """
        Clear all values from the cache matching a pattern.

        Args:
            pattern: Pattern to match keys

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        success = False

        # Clear from Redis if available
        if self.redis_client:
            try:
                redis_pattern = f"cache:{pattern}"
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(cursor, match=redis_pattern, count=100)
                    if keys:
                        self.redis_client.delete(*keys)
                    if cursor == 0:
                        break

                success = True

                CACHE_LATENCY_HISTOGRAM.labels(
                    cache_type="redis",
                    service=self.service_name,
                    operation="clear"
                ).observe(time.time() - start_time)
            except RedisError as e:
                logger.warning(f"Error clearing values from Redis for pattern '{pattern}': {e}")

                CACHE_ERROR_COUNTER.labels(
                    cache_type="redis",
                    service=self.service_name,
                    function="clear",
                    error_type=type(e).__name__
                ).inc()

        # Clear from local cache if enabled
        if self.enable_local_cache:
            with self.local_cache_lock:
                # Convert pattern to regex
                import re
                regex_pattern = pattern.replace("*", ".*")
                regex = re.compile(f"^{regex_pattern}$")

                # Find keys to delete
                keys_to_delete = [k for k in self.local_cache.keys() if regex.match(k)]

                # Delete keys
                for key in keys_to_delete:
                    del self.local_cache[key]

                success = True

            CACHE_LATENCY_HISTOGRAM.labels(
                cache_type="local",
                service=self.service_name,
                operation="clear"
            ).observe(time.time() - start_time)

        return success
