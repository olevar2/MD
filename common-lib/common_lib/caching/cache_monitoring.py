"""
Cache Monitoring Module for Forex Trading Platform.

This module provides monitoring for cache performance and health.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Union, List, Tuple, Set
from datetime import datetime, timedelta

import redis
from prometheus_client import Counter, Gauge, Histogram, Summary

# Import common library
from common_lib.monitoring.metrics import get_counter, get_gauge, get_histogram, get_summary

# Configure logging
logger = logging.getLogger(__name__)

# Cache metrics
CACHE_SIZE_GAUGE = get_gauge(
    name="cache_size",
    description="Number of items in the cache",
    labels=["cache_type", "service"]
)

CACHE_MEMORY_USAGE_GAUGE = get_gauge(
    name="cache_memory_usage_bytes",
    description="Memory usage of the cache in bytes",
    labels=["cache_type", "service"]
)

CACHE_HIT_RATIO_GAUGE = get_gauge(
    name="cache_hit_ratio",
    description="Cache hit ratio",
    labels=["cache_type", "service"]
)

CACHE_EVICTION_COUNTER = get_counter(
    name="cache_evictions_total",
    description="Total number of cache evictions",
    labels=["cache_type", "service", "reason"]
)

CACHE_EXPIRATION_COUNTER = get_counter(
    name="cache_expirations_total",
    description="Total number of cache expirations",
    labels=["cache_type", "service"]
)

class CacheMonitor:
    """
    Monitor for cache performance and health.
    
    This class provides monitoring for cache performance and health,
    including metrics collection and health checks.
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        service_name: str = "forex-service",
        update_interval: int = 60,  # 1 minute
        enable_metrics: bool = True
    ):
        """
        Initialize the cache monitor.
        
        Args:
            redis_client: Redis client
            service_name: Name of the service
            update_interval: Interval for updating metrics (in seconds)
            enable_metrics: Whether to enable metrics collection
        """
        self.redis_client = redis_client
        self.service_name = service_name
        self.update_interval = update_interval
        self.enable_metrics = enable_metrics
        
        # Cache statistics
        self.stats = {
            "redis": {
                "hits": 0,
                "misses": 0,
                "size": 0,
                "memory_usage": 0,
                "evictions": 0,
                "expirations": 0
            },
            "local": {
                "hits": 0,
                "misses": 0,
                "size": 0,
                "memory_usage": 0,
                "evictions": 0,
                "expirations": 0
            }
        }
        
        # Start monitoring thread
        if enable_metrics:
            self._start_monitoring_thread()
    
    def _start_monitoring_thread(self) -> None:
        """Start the monitoring thread."""
        def update_metrics():
    """
    Update metrics.
    
    """

            while True:
                try:
                    self.update_metrics()
                except Exception as e:
                    logger.warning(f"Error updating cache metrics: {e}")
                
                time.sleep(self.update_interval)
        
        thread = threading.Thread(target=update_metrics, daemon=True)
        thread.start()
        
        logger.info(f"Cache monitoring thread started with interval {self.update_interval}s")
    
    def update_metrics(self) -> None:
        """Update cache metrics."""
        # Update Redis metrics
        if self.redis_client:
            try:
                # Get Redis info
                info = self.redis_client.info()
                
                # Update Redis stats
                self.stats["redis"]["hits"] = info.get("keyspace_hits", 0)
                self.stats["redis"]["misses"] = info.get("keyspace_misses", 0)
                self.stats["redis"]["size"] = sum(db.get("keys", 0) for db_name, db in info.items() if db_name.startswith("db"))
                self.stats["redis"]["memory_usage"] = info.get("used_memory", 0)
                self.stats["redis"]["evictions"] = info.get("evicted_keys", 0)
                self.stats["redis"]["expirations"] = info.get("expired_keys", 0)
                
                # Update Redis metrics
                CACHE_SIZE_GAUGE.labels(
                    cache_type="redis",
                    service=self.service_name
                ).set(self.stats["redis"]["size"])
                
                CACHE_MEMORY_USAGE_GAUGE.labels(
                    cache_type="redis",
                    service=self.service_name
                ).set(self.stats["redis"]["memory_usage"])
                
                # Calculate hit ratio
                total_ops = self.stats["redis"]["hits"] + self.stats["redis"]["misses"]
                hit_ratio = self.stats["redis"]["hits"] / total_ops if total_ops > 0 else 0
                
                CACHE_HIT_RATIO_GAUGE.labels(
                    cache_type="redis",
                    service=self.service_name
                ).set(hit_ratio)
                
                # Update eviction and expiration counters
                CACHE_EVICTION_COUNTER.labels(
                    cache_type="redis",
                    service=self.service_name,
                    reason="memory"
                )._value.set(self.stats["redis"]["evictions"])
                
                CACHE_EXPIRATION_COUNTER.labels(
                    cache_type="redis",
                    service=self.service_name
                )._value.set(self.stats["redis"]["expirations"])
                
                logger.debug(f"Updated Redis cache metrics: {self.stats['redis']}")
            except redis.RedisError as e:
                logger.warning(f"Error getting Redis info: {e}")
    
    def record_hit(self, cache_type: str) -> None:
        """
        Record a cache hit.
        
        Args:
            cache_type: Type of cache (redis or local)
        """
        if cache_type in self.stats:
            self.stats[cache_type]["hits"] += 1
    
    def record_miss(self, cache_type: str) -> None:
        """
        Record a cache miss.
        
        Args:
            cache_type: Type of cache (redis or local)
        """
        if cache_type in self.stats:
            self.stats[cache_type]["misses"] += 1
    
    def record_eviction(self, cache_type: str, reason: str = "memory") -> None:
        """
        Record a cache eviction.
        
        Args:
            cache_type: Type of cache (redis or local)
            reason: Reason for eviction
        """
        if cache_type in self.stats:
            self.stats[cache_type]["evictions"] += 1
            
            CACHE_EVICTION_COUNTER.labels(
                cache_type=cache_type,
                service=self.service_name,
                reason=reason
            ).inc()
    
    def record_expiration(self, cache_type: str) -> None:
        """
        Record a cache expiration.
        
        Args:
            cache_type: Type of cache (redis or local)
        """
        if cache_type in self.stats:
            self.stats[cache_type]["expirations"] += 1
            
            CACHE_EXPIRATION_COUNTER.labels(
                cache_type=cache_type,
                service=self.service_name
            ).inc()
    
    def update_size(self, cache_type: str, size: int) -> None:
        """
        Update cache size.
        
        Args:
            cache_type: Type of cache (redis or local)
            size: Cache size
        """
        if cache_type in self.stats:
            self.stats[cache_type]["size"] = size
            
            CACHE_SIZE_GAUGE.labels(
                cache_type=cache_type,
                service=self.service_name
            ).set(size)
    
    def update_memory_usage(self, cache_type: str, memory_usage: int) -> None:
        """
        Update cache memory usage.
        
        Args:
            cache_type: Type of cache (redis or local)
            memory_usage: Memory usage in bytes
        """
        if cache_type in self.stats:
            self.stats[cache_type]["memory_usage"] = memory_usage
            
            CACHE_MEMORY_USAGE_GAUGE.labels(
                cache_type=cache_type,
                service=self.service_name
            ).set(memory_usage)
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return self.stats
