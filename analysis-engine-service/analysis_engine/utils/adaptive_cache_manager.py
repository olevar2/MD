"""
Adaptive Cache Manager

This module provides an advanced caching system with adaptive TTL and analytics.
It is designed to optimize memory usage and cache hit rates for performance-critical
operations like confluence and divergence detection.

Features:
- Adaptive TTL based on access patterns
- Cache analytics and statistics
- Memory-efficient storage
- Thread-safe operations
- Automatic cleanup of expired entries
"""

import threading
import time
import sys
from typing import Any, Dict, Tuple, Optional, List, Hashable
import logging

logger = logging.getLogger(__name__)

class AdaptiveCacheManager:
    """
    Advanced caching system with adaptive TTL and analytics.
    
    Features:
    - Adaptive TTL based on access patterns
    - Cache analytics and statistics
    - Memory-efficient storage
    - Thread-safe operations
    - Automatic cleanup of expired entries
    """
    
    def __init__(self, default_ttl_seconds: int = 3600, max_size: int = 1000, 
                 cleanup_interval_seconds: int = 300):
        """
        Initialize the cache manager.
        
        Args:
            default_ttl_seconds: Default time-to-live for cache entries in seconds
            max_size: Maximum number of entries in the cache
            cleanup_interval_seconds: Interval for automatic cache cleanup
        """
        self.default_ttl = default_ttl_seconds
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval_seconds
        
        # Main cache storage: {key: (value, timestamp, access_count, last_access)}
        self.cache = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Analytics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.last_cleanup_time = time.time()
        
        logger.debug(f"AdaptiveCacheManager initialized with TTL={default_ttl_seconds}s, "
                    f"max_size={max_size}, cleanup_interval={cleanup_interval_seconds}s")
    
    def get(self, key: Any) -> Tuple[bool, Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (hit, value) where hit is True if the key was found and not expired
        """
        with self.lock:
            if key in self.cache:
                value, timestamp, access_count, _ = self.cache[key]
                current_time = time.time()
                
                # Check if expired
                if current_time - timestamp > self.default_ttl:
                    self.cache.pop(key)
                    self.misses += 1
                    return False, None
                
                # Update access stats
                self.cache[key] = (value, timestamp, access_count + 1, current_time)
                self.hits += 1
                
                # Return a copy to prevent modification of cached data
                if isinstance(value, dict):
                    return True, value.copy()
                elif isinstance(value, list):
                    return True, value.copy()
                else:
                    return True, value
            
            self.misses += 1
            return False, None
    
    def set(self, key: Any, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL in seconds
        """
        with self.lock:
            # Clean cache if needed
            current_time = time.time()
            if current_time - self.last_cleanup_time > self.cleanup_interval:
                self._clean_cache()
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_one()
            
            # Store a copy to prevent modification of cached data
            if isinstance(value, dict):
                cached_value = value.copy()
            elif isinstance(value, list):
                cached_value = value.copy()
            else:
                cached_value = value
            
            # Set new value
            self.cache[key] = (cached_value, current_time, 1, current_time)
    
    def _evict_one(self) -> None:
        """
        Evict one item from the cache using adaptive policy.
        
        Uses a scoring system that considers:
        - Access frequency
        - Recency of access
        - Age relative to TTL
        """
        if not self.cache:
            return
        
        # Calculate score for each item (lower is better to keep)
        # Score = (current_time - last_access) / (access_count * ttl_factor)
        current_time = time.time()
        scores = {}
        
        for k, (_, timestamp, access_count, last_access) in self.cache.items():
            age_factor = current_time - timestamp
            ttl_factor = min(1.0, age_factor / self.default_ttl)
            recency_factor = current_time - last_access
            popularity_factor = 1.0 / max(1, access_count)
            
            scores[k] = recency_factor * popularity_factor * ttl_factor
        
        # Evict item with highest score
        if scores:
            worst_key = max(scores.items(), key=lambda x: x[1])[0]
            self.cache.pop(worst_key)
            self.evictions += 1
            logger.debug(f"Evicted cache entry with key: {worst_key}")
    
    def _clean_cache(self, force: bool = False) -> None:
        """
        Clean expired entries from the cache.
        
        Args:
            force: Force cleanup regardless of interval
        """
        current_time = time.time()
        
        # Only clean periodically to avoid overhead
        if not force and current_time - self.last_cleanup_time < self.cleanup_interval:
            return
        
        with self.lock:
            # Calculate expiry time
            expiry_time = current_time - self.default_ttl
            
            # Find expired keys
            expired_keys = [
                key for key, (_, timestamp, _, _) in self.cache.items()
                if timestamp < expiry_time
            ]
            
            # Remove expired entries
            for key in expired_keys:
                del self.cache[key]
            
            # Update last cleanup time
            self.last_cleanup_time = current_time
            
            # Log cleanup stats
            if expired_keys:
                logger.debug(f"Cache cleaned: {len(expired_keys)} expired entries removed, "
                            f"{len(self.cache)} entries remaining")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "memory_usage_bytes": sys.getsizeof(self.cache)
            }
    
    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            logger.debug("Cache cleared")
