"""
Cache Manager

This module provides caching functionality for frequently accessed data,
improving performance for repeated requests and heavy calculations.
"""

import time
import json
from typing import Dict, Any, Optional, Callable, Union, Tuple
import threading


class CacheManager:
    """
    Simple in-memory cache manager with TTL (Time To Live) support
    """
    
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float, float]] = {}  # {key: (value, timestamp, ttl)}
        self._lock = threading.RLock()
        
        # Start cache cleaning thread
        self._cleaner = threading.Thread(target=self._clean_expired, daemon=True)
        self._running = True
        self._cleaner.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache if it exists and is not expired"""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp, ttl = self._cache[key]
            if ttl > 0 and time.time() - timestamp > ttl:
                del self._cache[key]
                return None
                
            return value
    
    def set(self, key: str, value: Any, ttl: float = 300) -> None:
        """Set a value in cache with a TTL (in seconds)"""
        with self._lock:
            self._cache[key] = (value, time.time(), ttl)
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def flush(self) -> None:
        """Flush the entire cache"""
        with self._lock:
            self._cache.clear()
    
    def get_or_set(self, key: str, value_func: Callable[[], Any], ttl: float = 300) -> Any:
        """Get a value from cache or compute and store it if not available"""
        cached = self.get(key)
        if cached is not None:
            return cached
            
        value = value_func()
        self.set(key, value, ttl)
        return value
    
    def _clean_expired(self) -> None:
        """Clean expired cache entries periodically"""
        while self._running:
            time.sleep(60)  # Clean every minute
            with self._lock:
                current_time = time.time()
                keys_to_delete = [
                    key for key, (_, timestamp, ttl) in self._cache.items()
                    if ttl > 0 and current_time - timestamp > ttl
                ]
                
                for key in keys_to_delete:
                    del self._cache[key]
    
    def stop(self) -> None:
        """Stop the cleaner thread"""
        self._running = False
        if self._cleaner.is_alive():
            self._cleaner.join(timeout=1)


# Create a singleton instance
cache = CacheManager()


def cached(ttl: float = 300):
    """
    Decorator for caching function results
    
    Usage:
        @cached(ttl=60)
        def expensive_calculation(param1, param2):
            # ...calculations...
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create a cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = "|".join(key_parts)
            
            return cache.get_or_set(cache_key, lambda: func(*args, **kwargs), ttl)
            
        return wrapper
    return decorator
