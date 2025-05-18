"""
In-memory cache implementation for the forex trading platform.
"""

import logging
import time
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
import asyncio

from common_lib.resilience.decorators import (
    with_exception_handling,
    async_with_exception_handling
)

logger = logging.getLogger(__name__)


class MemoryCache:
    """
    In-memory cache implementation.
    """

    def __init__(
        self,
        prefix: str = "cache:",
        ttl: int = 3600,
        max_size: int = 1000,
        cleanup_interval: int = 60
    ):
        """
        Initialize the in-memory cache.

        Args:
            prefix: Key prefix
            ttl: Default TTL in seconds
            max_size: Maximum number of items in the cache
            cleanup_interval: Interval in seconds to clean up expired items
        """
        self.prefix = prefix
        self.ttl = ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval

        # Initialize cache
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.lock = threading.RLock()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        logger.info(f"Initialized in-memory cache with max size {max_size}")

    def _get_key(self, key: str) -> str:
        """
        Get the full key with prefix.

        Args:
            key: Key to get

        Returns:
            Full key with prefix
        """
        return f"{self.prefix}{key}"

    def _cleanup_loop(self) -> None:
        """
        Cleanup loop to remove expired items.
        """
        while True:
            time.sleep(self.cleanup_interval)
            self._cleanup()

    def _cleanup(self) -> None:
        """
        Remove expired items from the cache.
        """
        now = time.time()
        with self.lock:
            # Remove expired items
            expired_keys = [
                key for key, (_, expiry) in self.cache.items()
                if expiry < now
            ]
            for key in expired_keys:
                del self.cache[key]

            # If cache is still too large, remove oldest items
            if len(self.cache) > self.max_size:
                # Sort by expiry time
                sorted_items = sorted(
                    self.cache.items(),
                    key=lambda x: x[1][1]
                )
                # Remove oldest items
                for key, _ in sorted_items[:len(self.cache) - self.max_size]:
                    del self.cache[key]

    @with_exception_handling
    def get(self, key: str) -> Any:
        """
        Get a value from the cache.

        Args:
            key: Key to get

        Returns:
            Value from the cache
        """
        full_key = self._get_key(key)
        with self.lock:
            if full_key in self.cache:
                value, expiry = self.cache[full_key]
                if expiry > time.time():
                    return value
                else:
                    # Remove expired item
                    del self.cache[full_key]
        return None

    @with_exception_handling
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.

        Args:
            key: Key to set
            value: Value to set
            ttl: TTL in seconds

        Returns:
            True if the value was set
        """
        full_key = self._get_key(key)
        expiry = time.time() + (ttl or self.ttl)
        with self.lock:
            # If cache is full, remove oldest item
            if len(self.cache) >= self.max_size and full_key not in self.cache:
                self._cleanup()
                # If still full, remove oldest item
                if len(self.cache) >= self.max_size:
                    oldest_key = min(
                        self.cache.items(),
                        key=lambda x: x[1][1]
                    )[0]
                    del self.cache[oldest_key]
            
            self.cache[full_key] = (value, expiry)
        return True

    @with_exception_handling
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Key to delete

        Returns:
            True if the value was deleted
        """
        full_key = self._get_key(key)
        with self.lock:
            if full_key in self.cache:
                del self.cache[full_key]
                return True
        return False

    @with_exception_handling
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Key to check

        Returns:
            True if the key exists
        """
        full_key = self._get_key(key)
        with self.lock:
            if full_key in self.cache:
                _, expiry = self.cache[full_key]
                if expiry > time.time():
                    return True
                else:
                    # Remove expired item
                    del self.cache[full_key]
        return False

    @with_exception_handling
    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a value in the cache.

        Args:
            key: Key to increment
            amount: Amount to increment by

        Returns:
            New value
        """
        full_key = self._get_key(key)
        with self.lock:
            if full_key in self.cache:
                value, expiry = self.cache[full_key]
                if expiry > time.time():
                    if isinstance(value, (int, float)):
                        value += amount
                        self.cache[full_key] = (value, expiry)
                        return value
                    else:
                        raise TypeError(f"Cannot increment non-numeric value: {value}")
                else:
                    # Remove expired item
                    del self.cache[full_key]
            
            # Key doesn't exist, set to amount
            self.set(key, amount)
            return amount

    @with_exception_handling
    def decrement(self, key: str, amount: int = 1) -> int:
        """
        Decrement a value in the cache.

        Args:
            key: Key to decrement
            amount: Amount to decrement by

        Returns:
            New value
        """
        return self.increment(key, -amount)

    @with_exception_handling
    def expire(self, key: str, ttl: int) -> bool:
        """
        Set the TTL for a key.

        Args:
            key: Key to set TTL for
            ttl: TTL in seconds

        Returns:
            True if the TTL was set
        """
        full_key = self._get_key(key)
        with self.lock:
            if full_key in self.cache:
                value, _ = self.cache[full_key]
                expiry = time.time() + ttl
                self.cache[full_key] = (value, expiry)
                return True
        return False

    @with_exception_handling
    def ttl(self, key: str) -> int:
        """
        Get the TTL for a key.

        Args:
            key: Key to get TTL for

        Returns:
            TTL in seconds
        """
        full_key = self._get_key(key)
        with self.lock:
            if full_key in self.cache:
                _, expiry = self.cache[full_key]
                ttl = int(expiry - time.time())
                if ttl > 0:
                    return ttl
                else:
                    # Remove expired item
                    del self.cache[full_key]
        return -1

    @with_exception_handling
    def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching a pattern.

        Args:
            pattern: Pattern to match

        Returns:
            List of keys
        """
        import fnmatch
        
        full_pattern = self._get_key(pattern)
        with self.lock:
            # Get all keys
            all_keys = list(self.cache.keys())
            
            # Filter by pattern
            if pattern == "*":
                matching_keys = all_keys
            else:
                matching_keys = [
                    key for key in all_keys
                    if fnmatch.fnmatch(key, full_pattern)
                ]
            
            # Remove expired keys
            now = time.time()
            valid_keys = []
            for key in matching_keys:
                _, expiry = self.cache[key]
                if expiry > now:
                    valid_keys.append(key)
                else:
                    # Remove expired item
                    del self.cache[key]
            
            # Remove prefix from keys
            return [key[len(self.prefix):] for key in valid_keys]

    @with_exception_handling
    def flush(self) -> bool:
        """
        Flush all keys with the prefix.

        Returns:
            True if the keys were flushed
        """
        with self.lock:
            # Get all keys with prefix
            keys_to_delete = [
                key for key in self.cache.keys()
                if key.startswith(self.prefix)
            ]
            
            # Delete keys
            for key in keys_to_delete:
                del self.cache[key]
        
        return True

    @with_exception_handling
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from the cache.

        Args:
            keys: Keys to get

        Returns:
            Dictionary of keys and values
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    @with_exception_handling
    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in the cache.

        Args:
            mapping: Dictionary of keys and values
            ttl: TTL in seconds

        Returns:
            True if the values were set
        """
        for key, value in mapping.items():
            self.set(key, value, ttl)
        return True

    @with_exception_handling
    def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple values from the cache.

        Args:
            keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count

    @async_with_exception_handling
    async def async_get(self, key: str) -> Any:
        """
        Get a value from the cache asynchronously.

        Args:
            key: Key to get

        Returns:
            Value from the cache
        """
        return self.get(key)

    @async_with_exception_handling
    async def async_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache asynchronously.

        Args:
            key: Key to set
            value: Value to set
            ttl: TTL in seconds

        Returns:
            True if the value was set
        """
        return self.set(key, value, ttl)

    @async_with_exception_handling
    async def async_delete(self, key: str) -> bool:
        """
        Delete a value from the cache asynchronously.

        Args:
            key: Key to delete

        Returns:
            True if the value was deleted
        """
        return self.delete(key)

    @async_with_exception_handling
    async def async_exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache asynchronously.

        Args:
            key: Key to check

        Returns:
            True if the key exists
        """
        return self.exists(key)

    @async_with_exception_handling
    async def async_increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a value in the cache asynchronously.

        Args:
            key: Key to increment
            amount: Amount to increment by

        Returns:
            New value
        """
        return self.increment(key, amount)

    @async_with_exception_handling
    async def async_decrement(self, key: str, amount: int = 1) -> int:
        """
        Decrement a value in the cache asynchronously.

        Args:
            key: Key to decrement
            amount: Amount to decrement by

        Returns:
            New value
        """
        return self.decrement(key, amount)

    @async_with_exception_handling
    async def async_expire(self, key: str, ttl: int) -> bool:
        """
        Set the TTL for a key asynchronously.

        Args:
            key: Key to set TTL for
            ttl: TTL in seconds

        Returns:
            True if the TTL was set
        """
        return self.expire(key, ttl)

    @async_with_exception_handling
    async def async_ttl(self, key: str) -> int:
        """
        Get the TTL for a key asynchronously.

        Args:
            key: Key to get TTL for

        Returns:
            TTL in seconds
        """
        return self.ttl(key)

    @async_with_exception_handling
    async def async_keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching a pattern asynchronously.

        Args:
            pattern: Pattern to match

        Returns:
            List of keys
        """
        return self.keys(pattern)

    @async_with_exception_handling
    async def async_flush(self) -> bool:
        """
        Flush all keys with the prefix asynchronously.

        Returns:
            True if the keys were flushed
        """
        return self.flush()

    @async_with_exception_handling
    async def async_get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from the cache asynchronously.

        Args:
            keys: Keys to get

        Returns:
            Dictionary of keys and values
        """
        return self.get_many(keys)

    @async_with_exception_handling
    async def async_set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in the cache asynchronously.

        Args:
            mapping: Dictionary of keys and values
            ttl: TTL in seconds

        Returns:
            True if the values were set
        """
        return self.set_many(mapping, ttl)

    @async_with_exception_handling
    async def async_delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple values from the cache asynchronously.

        Args:
            keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        return self.delete_many(keys)