"""
Cache interface for the forex trading platform.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast


class CacheInterface(ABC):
    """
    Cache interface.
    """

    @abstractmethod
    def get(self, key: str) -> Any:
        """
        Get a value from the cache.

        Args:
            key: Key to get

        Returns:
            Value from the cache
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Key to delete

        Returns:
            True if the value was deleted
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Key to check

        Returns:
            True if the key exists
        """
        pass

    @abstractmethod
    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a value in the cache.

        Args:
            key: Key to increment
            amount: Amount to increment by

        Returns:
            New value
        """
        pass

    @abstractmethod
    def decrement(self, key: str, amount: int = 1) -> int:
        """
        Decrement a value in the cache.

        Args:
            key: Key to decrement
            amount: Amount to decrement by

        Returns:
            New value
        """
        pass

    @abstractmethod
    def expire(self, key: str, ttl: int) -> bool:
        """
        Set the TTL for a key.

        Args:
            key: Key to set TTL for
            ttl: TTL in seconds

        Returns:
            True if the TTL was set
        """
        pass

    @abstractmethod
    def ttl(self, key: str) -> int:
        """
        Get the TTL for a key.

        Args:
            key: Key to get TTL for

        Returns:
            TTL in seconds
        """
        pass

    @abstractmethod
    def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching a pattern.

        Args:
            pattern: Pattern to match

        Returns:
            List of keys
        """
        pass

    @abstractmethod
    def flush(self) -> bool:
        """
        Flush all keys with the prefix.

        Returns:
            True if the keys were flushed
        """
        pass

    @abstractmethod
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from the cache.

        Args:
            keys: Keys to get

        Returns:
            Dictionary of keys and values
        """
        pass

    @abstractmethod
    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in the cache.

        Args:
            mapping: Dictionary of keys and values
            ttl: TTL in seconds

        Returns:
            True if the values were set
        """
        pass

    @abstractmethod
    def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple values from the cache.

        Args:
            keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        pass

    @abstractmethod
    async def async_get(self, key: str) -> Any:
        """
        Get a value from the cache asynchronously.

        Args:
            key: Key to get

        Returns:
            Value from the cache
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def async_delete(self, key: str) -> bool:
        """
        Delete a value from the cache asynchronously.

        Args:
            key: Key to delete

        Returns:
            True if the value was deleted
        """
        pass

    @abstractmethod
    async def async_exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache asynchronously.

        Args:
            key: Key to check

        Returns:
            True if the key exists
        """
        pass

    @abstractmethod
    async def async_increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a value in the cache asynchronously.

        Args:
            key: Key to increment
            amount: Amount to increment by

        Returns:
            New value
        """
        pass

    @abstractmethod
    async def async_decrement(self, key: str, amount: int = 1) -> int:
        """
        Decrement a value in the cache asynchronously.

        Args:
            key: Key to decrement
            amount: Amount to decrement by

        Returns:
            New value
        """
        pass

    @abstractmethod
    async def async_expire(self, key: str, ttl: int) -> bool:
        """
        Set the TTL for a key asynchronously.

        Args:
            key: Key to set TTL for
            ttl: TTL in seconds

        Returns:
            True if the TTL was set
        """
        pass

    @abstractmethod
    async def async_ttl(self, key: str) -> int:
        """
        Get the TTL for a key asynchronously.

        Args:
            key: Key to get TTL for

        Returns:
            TTL in seconds
        """
        pass

    @abstractmethod
    async def async_keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching a pattern asynchronously.

        Args:
            pattern: Pattern to match

        Returns:
            List of keys
        """
        pass

    @abstractmethod
    async def async_flush(self) -> bool:
        """
        Flush all keys with the prefix asynchronously.

        Returns:
            True if the keys were flushed
        """
        pass

    @abstractmethod
    async def async_get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from the cache asynchronously.

        Args:
            keys: Keys to get

        Returns:
            Dictionary of keys and values
        """
        pass

    @abstractmethod
    async def async_set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in the cache asynchronously.

        Args:
            mapping: Dictionary of keys and values
            ttl: TTL in seconds

        Returns:
            True if the values were set
        """
        pass

    @abstractmethod
    async def async_delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple values from the cache asynchronously.

        Args:
            keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        pass