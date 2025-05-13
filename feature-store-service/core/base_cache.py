"""
Base cache interface for the feature store caching system.
"""
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Union, Pattern
import pandas as pd
from .cache_key import CacheKey


class BaseCache(ABC):
    """
    Abstract base class for all cache implementations.
    
    This defines the interface that all cache implementations must follow,
    regardless of whether they are memory-based, disk-based, or database-based.
    """
    
    @abstractmethod
    async def get(self, key: Union[str, CacheKey]) -> Optional[Any]:
        """
        Retrieve an item from the cache.
        
        Args:
            key: The cache key, either as a string or CacheKey object
            
        Returns:
            The cached data if found and valid, None otherwise
        """
        pass
    
    @abstractmethod
    async def put(self, key: Union[str, CacheKey], value: Any) -> bool:
        """
        Store an item in the cache.
        
        Args:
            key: The cache key, either as a string or CacheKey object
            value: The data to cache
            
        Returns:
            True if the operation was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def invalidate(self, key_pattern: Union[str, Pattern]) -> int:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            key_pattern: String pattern or regex pattern to match cache keys
            
        Returns:
            Number of invalidated cache entries
        """
        pass
    
    @abstractmethod
    async def clear(self) -> int:
        """
        Clear all items from the cache.
        
        Returns:
            Number of cleared cache entries
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """
        Get the current size of the cache in bytes.
        
        Returns:
            Size in bytes
        """
        pass
    
    @property
    @abstractmethod
    def max_size(self) -> int:
        """
        Get the maximum size of the cache in bytes.
        
        Returns:
            Maximum size in bytes
        """
        pass
    
    @property
    @abstractmethod
    def item_count(self) -> int:
        """
        Get the number of items in the cache.
        
        Returns:
            Number of items
        """
        pass
