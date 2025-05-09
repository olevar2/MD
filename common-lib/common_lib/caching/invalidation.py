"""
Cache Invalidation Strategies for Forex Trading Platform.

This module provides various cache invalidation strategies for the forex trading platform.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Union, Callable, List, Set, TypeVar, Generic
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Local imports
from common_lib.caching.cache_service import get_cache_service, CacheService

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')

class CacheInvalidationStrategy(ABC, Generic[T]):
    """
    Abstract base class for cache invalidation strategies.
    
    This class defines the interface for cache invalidation strategies.
    """
    
    @abstractmethod
    def should_invalidate(self, key: str, value: T) -> bool:
        """
        Check if a cache entry should be invalidated.
        
        Args:
            key: Cache key
            value: Cached value
            
        Returns:
            True if the entry should be invalidated, False otherwise
        """
        pass
    
    @abstractmethod
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def register_dependency(self, key: str, dependency: str) -> None:
        """
        Register a dependency for a cache key.
        
        Args:
            key: Cache key
            dependency: Dependency key
        """
        pass

class TimestampInvalidationStrategy(CacheInvalidationStrategy[Dict[str, Any]]):
    """
    Timestamp-based cache invalidation strategy.
    
    This strategy invalidates cache entries based on a timestamp field in the cached value.
    """
    
    def __init__(
        self,
        cache_service: Optional[CacheService] = None,
        timestamp_field: str = "timestamp",
        max_age: Optional[Union[int, timedelta]] = None
    ):
        """
        Initialize the timestamp invalidation strategy.
        
        Args:
            cache_service: Cache service
            timestamp_field: Field name for the timestamp
            max_age: Maximum age for cached entries (in seconds or timedelta)
        """
        self.cache_service = cache_service or get_cache_service()
        self.timestamp_field = timestamp_field
        self.max_age = max_age
    
    def should_invalidate(self, key: str, value: Dict[str, Any]) -> bool:
        """
        Check if a cache entry should be invalidated based on its timestamp.
        
        Args:
            key: Cache key
            value: Cached value
            
        Returns:
            True if the entry should be invalidated, False otherwise
        """
        # Check if the value has a timestamp field
        if not isinstance(value, dict) or self.timestamp_field not in value:
            return True
        
        # Get the timestamp
        timestamp = value[self.timestamp_field]
        
        # Convert timestamp to datetime if needed
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                return True
        
        # Check if the timestamp is a datetime
        if not isinstance(timestamp, datetime):
            return True
        
        # Check if the entry is too old
        if self.max_age is not None:
            max_age_seconds = self.max_age.total_seconds() if isinstance(self.max_age, timedelta) else self.max_age
            age = (datetime.now() - timestamp).total_seconds()
            return age > max_age_seconds
        
        return False
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        return self.cache_service.delete(key)
    
    def register_dependency(self, key: str, dependency: str) -> None:
        """
        Register a dependency for a cache key.
        
        This is a no-op for timestamp-based invalidation.
        
        Args:
            key: Cache key
            dependency: Dependency key
        """
        pass

class VersionInvalidationStrategy(CacheInvalidationStrategy[Dict[str, Any]]):
    """
    Version-based cache invalidation strategy.
    
    This strategy invalidates cache entries based on a version field in the cached value.
    """
    
    def __init__(
        self,
        cache_service: Optional[CacheService] = None,
        version_field: str = "version",
        current_version: Union[int, str] = 1
    ):
        """
        Initialize the version invalidation strategy.
        
        Args:
            cache_service: Cache service
            version_field: Field name for the version
            current_version: Current version
        """
        self.cache_service = cache_service or get_cache_service()
        self.version_field = version_field
        self.current_version = current_version
    
    def should_invalidate(self, key: str, value: Dict[str, Any]) -> bool:
        """
        Check if a cache entry should be invalidated based on its version.
        
        Args:
            key: Cache key
            value: Cached value
            
        Returns:
            True if the entry should be invalidated, False otherwise
        """
        # Check if the value has a version field
        if not isinstance(value, dict) or self.version_field not in value:
            return True
        
        # Get the version
        version = value[self.version_field]
        
        # Check if the version matches
        return version != self.current_version
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        return self.cache_service.delete(key)
    
    def register_dependency(self, key: str, dependency: str) -> None:
        """
        Register a dependency for a cache key.
        
        This is a no-op for version-based invalidation.
        
        Args:
            key: Cache key
            dependency: Dependency key
        """
        pass
