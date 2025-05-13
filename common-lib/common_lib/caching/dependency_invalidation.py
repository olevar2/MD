"""
Dependency-based Cache Invalidation Strategy for Forex Trading Platform.

This module provides a dependency-based cache invalidation strategy.
"""

import logging
import threading
from typing import Dict, Any, Optional, Union, Callable, List, Set, TypeVar
from datetime import datetime, timedelta

# Local imports
from common_lib.caching.cache_service import get_cache_service, CacheService
from common_lib.caching.invalidation import CacheInvalidationStrategy

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')

class DependencyInvalidationStrategy(CacheInvalidationStrategy[Any]):
    """
    Dependency-based cache invalidation strategy.
    
    This strategy invalidates cache entries based on dependencies between keys.
    """
    
    def __init__(
        self,
        cache_service: Optional[CacheService] = None
    ):
        """
        Initialize the dependency invalidation strategy.
        
        Args:
            cache_service: Cache service
        """
        self.cache_service = cache_service or get_cache_service()
        self.dependencies: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}
        self.lock = threading.RLock()
    
    def should_invalidate(self, key: str, value: Any) -> bool:
        """
        Check if a cache entry should be invalidated based on its dependencies.
        
        This is a no-op for dependency-based invalidation, as invalidation is
        triggered by changes to dependencies.
        
        Args:
            key: Cache key
            value: Cached value
            
        Returns:
            False (invalidation is triggered by changes to dependencies)
        """
        return False
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry and all entries that depend on it.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Get all keys that depend on this key
            dependent_keys = self.reverse_dependencies.get(key, set())
            
            # Invalidate the key itself
            success = self.cache_service.delete(key)
            
            # Invalidate all dependent keys
            for dependent_key in dependent_keys:
                self.cache_service.delete(dependent_key)
                
                # Remove the dependency
                if dependent_key in self.dependencies:
                    self.dependencies[dependent_key].discard(key)
                    if not self.dependencies[dependent_key]:
                        del self.dependencies[dependent_key]
            
            # Remove the key from reverse dependencies
            if key in self.reverse_dependencies:
                del self.reverse_dependencies[key]
            
            return success
    
    def register_dependency(self, key: str, dependency: str) -> None:
        """
        Register a dependency for a cache key.
        
        Args:
            key: Cache key
            dependency: Dependency key
        """
        with self.lock:
            # Add the dependency
            if key not in self.dependencies:
                self.dependencies[key] = set()
            self.dependencies[key].add(dependency)
            
            # Add the reverse dependency
            if dependency not in self.reverse_dependencies:
                self.reverse_dependencies[dependency] = set()
            self.reverse_dependencies[dependency].add(key)
