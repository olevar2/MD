"""
Query Caching Module for Data Pipeline Service.

This module provides caching for database query results with query-parameter-based keys.
"""

import logging
import time
import hashlib
import json
import pickle
from typing import Dict, Any, Optional, Union, List, Tuple, Set
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Import common library
from common_lib.caching.cache_service import get_cache_service, cache_result
from common_lib.caching.event_invalidation import EventInvalidationStrategy

# Configure logging
logger = logging.getLogger(__name__)

class QueryCache:
    """
    Cache for database query results with query-parameter-based keys.
    
    This class provides caching for database query results, with appropriate
    invalidation based on events.
    """
    
    def __init__(
        self,
        service_name: str = "data-pipeline-service",
        default_ttl: int = 300,  # 5 minutes
        enable_metrics: bool = True
    ):
        """
        Initialize the query cache.
        
        Args:
            service_name: Name of the service
            default_ttl: Default time-to-live for cached items (in seconds)
            enable_metrics: Whether to enable metrics collection
        """
        self.service_name = service_name
        self.default_ttl = default_ttl
        self.enable_metrics = enable_metrics
        
        # Get cache service
        self.cache_service = get_cache_service(
            service_name=service_name,
            default_ttl=default_ttl
        )
        
        # Create invalidation strategy
        self.invalidation_strategy = EventInvalidationStrategy(
            cache_service=self.cache_service
        )
        
        # Register event handlers
        self.invalidation_strategy.register_event_handler(
            "data_updated",
            self._handle_data_updated_event
        )
        
        logger.info(f"Query cache initialized with TTL {default_ttl}s")
    
    def get_query_result(
        self,
        query_type: str,
        table: str,
        query_params: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """
        Get a cached query result.
        
        Args:
            query_type: Type of query
            table: Table name
            query_params: Query parameters
            
        Returns:
            Cached query result or None if not found
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            query_type=query_type,
            table=table,
            query_params=query_params
        )
        
        # Get from cache
        cached_value = self.cache_service.get(cache_key)
        
        if cached_value is not None:
            # Return the cached value
            return cached_value.get("data")
        
        return None
    
    def set_query_result(
        self,
        query_type: str,
        table: str,
        query_params: Dict[str, Any],
        data: pd.DataFrame,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a query result.
        
        Args:
            query_type: Type of query
            table: Table name
            query_params: Query parameters
            data: Query result data
            ttl: Time-to-live (in seconds)
            
        Returns:
            True if successful, False otherwise
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            query_type=query_type,
            table=table,
            query_params=query_params
        )
        
        # Create cache value
        cache_value = {
            "query_type": query_type,
            "table": table,
            "query_params": query_params,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Set in cache
        success = self.cache_service.set(cache_key, cache_value, ttl=ttl or self.default_ttl)
        
        # Register key for data_updated event
        if success:
            self.invalidation_strategy.register_key_for_event(cache_key, f"data_updated:{table}")
        
        return success
    
    def invalidate_query_result(
        self,
        query_type: str,
        table: str,
        query_params: Dict[str, Any]
    ) -> bool:
        """
        Invalidate a cached query result.
        
        Args:
            query_type: Type of query
            table: Table name
            query_params: Query parameters
            
        Returns:
            True if successful, False otherwise
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            query_type=query_type,
            table=table,
            query_params=query_params
        )
        
        # Delete from cache
        return self.cache_service.delete(cache_key)
    
    def invalidate_table(self, table: str) -> None:
        """
        Invalidate all cached query results for a table.
        
        Args:
            table: Table name
        """
        # Trigger data_updated event
        self.invalidation_strategy.trigger_event(f"data_updated:{table}")
    
    def _handle_data_updated_event(self, event: str, data: Any) -> None:
        """
        Handle data_updated event.
        
        Args:
            event: Event name
            data: Event data
        """
        logger.debug(f"Handling event {event}")
    
    def _generate_cache_key(
        self,
        query_type: str,
        table: str,
        query_params: Dict[str, Any]
    ) -> str:
        """
        Generate a cache key for a query result.
        
        Args:
            query_type: Type of query
            table: Table name
            query_params: Query parameters
            
        Returns:
            Cache key
        """
        # Convert query params to a string
        params_str = json.dumps(query_params, sort_keys=True, default=str)
        
        # Generate a hash
        key_str = f"query:{query_type}:{table}:{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

# Create a decorator for caching query results
def cache_query(
    ttl: Optional[int] = None
):
    """
    Decorator for caching query results.
    
    Args:
        ttl: Time-to-live for the cached item (in seconds)
        
    Returns:
        Decorated function
    """
    def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """

        def wrapper(
            self,
            query_type: str,
            table: str,
            **kwargs
        ):
    """
    Wrapper.
    
    Args:
        query_type: Description of query_type
        table: Description of table
        kwargs: Description of kwargs
    
    """

            # Get query cache
            query_cache = getattr(self, "_query_cache", None)
            if query_cache is None:
                query_cache = QueryCache()
                setattr(self, "_query_cache", query_cache)
            
            # Extract query params from kwargs
            query_params = kwargs
            
            # Try to get from cache
            cached_result = query_cache.get_query_result(
                query_type=query_type,
                table=table,
                query_params=query_params
            )
            
            if cached_result is not None:
                return cached_result
            
            # Execute query
            result = func(
                self,
                query_type=query_type,
                table=table,
                **kwargs
            )
            
            # Cache result
            query_cache.set_query_result(
                query_type=query_type,
                table=table,
                query_params=query_params,
                data=result,
                ttl=ttl
            )
            
            return result
        
        return wrapper
    
    return decorator
