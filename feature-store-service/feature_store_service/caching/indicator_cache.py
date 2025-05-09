"""
Indicator Caching Module for Feature Store Service.

This module provides caching for technical indicators with timestamp-based invalidation.
"""

import logging
import time
import hashlib
import json
import pickle
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Import common library
from common_lib.caching.cache_service import get_cache_service, cache_result
from common_lib.caching.invalidation import TimestampInvalidationStrategy

# Configure logging
logger = logging.getLogger(__name__)

class IndicatorCache:
    """
    Cache for technical indicators with timestamp-based invalidation.
    
    This class provides caching for technical indicators, with appropriate
    invalidation based on the timestamp of the data.
    """
    
    def __init__(
        self,
        service_name: str = "feature-store-service",
        default_ttl: int = 3600,  # 1 hour
        enable_metrics: bool = True
    ):
        """
        Initialize the indicator cache.
        
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
        self.invalidation_strategy = TimestampInvalidationStrategy(
            cache_service=self.cache_service,
            timestamp_field="timestamp",
            max_age=timedelta(seconds=default_ttl)
        )
        
        logger.info(f"Indicator cache initialized with TTL {default_ttl}s")
    
    def get_indicator(
        self,
        indicator_name: str,
        symbol: str,
        timeframe: str,
        params: Dict[str, Any],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get a cached indicator.
        
        Args:
            indicator_name: Name of the indicator
            symbol: Symbol
            timeframe: Timeframe
            params: Indicator parameters
            start_time: Start time
            end_time: End time
            
        Returns:
            Cached indicator or None if not found
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            indicator_name=indicator_name,
            symbol=symbol,
            timeframe=timeframe,
            params=params,
            start_time=start_time,
            end_time=end_time
        )
        
        # Get from cache
        cached_value = self.cache_service.get(cache_key)
        
        if cached_value is not None:
            # Check if the value should be invalidated
            if self.invalidation_strategy.should_invalidate(cache_key, cached_value):
                logger.debug(f"Invalidating cached indicator {indicator_name} for {symbol} {timeframe}")
                self.cache_service.delete(cache_key)
                return None
            
            # Return the cached value
            return cached_value.get("data")
        
        return None
    
    def set_indicator(
        self,
        indicator_name: str,
        symbol: str,
        timeframe: str,
        params: Dict[str, Any],
        data: pd.DataFrame,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache an indicator.
        
        Args:
            indicator_name: Name of the indicator
            symbol: Symbol
            timeframe: Timeframe
            params: Indicator parameters
            data: Indicator data
            start_time: Start time
            end_time: End time
            ttl: Time-to-live (in seconds)
            
        Returns:
            True if successful, False otherwise
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            indicator_name=indicator_name,
            symbol=symbol,
            timeframe=timeframe,
            params=params,
            start_time=start_time,
            end_time=end_time
        )
        
        # Create cache value with timestamp
        cache_value = {
            "indicator_name": indicator_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Set in cache
        return self.cache_service.set(cache_key, cache_value, ttl=ttl or self.default_ttl)
    
    def invalidate_indicator(
        self,
        indicator_name: str,
        symbol: str,
        timeframe: str,
        params: Dict[str, Any],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """
        Invalidate a cached indicator.
        
        Args:
            indicator_name: Name of the indicator
            symbol: Symbol
            timeframe: Timeframe
            params: Indicator parameters
            start_time: Start time
            end_time: End time
            
        Returns:
            True if successful, False otherwise
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            indicator_name=indicator_name,
            symbol=symbol,
            timeframe=timeframe,
            params=params,
            start_time=start_time,
            end_time=end_time
        )
        
        # Delete from cache
        return self.cache_service.delete(cache_key)
    
    def invalidate_symbol(self, symbol: str) -> bool:
        """
        Invalidate all cached indicators for a symbol.
        
        Args:
            symbol: Symbol
            
        Returns:
            True if successful, False otherwise
        """
        # Generate cache key pattern
        cache_key_pattern = f"indicator:{symbol}:*"
        
        # Clear from cache
        return self.cache_service.clear(cache_key_pattern)
    
    def _generate_cache_key(
        self,
        indicator_name: str,
        symbol: str,
        timeframe: str,
        params: Dict[str, Any],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """
        Generate a cache key for an indicator.
        
        Args:
            indicator_name: Name of the indicator
            symbol: Symbol
            timeframe: Timeframe
            params: Indicator parameters
            start_time: Start time
            end_time: End time
            
        Returns:
            Cache key
        """
        # Convert params to a string
        params_str = json.dumps(params, sort_keys=True, default=str)
        
        # Convert start and end times to strings
        start_time_str = start_time.isoformat() if start_time else "none"
        end_time_str = end_time.isoformat() if end_time else "none"
        
        # Generate a hash
        key_str = f"indicator:{symbol}:{timeframe}:{indicator_name}:{params_str}:{start_time_str}:{end_time_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

# Create a decorator for caching indicator calculations
def cache_indicator(
    ttl: Optional[int] = None,
    invalidate_on_params: Optional[List[str]] = None
):
    """
    Decorator for caching indicator calculations.
    
    Args:
        ttl: Time-to-live for the cached item (in seconds)
        invalidate_on_params: Parameters that should trigger invalidation
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(
            self,
            symbol: str,
            timeframe: str,
            data: pd.DataFrame,
            **kwargs
        ):
            # Get indicator name from function name
            indicator_name = func.__name__
            
            # Get indicator cache
            indicator_cache = getattr(self, "_indicator_cache", None)
            if indicator_cache is None:
                indicator_cache = IndicatorCache()
                setattr(self, "_indicator_cache", indicator_cache)
            
            # Get start and end times from data
            start_time = data.index.min() if isinstance(data.index, pd.DatetimeIndex) else None
            end_time = data.index.max() if isinstance(data.index, pd.DatetimeIndex) else None
            
            # Try to get from cache
            cached_result = indicator_cache.get_indicator(
                indicator_name=indicator_name,
                symbol=symbol,
                timeframe=timeframe,
                params=kwargs,
                start_time=start_time,
                end_time=end_time
            )
            
            if cached_result is not None:
                return cached_result
            
            # Calculate indicator
            result = func(self, symbol, timeframe, data, **kwargs)
            
            # Cache result
            indicator_cache.set_indicator(
                indicator_name=indicator_name,
                symbol=symbol,
                timeframe=timeframe,
                params=kwargs,
                data=result,
                start_time=start_time,
                end_time=end_time,
                ttl=ttl
            )
            
            return result
        
        return wrapper
    
    return decorator
