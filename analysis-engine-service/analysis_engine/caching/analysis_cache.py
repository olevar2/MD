"""
Analysis Results Caching Module for Analysis Engine Service.

This module provides caching for analysis results with version-based invalidation.
"""
import logging
import time
import hashlib
import json
from typing import Dict, Any, Optional, Union, List, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from common_lib.caching.cache_service import get_cache_service, cache_result
from common_lib.caching.invalidation import VersionInvalidationStrategy
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AnalysisCache:
    """
    Cache for analysis results with version-based invalidation.

    This class provides caching for analysis results, with appropriate
    invalidation based on version changes.
    """

    def __init__(self, service_name: str='analysis-engine-service',
        default_ttl: int=3600, enable_metrics: bool=True, current_version:
        str='1.0.0'):
        """
        Initialize the analysis cache.

        Args:
            service_name: Name of the service
            default_ttl: Default time-to-live for cached items (in seconds)
            enable_metrics: Whether to enable metrics collection
            current_version: Current version of the analysis engine
        """
        self.service_name = service_name
        self.default_ttl = default_ttl
        self.enable_metrics = enable_metrics
        self.current_version = current_version
        self.cache_service = get_cache_service(service_name=service_name,
            default_ttl=default_ttl)
        self.invalidation_strategy = VersionInvalidationStrategy(cache_service
            =self.cache_service, version_field='version', current_version=
            current_version)
        logger.info(
            f'Analysis cache initialized with TTL {default_ttl}s and version {current_version}'
            )

    @with_resilience('get_analysis')
    def get_analysis(self, analysis_type: str, symbol: str, timeframe: str,
        params: Dict[str, Any], start_time: Optional[datetime]=None,
        end_time: Optional[datetime]=None) ->Optional[Dict[str, Any]]:
        """
        Get a cached analysis result.

        Args:
            analysis_type: Type of analysis
            symbol: Symbol
            timeframe: Timeframe
            params: Analysis parameters
            start_time: Start time
            end_time: End time

        Returns:
            Cached analysis result or None if not found
        """
        cache_key = self._generate_cache_key(analysis_type=analysis_type,
            symbol=symbol, timeframe=timeframe, params=params, start_time=
            start_time, end_time=end_time)
        cached_value = self.cache_service.get(cache_key)
        if cached_value is not None:
            if self.invalidation_strategy.should_invalidate(cache_key,
                cached_value):
                logger.debug(
                    f'Invalidating cached analysis {analysis_type} for {symbol} {timeframe}'
                    )
                self.cache_service.delete(cache_key)
                return None
            return cached_value.get('data')
        return None

    def set_analysis(self, analysis_type: str, symbol: str, timeframe: str,
        params: Dict[str, Any], data: Dict[str, Any], start_time: Optional[
        datetime]=None, end_time: Optional[datetime]=None, ttl: Optional[
        int]=None) ->bool:
        """
        Cache an analysis result.

        Args:
            analysis_type: Type of analysis
            symbol: Symbol
            timeframe: Timeframe
            params: Analysis parameters
            data: Analysis data
            start_time: Start time
            end_time: End time
            ttl: Time-to-live (in seconds)

        Returns:
            True if successful, False otherwise
        """
        cache_key = self._generate_cache_key(analysis_type=analysis_type,
            symbol=symbol, timeframe=timeframe, params=params, start_time=
            start_time, end_time=end_time)
        cache_value = {'analysis_type': analysis_type, 'symbol': symbol,
            'timeframe': timeframe, 'params': params, 'start_time': 
            start_time.isoformat() if start_time else None, 'end_time': 
            end_time.isoformat() if end_time else None, 'timestamp':
            datetime.now().isoformat(), 'version': self.current_version,
            'data': data}
        return self.cache_service.set(cache_key, cache_value, ttl=ttl or
            self.default_ttl)

    def invalidate_analysis(self, analysis_type: str, symbol: str,
        timeframe: str, params: Dict[str, Any], start_time: Optional[
        datetime]=None, end_time: Optional[datetime]=None) ->bool:
        """
        Invalidate a cached analysis result.

        Args:
            analysis_type: Type of analysis
            symbol: Symbol
            timeframe: Timeframe
            params: Analysis parameters
            start_time: Start time
            end_time: End time

        Returns:
            True if successful, False otherwise
        """
        cache_key = self._generate_cache_key(analysis_type=analysis_type,
            symbol=symbol, timeframe=timeframe, params=params, start_time=
            start_time, end_time=end_time)
        return self.cache_service.delete(cache_key)

    def invalidate_analysis_type(self, analysis_type: str) ->bool:
        """
        Invalidate all cached analysis results of a specific type.

        Args:
            analysis_type: Type of analysis

        Returns:
            True if successful, False otherwise
        """
        cache_key_pattern = f'analysis:{analysis_type}:*'
        return self.cache_service.clear(cache_key_pattern)

    @with_resilience('update_version')
    def update_version(self, new_version: str) ->None:
        """
        Update the current version of the analysis engine.

        This will cause all cached analysis results with a different version
        to be invalidated.

        Args:
            new_version: New version
        """
        self.current_version = new_version
        self.invalidation_strategy.current_version = new_version
        logger.info(f'Analysis cache version updated to {new_version}')

    def _generate_cache_key(self, analysis_type: str, symbol: str,
        timeframe: str, params: Dict[str, Any], start_time: Optional[
        datetime]=None, end_time: Optional[datetime]=None) ->str:
        """
        Generate a cache key for an analysis result.

        Args:
            analysis_type: Type of analysis
            symbol: Symbol
            timeframe: Timeframe
            params: Analysis parameters
            start_time: Start time
            end_time: End time

        Returns:
            Cache key
        """
        params_str = json.dumps(params, sort_keys=True, default=str)
        start_time_str = start_time.isoformat() if start_time else 'none'
        end_time_str = end_time.isoformat() if end_time else 'none'
        key_str = (
            f'analysis:{analysis_type}:{symbol}:{timeframe}:{params_str}:{start_time_str}:{end_time_str}'
            )
        return hashlib.md5(key_str.encode()).hexdigest()


def cache_analysis(ttl: Optional[int]=None):
    """
    Decorator for caching analysis calculations.

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


        def wrapper(self, analysis_type: str, symbol: str, timeframe: str,
            **kwargs):
    """
    Wrapper.
    
    Args:
        analysis_type: Description of analysis_type
        symbol: Description of symbol
        timeframe: Description of timeframe
        kwargs: Description of kwargs
    
    """

            analysis_cache = getattr(self, '_analysis_cache', None)
            if analysis_cache is None:
                analysis_cache = AnalysisCache()
                setattr(self, '_analysis_cache', analysis_cache)
            start_time = kwargs.get('start_time')
            end_time = kwargs.get('end_time')
            params = {k: v for k, v in kwargs.items() if k not in [
                'start_time', 'end_time']}
            cached_result = analysis_cache.get_analysis(analysis_type=
                analysis_type, symbol=symbol, timeframe=timeframe, params=
                params, start_time=start_time, end_time=end_time)
            if cached_result is not None:
                return cached_result
            result = func(self, analysis_type=analysis_type, symbol=symbol,
                timeframe=timeframe, **kwargs)
            analysis_cache.set_analysis(analysis_type=analysis_type, symbol
                =symbol, timeframe=timeframe, params=params, data=result,
                start_time=start_time, end_time=end_time, ttl=ttl)
            return result
        return wrapper
    return decorator
