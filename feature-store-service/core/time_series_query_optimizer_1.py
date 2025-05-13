"""
Time Series Query Optimizer

This module provides optimization functionality for time series queries,
focusing on improving query performance through caching, indexing strategies,
and query pattern analysis.
"""
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import time
import hashlib
import json
import pandas as pd
import numpy as np
from functools import lru_cache


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TimeSeriesQueryOptimizer:
    """
    Optimizes time series queries by implementing various strategies:
    - Query result caching with TTL
    - Query pattern analysis to optimize future queries
    - Index utilization strategy
    - Time range partitioning for parallel query execution
    """

    def __init__(self, max_cache_size: int=100, default_ttl: int=300):
        """
        Initialize the optimizer with configuration parameters
        
        Args:
            max_cache_size: Maximum number of query results to cache
            default_ttl: Default time-to-live for cached results in seconds
        """
        self.logger = logging.getLogger(__name__)
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        self.query_cache = {}
        self.query_patterns = {}
        self.pattern_counts = {}
        self.query_times = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def optimize_query(self, query_params: Dict[str, Any]) ->Dict[str, Any]:
        """
        Optimize a query based on its parameters and historical patterns
        
        Args:
            query_params: Dictionary containing query parameters
            
        Returns:
            Modified query parameters for optimal execution
        """
        optimized_params = query_params.copy()
        if 'start_time' in query_params and 'end_time' in query_params:
            optimized_params = self._optimize_time_range(optimized_params)
        optimized_params = self._add_index_hints(optimized_params)
        self._track_query_pattern(query_params)
        return optimized_params

    def get_cached_result(self, query_params: Dict[str, Any]) ->Tuple[bool,
        Optional[Any]]:
        """
        Try to retrieve a cached result for the given query parameters
        
        Args:
            query_params: Dictionary containing query parameters
            
        Returns:
            Tuple of (cache_hit, result). If cache_hit is False, result is None
        """
        query_hash = self._generate_query_hash(query_params)
        if query_hash in self.query_cache:
            result, timestamp, ttl = self.query_cache[query_hash]
            if datetime.now().timestamp() - timestamp < ttl:
                self.cache_hits += 1
                self.logger.debug(f'Cache hit for query: {query_hash}')
                return True, result
            else:
                del self.query_cache[query_hash]
        self.cache_misses += 1
        self.logger.debug(f'Cache miss for query: {query_hash}')
        return False, None

    def cache_result(self, query_params: Dict[str, Any], result: Any, ttl:
        Optional[int]=None):
        """
        Cache a query result with an expiration time
        
        Args:
            query_params: Dictionary containing query parameters
            result: Query result to cache
            ttl: Time-to-live in seconds, uses default_ttl if None
        """
        if ttl is None:
            ttl = self.default_ttl
        query_hash = self._generate_query_hash(query_params)
        if len(self.query_cache) >= self.max_cache_size:
            self._clean_cache()
        self.query_cache[query_hash] = result, datetime.now().timestamp(), ttl
        self.logger.debug(f'Cached result for query: {query_hash}')

    def record_query_time(self, query_params: Dict[str, Any],
        execution_time: float):
        """
        Record the execution time for a query to improve future optimizations
        
        Args:
            query_params: Dictionary containing query parameters
            execution_time: Query execution time in seconds
        """
        query_hash = self._generate_query_hash(query_params)
        if query_hash in self.query_times:
            prev_time = self.query_times[query_hash]
            self.query_times[query_hash
                ] = 0.7 * execution_time + 0.3 * prev_time
        else:
            self.query_times[query_hash] = execution_time

    def get_query_statistics(self) ->Dict[str, Any]:
        """
        Get statistics about query optimization performance
        
        Returns:
            Dictionary with cache hit rate, pattern frequencies, and timing data
        """
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_queries if total_queries > 0 else 0
        return {'cache_hit_rate': hit_rate, 'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses, 'cached_queries': len(self.
            query_cache), 'tracked_patterns': len(self.query_patterns),
            'common_patterns': self._get_common_patterns(5),
            'query_time_stats': self._get_query_time_stats()}

    def partition_time_range(self, start_time: datetime, end_time: datetime,
        parts: int=4) ->List[Tuple[datetime, datetime]]:
        """
        Partition a time range into smaller chunks for parallel query execution
        
        Args:
            start_time: Start datetime of the range
            end_time: End datetime of the range
            parts: Number of partitions to create
            
        Returns:
            List of (start, end) datetime tuples representing partitions
        """
        if parts <= 1:
            return [(start_time, end_time)]
        total_seconds = (end_time - start_time).total_seconds()
        chunk_size = total_seconds / parts
        partitions = []
        for i in range(parts):
            chunk_start = start_time + timedelta(seconds=i * chunk_size)
            chunk_end = start_time + timedelta(seconds=(i + 1) * chunk_size)
            if i == parts - 1:
                chunk_end = end_time
            partitions.append((chunk_start, chunk_end))
        return partitions

    @with_exception_handling
    def invalidate_cache(self, symbol: Optional[str]=None, indicator_type:
        Optional[str]=None):
        """
        Invalidate cache entries matching the given criteria
        
        Args:
            symbol: If provided, invalidate entries for this symbol
            indicator_type: If provided, invalidate entries for this indicator type
        """
        keys_to_remove = []
        for query_hash, (result, timestamp, ttl) in self.query_cache.items():
            try:
                if symbol and any(symbol in str(item) for item in result.
                    items()):
                    keys_to_remove.append(query_hash)
                elif indicator_type and any(indicator_type in str(item) for
                    item in result.items()):
                    keys_to_remove.append(query_hash)
            except:
                continue
        for key in keys_to_remove:
            if key in self.query_cache:
                del self.query_cache[key]
        self.logger.info(f'Invalidated {len(keys_to_remove)} cache entries')

    def _optimize_time_range(self, query_params: Dict[str, Any]) ->Dict[str,
        Any]:
        """
        Optimize time range based on common patterns and typical time boundaries
        
        Args:
            query_params: Original query parameters
            
        Returns:
            Optimized query parameters
        """
        optimized = query_params.copy()
        start_time = query_params['start_time']
        end_time = query_params['end_time']
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        time_delta = end_time - start_time
        if time_delta.total_seconds() > 86400 * 30:
            start_time = start_time.replace(hour=0, minute=0, second=0,
                microsecond=0)
            end_time = end_time.replace(hour=23, minute=59, second=59,
                microsecond=999999)
        elif time_delta.total_seconds() > 86400:
            start_time = start_time.replace(minute=0, second=0, microsecond=0)
            end_time = end_time.replace(minute=59, second=59, microsecond=
                999999)
        elif time_delta.total_seconds() > 3600:
            start_time = start_time.replace(second=0, microsecond=0)
            end_time = end_time.replace(second=59, microsecond=999999)
        optimized['start_time'] = start_time
        optimized['end_time'] = end_time
        return optimized

    def _add_index_hints(self, query_params: Dict[str, Any]) ->Dict[str, Any]:
        """
        Add index hints based on query parameters
        
        Args:
            query_params: Original query parameters
            
        Returns:
            Query parameters with index hints
        """
        optimized = query_params.copy()
        if 'time_index' not in optimized and 'start_time' in optimized:
            optimized['time_index'] = True
        if 'symbol' in optimized and 'symbol_index' not in optimized:
            optimized['symbol_index'] = True
        return optimized

    def _track_query_pattern(self, query_params: Dict[str, Any]):
        """
        Track query patterns to identify common queries
        
        Args:
            query_params: Query parameters to track
        """
        pattern = {}
        for key, value in query_params.items():
            if key in ['start_time', 'end_time']:
                if 'start_time' in query_params and 'end_time' in query_params:
                    start = query_params['start_time']
                    end = query_params['end_time']
                    if isinstance(start, str):
                        start = pd.to_datetime(start)
                    if isinstance(end, str):
                        end = pd.to_datetime(end)
                    duration = (end - start).total_seconds()
                    if duration <= 3600:
                        pattern['duration'] = '<=1h'
                    elif duration <= 86400:
                        pattern['duration'] = '<=1d'
                    elif duration <= 604800:
                        pattern['duration'] = '<=1w'
                    elif duration <= 2592000:
                        pattern['duration'] = '<=30d'
                    else:
                        pattern['duration'] = '>30d'
            else:
                pattern[key] = type(value).__name__
        pattern_key = json.dumps(pattern, sort_keys=True)
        if pattern_key in self.pattern_counts:
            self.pattern_counts[pattern_key] += 1
        else:
            self.pattern_counts[pattern_key] = 1
            self.query_patterns[pattern_key] = pattern

    def _clean_cache(self):
        """
        Clean expired or least recently used items from cache
        """
        now = datetime.now().timestamp()
        expired_keys = [k for k, (_, timestamp, ttl) in self.query_cache.
            items() if now - timestamp > ttl]
        for key in expired_keys:
            del self.query_cache[key]
        if len(self.query_cache) >= self.max_cache_size:
            sorted_items = sorted(self.query_cache.items(), key=lambda item:
                item[1][1])
            items_to_remove = int(self.max_cache_size * 0.1)
            for i in range(items_to_remove):
                if i < len(sorted_items):
                    key = sorted_items[i][0]
                    if key in self.query_cache:
                        del self.query_cache[key]

    def _generate_query_hash(self, query_params: Dict[str, Any]) ->str:
        """
        Generate a unique hash for query parameters
        
        Args:
            query_params: Query parameters
            
        Returns:
            Hash string representing the query
        """
        standard_params = {}
        for key, value in sorted(query_params.items()):
            if isinstance(value, datetime):
                standard_params[key] = value.isoformat()
            elif isinstance(value, (list, tuple, set)):
                standard_params[key] = tuple(sorted(value))
            else:
                standard_params[key] = value
        json_str = json.dumps(standard_params, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()

    def _get_common_patterns(self, limit: int=5) ->List[Dict[str, Any]]:
        """
        Get the most common query patterns
        
        Args:
            limit: Maximum number of patterns to return
            
        Returns:
            List of most common patterns with their counts
        """
        sorted_patterns = sorted(self.pattern_counts.items(), key=lambda
            item: item[1], reverse=True)
        result = []
        for pattern_key, count in sorted_patterns[:limit]:
            pattern = self.query_patterns.get(pattern_key, {})
            result.append({'pattern': pattern, 'count': count})
        return result

    def _get_query_time_stats(self) ->Dict[str, float]:
        """
        Get statistics on query execution times
        
        Returns:
            Dictionary with min, max, avg execution times
        """
        if not self.query_times:
            return {'min': 0, 'max': 0, 'avg': 0}
        times = list(self.query_times.values())
        return {'min': min(times), 'max': max(times), 'avg': sum(times) /
            len(times)}


class TimeSeriesQueryContext:
    """
    Context manager for measuring and optimizing time series queries
    """

    def __init__(self, optimizer: TimeSeriesQueryOptimizer, query_params:
        Dict[str, Any]):
        """
        Initialize the query context
        
        Args:
            optimizer: The query optimizer instance
            query_params: Original query parameters
        """
        self.optimizer = optimizer
        self.original_params = query_params
        self.optimized_params = None
        self.result = None
        self.start_time = None

    def __enter__(self):
        """
        Enter the context: optimize query and check cache
        
        Returns:
            Tuple of (cache_hit, result, optimized_params)
        """
        self.start_time = time.time()
        self.optimized_params = self.optimizer.optimize_query(self.
            original_params)
        cache_hit, result = self.optimizer.get_cached_result(self.
            optimized_params)
        self.result = result
        return cache_hit, result, self.optimized_params

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context: record query time
        """
        if self.start_time and not exc_type:
            execution_time = time.time() - self.start_time
            self.optimizer.record_query_time(self.original_params,
                execution_time)

    @with_exception_handling
    def cache_result(self, result: Any, ttl: Optional[int]=None) ->bool:
        """
        Cache the query result
        
        Args:
            result: Query result to cache
            ttl: Time-to-live in seconds
            
        Returns:
            Boolean indicating whether caching was successful
        """
        if result is None:
            self.logger.warning(
                'Attempted to cache None result - skipping cache operation')
            return False
        try:
            if hasattr(result, '__len__') and len(result) == 0:
                self.logger.debug('Skipping cache for empty result')
                return False
            cache_start = time.time()
            self.optimizer.cache_result(self.optimized_params, result, ttl)
            cache_duration = time.time() - cache_start
            self.optimizer.record_query_time(self.optimized_params,
                cache_duration)
            self.result = result
            self.logger.debug(
                f"Successfully cached result (size: {len(result) if hasattr(result, '__len__') else 'unknown'}, ttl: {ttl or self.optimizer.default_ttl}s)"
                )
            return True
        except Exception as e:
            self.logger.error(f'Failed to cache query result: {str(e)}')
            return False
