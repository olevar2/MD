"""
Time Series Access Module

This module provides optimized storage and retrieval mechanisms for time-series data at multiple resolutions.
It implements intelligent data aggregation, predictive loading, and efficient window-based querying capabilities.

The module is designed to work with time-series databases like TimescaleDB or InfluxDB and provides
abstractions to make these interactions efficient and consistent across the platform.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel

from core_foundations.models.market_data import TimeSeriesDataPoint
from core_foundations.utils.caching import LRUCache
from core_foundations.config import settings

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
TimeSeriesData = Union[pd.DataFrame, pd.Series, List[TimeSeriesDataPoint]]
TimeStamp = Union[datetime, pd.Timestamp, str, int]


class Resolution(str, Enum):
    """Defines standard time resolutions for time series data."""
    TICK = "tick"
    SECOND = "1S"
    MINUTE = "1T"
    FIVE_MINUTES = "5T"
    FIFTEEN_MINUTES = "15T"
    HOUR = "1H"
    FOUR_HOURS = "4H"
    DAY = "1D"
    WEEK = "1W"
    MONTH = "1M"

    @classmethod
    def get_seconds(cls, resolution: "Resolution") -> int:
        """Convert resolution to seconds."""
        if resolution == cls.TICK:
            return 0
        
        mapping = {
            cls.SECOND: 1,
            cls.MINUTE: 60,
            cls.FIVE_MINUTES: 300,
            cls.FIFTEEN_MINUTES: 900,
            cls.HOUR: 3600,
            cls.FOUR_HOURS: 14400,
            cls.DAY: 86400,
            cls.WEEK: 604800,
            cls.MONTH: 2592000,  # Approximate
        }
        return mapping.get(resolution, 0)


class AggregationMethod(str, Enum):
    """Available methods for aggregating time-series data."""
    LAST = "last"
    FIRST = "first"
    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    OHLC = "ohlc"
    OHLCV = "ohlcv"
    TIME_WEIGHTED = "time_weighted"
    VOLUME_WEIGHTED = "volume_weighted"


class QueryPattern(BaseModel):
    """Model for tracking and predicting query patterns."""
    resolution: Resolution
    symbol: str
    start_time: datetime
    end_time: datetime
    frequency: int = 1  # Number of times this pattern has been observed
    last_queried: datetime = datetime.now()


class TimeSeriesDataStore(ABC, Generic[T]):
    """Abstract base class for time series data storage implementations."""

    @abstractmethod
    def store(self, data: TimeSeriesData, symbol: str, resolution: Resolution) -> bool:
        """Store time series data for a given symbol and resolution."""
        pass

    @abstractmethod
    def retrieve(
        self, 
        symbol: str, 
        resolution: Resolution,
        start_time: TimeStamp,
        end_time: TimeStamp,
        columns: Optional[List[str]] = None
    ) -> TimeSeriesData:
        """Retrieve time series data for a specific time window."""
        pass

    @abstractmethod
    def aggregate(
        self,
        symbol: str,
        source_resolution: Resolution,
        target_resolution: Resolution,
        method: AggregationMethod,
        start_time: TimeStamp,
        end_time: TimeStamp
    ) -> TimeSeriesData:
        """Aggregate data from source resolution to target resolution."""
        pass

    @abstractmethod
    def batch_store(
        self, 
        data_batch: Dict[str, Dict[Resolution, TimeSeriesData]]
    ) -> Dict[str, List[bool]]:
        """Batch store operation for multiple symbols and resolutions."""
        pass

    @abstractmethod
    def batch_retrieve(
        self,
        queries: Dict[str, Dict[Resolution, Tuple[TimeStamp, TimeStamp]]],
        columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[Resolution, TimeSeriesData]]:
        """Batch retrieve operation for multiple symbols and resolutions."""
        pass


class TimeSeriesAccessManager:
    """
    Manager class for efficient time series data access with multi-resolution storage,
    predictive loading, and intelligent caching.
    """

    def __init__(
        self, 
        data_store: TimeSeriesDataStore,
        cache_size: int = 100,
        enable_prefetch: bool = True,
        prediction_threshold: int = 3
    ):
        """
        Initialize the time series access manager.
        
        Args:
            data_store: The underlying data store implementation
            cache_size: Maximum number of items to keep in the cache
            enable_prefetch: Whether to enable predictive data loading
            prediction_threshold: Minimum frequency of a query pattern to trigger prefetch
        """
        self.data_store = data_store
        self.cache = LRUCache(max_size=cache_size)
        self.query_patterns = {}  # Tracks historical query patterns
        self.enable_prefetch = enable_prefetch
        self.prediction_threshold = prediction_threshold

    def get_data(
        self,
        symbol: str,
        resolution: Resolution,
        start_time: TimeStamp,
        end_time: TimeStamp,
        columns: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> TimeSeriesData:
        """
        Retrieve time series data with efficient caching and prefetching.
        
        Args:
            symbol: The asset symbol
            resolution: Time resolution of the data
            start_time: Start of the time window
            end_time: End of the time window
            columns: Specific columns to retrieve
            force_refresh: Whether to bypass the cache
            
        Returns:
            The requested time series data
        """
        # Standardize timestamps
        start_time_dt = pd.Timestamp(start_time)
        end_time_dt = pd.Timestamp(end_time)
        
        # Generate cache key
        cache_key = self._generate_cache_key(symbol, resolution, start_time_dt, end_time_dt, columns)
        
        # Check cache first if not forced to refresh
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {symbol} at {resolution}")
                return cached_data
        
        # Track this query pattern for future prefetching
        self._track_query_pattern(symbol, resolution, start_time_dt, end_time_dt)
        
        # Retrieve from data store
        data = self.data_store.retrieve(
            symbol=symbol,
            resolution=resolution,
            start_time=start_time_dt,
            end_time=end_time_dt,
            columns=columns
        )
        
        # Cache the result
        self.cache.put(cache_key, data)
        
        # Check if we should prefetch related data
        if self.enable_prefetch:
            self._prefetch_related_data(symbol, resolution, start_time_dt, end_time_dt, columns)
        
        return data

    def aggregate_data(
        self,
        symbol: str,
        source_resolution: Resolution,
        target_resolution: Resolution,
        method: AggregationMethod = AggregationMethod.OHLCV,
        start_time: Optional[TimeStamp] = None,
        end_time: Optional[TimeStamp] = None
    ) -> TimeSeriesData:
        """
        Aggregate data from one resolution to another.
        
        Args:
            symbol: The asset symbol
            source_resolution: Original data resolution
            target_resolution: Target data resolution
            method: Aggregation method to use
            start_time: Start of time window (optional)
            end_time: End of time window (optional)
            
        Returns:
            Aggregated time series data
        """
        # Set default time window if not provided
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)
        if end_time is None:
            end_time = datetime.now()
            
        # Standardize timestamps
        start_time_dt = pd.Timestamp(start_time)
        end_time_dt = pd.Timestamp(end_time)
        
        # Validate resolutions
        self._validate_resolution_conversion(source_resolution, target_resolution)
        
        # Generate cache key for aggregated data
        cache_key = self._generate_cache_key(
            symbol, 
            f"{source_resolution}_to_{target_resolution}_{method}", 
            start_time_dt, 
            end_time_dt
        )
        
        # Check if we have this aggregation cached
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
            
        # Perform aggregation
        aggregated_data = self.data_store.aggregate(
            symbol=symbol,
            source_resolution=source_resolution,
            target_resolution=target_resolution,
            method=method,
            start_time=start_time_dt,
            end_time=end_time_dt
        )
        
        # Cache the result
        self.cache.put(cache_key, aggregated_data)
        
        return aggregated_data

    def store_data(
        self,
        data: TimeSeriesData,
        symbol: str,
        resolution: Resolution
    ) -> bool:
        """
        Store time series data and update appropriate caches.
        
        Args:
            data: The time series data to store
            symbol: The asset symbol
            resolution: Time resolution of the data
            
        Returns:
            Success status of the store operation
        """
        # Store in the data store
        success = self.data_store.store(data, symbol, resolution)
        
        if success:
            # Invalidate related cache entries
            self._invalidate_related_cache_entries(symbol, resolution)
            
            # Automatically update lower resolutions if needed
            self._update_derived_resolutions(data, symbol, resolution)
            
        return success

    def batch_query(
        self,
        queries: Dict[str, Dict[Resolution, Tuple[TimeStamp, TimeStamp]]],
        columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[Resolution, TimeSeriesData]]:
        """
        Perform batch queries for multiple symbols and resolutions.
        
        Args:
            queries: Dictionary mapping symbols to resolutions and time windows
            columns: Specific columns to retrieve
            
        Returns:
            Dictionary of retrieved data by symbol and resolution
        """
        # Check cache first for each query
        results = {}
        uncached_queries = {}
        
        for symbol, resolution_queries in queries.items():
            results[symbol] = {}
            uncached_queries[symbol] = {}
            
            for resolution, (start_time, end_time) in resolution_queries.items():
                start_time_dt = pd.Timestamp(start_time)
                end_time_dt = pd.Timestamp(end_time)
                
                cache_key = self._generate_cache_key(
                    symbol, resolution, start_time_dt, end_time_dt, columns
                )
                
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    results[symbol][resolution] = cached_data
                    # Track this hit for prediction
                    self._track_query_pattern(symbol, resolution, start_time_dt, end_time_dt)
                else:
                    # Need to fetch this data
                    uncached_queries[symbol][resolution] = (start_time_dt, end_time_dt)
        
        # If we have any queries that weren't cached, fetch them in batch
        if any(resolutions for resolutions in uncached_queries.values()):
            fetched_data = self.data_store.batch_retrieve(uncached_queries, columns)
            
            # Update results and cache
            for symbol, resolution_data in fetched_data.items():
                for resolution, data in resolution_data.items():
                    start_time, end_time = uncached_queries[symbol][resolution]
                    cache_key = self._generate_cache_key(
                        symbol, resolution, start_time, end_time, columns
                    )
                    self.cache.put(cache_key, data)
                    results[symbol][resolution] = data
                    
                    # Track for prediction
                    self._track_query_pattern(symbol, resolution, start_time, end_time)
        
        return results

    def get_window_data(
        self,
        symbol: str,
        resolution: Resolution,
        end_time: TimeStamp,
        window_size: int,
        window_unit: str = 'D',  # e.g., 'D' for days, 'H' for hours
        columns: Optional[List[str]] = None
    ) -> TimeSeriesData:
        """
        Get data for a specific window looking back from end_time.
        
        Args:
            symbol: The asset symbol
            resolution: Time resolution of the data
            end_time: End of the time window
            window_size: Size of the window
            window_unit: Unit for window size (e.g., 'D', 'H', 'T')
            columns: Specific columns to retrieve
            
        Returns:
            The requested time series data
        """
        end_time_dt = pd.Timestamp(end_time)
        if window_unit == 'D':
            start_time_dt = end_time_dt - pd.Timedelta(days=window_size)
        elif window_unit == 'H':
            start_time_dt = end_time_dt - pd.Timedelta(hours=window_size)
        elif window_unit == 'T' or window_unit == 'min':
            start_time_dt = end_time_dt - pd.Timedelta(minutes=window_size)
        else:
            raise ValueError(f"Unsupported window unit: {window_unit}")
            
        return self.get_data(
            symbol=symbol,
            resolution=resolution,
            start_time=start_time_dt,
            end_time=end_time_dt,
            columns=columns
        )
        
    def get_latest_data_point(
        self,
        symbol: str,
        resolution: Resolution = Resolution.TICK,
        columns: Optional[List[str]] = None
    ) -> Union[TimeSeriesDataPoint, Dict[str, Any], None]:
        """
        Get the most recent data point for a symbol at specified resolution.
        
        Args:
            symbol: The asset symbol
            resolution: Time resolution of the data
            columns: Specific columns to retrieve
            
        Returns:
            Latest data point or None if not available
        """
        # This is a common operation that benefits from special handling
        now = datetime.now()
        # For tick data, we can use a smaller window
        if resolution == Resolution.TICK:
            window_size = 1
            window_unit = 'H'
        else:
            # For bar data, use a window based on resolution
            resolution_seconds = Resolution.get_seconds(resolution)
            if resolution_seconds < 3600:  # Less than 1 hour
                window_size = 6
                window_unit = 'H'
            else:
                window_size = 1
                window_unit = 'D'
                
        data = self.get_window_data(
            symbol=symbol,
            resolution=resolution,
            end_time=now,
            window_size=window_size,
            window_unit=window_unit,
            columns=columns
        )
        
        if data is None or (isinstance(data, (pd.DataFrame, pd.Series)) and data.empty):
            return None
        
        if isinstance(data, pd.DataFrame):
            # Return the last row as a dictionary
            return data.iloc[-1].to_dict()
        elif isinstance(data, list) and data:
            # Return the last data point in the list
            return data[-1]
        
        return None
        
    def clear_cache(self, symbol: Optional[str] = None, resolution: Optional[Resolution] = None) -> None:
        """
        Clear cache entries, optionally filtered by symbol and resolution.
        
        Args:
            symbol: Optional symbol to filter
            resolution: Optional resolution to filter
        """
        if symbol is None and resolution is None:
            # Clear entire cache
            self.cache.clear()
        else:
            # Clear specific entries
            keys_to_remove = []
            for key in self.cache.keys():
                key_parts = key.split(':')
                if len(key_parts) >= 2:
                    cache_symbol, cache_resolution = key_parts[0], key_parts[1]
                    if (symbol is None or cache_symbol == symbol) and \
                       (resolution is None or cache_resolution == str(resolution)):
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.cache.remove(key)

    def _update_derived_resolutions(
        self,
        data: TimeSeriesData,
        symbol: str,
        resolution: Resolution
    ) -> None:
        """
        Update derived lower resolutions when higher resolution data is updated.
        For example, when new tick data arrives, update 1-min bars if needed.
        
        Args:
            data: The new time series data
            symbol: The asset symbol
            resolution: Resolution of the data
        """
        if isinstance(data, pd.DataFrame) and not data.empty:
            # Determine which derived resolutions need updates
            update_resolutions = self._get_derived_resolutions(resolution)
            if not update_resolutions:
                return
                
            min_timestamp = data.index.min()
            max_timestamp = data.index.max()
            
            for target_res in update_resolutions:
                # Only aggregate data if we have a complete bar for the target resolution
                resolution_seconds = Resolution.get_seconds(target_res)
                # Check if our data spans at least one full bar of the target resolution
                if (max_timestamp - min_timestamp).total_seconds() >= resolution_seconds:
                    # Align to bars
                    aligned_start = min_timestamp.floor(target_res.value)
                    aligned_end = max_timestamp.ceil(target_res.value)
                    
                    # Aggregate the data to the target resolution
                    aggregated = self.data_store.aggregate(
                        symbol=symbol,
                        source_resolution=resolution,
                        target_resolution=target_res,
                        method=AggregationMethod.OHLCV,
                        start_time=aligned_start,
                        end_time=aligned_end
                    )
                    
                    if not aggregated.empty:
                        # Store the aggregated data
                        self.store_data(aggregated, symbol, target_res)

    def _get_derived_resolutions(self, resolution: Resolution) -> List[Resolution]:
        """
        Get list of resolutions that can be derived from the provided resolution.
        
        Args:
            resolution: Source resolution
            
        Returns:
            List of resolutions that can be derived
        """
        # Define the resolution hierarchy
        hierarchy = [
            Resolution.TICK,
            Resolution.SECOND,
            Resolution.MINUTE, 
            Resolution.FIVE_MINUTES,
            Resolution.FIFTEEN_MINUTES,
            Resolution.HOUR,
            Resolution.FOUR_HOURS,
            Resolution.DAY,
            Resolution.WEEK,
            Resolution.MONTH
        ]
        
        # Find the index of the current resolution
        try:
            index = hierarchy.index(resolution)
        except ValueError:
            return []
            
        # Return all resolutions that come after the current one
        if index < len(hierarchy) - 1:
            return hierarchy[index+1:]
        return []

    def _track_query_pattern(
        self, 
        symbol: str, 
        resolution: Resolution,
        start_time: TimeStamp,
        end_time: TimeStamp
    ) -> None:
        """
        Track query patterns for predictive data loading.
        
        Args:
            symbol: The asset symbol
            resolution: Time resolution of the data
            start_time: Start of the time window
            end_time: End of the time window
        """
        if not self.enable_prefetch:
            return
            
        start_time_dt = pd.Timestamp(start_time)
        end_time_dt = pd.Timestamp(end_time)
        
        # Create a pattern key
        pattern_key = f"{symbol}:{resolution}:{start_time_dt.round('H')}:{end_time_dt.round('H')}"
        
        # Update or create the pattern
        if pattern_key in self.query_patterns:
            pattern = self.query_patterns[pattern_key]
            pattern.frequency += 1
            pattern.last_queried = datetime.now()
        else:
            pattern = QueryPattern(
                resolution=resolution,
                symbol=symbol,
                start_time=start_time_dt.to_pydatetime(),
                end_time=end_time_dt.to_pydatetime()
            )
            self.query_patterns[pattern_key] = pattern
            
        # Prune old patterns occasionally
        if len(self.query_patterns) > 1000:  # Arbitrary threshold
            self._prune_old_patterns()

    def _prune_old_patterns(self, max_age_days: int = 7) -> None:
        """
        Remove old query patterns that haven't been used recently.
        
        Args:
            max_age_days: Maximum age in days for patterns to keep
        """
        now = datetime.now()
        cutoff = now - timedelta(days=max_age_days)
        
        keys_to_remove = []
        for key, pattern in self.query_patterns.items():
            if pattern.last_queried < cutoff:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.query_patterns[key]

    def _prefetch_related_data(
        self,
        symbol: str,
        resolution: Resolution,
        start_time: TimeStamp,
        end_time: TimeStamp,
        columns: Optional[List[str]] = None
    ) -> None:
        """
        Prefetch data based on predicted query patterns.
        
        Args:
            symbol: The current symbol being queried
            resolution: Current resolution being queried
            start_time: Start of current time window
            end_time: End of current time window
            columns: Columns being queried
        """
        # Look for related symbols often queried together
        related_symbols = self._find_related_symbols(symbol)
        
        # Look for related time windows
        next_window = self._predict_next_window(symbol, resolution, end_time)
        
        prefetch_tasks = []
        
        # Prefetch data for the same symbol but different resolutions
        for related_res in self._get_common_resolutions(symbol, resolution):
            prefetch_tasks.append({
                'symbol': symbol,
                'resolution': related_res,
                'start_time': start_time,
                'end_time': end_time,
                'columns': columns
            })
            
        # Prefetch data for related symbols at the same resolution
        for related_symbol in related_symbols:
            prefetch_tasks.append({
                'symbol': related_symbol,
                'resolution': resolution,
                'start_time': start_time,
                'end_time': end_time,
                'columns': columns
            })
            
        # Prefetch next time window if predicted
        if next_window:
            prefetch_tasks.append({
                'symbol': symbol,
                'resolution': resolution,
                'start_time': next_window[0],
                'end_time': next_window[1],
                'columns': columns
            })
            
        # Execute prefetch tasks in background (simplified - in real implementation use async)
        for task in prefetch_tasks:
            try:
                cache_key = self._generate_cache_key(
                    task['symbol'], 
                    task['resolution'],
                    task['start_time'],
                    task['end_time'],
                    task['columns']
                )
                
                # Only prefetch if not already cached
                if self.cache.get(cache_key) is None:
                    # In real implementation, this would be done asynchronously
                    logger.debug(f"Prefetching data for {task['symbol']} at {task['resolution']}")
                    data = self.data_store.retrieve(
                        symbol=task['symbol'],
                        resolution=task['resolution'],
                        start_time=task['start_time'],
                        end_time=task['end_time'],
                        columns=task['columns']
                    )
                    self.cache.put(cache_key, data)
            except Exception as e:
                logger.warning(f"Error during prefetch: {str(e)}")

    def _find_related_symbols(self, symbol: str) -> List[str]:
        """
        Find symbols that are often queried together with the given symbol.
        
        Args:
            symbol: The reference symbol
            
        Returns:
            List of related symbols
        """
        # Count how many times other symbols appear with this symbol in patterns
        symbol_counts = {}
        
        # First pass: find all patterns with this symbol
        symbol_patterns = []
        for pattern_key, pattern in self.query_patterns.items():
            if pattern.symbol == symbol:
                symbol_patterns.append(pattern_key)
                
        # Second pass: find symbols that appear close in time to this symbol
        now = datetime.now()
        recent_cutoff = now - timedelta(minutes=30)  # Consider patterns in the last 30 minutes
        
        for pattern_key, pattern in self.query_patterns.items():
            if pattern_key in symbol_patterns or pattern.symbol == symbol:
                continue
            
            if pattern.last_queried >= recent_cutoff:
                symbol_counts[pattern.symbol] = symbol_counts.get(pattern.symbol, 0) + 1
                
        # Return top related symbols
        related = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
        return [s for s, _ in related[:3]]  # Return top 3 related symbols

    def _predict_next_window(
        self, 
        symbol: str, 
        resolution: Resolution,
        current_end_time: TimeStamp
    ) -> Optional[Tuple[TimeStamp, TimeStamp]]:
        """
        Predict the next time window likely to be queried.
        
        Args:
            symbol: The symbol being queried
            resolution: Resolution of the data
            current_end_time: End time of current query
            
        Returns:
            Tuple of (start_time, end_time) for the next window, or None
        """
        current_end = pd.Timestamp(current_end_time)
        
        # Find matching patterns with this symbol and resolution
        matching_patterns = []
        for pattern in self.query_patterns.values():
            if pattern.symbol == symbol and pattern.resolution == resolution:
                matching_patterns.append(pattern)
                
        if not matching_patterns:
            return None
            
        # Look for patterns that often follow the current window
        window_delta = None
        max_frequency = 0
        
        for pattern in matching_patterns:
            pattern_end = pd.Timestamp(pattern.end_time)
            
            # Skip patterns that end before our current window
            if pattern_end <= current_end:
                continue
                
            # If this pattern has been seen frequently, consider it
            if pattern.frequency >= self.prediction_threshold:
                delta = (pattern_end - current_end).total_seconds()
                if window_delta is None or pattern.frequency > max_frequency:
                    window_delta = delta
                    max_frequency = pattern.frequency
        
        # If we found a likely next window
        if window_delta is not None:
            # Calculate the start and end times for the next window
            # This is simplified - would be more sophisticated in real implementation
            window_size = (current_end - pd.Timestamp(matching_patterns[0].start_time)).total_seconds()
            next_start = current_end
            next_end = next_start + pd.Timedelta(seconds=window_size)
            return (next_start, next_end)
            
        # Alternative: just return the next window based on the resolution
        resolution_seconds = Resolution.get_seconds(resolution)
        if resolution_seconds > 0:
            next_start = current_end
            next_end = next_start + pd.Timedelta(seconds=resolution_seconds * 10)  # Get the next 10 bars
            return (next_start, next_end)
            
        return None

    def _get_common_resolutions(self, symbol: str, current_resolution: Resolution) -> List[Resolution]:
        """
        Find resolutions commonly queried with this symbol.
        
        Args:
            symbol: The asset symbol
            current_resolution: The current resolution
            
        Returns:
            List of commonly queried resolutions
        """
        resolution_counts = {}
        
        for pattern in self.query_patterns.values():
            if pattern.symbol == symbol and pattern.resolution != current_resolution:
                res_str = str(pattern.resolution)
                resolution_counts[res_str] = resolution_counts.get(res_str, 0) + 1
                
        # Return top 2 most common resolutions
        common = sorted(resolution_counts.items(), key=lambda x: x[1], reverse=True)
        return [Resolution(r) for r, _ in common[:2]]  # Return top 2 common resolutions

    def _generate_cache_key(
        self,
        symbol: str,
        resolution: Union[Resolution, str],
        start_time: TimeStamp,
        end_time: TimeStamp,
        columns: Optional[List[str]] = None
    ) -> str:
        """
        Generate a unique cache key for the query parameters.
        
        Args:
            symbol: The asset symbol
            resolution: Time resolution of the data
            start_time: Start of the time window
            end_time: End of the time window
            columns: Specific columns included
            
        Returns:
            A unique cache key string
        """
        start_str = pd.Timestamp(start_time).strftime('%Y%m%d%H%M%S')
        end_str = pd.Timestamp(end_time).strftime('%Y%m%d%H%M%S')
        col_str = '_'.join(columns) if columns else 'all'
        
        return f"{symbol}:{resolution}:{start_str}:{end_str}:{col_str}"

    def _invalidate_related_cache_entries(self, symbol: str, resolution: Resolution) -> None:
        """
        Invalidate cache entries that might be affected by new data.
        
        Args:
            symbol: The asset symbol
            resolution: Time resolution of the updated data
        """
        keys_to_remove = []
        
        for key in self.cache.keys():
            key_parts = key.split(':')
            if len(key_parts) >= 2:
                cache_symbol, cache_resolution = key_parts[0], key_parts[1]
                
                # Clear entries for this symbol and resolution
                if cache_symbol == symbol and (
                    cache_resolution == str(resolution) or 
                    # If this is derived data
                    cache_resolution.startswith(f"{resolution}_to_") or
                    # If this data derives from the updated resolution
                    cache_resolution.endswith(f"_to_{resolution}")
                ):
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.cache.remove(key)

    def _validate_resolution_conversion(
        self, 
        source_resolution: Resolution, 
        target_resolution: Resolution
    ) -> bool:
        """
        Validate that the source resolution can be converted to the target resolution.
        
        Args:
            source_resolution: Original data resolution
            target_resolution: Target data resolution
            
        Returns:
            True if conversion is valid, raises ValueError otherwise
        """
        source_seconds = Resolution.get_seconds(source_resolution)
        target_seconds = Resolution.get_seconds(target_resolution)
        
        if source_resolution == Resolution.TICK and target_seconds > 0:
            # Can always aggregate tick data to bar data
            return True
            
        if source_seconds == 0 or target_seconds == 0:
            # Can't convert to or from tick data (except as above)
            if source_resolution != Resolution.TICK:
                raise ValueError(f"Cannot convert {source_resolution} to {target_resolution}")
        
        if target_seconds <= source_seconds:
            raise ValueError(
                f"Cannot aggregate to a higher resolution: {source_resolution} to {target_resolution}"
            )
            
        if target_seconds % source_seconds != 0:
            raise ValueError(
                f"Target resolution must be a multiple of source resolution: "
                f"{source_resolution} to {target_resolution}"
            )
            
        return True


# Example implementation for a specific database
class TimeScaleDBDataStore(TimeSeriesDataStore):
    """TimescaleDB-specific implementation of TimeSeriesDataStore."""
    
    def __init__(self, connection_string: str):
        """
        Initialize TimescaleDB data store.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        # In a real implementation, initialize the database connection
        
    def store(self, data: TimeSeriesData, symbol: str, resolution: Resolution) -> bool:
        """Store time series data for a given symbol and resolution."""
        # Implementation for storing data in TimescaleDB
        logger.info(f"Storing {len(data)} data points for {symbol} at {resolution}")
        # In a real implementation, convert data to appropriate format and store
        return True
        
    def retrieve(
        self, 
        symbol: str, 
        resolution: Resolution,
        start_time: TimeStamp,
        end_time: TimeStamp,
        columns: Optional[List[str]] = None
    ) -> TimeSeriesData:
        """Retrieve time series data for a specific time window."""
        # Implementation for retrieving data from TimescaleDB
        logger.info(f"Retrieving data for {symbol} at {resolution} from {start_time} to {end_time}")
        # In a real implementation, build and execute query
        return pd.DataFrame()  # Placeholder
        
    def aggregate(
        self,
        symbol: str,
        source_resolution: Resolution,
        target_resolution: Resolution,
        method: AggregationMethod,
        start_time: TimeStamp,
        end_time: TimeStamp
    ) -> TimeSeriesData:
        """Aggregate data from source resolution to target resolution."""
        # Implementation for aggregating data in TimescaleDB
        logger.info(f"Aggregating {symbol} data from {source_resolution} to {target_resolution}")
        # In a real implementation, use TimescaleDB's time_bucket function
        return pd.DataFrame()  # Placeholder
        
    def batch_store(
        self, 
        data_batch: Dict[str, Dict[Resolution, TimeSeriesData]]
    ) -> Dict[str, List[bool]]:
        """Batch store operation for multiple symbols and resolutions."""
        # Implementation for batch storing data
        results = {}
        for symbol, resolution_data in data_batch.items():
            results[symbol] = []
            for resolution, data in resolution_data.items():
                success = self.store(data, symbol, resolution)
                results[symbol].append(success)
        return results
        
    def batch_retrieve(
        self,
        queries: Dict[str, Dict[Resolution, Tuple[TimeStamp, TimeStamp]]],
        columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[Resolution, TimeSeriesData]]:
        """Batch retrieve operation for multiple symbols and resolutions."""
        # Implementation for batch retrieving data
        results = {}
        for symbol, resolution_queries in queries.items():
            results[symbol] = {}
            for resolution, (start_time, end_time) in resolution_queries.items():
                data = self.retrieve(symbol, resolution, start_time, end_time, columns)
                results[symbol][resolution] = data
        return results


# Example implementation for InfluxDB
class InfluxDBDataStore(TimeSeriesDataStore):
    """InfluxDB-specific implementation of TimeSeriesDataStore."""
    
    def __init__(self, url: str, token: str, org: str, bucket: str):
        """
        Initialize InfluxDB data store.
        
        Args:
            url: InfluxDB server URL
            token: Authentication token
            org: Organization name
            bucket: Bucket name
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        # In a real implementation, initialize the InfluxDB client
        
    def store(self, data: TimeSeriesData, symbol: str, resolution: Resolution) -> bool:
        """Store time series data for a given symbol and resolution."""
        # Implementation for storing data in InfluxDB
        logger.info(f"Storing {len(data)} data points for {symbol} at {resolution} in InfluxDB")
        # In a real implementation, convert data to line protocol and write
        return True
        
    def retrieve(
        self, 
        symbol: str, 
        resolution: Resolution,
        start_time: TimeStamp,
        end_time: TimeStamp,
        columns: Optional[List[str]] = None
    ) -> TimeSeriesData:
        """Retrieve time series data for a specific time window."""
        # Implementation for retrieving data from InfluxDB
        logger.info(f"Retrieving data for {symbol} at {resolution} from {start_time} to {end_time}")
        # In a real implementation, build and execute Flux query
        return pd.DataFrame()  # Placeholder
        
    def aggregate(
        self,
        symbol: str,
        source_resolution: Resolution,
        target_resolution: Resolution,
        method: AggregationMethod,
        start_time: TimeStamp,
        end_time: TimeStamp
    ) -> TimeSeriesData:
        """Aggregate data from source resolution to target resolution."""
        # Implementation for aggregating data in InfluxDB
        logger.info(f"Aggregating {symbol} data from {source_resolution} to {target_resolution}")
        # In a real implementation, use Flux aggregation functions
        return pd.DataFrame()  # Placeholder
        
    def batch_store(
        self, 
        data_batch: Dict[str, Dict[Resolution, TimeSeriesData]]
    ) -> Dict[str, List[bool]]:
        """Batch store operation for multiple symbols and resolutions."""
        # Similar implementation to TimescaleDB version
        results = {}
        for symbol, resolution_data in data_batch.items():
            results[symbol] = []
            for resolution, data in resolution_data.items():
                success = self.store(data, symbol, resolution)
                results[symbol].append(success)
        return results
        
    def batch_retrieve(
        self,
        queries: Dict[str, Dict[Resolution, Tuple[TimeStamp, TimeStamp]]],
        columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[Resolution, TimeSeriesData]]:
        """Batch retrieve operation for multiple symbols and resolutions."""
        # Similar implementation to TimescaleDB version
        results = {}
        for symbol, resolution_queries in queries.items():
            results[symbol] = {}
            for resolution, (start_time, end_time) in resolution_queries.items():
                data = self.retrieve(symbol, resolution, start_time, end_time, columns)
                results[symbol][resolution] = data
        return results
