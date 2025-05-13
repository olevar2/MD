"""
Optimized Time Series Query Manager

This module provides optimized query functionality for time-series data in TimescaleDB.
It implements caching, query optimization, and streaming capabilities for efficient data access.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import select, func, text
from sqlalchemy.orm import Session
from sqlalchemy.sql import expression
from feature_store_service.caching import CacheManager


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
    Optimizes database queries for time-series data, focusing on timestamp-based queries.
    
    Features:
    - Query optimization using TimescaleDB hypertable capabilities
    - Multi-level caching (memory and disk) for frequently accessed data
    - Support for streaming data for real-time applications
    - Automatic chunk selection for improved query performance
    """

    def __init__(self, db_session: Session, cache_manager: Optional[
        CacheManager]=None, cache_ttl: int=3600, enable_streaming: bool=False):
        """
        Initialize the query optimizer
        
        Args:
            db_session: Database session for queries
            cache_manager: Optional cache manager instance (creates one if not provided)
            cache_ttl: Time-to-live for cached results in seconds
            enable_streaming: Whether to enable streaming capabilities
        """
        self.db = db_session
        self.logger = logging.getLogger(__name__)
        if cache_manager:
            self.cache_manager = cache_manager
        else:
            from feature_store_service.caching import create_default_cache_manager
            self.cache_manager = create_default_cache_manager()
        self.cache_ttl = cache_ttl
        self.enable_streaming = enable_streaming
        self.stats = {'cache_hits': 0, 'cache_misses': 0, 'query_times': [],
            'streaming_sessions': 0}

    @with_exception_handling
    def query_time_series(self, table, columns: List[str], start_time:
        Optional[datetime]=None, end_time: Optional[datetime]=None, filters:
        Optional[Dict]=None, order_by: str='timestamp', order_desc: bool=
        False, limit: Optional[int]=None, use_cache: bool=True,
        chunk_time_interval: Optional[str]=None) ->pd.DataFrame:
        """
        Execute an optimized time-series query
        
        Args:
            table: SQLAlchemy model or table
            columns: List of column names to select
            start_time: Start time for the query range
            end_time: End time for the query range
            filters: Additional filter conditions
            order_by: Column to order by (default: timestamp)
            order_desc: Whether to order in descending order
            limit: Maximum number of rows to return
            use_cache: Whether to use cache for this query
            chunk_time_interval: Optional chunk time interval for TimescaleDB
        
        Returns:
            DataFrame with query results
        """
        import time
        start = time.time()
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(table.__tablename__,
                columns, start_time, end_time, filters, limit, order_desc)
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f'Cache hit for key {cache_key}')
                self.stats['cache_hits'] += 1
                elapsed = time.time() - start
                self.logger.debug(
                    f'Query completed in {elapsed:.4f}s (from cache)')
                return cached_data
        self.stats['cache_misses'] += 1
        query = self._build_optimized_query(table, columns, start_time,
            end_time, filters, order_by, order_desc, limit, chunk_time_interval
            )
        try:
            result = pd.read_sql(query.statement, self.db.connection())
            if use_cache and cache_key:
                self.cache_manager.set(cache_key, result, self.cache_ttl)
            elapsed = time.time() - start
            self.stats['query_times'].append(elapsed)
            self.logger.debug(
                f'Query completed in {elapsed:.4f}s (from database)')
            return result
        except Exception as e:
            self.logger.error(f'Error executing time-series query: {str(e)}')
            raise

    @with_exception_handling
    def stream_time_series(self, table, columns: List[str], batch_size: int
        =1000, start_time: Optional[datetime]=None, filters: Optional[Dict]
        =None, chunk_time_interval: Optional[str]=None):
        """
        Stream time-series data with a generator to handle large datasets
        
        Args:
            table: SQLAlchemy model or table
            columns: List of column names to select
            batch_size: Number of rows per batch
            start_time: Optional start time (defaults to now for real-time streaming)
            filters: Additional filter conditions
            chunk_time_interval: Optional chunk time interval for TimescaleDB
        
        Yields:
            DataFrame batches of time-series data
        """
        if not self.enable_streaming:
            self.logger.warning(
                'Streaming is not enabled. Please initialize with enable_streaming=True'
                )
            return
        self.stats['streaming_sessions'] += 1
        current_time = start_time or datetime.utcnow()
        while True:
            try:
                query = self._build_optimized_query(table, columns,
                    start_time=current_time, end_time=None, filters=filters,
                    limit=batch_size, chunk_time_interval=chunk_time_interval)
                batch = pd.read_sql(query.statement, self.db.connection())
                if batch.empty:
                    import time
                    time.sleep(0.1)
                    continue
                if 'timestamp' in batch.columns and not batch['timestamp'
                    ].empty:
                    current_time = batch['timestamp'].max() + timedelta(
                        milliseconds=1)
                yield batch
            except Exception as e:
                self.logger.error(
                    f'Error in streaming time-series data: {str(e)}')
                import time
                time.sleep(1)

    @with_exception_handling
    def get_optimized_time_ranges(self, table, interval: str='1h',
        start_time: Optional[datetime]=None, end_time: Optional[datetime]=None
        ) ->Dict[str, Any]:
        """
        Get optimized time ranges based on continuous aggregates in TimescaleDB
        
        Args:
            table: SQLAlchemy model or table
            interval: Time interval for the aggregates
            start_time: Start time of the range
            end_time: End time of the range
        
        Returns:
            Dictionary with optimized time ranges and statistics
        """
        try:
            continuous_aggregates = self._get_continuous_aggregates(table.
                __tablename__)
            if not continuous_aggregates:
                return {'has_optimized_ranges': False, 'message':
                    'No continuous aggregates available for this table'}
            best_aggregate = None
            for agg in continuous_aggregates:
                if agg['interval'] == interval:
                    best_aggregate = agg
                    break
            if not best_aggregate:
                return {'has_optimized_ranges': False, 'message':
                    f'No continuous aggregate with interval {interval} found'}
            stats_query = text(
                f"""
                SELECT 
                    min(time_bucket('{interval}', timestamp)) as min_time,
                    max(time_bucket('{interval}', timestamp)) as max_time,
                    count(*) as bucket_count
                FROM {best_aggregate['view_name']}
                WHERE 
                    timestamp >= :start_time AND 
                    timestamp <= :end_time
            """
                )
            params = {}
            if start_time:
                params['start_time'] = start_time
            else:
                params['start_time'] = text("'-infinity'::timestamptz")
            if end_time:
                params['end_time'] = end_time
            else:
                params['end_time'] = text("'infinity'::timestamptz")
            result = self.db.execute(stats_query, params).fetchone()
            return {'has_optimized_ranges': True, 'aggregate_view':
                best_aggregate['view_name'], 'interval': interval,
                'min_time': result[0], 'max_time': result[1],
                'bucket_count': result[2]}
        except Exception as e:
            self.logger.error(f'Error getting optimized time ranges: {str(e)}')
            return {'has_optimized_ranges': False, 'error': str(e)}

    def get_performance_stats(self) ->Dict[str, Any]:
        """
        Get performance statistics for the query optimizer
        
        Returns:
            Dictionary with performance statistics
        """
        stats = self.stats.copy()
        if stats['query_times']:
            stats['avg_query_time'] = sum(stats['query_times']) / len(stats
                ['query_times'])
            stats['min_query_time'] = min(stats['query_times'])
            stats['max_query_time'] = max(stats['query_times'])
        else:
            stats['avg_query_time'] = 0
            stats['min_query_time'] = 0
            stats['max_query_time'] = 0
        total_queries = stats['cache_hits'] + stats['cache_misses']
        if total_queries > 0:
            stats['cache_hit_ratio'] = stats['cache_hits'] / total_queries
        else:
            stats['cache_hit_ratio'] = 0
        stats['cache_stats'] = self.cache_manager.get_stats()
        return stats

    def _build_optimized_query(self, table, columns: List[str], start_time:
        Optional[datetime]=None, end_time: Optional[datetime]=None, filters:
        Optional[Dict]=None, order_by: str='timestamp', order_desc: bool=
        False, limit: Optional[int]=None, chunk_time_interval: Optional[str
        ]=None):
        """
        Build an optimized SQLAlchemy query for time-series data
        
        Args:
            table: SQLAlchemy model or table
            columns: List of column names to select
            start_time: Start time for the query range
            end_time: End time for the query range
            filters: Additional filter conditions
            order_by: Column to order by
            order_desc: Whether to order in descending order
            limit: Maximum number of rows to return
            chunk_time_interval: Optional chunk time interval for TimescaleDB
        
        Returns:
            SQLAlchemy query object
        """
        if '*' in columns:
            query = select(table)
        else:
            query = select([getattr(table, col) for col in columns])
        if start_time:
            query = query.where(table.timestamp >= start_time)
        if end_time:
            query = query.where(table.timestamp <= end_time)
        if filters:
            for column, value in filters.items():
                if hasattr(table, column):
                    query = query.where(getattr(table, column) == value)
                else:
                    self.logger.warning(
                        f'Ignoring filter for unknown column: {column}')
        if hasattr(table, order_by):
            order_col = getattr(table, order_by)
            if order_desc:
                query = query.order_by(order_col.desc())
            else:
                query = query.order_by(order_col)
        if limit:
            query = query.limit(limit)
        if chunk_time_interval:
            pass
        return query

    def _generate_cache_key(self, table_name: str, columns: List[str],
        start_time: Optional[datetime], end_time: Optional[datetime],
        filters: Optional[Dict], limit: Optional[int], order_desc: bool) ->str:
        """
        Generate a unique cache key for a query
        
        Args:
            table_name: Name of the table
            columns: Columns to select
            start_time: Start time of the query
            end_time: End time of the query
            filters: Additional filters
            limit: Query limit
            order_desc: Whether ordering is descending
            
        Returns:
            Cache key string
        """
        import hashlib
        import json
        key_dict = {'table': table_name, 'columns': sorted(columns),
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': end_time.isoformat() if end_time else None,
            'filters': json.dumps(filters, sort_keys=True) if filters else
            None, 'limit': limit, 'order_desc': order_desc}
        key_str = json.dumps(key_dict, sort_keys=True)
        return f'tsq:{hashlib.md5(key_str.encode()).hexdigest()}'

    @with_exception_handling
    def _get_continuous_aggregates(self, table_name: str) ->List[Dict[str, str]
        ]:
        """
        Get continuous aggregates defined for a table in TimescaleDB
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of dictionaries with aggregate information
        """
        try:
            query = text(
                """
                SELECT 
                    view_name,
                    materialization_hypertable,
                    view_definition,
                    refresh_interval
                FROM timescaledb_information.continuous_aggregates
                WHERE materialization_hypertable = :table_name
                ORDER BY refresh_interval
            """
                )
            results = self.db.execute(query, {'table_name': table_name}
                ).fetchall()
            aggregates = []
            for row in results:
                view_def = row[2].lower()
                interval = None
                import re
                interval_match = re.search("time_bucket\\('([^']+)'", view_def)
                if interval_match:
                    interval = interval_match.group(1)
                aggregates.append({'view_name': row[0], 'hypertable': row[1
                    ], 'view_definition': row[2], 'refresh_interval': row[3
                    ], 'interval': interval})
            return aggregates
        except Exception as e:
            self.logger.error(f'Error fetching continuous aggregates: {str(e)}'
                )
            return []
