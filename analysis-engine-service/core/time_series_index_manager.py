"""
Time Series Index Manager

This module provides functionality to optimize time series database queries
by efficiently managing indexes on timestamp columns.
"""
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from sqlalchemy import Column, Table, Index, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.schema import MetaData
from sqlalchemy.sql import Select, select
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class TimeSeriesIndexManager:
    """
    A service that optimizes time series queries by managing database indexes
    and providing strategies for efficient timestamp-based queries.
    """

    def __init__(self, engine: Optional[Engine]=None):
        """Initialize the index manager with an optional SQLAlchemy engine."""
        self.engine = engine
        self.metadata = MetaData()
        self.indexed_tables = set()
        self.index_statistics = {}

    def register_table(self, table: Table, timestamp_column: str) ->None:
        """
        Register a table for time series index management.
        
        Args:
            table: SQLAlchemy Table object
            timestamp_column: Name of the timestamp column to index
        """
        if table.name in self.indexed_tables:
            logger.debug(f'Table {table.name} already registered for indexing')
            return
        self.indexed_tables.add(table.name)
        self.index_statistics[table.name] = {'table': table,
            'timestamp_column': timestamp_column, 'indexed': False,
            'query_count': 0, 'avg_query_time': 0}
        logger.info(
            f'Registered table {table.name} with timestamp column {timestamp_column}'
            )

    @with_exception_handling
    def ensure_index(self, table_name: str) ->bool:
        """
        Ensure that the timestamp index exists for the given table.
        
        Args:
            table_name: Name of the table to index
            
        Returns:
            bool: True if index was created or already exists, False otherwise
        """
        if table_name not in self.indexed_tables or not self.engine:
            logger.warning(
                f'Cannot create index for {table_name}: table not registered or no engine'
                )
            return False
        stats = self.index_statistics[table_name]
        if stats['indexed']:
            return True
        table = stats['table']
        timestamp_col = stats['timestamp_column']
        try:
            idx_name = f'idx_{table_name}_{timestamp_col}'
            idx = Index(idx_name, table.c[timestamp_col])
            idx.create(self.engine)
            stats['indexed'] = True
            logger.info(
                f'Created timestamp index {idx_name} on {table_name}.{timestamp_col}'
                )
            return True
        except Exception as e:
            logger.error(
                f'Failed to create index on {table_name}.{timestamp_col}: {str(e)}'
                )
            return False

    def optimize_query(self, query: Select, table_name: str) ->Select:
        """
        Optimize a query for time series performance.
        
        Args:
            query: SQLAlchemy Select object to optimize
            table_name: Name of the table being queried
            
        Returns:
            Select: Optimized query
        """
        if table_name not in self.indexed_tables:
            return query
        self.ensure_index(table_name)
        self.index_statistics[table_name]['query_count'] += 1
        return query

    @with_resilience('get_optimal_time_range_strategy')
    def get_optimal_time_range_strategy(self, table_name: str, from_date:
        Optional[datetime]=None, to_date: Optional[datetime]=None) ->Dict[
        str, Any]:
        """
        Get the optimal query strategy for a time range query.
        
        Args:
            table_name: Name of the table to query
            from_date: Start date for the range query
            to_date: End date for the range query
            
        Returns:
            Dict with strategy information
        """
        if table_name not in self.indexed_tables:
            return {'strategy': 'full_scan'}
        strategy = {'strategy': 'indexed_range', 'use_index': True}
        if from_date and to_date and to_date - from_date > timedelta(days=30):
            strategy['strategy'] = 'partitioned_range'
            partition_days = 30
            partition_count = (to_date - from_date).days // partition_days + 1
            strategy.update({'partition_days': partition_days,
                'partition_count': partition_count})
        return strategy

    def partition_time_range(self, from_date: datetime, to_date: datetime,
        max_partition_days: int=30) ->List[Tuple[datetime, datetime]]:
        """
        Split a large time range into smaller partitions for better query performance.
        
        Args:
            from_date: Start date
            to_date: End date
            max_partition_days: Maximum days per partition
            
        Returns:
            List of (start_date, end_date) tuples for each partition
        """
        partitions = []
        current_start = from_date
        while current_start < to_date:
            current_end = min(current_start + timedelta(days=
                max_partition_days), to_date)
            partitions.append((current_start, current_end))
            current_start = current_end
        return partitions

    @with_resilience('get_statistics')
    def get_statistics(self) ->Dict[str, Dict[str, Any]]:
        """
        Get statistics about indexed tables and query performance.
        
        Returns:
            Dict with index statistics
        """
        return self.index_statistics
