"""
Time Series Query Optimizer

This module provides optimized query capabilities for time series data,
specifically designed for efficient retrieval and processing of financial
market data stored in TimescaleDB.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import json
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class QueryPlan:
    """
    Represents an optimized query plan for time series data.
    """

    def __init__(self, table_name: str, columns: List[str], time_column:
        str='timestamp', where_conditions: Optional[List[str]]=None,
        sort_columns: Optional[List[str]]=None, sort_direction: str='ASC',
        group_by: Optional[List[str]]=None, limit: Optional[int]=None,
        time_bucket: Optional[str]=None, chunk_size: Optional[int]=None):
    """
      init  .
    
    Args:
        table_name: Description of table_name
        columns: Description of columns
        time_column: Description of time_column
        where_conditions: Description of where_conditions
        sort_columns: Description of sort_columns
        sort_direction: Description of sort_direction
        group_by: Description of group_by
        limit: Description of limit
        time_bucket: Description of time_bucket
        chunk_size: Description of chunk_size
    
    """

        self.table_name = table_name
        self.columns = columns
        self.time_column = time_column
        self.where_conditions = where_conditions or []
        self.sort_columns = sort_columns or [time_column]
        self.sort_direction = sort_direction
        self.group_by = group_by or []
        self.limit = limit
        self.time_bucket = time_bucket
        self.chunk_size = chunk_size
        self.estimated_rows = None
        self.estimated_chunks = None
        self.use_hypertable_cache = False
        self.use_index_scan = False
        self.use_parallel_scan = False

    def to_sql(self) ->str:
        """
        Convert the query plan to a SQL statement.
        
        Returns:
            SQL statement string
        """
        select_cols = []
        if self.time_bucket:
            select_cols.append(
                f"time_bucket('{self.time_bucket}', {self.time_column}) AS {self.time_column}"
                )
            if self.time_column not in self.group_by and self.group_by:
                self.group_by.insert(0,
                    f"time_bucket('{self.time_bucket}', {self.time_column})")
        for col in self.columns:
            if col != self.time_column or not self.time_bucket:
                select_cols.append(col)
        sql = f"SELECT {', '.join(select_cols)} FROM {self.table_name}"
        if self.where_conditions:
            sql += f" WHERE {' AND '.join(self.where_conditions)}"
        if self.group_by and (self.time_bucket or len(self.group_by) > 1):
            sql += f" GROUP BY {', '.join(self.group_by)}"
        if self.sort_columns:
            sql += (
                f" ORDER BY {', '.join(self.sort_columns)} {self.sort_direction}"
                )
        if self.limit:
            sql += f' LIMIT {self.limit}'
        return sql

    @with_exception_handling
    def optimize(self, db_session: Session) ->'QueryPlan':
        """
        Optimize the query plan based on table statistics and TimescaleDB features.
        
        Args:
            db_session: SQLAlchemy session for executing metadata queries
            
        Returns:
            Self with optimized parameters
        """
        try:
            table_name_safe = self.table_name.split('.')[-1]
            stats_query = """
            SELECT 
                reltuples::bigint as approximate_row_count,
                pg_total_relation_size(:table_name) as table_size_bytes
            FROM pg_class
            WHERE relname = :table_name_safe
            """
            result = db_session.execute(text(stats_query), {'table_name':
                self.table_name, 'table_name_safe': table_name_safe}).fetchone(
                )
            if result:
                self.estimated_rows = result[0]
                table_size_mb = result[1] / (1024 * 1024)
                logger.info(
                    f'Table {self.table_name} has approximately {self.estimated_rows} rows and size {table_size_mb:.2f} MB'
                    )
            hypertable_query = """
            SELECT * FROM timescaledb_information.hypertables
            WHERE hypertable_name = :table_name_safe
            """
            hypertable_info = db_session.execute(text(hypertable_query), {
                'table_name_safe': table_name_safe}).fetchone()
            if hypertable_info:
                logger.info(
                    f'Table {self.table_name} is a TimescaleDB hypertable')
                chunks_query = """
                SELECT count(*) as chunk_count
                FROM timescaledb_information.chunks
                WHERE hypertable_name = :table_name_safe
                """
                chunks_result = db_session.execute(text(chunks_query), {
                    'table_name_safe': table_name_safe}).fetchone()
                if chunks_result:
                    self.estimated_chunks = chunks_result[0]
                    logger.info(
                        f'Hypertable has {self.estimated_chunks} chunks')
                if self.time_column and any(cond for cond in self.
                    where_conditions if self.time_column in cond):
                    self.use_hypertable_cache = True
                    logger.info('Enabling chunk exclusion for time range query'
                        )
                if self.estimated_rows and self.estimated_rows > 1000000:
                    self.use_parallel_scan = True
                    logger.info('Enabling parallel scan for large hypertable')
            index_query = """
            SELECT
                i.relname as index_name,
                a.attname as column_name
            FROM
                pg_class t,
                pg_class i,
                pg_index ix,
                pg_attribute a
            WHERE
                t.relname = :table_name_safe AND
                t.oid = ix.indrelid AND
                i.oid = ix.indexrelid AND
                a.attrelid = t.oid AND
                a.attnum = ANY(ix.indkey)
            ORDER BY
                i.relname
            """
            indexes = db_session.execute(text(index_query), {
                'table_name_safe': table_name_safe}).fetchall()
            indexed_columns = [idx[1] for idx in indexes]
            for condition in self.where_conditions:
                for col in indexed_columns:
                    if col in condition:
                        self.use_index_scan = True
                        logger.info(f'Query can use index on column {col}')
                        break
            for sort_col in self.sort_columns:
                if sort_col in indexed_columns:
                    logger.info(f'Sort on {sort_col} can use index')
            return self
        except Exception as e:
            logger.error(f'Error during query optimization: {str(e)}')
            return self

    def generate_optimized_sql(self) ->str:
        """
        Generate SQL with TimescaleDB-specific optimizations.
        
        Returns:
            Optimized SQL statement string
        """
        base_sql = self.to_sql()
        hints = []
        if self.use_hypertable_cache:
            hints.append('TimescaleDB.enable_chunk_exclusion true')
        if self.use_parallel_scan and not self.use_index_scan:
            hints.append('enable_parallel_append true')
            hints.append('enable_parallel_seq_scan true')
            hints.append('parallel_workers 4')
        if hints:
            hint_string = ', '.join(hints)
            optimized_sql = base_sql.replace('SELECT',
                f'SELECT /*+ {hint_string} */', 1)
            return optimized_sql
        else:
            return base_sql


class QueryCache:
    """
    Cache for storing and retrieving time series query results.
    """

    def __init__(self, max_size: int=100):
        """
        Initialize the query cache.
        
        Args:
            max_size: Maximum number of queries to cache
        """
        self.cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self.max_size = max_size
        self.logger = logging.getLogger(__name__)

    def get(self, sql: str) ->Optional[pd.DataFrame]:
        """
        Get a cached query result if it exists and is not expired.
        
        Args:
            sql: SQL query string as cache key
            
        Returns:
            Cached DataFrame or None if not found
        """
        if sql in self.cache:
            df, timestamp = self.cache[sql]
            if datetime.now() - timestamp < timedelta(minutes=5):
                self.logger.info(f'Cache hit for query')
                return df
            else:
                del self.cache[sql]
                self.logger.info(f'Expired cache entry removed')
        return None

    def set(self, sql: str, df: pd.DataFrame) ->None:
        """
        Cache a query result.
        
        Args:
            sql: SQL query string as cache key
            df: DataFrame result to cache
        """
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            self.logger.info(f'Removed oldest cache entry to make room')
        self.cache[sql] = df, datetime.now()
        self.logger.info(f'Cached query result ({len(df)} rows)')

    def invalidate(self, table_name: str) ->None:
        """
        Invalidate cache entries for a specific table.
        
        Args:
            table_name: Name of the table to invalidate
        """
        keys_to_remove = [key for key in self.cache.keys() if 
            f'FROM {table_name}' in key or f'from {table_name}' in key]
        for key in keys_to_remove:
            del self.cache[key]
        self.logger.info(
            f'Invalidated {len(keys_to_remove)} cache entries for table {table_name}'
            )

    def clear(self) ->None:
        """Clear all cache entries."""
        self.cache.clear()
        self.logger.info('Cleared query cache')


class TimeSeriesQueryOptimizer:
    """
    Optimizes time series queries for TimescaleDB.
    """

    def __init__(self, engine=None, connection_string: Optional[str]=None):
        """
        Initialize the query optimizer.
        
        Args:
            engine: SQLAlchemy engine
            connection_string: Database connection string (alternative to engine)
        """
        self.engine = engine
        self.connection_string = connection_string
        self.cache = QueryCache()
        self.logger = logging.getLogger(__name__)
        self.default_chunk_size = 10000

    @with_exception_handling
    def optimize_query(self, table_name: str, columns: List[str],
        time_range: Optional[Tuple[datetime, datetime]]=None, filters:
        Optional[Dict[str, Any]]=None, sort_by: Optional[str]=None, limit:
        Optional[int]=None, time_bucket: Optional[str]=None, use_cache:
        bool=True) ->pd.DataFrame:
        """
        Execute an optimized time series query.
        
        Args:
            table_name: Name of the table
            columns: List of columns to select
            time_range: Optional tuple of (start_time, end_time)
            filters: Optional dictionary of column filters
            sort_by: Optional column to sort by
            limit: Optional limit on the number of results
            time_bucket: Optional time bucket for aggregation (e.g. '1 hour')
            use_cache: Whether to use query cache
            
        Returns:
            DataFrame containing query results
        """
        query_plan = self._build_query_plan(table_name, columns, time_range,
            filters, sort_by, limit, time_bucket)
        sql = query_plan.to_sql()
        if use_cache:
            cached_result = self.cache.get(sql)
            if cached_result is not None:
                return cached_result
        try:
            if self.engine:
                with self.engine.connect() as conn:
                    with Session(conn) as session:
                        query_plan = query_plan.optimize(session)
                    optimized_sql = query_plan.generate_optimized_sql()
                    self.logger.info(
                        f'Executing optimized query: {optimized_sql}')
                    start_time = datetime.now()
                    df = pd.read_sql(optimized_sql, conn)
                    query_time = (datetime.now() - start_time).total_seconds()
                    self.logger.info(
                        f'Query completed in {query_time:.2f} seconds, returned {len(df)} rows'
                        )
            else:
                with psycopg2.connect(self.connection_string) as conn:
                    with conn.cursor() as cursor:


                        class SimpleCursor:
    """
    SimpleCursor class.
    
    Attributes:
        Add attributes here
    """


                            def execute(self, query):
    """
    Execute.
    
    Args:
        query: Description of query
    
    """

                                cursor.execute(query)
                                return cursor
                        simple_cursor = SimpleCursor()
                        query_plan = query_plan.optimize(simple_cursor)
                    optimized_sql = query_plan.generate_optimized_sql()
                    self.logger.info(
                        f'Executing optimized query: {optimized_sql}')
                    start_time = datetime.now()
                    df = pd.read_sql(optimized_sql, conn)
                    query_time = (datetime.now() - start_time).total_seconds()
                    self.logger.info(
                        f'Query completed in {query_time:.2f} seconds, returned {len(df)} rows'
                        )
            if use_cache:
                self.cache.set(sql, df)
            return df
        except Exception as e:
            self.logger.error(f'Query execution error: {str(e)}')
            raise

    def build_query_plan(self, table_name: str, columns: List[str],
        time_range: Optional[Tuple[datetime, datetime]]=None, filters:
        Optional[Dict[str, Any]]=None, sort_by: Optional[str]=None, limit:
        Optional[int]=None, time_bucket: Optional[str]=None) ->QueryPlan:
        """
        Build a QueryPlan object from query parameters.
        
        Args:
            table_name: Name of the table
            columns: List of columns to select
            time_range: Optional tuple of (start_time, end_time)
            filters: Optional dictionary of column filters
            sort_by: Optional column to sort by
            limit: Optional limit on the number of results
            time_bucket: Optional time bucket for aggregation
            
        Returns:
            QueryPlan object
        """
        self._validate_table_name(table_name)
        self._validate_columns(columns)
        if sort_by:
            self._validate_column_name(sort_by)
        if time_bucket:
            self._validate_time_bucket(time_bucket)
        where_conditions = []
        if time_range:
            start_time, end_time = time_range
            where_conditions.append('timestamp >= :start_time')
            where_conditions.append('timestamp <= :end_time')
        if filters:
            for column, value in filters.items():
                self._validate_column_name(column)
                if isinstance(value, (list, tuple)):
                    where_conditions.append(f'{column} IN (:value_{column})')
                else:
                    where_conditions.append(f'{column} = :value_{column}')
        sort_columns = [sort_by] if sort_by else ['timestamp']
        group_by = columns.copy() if time_bucket else None
        return QueryPlan(table_name=table_name, columns=columns,
            time_column='timestamp', where_conditions=where_conditions,
            sort_columns=sort_columns, limit=limit, time_bucket=time_bucket,
            group_by=group_by, chunk_size=self.default_chunk_size)

    def validate_table_name(self, table_name: str) ->None:
        """
        Validate table name to prevent SQL injection.
        
        Args:
            table_name: Table name to validate
            
        Raises:
            ValueError: If table name contains invalid characters
        """
        if not table_name or not isinstance(table_name, str):
            raise ValueError('Table name must be a non-empty string')
        if not all(c.isalnum() or c == '_' or c == '.' for c in table_name):
            raise ValueError(f'Invalid table name: {table_name}')

    def validate_column_name(self, column_name: str) ->None:
        """
        Validate column name to prevent SQL injection.
        
        Args:
            column_name: Column name to validate
            
        Raises:
            ValueError: If column name contains invalid characters
        """
        if not column_name or not isinstance(column_name, str):
            raise ValueError('Column name must be a non-empty string')
        if not all(c.isalnum() or c == '_' for c in column_name):
            raise ValueError(f'Invalid column name: {column_name}')

    def validate_columns(self, columns: List[str]) ->None:
        """
        Validate a list of column names.
        
        Args:
            columns: List of column names to validate
            
        Raises:
            ValueError: If any column name is invalid
        """
        if not columns or not isinstance(columns, list):
            raise ValueError('Columns must be a non-empty list')
        for column in columns:
            self._validate_column_name(column)

    @with_exception_handling
    def validate_time_bucket(self, time_bucket: str) ->None:
        """
        Validate time bucket string.
        
        Args:
            time_bucket: Time bucket string to validate
            
        Raises:
            ValueError: If time bucket format is invalid
        """
        if not time_bucket or not isinstance(time_bucket, str):
            raise ValueError('Time bucket must be a non-empty string')
        valid_units = ['microsecond', 'microseconds', 'millisecond',
            'milliseconds', 'second', 'seconds', 'minute', 'minutes',
            'hour', 'hours', 'day', 'days', 'week', 'weeks', 'month',
            'months', 'year', 'years']
        parts = time_bucket.split()
        if len(parts) != 2:
            raise ValueError(f'Invalid time bucket format: {time_bucket}')
        try:
            float(parts[0])
            if parts[1].lower() not in valid_units:
                raise ValueError(
                    f'Invalid time unit in time bucket: {parts[1]}')
        except ValueError:
            raise ValueError(f'Invalid time bucket format: {time_bucket}')

    @with_exception_handling
    def get_timescale_continuous_aggregates(self) ->List[Dict[str, Any]]:
        """
        Get information about available TimescaleDB continuous aggregates.
        
        Returns:
            List of dictionaries with continuous aggregate information
        """
        try:
            if self.engine:
                with self.engine.connect() as conn:
                    with Session(conn) as session:
                        query = """
                        SELECT * FROM timescaledb_information.continuous_aggregates
                        """
                        result = session.execute(text(query)).fetchall()
                        return [dict(row) for row in result]
            else:
                with psycopg2.connect(self.connection_string) as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute(
                            """
                        SELECT * FROM timescaledb_information.continuous_aggregates
                        """
                            )
                        result = cursor.fetchall()
                        return list(result)
        except Exception as e:
            self.logger.error(f'Error fetching continuous aggregates: {str(e)}'
                )
            return []

    def query_continuous_aggregate(self, view_name: str, columns: List[str],
        time_range: Optional[Tuple[datetime, datetime]]=None, filters:
        Optional[Dict[str, Any]]=None, sort_by: Optional[str]=None, limit:
        Optional[int]=None) ->pd.DataFrame:
        """
        Query a TimescaleDB continuous aggregate view.
        
        Args:
            view_name: Name of the continuous aggregate view
            columns: List of columns to select
            time_range: Optional tuple of (start_time, end_time)
            filters: Optional dictionary of column filters
            sort_by: Optional column to sort by
            limit: Optional limit on the number of results
            
        Returns:
            DataFrame containing query results
        """
        return self.optimize_query(table_name=view_name, columns=columns,
            time_range=time_range, filters=filters, sort_by=sort_by, limit=
            limit, time_bucket=None)

    @with_exception_handling
    def suggest_optimal_query(self, table_name: str, columns: List[str],
        time_range: Optional[Tuple[datetime, datetime]]=None, filters:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Suggest the optimal query approach based on the query parameters.
        
        Args:
            table_name: Name of the table
            columns: List of columns to select
            time_range: Optional tuple of (start_time, end_time)
            filters: Optional dictionary of column filters
            
        Returns:
            Dictionary with suggestions for optimal query
        """
        suggestions = {'original_params': {'table_name': table_name,
            'columns': columns, 'time_range': [(t.isoformat() if t else
            None) for t in time_range] if time_range else None, 'filters':
            filters}, 'suggestions': []}
        try:
            continuous_aggregates = self.get_timescale_continuous_aggregates()
            matching_aggs = []
            for agg in continuous_aggregates:
                if agg.get('materialization_hypertable') == table_name:
                    matching_aggs.append(agg)
            if matching_aggs:
                if time_range:
                    start_time, end_time = time_range
                    time_diff = end_time - start_time
                    for agg in matching_aggs:
                        bucket = agg.get('materialization_interval')
                        if bucket:
                            parts = bucket.split()
                            if len(parts) >= 2:
                                quantity = int(parts[0])
                                unit = parts[1].lower()
                                if unit.startswith('hour'):
                                    bucket_seconds = quantity * 3600
                                elif unit.startswith('day'):
                                    bucket_seconds = quantity * 86400
                                elif unit.startswith('minute'):
                                    bucket_seconds = quantity * 60
                                elif unit.startswith('week'):
                                    bucket_seconds = quantity * 604800
                                else:
                                    bucket_seconds = None
                                if bucket_seconds and time_diff.total_seconds(
                                    ) > bucket_seconds * 10:
                                    suggestions['suggestions'].append({
                                        'type': 'use_continuous_aggregate',
                                        'view_name': agg.get('view_name'),
                                        'reason':
                                        f'Query spans {time_diff.total_seconds() / 86400:.1f} days, which is much larger than the bucket size ({bucket})'
                                        })
            if time_range:
                start_time, end_time = time_range
                time_diff = end_time - start_time
                if time_diff.total_seconds() > 86400 * 30:
                    suggestions['suggestions'].append({'type':
                        'use_time_bucket', 'bucket_size': '1 day', 'reason':
                        'Query spans more than 30 days'})
                elif time_diff.total_seconds() > 86400 * 7:
                    suggestions['suggestions'].append({'type':
                        'use_time_bucket', 'bucket_size': '1 hour',
                        'reason': 'Query spans more than 7 days'})
            if len(columns) > 5:
                suggestions['suggestions'].append({'type': 'limit_columns',
                    'reason':
                    f'Query selects {len(columns)} columns, which may impact performance'
                    })
            return suggestions
        except Exception as e:
            self.logger.error(f'Error generating query suggestions: {str(e)}')
            suggestions['error'] = str(e)
            return suggestions
