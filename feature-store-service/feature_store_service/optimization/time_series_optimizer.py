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


# Setup logging
logger = logging.getLogger(__name__)


class QueryPlan:
    """
    Represents an optimized query plan for time series data.
    """
    
    def __init__(
        self,
        table_name: str,
        columns: List[str],
        time_column: str = "timestamp",
        where_conditions: Optional[List[str]] = None,
        sort_columns: Optional[List[str]] = None,
        sort_direction: str = "ASC",
        group_by: Optional[List[str]] = None,
        limit: Optional[int] = None,
        time_bucket: Optional[str] = None,
        chunk_size: Optional[int] = None
    ):
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
        
        # Statistics for query optimization
        self.estimated_rows = None
        self.estimated_chunks = None
        self.use_hypertable_cache = False
        self.use_index_scan = False
        self.use_parallel_scan = False
    
    def to_sql(self) -> str:
        """
        Convert the query plan to a SQL statement.
        
        Returns:
            SQL statement string
        """
        # Handle time bucketing for aggregated data
        select_cols = []
        if self.time_bucket:
            select_cols.append(f"time_bucket('{self.time_bucket}', {self.time_column}) AS {self.time_column}")
            if self.time_column not in self.group_by and self.group_by:
                self.group_by.insert(0, f"time_bucket('{self.time_bucket}', {self.time_column})")
        
        # Add all requested columns to select
        for col in self.columns:
            if col != self.time_column or not self.time_bucket:
                select_cols.append(col)
        
        # Build the base SQL query
        sql = f"SELECT {', '.join(select_cols)} FROM {self.table_name}"
        
        # Add WHERE clause if conditions exist
        if self.where_conditions:
            sql += f" WHERE {' AND '.join(self.where_conditions)}"
        
        # Add GROUP BY if specified
        if self.group_by and (self.time_bucket or len(self.group_by) > 1):
            sql += f" GROUP BY {', '.join(self.group_by)}"
        
        # Add ORDER BY
        if self.sort_columns:
            sql += f" ORDER BY {', '.join(self.sort_columns)} {self.sort_direction}"
        
        # Add LIMIT if specified
        if self.limit:
            sql += f" LIMIT {self.limit}"
        
        return sql
    
    def optimize(self, db_session: Session) -> 'QueryPlan':
        """
        Optimize the query plan based on table statistics and TimescaleDB features.
        
        Args:
            db_session: SQLAlchemy session for executing metadata queries
            
        Returns:
            Self with optimized parameters
        """
        try:
            # Get table statistics to estimate query size
            stats_query = f"""
            SELECT 
                reltuples::bigint as approximate_row_count,
                pg_total_relation_size('{self.table_name}') as table_size_bytes
            FROM pg_class
            WHERE relname = '{self.table_name.split('.')[-1]}'
            """
            
            result = db_session.execute(text(stats_query)).fetchone()
            if result:
                self.estimated_rows = result[0]
                table_size_mb = result[1] / (1024 * 1024)
                logger.info(f"Table {self.table_name} has approximately {self.estimated_rows} rows "
                           f"and size {table_size_mb:.2f} MB")
            
            # Check if this is a TimescaleDB hypertable
            hypertable_query = f"""
            SELECT * FROM timescaledb_information.hypertables
            WHERE hypertable_name = '{self.table_name.split('.')[-1]}'
            """
            
            hypertable_info = db_session.execute(text(hypertable_query)).fetchone()
            if hypertable_info:
                logger.info(f"Table {self.table_name} is a TimescaleDB hypertable")
                
                # Get chunk information
                chunks_query = f"""
                SELECT count(*) as chunk_count
                FROM timescaledb_information.chunks
                WHERE hypertable_name = '{self.table_name.split('.')[-1]}'
                """
                
                chunks_result = db_session.execute(text(chunks_query)).fetchone()
                if chunks_result:
                    self.estimated_chunks = chunks_result[0]
                    logger.info(f"Hypertable has {self.estimated_chunks} chunks")
                
                # Enable optimizations for TimescaleDB
                if self.time_column and any(cond for cond in self.where_conditions if self.time_column in cond):
                    # Time range queries benefit from chunk exclusion
                    self.use_hypertable_cache = True
                    logger.info("Enabling chunk exclusion for time range query")
                
                # For large tables, use parallel scans
                if self.estimated_rows and self.estimated_rows > 1000000:
                    self.use_parallel_scan = True
                    logger.info("Enabling parallel scan for large hypertable")
            
            # Check for indexes that can help this query
            index_query = f"""
            SELECT
                i.relname as index_name,
                a.attname as column_name
            FROM
                pg_class t,
                pg_class i,
                pg_index ix,
                pg_attribute a
            WHERE
                t.relname = '{self.table_name.split('.')[-1]}' AND
                t.oid = ix.indrelid AND
                i.oid = ix.indexrelid AND
                a.attrelid = t.oid AND
                a.attnum = ANY(ix.indkey)
            ORDER BY
                i.relname
            """
            
            indexes = db_session.execute(text(index_query)).fetchall()
            indexed_columns = [idx[1] for idx in indexes]
            
            # Check if our WHERE conditions use indexed columns
            for condition in self.where_conditions:
                for col in indexed_columns:
                    if col in condition:
                        self.use_index_scan = True
                        logger.info(f"Query can use index on column {col}")
                        break
            
            # If sorting on indexed column, query will be more efficient
            for sort_col in self.sort_columns:
                if sort_col in indexed_columns:
                    logger.info(f"Sort on {sort_col} can use index")
            
            return self
            
        except Exception as e:
            logger.error(f"Error during query optimization: {str(e)}")
            # Return the unoptimized plan if optimization fails
            return self
    
    def generate_optimized_sql(self) -> str:
        """
        Generate SQL with TimescaleDB-specific optimizations.
        
        Returns:
            Optimized SQL statement string
        """
        base_sql = self.to_sql()
        
        # Add TimescaleDB-specific optimizations
        hints = []
        
        if self.use_hypertable_cache:
            hints.append("TimescaleDB.enable_chunk_exclusion true")
        
        if self.use_parallel_scan and not self.use_index_scan:
            hints.append("enable_parallel_append true")
            hints.append("enable_parallel_seq_scan true")
            hints.append("parallel_workers 4")
        
        if hints:
            # Add hints to the SQL query
            hint_string = ", ".join(hints)
            # Insert hints after the SELECT keyword
            optimized_sql = base_sql.replace("SELECT", f"SELECT /*+ {hint_string} */", 1)
            return optimized_sql
        else:
            return base_sql


class QueryCache:
    """
    Cache for storing and retrieving time series query results.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the query cache.
        
        Args:
            max_size: Maximum number of queries to cache
        """
        self.cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self.max_size = max_size
        self.logger = logging.getLogger(__name__)
    
    def get(self, sql: str) -> Optional[pd.DataFrame]:
        """
        Get a cached query result if it exists and is not expired.
        
        Args:
            sql: SQL query string as cache key
            
        Returns:
            Cached DataFrame or None if not found
        """
        if sql in self.cache:
            df, timestamp = self.cache[sql]
            
            # Check if cache is valid (less than 5 minutes old)
            if datetime.now() - timestamp < timedelta(minutes=5):
                self.logger.info(f"Cache hit for query")
                return df
            else:
                # Remove expired cache entry
                del self.cache[sql]
                self.logger.info(f"Expired cache entry removed")
        
        return None
    
    def set(self, sql: str, df: pd.DataFrame) -> None:
        """
        Cache a query result.
        
        Args:
            sql: SQL query string as cache key
            df: DataFrame result to cache
        """
        # Enforce cache size limit
        if len(self.cache) >= self.max_size:
            # Remove the oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            self.logger.info(f"Removed oldest cache entry to make room")
        
        # Store the result with current timestamp
        self.cache[sql] = (df, datetime.now())
        self.logger.info(f"Cached query result ({len(df)} rows)")
    
    def invalidate(self, table_name: str) -> None:
        """
        Invalidate cache entries for a specific table.
        
        Args:
            table_name: Name of the table to invalidate
        """
        keys_to_remove = [
            key for key in self.cache.keys() 
            if f"FROM {table_name}" in key or f"from {table_name}" in key
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        self.logger.info(f"Invalidated {len(keys_to_remove)} cache entries for table {table_name}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.logger.info("Cleared query cache")


class TimeSeriesQueryOptimizer:
    """
    Optimizes time series queries for TimescaleDB.
    """
    
    def __init__(self, engine=None, connection_string: Optional[str] = None):
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
        
        # Configure default chunk size based on data volume and query patterns
        self.default_chunk_size = 10000
    
    def optimize_query(
        self,
        table_name: str,
        columns: List[str],
        time_range: Optional[Tuple[datetime, datetime]] = None,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None,
        time_bucket: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
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
        # Build the query plan
        query_plan = self._build_query_plan(
            table_name, columns, time_range, filters, sort_by, limit, time_bucket
        )
        
        # Generate SQL
        sql = query_plan.to_sql()
        
        # Check cache if enabled
        if use_cache:
            cached_result = self.cache.get(sql)
            if cached_result is not None:
                return cached_result
        
        # Execute the query
        try:
            # Create a session if engine provided, otherwise use connection string
            if self.engine:
                with self.engine.connect() as conn:
                    # Optimize the query based on database statistics
                    with Session(conn) as session:
                        query_plan = query_plan.optimize(session)
                    
                    # Get the optimized SQL
                    optimized_sql = query_plan.generate_optimized_sql()
                    
                    # Execute with parameters if needed
                    self.logger.info(f"Executing optimized query: {optimized_sql}")
                    start_time = datetime.now()
                    
                    df = pd.read_sql(optimized_sql, conn)
                    
                    query_time = (datetime.now() - start_time).total_seconds()
                    self.logger.info(f"Query completed in {query_time:.2f} seconds, returned {len(df)} rows")
            else:
                # Use direct psycopg2 connection
                with psycopg2.connect(self.connection_string) as conn:
                    # Optimize the query based on database statistics
                    with conn.cursor() as cursor:
                        # Create a simple session-like object to use with optimize
                        class SimpleCursor:
                            def execute(self, query):
                                cursor.execute(query)
                                return cursor
                        
                        simple_cursor = SimpleCursor()
                        query_plan = query_plan.optimize(simple_cursor)
                    
                    # Get the optimized SQL
                    optimized_sql = query_plan.generate_optimized_sql()
                    
                    # Execute with parameters if needed
                    self.logger.info(f"Executing optimized query: {optimized_sql}")
                    start_time = datetime.now()
                    
                    df = pd.read_sql(optimized_sql, conn)
                    
                    query_time = (datetime.now() - start_time).total_seconds()
                    self.logger.info(f"Query completed in {query_time:.2f} seconds, returned {len(df)} rows")
            
            # Cache the result if enabled
            if use_cache:
                self.cache.set(sql, df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Query execution error: {str(e)}")
            raise
    
    def _build_query_plan(
        self,
        table_name: str,
        columns: List[str],
        time_range: Optional[Tuple[datetime, datetime]] = None,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None,
        time_bucket: Optional[str] = None
    ) -> QueryPlan:
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
        # Build WHERE conditions
        where_conditions = []
        
        # Add time range condition if provided
        if time_range:
            start_time, end_time = time_range
            where_conditions.append(f"timestamp >= '{start_time.isoformat()}'")
            where_conditions.append(f"timestamp <= '{end_time.isoformat()}'")
        
        # Add filters if provided
        if filters:
            for column, value in filters.items():
                if isinstance(value, str):
                    where_conditions.append(f"{column} = '{value}'")
                elif isinstance(value, (list, tuple)):
                    value_str = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
                    where_conditions.append(f"{column} IN ({value_str})")
                else:
                    where_conditions.append(f"{column} = {value}")
        
        # Set sort columns
        sort_columns = [sort_by] if sort_by else ["timestamp"]
        
        # Set group by columns if using time_bucket
        group_by = columns.copy() if time_bucket else None
        
        # Create the query plan
        return QueryPlan(
            table_name=table_name,
            columns=columns,
            time_column="timestamp",
            where_conditions=where_conditions,
            sort_columns=sort_columns,
            limit=limit,
            time_bucket=time_bucket,
            group_by=group_by,
            chunk_size=self.default_chunk_size
        )
    
    def get_timescale_continuous_aggregates(self) -> List[Dict[str, Any]]:
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
                        cursor.execute("""
                        SELECT * FROM timescaledb_information.continuous_aggregates
                        """)
                        result = cursor.fetchall()
                        return list(result)
                        
        except Exception as e:
            self.logger.error(f"Error fetching continuous aggregates: {str(e)}")
            return []
    
    def query_continuous_aggregate(
        self,
        view_name: str,
        columns: List[str],
        time_range: Optional[Tuple[datetime, datetime]] = None,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
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
        # TimescaleDB continuous aggregates are materialized views
        # They can be queried like regular tables but with better performance
        return self.optimize_query(
            table_name=view_name,
            columns=columns,
            time_range=time_range,
            filters=filters,
            sort_by=sort_by,
            limit=limit,
            time_bucket=None  # No need for time_bucket as it's already aggregated
        )
    
    def suggest_optimal_query(
        self,
        table_name: str,
        columns: List[str],
        time_range: Optional[Tuple[datetime, datetime]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
        suggestions = {
            "original_params": {
                "table_name": table_name,
                "columns": columns,
                "time_range": [t.isoformat() if t else None for t in time_range] if time_range else None,
                "filters": filters
            },
            "suggestions": []
        }
        
        try:
            # Get continuous aggregates that might be useful
            continuous_aggregates = self.get_timescale_continuous_aggregates()
            
            # Find potential matches for this query
            matching_aggs = []
            for agg in continuous_aggregates:
                # Check if this aggregate is for our table
                if agg.get("materialization_hypertable") == table_name:
                    matching_aggs.append(agg)
            
            if matching_aggs:
                # Check for time range optimization
                if time_range:
                    start_time, end_time = time_range
                    time_diff = end_time - start_time
                    
                    # Find the best matching time bucket
                    for agg in matching_aggs:
                        bucket = agg.get("materialization_interval")
                        if bucket:
                            # Parse the bucket interval (e.g., '1 hour', '1 day')
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
                                
                                if bucket_seconds and time_diff.total_seconds() > bucket_seconds * 10:
                                    suggestions["suggestions"].append({
                                        "type": "use_continuous_aggregate",
                                        "view_name": agg.get("view_name"),
                                        "reason": f"Query spans {time_diff.total_seconds() / 86400:.1f} days, which is much larger than the bucket size ({bucket})"
                                    })
            
            # Check for time range query optimization
            if time_range:
                start_time, end_time = time_range
                time_diff = end_time - start_time
                
                # For large time ranges, suggest time bucketing
                if time_diff.total_seconds() > 86400 * 30:  # More than 30 days
                    suggestions["suggestions"].append({
                        "type": "use_time_bucket",
                        "bucket_size": "1 day",
                        "reason": "Query spans more than 30 days"
                    })
                elif time_diff.total_seconds() > 86400 * 7:  # More than 7 days
                    suggestions["suggestions"].append({
                        "type": "use_time_bucket",
                        "bucket_size": "1 hour",
                        "reason": "Query spans more than 7 days"
                    })
                    
            # If many columns are selected, suggest limiting columns
            if len(columns) > 5:
                suggestions["suggestions"].append({
                    "type": "limit_columns",
                    "reason": f"Query selects {len(columns)} columns, which may impact performance"
                })
                
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating query suggestions: {str(e)}")
            suggestions["error"] = str(e)
            return suggestions
