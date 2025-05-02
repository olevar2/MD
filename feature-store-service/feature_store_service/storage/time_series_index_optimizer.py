"""
Time Series Index Optimizer

This module provides specialized indexing strategies for timestamp-based queries,
implementing various techniques for low-latency time series data access.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import json
import hashlib
import asyncpg
import asyncio
from functools import lru_cache
from enum import Enum


class TimePrecision(Enum):
    """Enumeration of time precision levels for index optimization"""
    MILLISECOND = "millisecond"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class TimeBucket(Enum):
    """Enumeration of time bucket sizes for TimescaleDB hypertables"""
    MINUTE = "1 minute"
    HOUR = "1 hour"
    DAY = "1 day"
    WEEK = "7 days"
    MONTH = "30 days"
    QUARTER = "90 days"


class IndexType(Enum):
    """Types of indexes available for time series data"""
    B_TREE = "btree"
    BRIN = "brin"
    HASH = "hash"  # For exact equality matches
    GIN = "gin"    # For array or jsonb data
    HYPERTABLE = "hypertable_partitioning"
    COMPOSITE = "composite"  # For multi-column indexes


class TimeSeriesIndexManager:
    """
    Manages index creation, maintenance and utilization strategies for timestamp-based queries
    
    This class provides advanced index management specifically optimized for
    time series data, with specialized support for TimescaleDB hypertables.
    """
    
    def __init__(
        self, 
        db_pool: asyncpg.Pool,
        schema_name: str = "public",
        metadata_table: str = "timeseries_index_metadata"
    ):
        """
        Initialize the time series index manager
        
        Args:
            db_pool: Connection pool to the database
            schema_name: Database schema name
            metadata_table: Table name for storing index metadata
        """
        self.db_pool = db_pool
        self.schema_name = schema_name
        self.metadata_table = metadata_table
        self.logger = logging.getLogger(__name__)
        
        # Cache of index information
        self.index_cache: Dict[str, Dict[str, Any]] = {}
        self.query_stats: Dict[str, Dict[str, Any]] = {}
        
        # Flag to track initialization
        self._initialized = False
    
    async def initialize(self):
        """
        Initialize the index manager, ensuring metadata tables exist
        """
        if self._initialized:
            return
            
        async with self.db_pool.acquire() as conn:
            # Create metadata table if it doesn't exist
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.metadata_table} (
                    table_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    index_name TEXT NOT NULL,
                    index_type TEXT NOT NULL,
                    time_precision TEXT,
                    time_bucket TEXT,
                    is_hypertable BOOLEAN DEFAULT FALSE,
                    is_compressed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_used TIMESTAMPTZ,
                    usage_count INTEGER DEFAULT 0,
                    avg_query_time FLOAT DEFAULT 0.0,
                    config JSONB DEFAULT '{{}}'::jsonb,
                    PRIMARY KEY (table_name, index_name)
                )
            """)

            # Create statistics table for query patterns
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.timeseries_query_stats (
                    query_hash TEXT PRIMARY KEY,
                    query_pattern TEXT NOT NULL,
                    execution_count INTEGER DEFAULT 0,
                    avg_execution_time FLOAT DEFAULT 0.0,
                    last_executed TIMESTAMPTZ DEFAULT NOW(),
                    time_range_seconds FLOAT,
                    parameter_stats JSONB DEFAULT '{{}}'::jsonb
                )
            """)
            
            # Load existing indexes into cache
            rows = await conn.fetch(f"""
                SELECT * FROM {self.schema_name}.{self.metadata_table}
            """)
            
            for row in rows:
                table_name = row['table_name']
                index_name = row['index_name']
                if table_name not in self.index_cache:
                    self.index_cache[table_name] = {}
                self.index_cache[table_name][index_name] = dict(row)
        
        self._initialized = True
    
    async def create_time_index(
        self,
        table_name: str,
        time_column: str,
        precision: TimePrecision = TimePrecision.MINUTE,
        index_type: IndexType = IndexType.B_TREE,
        include_columns: Optional[List[str]] = None
    ) -> str:
        """
        Create an optimized timestamp-based index
        
        Args:
            table_name: Name of the table to index
            time_column: Name of the timestamp column
            precision: Time precision level for optimizing the index
            index_type: Type of index to create
            include_columns: Additional columns to include in a composite index
            
        Returns:
            Name of the created index
        """
        await self.initialize()
        
        # Generate index name
        base_name = f"idx_{table_name}_{time_column}"
        if include_columns:
            base_name += "_" + "_".join(include_columns)
        index_name = f"{base_name}_{index_type.value}_{precision.value}"
        
        # Prevent duplicate index creation
        if table_name in self.index_cache and index_name in self.index_cache[table_name]:
            self.logger.info(f"Index {index_name} already exists on {table_name}")
            return index_name
        
        async with self.db_pool.acquire() as conn:
            # Different index creation strategies based on type and precision
            if index_type == IndexType.HYPERTABLE:
                # For TimescaleDB hypertables, we create the hypertable partitioning
                time_bucket = self._get_time_bucket_from_precision(precision)
                await conn.execute(f"""
                    SELECT create_hypertable(
                        '{table_name}', '{time_column}',
                        chunk_time_interval => INTERVAL '{time_bucket.value}'
                    )
                """)
            elif index_type == IndexType.COMPOSITE and include_columns:
                # Create composite index with time and additional columns
                columns = [time_column] + include_columns
                columns_sql = ", ".join(columns)
                await conn.execute(f"""
                    CREATE INDEX {index_name} ON {self.schema_name}.{table_name} 
                    USING {index_type.value} ({columns_sql})
                """)
            elif index_type == IndexType.BRIN:
                # BRIN indexes are efficient for time series with timestamps
                # that correlate with physical storage
                await conn.execute(f"""
                    CREATE INDEX {index_name} ON {self.schema_name}.{table_name} 
                    USING brin ({time_column})
                """)
            else:
                # Standard B-tree index
                await conn.execute(f"""
                    CREATE INDEX {index_name} ON {self.schema_name}.{table_name} 
                    USING btree ({time_column})
                """)
            
            # Record index metadata
            await conn.execute(f"""
                INSERT INTO {self.schema_name}.{self.metadata_table}
                (table_name, column_name, index_name, index_type, time_precision, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (table_name, index_name) DO NOTHING
            """, table_name, time_column, index_name, index_type.value, precision.value)
            
            # Update cache
            if table_name not in self.index_cache:
                self.index_cache[table_name] = {}
                
            self.index_cache[table_name][index_name] = {
                'table_name': table_name,
                'column_name': time_column,
                'index_name': index_name,
                'index_type': index_type.value,
                'time_precision': precision.value,
                'created_at': datetime.now()
            }
            
            self.logger.info(f"Created {index_type.value} index {index_name} on {table_name}.{time_column}")
            
        return index_name
    
    async def recommend_index_strategy(
        self,
        table_name: str,
        time_column: str,
        sample_queries: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Recommend optimal indexing strategy based on table characteristics and query patterns
        
        Args:
            table_name: Table to analyze
            time_column: Timestamp column
            sample_queries: List of sample queries to analyze
            
        Returns:
            Dictionary with recommended indexing strategy
        """
        await self.initialize()
        
        async with self.db_pool.acquire() as conn:
            # Analyze table size and distribution
            table_stats = await conn.fetchrow(f"""
                SELECT 
                    pg_total_relation_size('{self.schema_name}.{table_name}') as total_size,
                    (SELECT COUNT(*) FROM {self.schema_name}.{table_name}) as row_count
            """)
            
            # Analyze time distribution if available
            time_stats = await conn.fetchrow(f"""
                SELECT 
                    MIN({time_column}) as min_time,
                    MAX({time_column}) as max_time,
                    (MAX({time_column}) - MIN({time_column})) as time_range
                FROM {self.schema_name}.{table_name}
                WHERE {time_column} IS NOT NULL
            """)
            
            # Get information about existing indexes
            existing_indexes = await conn.fetch(f"""
                SELECT 
                    i.indexname as index_name,
                    i.indexdef as index_definition
                FROM 
                    pg_indexes i
                WHERE 
                    i.schemaname = '{self.schema_name}' AND 
                    i.tablename = '{table_name}'
            """)
            
            # Check if TimescaleDB is installed and if this is already a hypertable
            is_timescale_installed = False
            is_hypertable = False
            try:
                ts_check = await conn.fetchval("SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'")
                is_timescale_installed = ts_check > 0
                
                if is_timescale_installed:
                    ht_check = await conn.fetchval(f"""
                        SELECT COUNT(*) FROM timescaledb_information.hypertables 
                        WHERE hypertable_name = '{table_name}'
                    """)
                    is_hypertable = ht_check > 0
            except:
                # Ignore errors - TimescaleDB might not be available
                pass
        
        # Make recommendations based on gathered information
        recommendations = {
            "table_name": table_name,
            "time_column": time_column,
            "table_size_bytes": table_stats['total_size'] if table_stats else None,
            "row_count": table_stats['row_count'] if table_stats else None,
            "time_range": str(time_stats['time_range']) if time_stats and time_stats['time_range'] else None,
            "existing_indexes": [idx['index_name'] for idx in existing_indexes],
            "recommended_strategies": []
        }
        
        # Define table size thresholds (in bytes)
        SMALL_TABLE = 1024 * 1024 * 100  # 100 MB
        MEDIUM_TABLE = 1024 * 1024 * 1024  # 1 GB
        LARGE_TABLE = 1024 * 1024 * 1024 * 10  # 10 GB
        
        table_size = table_stats['total_size'] if table_stats else 0
        row_count = table_stats['row_count'] if table_stats else 0
        
        # Make size-based recommendations
        if is_timescale_installed and table_size >= MEDIUM_TABLE and not is_hypertable:
            recommendations["recommended_strategies"].append({
                "strategy": "convert_to_hypertable",
                "index_type": IndexType.HYPERTABLE.value,
                "reason": "Large time-series table would benefit from TimescaleDB hypertable partitioning",
                "priority": "high" 
            })
        
        if table_size <= SMALL_TABLE:
            recommendations["recommended_strategies"].append({
                "strategy": "btree_index",
                "index_type": IndexType.B_TREE.value,
                "reason": "Small table will perform well with standard B-tree index on timestamp",
                "priority": "medium"
            })
        elif table_size > LARGE_TABLE:
            recommendations["recommended_strategies"].append({
                "strategy": "brin_index",
                "index_type": IndexType.BRIN.value,
                "reason": "Very large table would benefit from space-efficient BRIN index on timestamp",
                "priority": "high"
            })
        
        # Analyze query patterns if provided
        if sample_queries:
            common_filters = self._analyze_query_filters(sample_queries)
            if common_filters:
                # Recommend composite indexes for commonly filtered columns
                composite_columns = [col for col, count in common_filters.items() 
                                    if count >= len(sample_queries) * 0.3 and col != time_column]
                
                if composite_columns:
                    recommendations["recommended_strategies"].append({
                        "strategy": "composite_index",
                        "index_type": IndexType.COMPOSITE.value,
                        "columns": [time_column] + composite_columns[:2],  # Limit to time + 2 most common columns
                        "reason": f"Queries frequently filter on these columns along with timestamp",
                        "priority": "high"
                    })
        
        return recommendations
    
    def get_optimal_query_plan(
        self,
        table_name: str,
        time_column: str,
        start_time: datetime,
        end_time: datetime,
        filters: Dict[str, Any] = None,
        select_columns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an optimized query plan for a time series query
        
        Args:
            table_name: Table to query
            time_column: Timestamp column name
            start_time: Query start time
            end_time: Query end time
            filters: Additional filters for the query
            select_columns: Columns to select
            
        Returns:
            Dictionary with optimized query components
        """
        if table_name not in self.index_cache:
            return self._get_default_query_plan(
                table_name, time_column, start_time, end_time, filters, select_columns
            )
        
        # Available indexes for this table
        indexes = self.index_cache[table_name]
        
        # Find optimal index for this query
        best_index = None
        best_score = -1
        
        for idx_name, idx_info in indexes.items():
            # Skip non-time indexes
            if idx_info['column_name'] != time_column:
                continue
                
            # Calculate a score for this index based on:
            # 1. Index type suitability
            # 2. Time range size vs. index precision
            # 3. Usage statistics
            
            # Base score by index type
            score = 0
            if idx_info['index_type'] == IndexType.HYPERTABLE.value:
                score += 10  # Hypertable is generally best for time series
            elif idx_info['index_type'] == IndexType.BRIN.value:
                score += 5   # BRIN is good for sequential scans
            elif idx_info['index_type'] == IndexType.B_TREE.value:
                score += 3   # B-tree is good for range scans
            
            # Adjust score based on time range
            time_diff = (end_time - start_time).total_seconds()
            if time_diff < 3600 and idx_info['time_precision'] in [TimePrecision.SECOND.value, TimePrecision.MINUTE.value]:
                score += 5  # Short time range benefits from fine precision
            elif time_diff < 86400 and idx_info['time_precision'] == TimePrecision.HOUR.value:
                score += 3  # Day range works well with hour precision
            elif time_diff > 86400 * 30 and idx_info['time_precision'] in [TimePrecision.MONTH.value, TimePrecision.QUARTER.value]:
                score += 3  # Large ranges work well with coarse precision
            
            # Composite index bonus if we have matching filters
            if idx_info['index_type'] == IndexType.COMPOSITE.value and filters:
                if 'config' in idx_info and 'include_columns' in idx_info['config']:
                    filter_columns = set(filters.keys())
                    index_columns = set(idx_info['config']['include_columns'])
                    matching_columns = filter_columns.intersection(index_columns)
                    score += len(matching_columns) * 2
            
            if score > best_score:
                best_score = score
                best_index = idx_info
        
        # If no suitable index found, use default query plan
        if not best_index:
            return self._get_default_query_plan(
                table_name, time_column, start_time, end_time, filters, select_columns
            )
        
        # Create optimized query plan using the selected index
        query_plan = {
            "table": table_name,
            "time_column": time_column,
            "start_time": start_time,
            "end_time": end_time,
            "filters": filters or {},
            "select_columns": select_columns or ["*"],
            "index_name": best_index['index_name'],
            "index_type": best_index['index_type'],
            "use_index_hint": True
        }
        
        # Add specific optimizations based on index type
        if best_index['index_type'] == IndexType.HYPERTABLE.value:
            # For TimescaleDB, add time bucket function for aggregations
            precision = best_index.get('time_precision', TimePrecision.HOUR.value)
            time_bucket = self._get_time_bucket_from_precision(TimePrecision(precision))
            query_plan["timescaledb_options"] = {
                "time_bucket": time_bucket.value,
                "use_chunks_optimization": True
            }
        elif best_index['index_type'] == IndexType.BRIN.value:
            # BRIN performs best with sequential scans and simple predicates
            query_plan["query_options"] = {
                "use_sequential_scan": True,
                "simplify_predicates": True
            }
        
        return query_plan
    
    def generate_sql(self, query_plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate optimized SQL from a query plan
        
        Args:
            query_plan: Query plan dictionary
            
        Returns:
            Tuple of (sql_query, parameters)
        """
        # Extract plan components
        table = query_plan["table"]
        time_column = query_plan["time_column"]
        start_time = query_plan["start_time"]
        end_time = query_plan["end_time"]
        filters = query_plan.get("filters", {})
        select_columns = query_plan.get("select_columns", ["*"])
        
        # Build column list
        if "*" in select_columns:
            columns_sql = "*"
        else:
            columns_sql = ", ".join(select_columns)
        
        # Build WHERE clause
        where_clauses = [f"{time_column} >= $1", f"{time_column} <= $2"]
        params = [start_time, end_time]
        
        # Add additional filters
        param_idx = 3
        for column, value in filters.items():
            where_clauses.append(f"{column} = ${param_idx}")
            params.append(value)
            param_idx += 1
        
        where_sql = " AND ".join(where_clauses)
        
        # Check for TimescaleDB-specific optimizations
        if "timescaledb_options" in query_plan:
            # Use time_bucket for aggregation if needed
            if "group_by" in query_plan:
                time_bucket = query_plan["timescaledb_options"]["time_bucket"]
                columns_sql = f"time_bucket('{time_bucket}', {time_column}) as time_bucket, {columns_sql}"
                order_by = "time_bucket"
            else:
                order_by = time_column
        else:
            order_by = time_column
        
        # Generate the final query
        sql = f"""
            SELECT {columns_sql}
            FROM {self.schema_name}.{table}
            WHERE {where_sql}
            ORDER BY {order_by}
        """
        
        # Add index hint if needed and supported by database
        if query_plan.get("use_index_hint", False) and query_plan.get("index_name"):
            # PostgreSQL doesn't support index hints directly
            # In TimescaleDB, we'd use other optimization techniques
            pass
        
        return sql.strip(), params
    
    async def track_query_execution(
        self,
        query_plan: Dict[str, Any],
        execution_time_ms: float
    ):
        """
        Track query execution statistics
        
        Args:
            query_plan: The executed query plan
            execution_time_ms: Execution time in milliseconds
        """
        if not self._initialized:
            await self.initialize()
        
        query_hash = self._generate_query_hash(query_plan)
        
        # Update index usage statistics if an index was used
        if "index_name" in query_plan:
            table_name = query_plan["table"]
            index_name = query_plan["index_name"]
            
            if table_name in self.index_cache and index_name in self.index_cache[table_name]:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(f"""
                        UPDATE {self.schema_name}.{self.metadata_table}
                        SET 
                            usage_count = usage_count + 1,
                            last_used = NOW(),
                            avg_query_time = (avg_query_time * usage_count + $1) / (usage_count + 1)
                        WHERE table_name = $2 AND index_name = $3
                    """, execution_time_ms, table_name, index_name)
        
        # Track query pattern
        time_range_seconds = (query_plan["end_time"] - query_plan["start_time"]).total_seconds()
        
        # Extract query pattern (anonymize specific values, keep structure)
        pattern = {
            "table": query_plan["table"],
            "time_column": query_plan["time_column"],
            "time_range_seconds": time_range_seconds,
            "has_filters": bool(query_plan.get("filters")),
            "filter_columns": list(query_plan.get("filters", {}).keys()),
            "select_columns": query_plan.get("select_columns", ["*"])
        }
        
        pattern_json = json.dumps(pattern, sort_keys=True)
        
        # Store query statistics
        async with self.db_pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self.schema_name}.timeseries_query_stats
                (query_hash, query_pattern, execution_count, avg_execution_time, 
                 last_executed, time_range_seconds)
                VALUES ($1, $2, 1, $3, NOW(), $4)
                ON CONFLICT (query_hash) DO UPDATE SET
                    execution_count = timeseries_query_stats.execution_count + 1,
                    avg_execution_time = (timeseries_query_stats.avg_execution_time * 
                                         timeseries_query_stats.execution_count + $3) / 
                                         (timeseries_query_stats.execution_count + 1),
                    last_executed = NOW(),
                    time_range_seconds = $4
            """, query_hash, pattern_json, execution_time_ms, time_range_seconds)
    
    async def get_query_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated query statistics
        
        Returns:
            Dictionary with query execution statistics
        """
        if not self._initialized:
            await self.initialize()
        
        async with self.db_pool.acquire() as conn:
            # Get most frequent query patterns
            frequent_patterns = await conn.fetch(f"""
                SELECT 
                    query_pattern,
                    execution_count,
                    avg_execution_time,
                    time_range_seconds
                FROM {self.schema_name}.timeseries_query_stats
                ORDER BY execution_count DESC
                LIMIT 10
            """)
            
            # Get slowest query patterns
            slow_patterns = await conn.fetch(f"""
                SELECT 
                    query_pattern,
                    execution_count,
                    avg_execution_time,
                    time_range_seconds
                FROM {self.schema_name}.timeseries_query_stats
                WHERE execution_count > 5
                ORDER BY avg_execution_time DESC
                LIMIT 10
            """)
            
            # Get index usage statistics
            index_usage = await conn.fetch(f"""
                SELECT 
                    table_name,
                    index_name,
                    index_type,
                    usage_count,
                    avg_query_time,
                    last_used
                FROM {self.schema_name}.{self.metadata_table}
                ORDER BY usage_count DESC
            """)
        
        # Process and return statistics
        return {
            "frequent_queries": [dict(row) for row in frequent_patterns],
            "slow_queries": [dict(row) for row in slow_patterns],
            "index_usage": [dict(row) for row in index_usage]
        }
    
    async def analyze_time_patterns(self, table_name: str, time_column: str) -> Dict[str, Any]:
        """
        Analyze time series patterns in the data
        
        Args:
            table_name: Table to analyze
            time_column: Timestamp column
            
        Returns:
            Dictionary with time pattern analysis
        """
        if not self._initialized:
            await self.initialize()
        
        async with self.db_pool.acquire() as conn:
            # Time distribution analysis
            time_stats = await conn.fetchrow(f"""
                SELECT 
                    MIN({time_column}) as min_time,
                    MAX({time_column}) as max_time,
                    MAX({time_column}) - MIN({time_column}) as time_range,
                    COUNT(*) as total_points
                FROM {self.schema_name}.{table_name}
                WHERE {time_column} IS NOT NULL
            """)
            
            # Check for gaps in time series
            has_timescaledb = False
            try:
                tsdb_check = await conn.fetchval("""
                    SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'
                """)
                has_timescaledb = tsdb_check > 0
            except:
                pass
            
            if has_timescaledb:
                # Use TimescaleDB's time_bucket for gap analysis
                gap_analysis = await conn.fetch(f"""
                    WITH time_buckets AS (
                        SELECT time_bucket('1 hour', {time_column}) AS bucket,
                               COUNT(*) as point_count
                        FROM {self.schema_name}.{table_name}
                        WHERE {time_column} >= $1 AND {time_column} <= $2
                        GROUP BY 1
                        ORDER BY 1
                    ),
                    gaps AS (
                        SELECT 
                            bucket,
                            point_count,
                            bucket - LAG(bucket) OVER (ORDER BY bucket) as gap
                        FROM time_buckets
                    )
                    SELECT 
                        COUNT(*) as total_buckets,
                        COUNT(*) FILTER (WHERE point_count = 0) as empty_buckets,
                        COUNT(*) FILTER (WHERE gap > INTERVAL '1 hour') as gaps,
                        MAX(gap) as max_gap
                    FROM gaps
                """, time_stats['min_time'], time_stats['max_time'])
            else:
                # Simplified gap analysis for non-TimescaleDB
                # This is less efficient but provides basic analysis
                # For real time series, we strongly recommend TimescaleDB
                timestamp_rows = await conn.fetch(f"""
                    SELECT {time_column}
                    FROM {self.schema_name}.{table_name}
                    WHERE {time_column} IS NOT NULL
                    ORDER BY {time_column}
                    LIMIT 10000  -- Sampling for efficiency
                """)
                
                # Simple gap analysis from sampled data
                timestamps = [row[0] for row in timestamp_rows]
                gaps = []
                for i in range(1, len(timestamps)):
                    gap = (timestamps[i] - timestamps[i-1]).total_seconds()
                    gaps.append(gap)
                
                gap_analysis = [{
                    "total_samples": len(timestamps),
                    "avg_gap_seconds": sum(gaps) / len(gaps) if gaps else 0,
                    "max_gap_seconds": max(gaps) if gaps else 0,
                    "min_gap_seconds": min(gaps) if gaps else 0
                }]
        
        # Process and return time pattern analysis
        return {
            "table_name": table_name,
            "time_column": time_column,
            "time_range": {
                "start": time_stats['min_time'],
                "end": time_stats['max_time'],
                "duration": str(time_stats['time_range'])
            },
            "data_points": time_stats['total_points'],
            "gap_analysis": [dict(row) for row in gap_analysis],
            "has_timescaledb": has_timescaledb
        }
            
    def _get_default_query_plan(
        self,
        table_name: str,
        time_column: str,
        start_time: datetime,
        end_time: datetime,
        filters: Dict[str, Any] = None,
        select_columns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get a default query plan when no optimized index is available
        
        Args:
            Same as get_optimal_query_plan
            
        Returns:
            Default query plan
        """
        return {
            "table": table_name,
            "time_column": time_column,
            "start_time": start_time,
            "end_time": end_time,
            "filters": filters or {},
            "select_columns": select_columns or ["*"],
            "index_name": None,
            "index_type": None,
            "use_index_hint": False
        }
    
    def _generate_query_hash(self, query_plan: Dict[str, Any]) -> str:
        """
        Generate a stable hash for a query plan
        
        Args:
            query_plan: Query plan to hash
            
        Returns:
            Hash string
        """
        # Create a serializable version of the query plan
        hashable_plan = {
            "table": query_plan["table"],
            "time_column": query_plan["time_column"],
            "start_time_diff": (query_plan["end_time"] - query_plan["start_time"]).total_seconds(),
            "filters": sorted(query_plan.get("filters", {}).items()),
            "select_columns": sorted(query_plan.get("select_columns", ["*"]))
        }
        
        # Generate hash
        hash_input = json.dumps(hashable_plan, sort_keys=True)
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _analyze_query_filters(self, queries: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze common filter columns in sample queries
        
        Args:
            queries: List of query dictionaries
            
        Returns:
            Dictionary with column counts
        """
        filter_counts = {}
        
        for query in queries:
            if "filters" in query:
                for column in query["filters"].keys():
                    filter_counts[column] = filter_counts.get(column, 0) + 1
        
        return filter_counts
    
    def _get_time_bucket_from_precision(self, precision: TimePrecision) -> TimeBucket:
        """
        Map time precision to appropriate time bucket
        
        Args:
            precision: Time precision enum
            
        Returns:
            Corresponding time bucket enum
        """
        precision_to_bucket = {
            TimePrecision.MILLISECOND: TimeBucket.MINUTE,
            TimePrecision.SECOND: TimeBucket.MINUTE,
            TimePrecision.MINUTE: TimeBucket.HOUR,
            TimePrecision.HOUR: TimeBucket.DAY,
            TimePrecision.DAY: TimeBucket.WEEK,
            TimePrecision.WEEK: TimeBucket.MONTH,
            TimePrecision.MONTH: TimeBucket.QUARTER,
            TimePrecision.QUARTER: TimeBucket.QUARTER,
            TimePrecision.YEAR: TimeBucket.QUARTER
        }
        
        return precision_to_bucket.get(precision, TimeBucket.DAY)
