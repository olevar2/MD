"""
Time Series Optimized Queries Module.

This module provides optimized query patterns for time-series data in TimescaleDB,
focusing on efficient timestamp-based access patterns for low-latency applications.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import asyncio
import asyncpg
import pandas as pd
from core_foundations.utils.logger import get_logger
from core_foundations.config.config_loader import ConfigLoader
from services.time_series_index_optimizer import TimeSeriesIndexManager, TimePrecision, IndexType
logger = get_logger('feature-store-service.timeseries-queries')


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
    Specialized optimizer for time series queries designed to work with TimescaleDB
    or other time series databases. Provides query optimizations for time range queries,
    aggregations, and downsampling operations.
    """

    def __init__(self, db_connection, config: Dict[str, Any]=None,
        enable_caching: bool=True, index_manager: Optional[
        TimeSeriesIndexManager]=None):
        """
        Initialize the time series query optimizer.

        Args:
            db_connection: The database connection to use
            config: Configuration options for the optimizer
            enable_caching: Whether to enable query plan caching
            index_manager: Optional index manager to use, will create one if not provided
        """
        self.db_connection = db_connection
        self.config = config or {}
        self.enable_caching = enable_caching
        self.query_plan_cache = {}
        self.index_manager = index_manager or TimeSeriesIndexManager(
            db_connection)
        self._index_manager_initialized = False
        self.chunk_time_interval = self.config.get('chunk_time_interval',
            timedelta(days=7))
        self.default_time_precision = TimePrecision(self.config.get(
            'default_time_precision', TimePrecision.MILLISECOND))
        self.default_index_type = IndexType(self.config.get(
            'default_index_type', IndexType.BTREE))
        self.query_stats = {'optimized_queries': 0, 'cache_hits': 0,
            'total_queries': 0, 'optimization_time': 0.0}

    async def initialize(self):
        """
        Asynchronously initialize the query optimizer and its dependencies.

        This method must be called before using any optimization features.
        It initializes the index manager and prepares the optimizer for use.
        """
        if not self._index_manager_initialized:
            await self.index_manager.initialize()
            self._index_manager_initialized = True
            logger.info('Time Series Index Manager initialized successfully')

    async def ensure_initialized(self):
        """
        Ensures that the optimizer is initialized before operation.

        This is a convenience method that can be called before operations
        that require initialization.
        """
        if not self._index_manager_initialized:
            await self.initialize()

    async def optimize_query(self, query: str, params: Dict[str, Any]=None,
        start_time: Optional[datetime]=None, end_time: Optional[datetime]=
        None, table_name: Optional[str]=None, time_column: str='timestamp',
        time_precision: Optional[TimePrecision]=None) ->Tuple[str, Dict[str,
        Any]]:
        """
        Optimize a time series query using the index manager and caching strategies.

        Args:
            query: The SQL query string to optimize
            params: Query parameters
            start_time: Start time for the time range
            end_time: End time for the time range
            table_name: Name of the primary time series table
            time_column: Name of the timestamp column
            time_precision: Time precision to use for optimization

        Returns:
            Tuple of (optimized_query, updated_params)
        """
        import time
        start = time.time()
        await self.ensure_initialized()
        self.query_stats['total_queries'] += 1
        cache_key = None
        if self.enable_caching and params:
            cache_key = (
                f'{query}:{start_time}:{end_time}:{table_name}:{time_precision}'
                )
            if cache_key in self.query_plan_cache:
                self.query_stats['cache_hits'] += 1
                optimized_query, updated_params = self.query_plan_cache[
                    cache_key]
                return optimized_query, updated_params
        params = params or {}
        time_precision = time_precision or self.default_time_precision
        if start_time and end_time and table_name:
            best_index = await self.index_manager.get_optimal_index(table_name,
                time_column, start_time, end_time, time_precision)
            optimized_query, updated_params = (await self.index_manager.
                optimize_query_with_index(query, params, best_index,
                time_column, start_time, end_time))
        else:
            optimized_query = query
            updated_params = params.copy()
        self.query_stats['optimized_queries'] += 1
        self.query_stats['optimization_time'] += time.time() - start
        if self.enable_caching and cache_key:
            self.query_plan_cache[cache_key] = optimized_query, updated_params
        return optimized_query, updated_params

    @async_with_exception_handling
    async def execute_optimized_query(self, query: str, params: Dict[str,
        Any]=None, start_time: Optional[datetime]=None, end_time: Optional[
        datetime]=None, table_name: Optional[str]=None, time_column: str=
        'timestamp', as_dataframe: bool=False) ->Union[List[Dict[str, Any]],
        pd.DataFrame]:
        """
        Optimize and execute a time series query, returning the results.

        Args:
            query: The SQL query string to optimize and execute
            params: Query parameters
            start_time: Start time for the time range
            end_time: End time for the time range
            table_name: Name of the primary time series table
            time_column: Name of the timestamp column
            as_dataframe: Whether to return the results as a pandas DataFrame

        Returns:
            Query results as a list of dictionaries or as a pandas DataFrame
        """
        optimized_query, updated_params = await self.optimize_query(query,
            params, start_time, end_time, table_name, time_column)
        try:
            async with self.db_connection.acquire() as conn:
                results = await conn.fetch(optimized_query, *updated_params
                    .values())
            if as_dataframe:
                import pandas as pd
                return pd.DataFrame([dict(row) for row in results])
            else:
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f'Error executing optimized query: {str(e)}')
            raise
