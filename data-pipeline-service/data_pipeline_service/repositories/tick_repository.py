"""
Repository for tick data.

This module provides database operations for storing, retrieving,
and managing tick-level market data.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import asyncpg
from asyncpg.pool import Pool
from ..models.schemas import TickData
from data_pipeline_service.monitoring import track_query_performance
from data_pipeline_service.optimization.connection_pool import get_optimized_asyncpg_connection
logger = logging.getLogger(__name__)


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TickRepository:
    """
    Repository class for tick data operations.
    Handles database operations for tick-level market data.
    """

    def __init__(self, pool: Pool):
        """Initialize with database connection pool."""
        self.pool = pool

    @track_query_performance(query_type='select', table='ticks')
    @async_with_exception_handling
    async def fetch_tick_data(self, instrument: str, start_time: datetime,
        end_time: datetime, use_optimized_pool: bool=False) ->List[TickData]:
        """
        Fetch tick data for specified instrument and time range.

        Args:
            instrument: Trading instrument identifier
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            use_optimized_pool: Whether to use the optimized connection pool for better performance

        Returns:
            List of tick data points
        """
        try:
            query = """
            SELECT
                timestamp,
                instrument,
                bid,
                ask,
                bid_volume,
                ask_volume
            FROM ticks
            WHERE
                instrument = $1
                AND timestamp >= $2
                AND timestamp < $3
            ORDER BY timestamp ASC
            """
            from data_pipeline_service.optimization import optimize_query
            optimized_query, _ = optimize_query(query)
            if use_optimized_pool:
                async with get_optimized_asyncpg_connection() as conn:
                    result = await conn.fetch(optimized_query, instrument,
                        start_time, end_time)
            else:
                result = await self.pool.fetch(optimized_query, instrument,
                    start_time, end_time)
            return [TickData.from_record(record) for record in result]
        except Exception as e:
            logger.error(f'Error fetching tick data: {e}')
            raise

    @track_query_performance(query_type='insert', table='ticks')
    @async_with_exception_handling
    async def insert_tick_data(self, data: List[TickData]) ->int:
        """
        Insert tick data into the database.

        Args:
            data: List of tick data points to insert

        Returns:
            Number of records inserted
        """
        if not data:
            return 0
        try:
            values = []
            for point in data:
                values.append((point.timestamp, point.instrument, point.bid,
                    point.ask, point.bid_volume, point.ask_volume))
            stmt = """
            INSERT INTO ticks (
                timestamp, instrument, bid, ask, bid_volume, ask_volume
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (timestamp, instrument) DO UPDATE
            SET
                bid = EXCLUDED.bid,
                ask = EXCLUDED.ask,
                bid_volume = EXCLUDED.bid_volume,
                ask_volume = EXCLUDED.ask_volume,
                updated_at = NOW()
            """
            async with self.pool.acquire() as conn:
                result = await conn.executemany(stmt, values)
            return len(values)
        except Exception as e:
            logger.error(f'Error inserting tick data: {e}')
            raise

    @track_query_performance(query_type='select', table='ticks')
    @async_with_exception_handling
    async def fetch_bulk_tick_data(self, instruments: List[str], start_time:
        datetime, end_time: datetime) ->Dict[str, List[TickData]]:
        """
        Fetch tick data for multiple instruments in a single optimized query.

        This method uses the optimized connection pool for better performance.

        Args:
            instruments: List of trading instrument identifiers
            start_time: Start time for data retrieval
            end_time: End time for data retrieval

        Returns:
            Dictionary mapping instrument to list of tick data points
        """
        if not instruments:
            return {}
        try:
            query = """
            SELECT
                timestamp,
                instrument,
                bid,
                ask,
                bid_volume,
                ask_volume
            FROM ticks
            WHERE
                instrument = ANY($1)
                AND timestamp >= $2
                AND timestamp < $3
            ORDER BY instrument, timestamp ASC
            """
            from data_pipeline_service.optimization import optimize_query
            optimized_query, _ = optimize_query(query)
            async with get_optimized_asyncpg_connection() as conn:
                result = await conn.fetch(optimized_query, instruments,
                    start_time, end_time)
            grouped_results: Dict[str, List[TickData]] = {}
            for record in result:
                instrument = record['instrument']
                if instrument not in grouped_results:
                    grouped_results[instrument] = []
                grouped_results[instrument].append(TickData.from_record(record)
                    )
            return grouped_results
        except Exception as e:
            logger.error(f'Error fetching bulk tick data: {e}')
            raise

    @track_query_performance(query_type='select', table='ticks')
    @async_with_exception_handling
    async def get_latest_tick(self, instrument: str) ->Optional[TickData]:
        """
        Get the latest tick for a specific instrument.

        Args:
            instrument: Trading instrument identifier

        Returns:
            Latest tick data or None if not found
        """
        try:
            query = """
            SELECT
                timestamp,
                instrument,
                bid,
                ask,
                bid_volume,
                ask_volume
            FROM ticks
            WHERE
                instrument = $1
            ORDER BY timestamp DESC
            LIMIT 1
            """
            async with self.pool.acquire() as conn:
                record = await conn.fetchrow(query, instrument)
            if record:
                return TickData.from_record(record)
            return None
        except Exception as e:
            logger.error(f'Error fetching latest tick: {e}')
            raise
