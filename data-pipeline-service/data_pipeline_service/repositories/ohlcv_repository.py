"""
Repository for OHLCV (Open, High, Low, Close, Volume) data.

This module provides database operations for storing, retrieving,
and managing OHLCV candle data.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

import asyncpg
from asyncpg.pool import Pool

from ..models.schemas import OHLCVData, TimeframeEnum
from data_pipeline_service.monitoring import track_query_performance
from data_pipeline_service.optimization.connection_pool import get_optimized_asyncpg_connection

logger = logging.getLogger(__name__)


class OHLCVRepository:
    """
    Repository class for OHLCV data operations.
    Handles database operations for historical price data.
    """

    def __init__(self, pool: Pool):
        """Initialize with database connection pool."""
        self.pool = pool

    @track_query_performance(query_type="select", table="ohlcv")
    async def fetch_historical_ohlcv(
        self,
        instrument: str,
        start_time: datetime,
        end_time: datetime,
        timeframe: TimeframeEnum,
        include_incomplete: bool = False,
        use_optimized_pool: bool = False
    ) -> List[OHLCVData]:
        """
        Fetch historical OHLCV data for specified instrument and timeframe.

        Args:
            instrument: Trading instrument identifier
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            timeframe: Candle timeframe
            include_incomplete: Whether to include incomplete candles
            use_optimized_pool: Whether to use the optimized connection pool for better performance

        Returns:
            List of OHLCV data points
        """
        try:
            # Determine appropriate table/hypertable based on timeframe
            table_name = self._get_table_for_timeframe(timeframe)

            # For frequent timeframes, check if continuous aggregates exist
            continuous_aggregate = self._get_continuous_aggregate(timeframe)
            if continuous_aggregate and (end_time - start_time).days > 7:
                # Use continuous aggregate for long time periods
                query_table = continuous_aggregate
            else:
                query_table = table_name

            # Build query with proper time bucket alignment
            query = f"""
            SELECT
                time_bucket('{timeframe.value}'::interval, timestamp) AS time,
                instrument,
                first(open, timestamp) AS open,
                max(high) AS high,
                min(low) AS low,
                last(close, timestamp) AS close,
                sum(volume) AS volume
            FROM {query_table}
            WHERE
                instrument = $1
                AND timestamp >= $2
                AND timestamp < $3
                {'' if include_incomplete else "AND is_complete = TRUE"}
            GROUP BY time, instrument
            ORDER BY time ASC
            """

            # Optimize the query
            from data_pipeline_service.optimization import optimize_query
            optimized_query, _ = optimize_query(query)

            # Execute query with parameterized inputs
            if use_optimized_pool:
                # Use optimized connection pool for better performance
                async with get_optimized_asyncpg_connection() as conn:
                    result = await conn.fetch(
                        optimized_query,
                        instrument,
                        start_time,
                        end_time
                    )
            else:
                # Use regular connection pool
                result = await self.pool.fetch(
                    optimized_query,
                    instrument,
                    start_time,
                    end_time
                )

            return [OHLCVData.from_record(record) for record in result]

        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise

    def _get_table_for_timeframe(self, timeframe: TimeframeEnum) -> str:
        """
        Get the appropriate database table for a timeframe.

        Args:
            timeframe: The timeframe to get table for

        Returns:
            Table name for the timeframe
        """
        # Map timeframes to tables
        # In a real system, this might be configured elsewhere
        timeframe_tables = {
            TimeframeEnum.ONE_MINUTE: "ohlcv_1m",
            TimeframeEnum.FIVE_MINUTES: "ohlcv_5m",
            TimeframeEnum.FIFTEEN_MINUTES: "ohlcv_15m",
            TimeframeEnum.THIRTY_MINUTES: "ohlcv_30m",
            TimeframeEnum.ONE_HOUR: "ohlcv_1h",
            TimeframeEnum.FOUR_HOURS: "ohlcv_4h",
            TimeframeEnum.ONE_DAY: "ohlcv_1d",
            TimeframeEnum.ONE_WEEK: "ohlcv_1w",
        }

        return timeframe_tables.get(timeframe, "ohlcv_1m")

    def _get_continuous_aggregate(self, timeframe: TimeframeEnum) -> Optional[str]:
        """
        Get the continuous aggregate view for a timeframe if it exists.

        Args:
            timeframe: The timeframe to get continuous aggregate for

        Returns:
            Continuous aggregate view name or None if it doesn't exist
        """
        # Map timeframes to continuous aggregates
        # Only certain timeframes may have continuous aggregates
        continuous_aggregates = {
            TimeframeEnum.ONE_HOUR: "ohlcv_1h_agg",
            TimeframeEnum.FOUR_HOURS: "ohlcv_4h_agg",
            TimeframeEnum.ONE_DAY: "ohlcv_1d_agg",
        }

        return continuous_aggregates.get(timeframe)

    @track_query_performance(query_type="insert", table="ohlcv")
    async def insert_ohlcv_data(self, data: List[OHLCVData], timeframe: TimeframeEnum) -> int:
        """
        Insert OHLCV data into the database.

        Args:
            data: List of OHLCV data points to insert
            timeframe: Timeframe of the data

        Returns:
            Number of records inserted
        """
        if not data:
            return 0

        table = self._get_table_for_timeframe(timeframe)

        try:
            # Prepare values for bulk insert
            values = []
            for point in data:
                values.append((
                    point.timestamp,
                    point.instrument,
                    point.open,
                    point.high,
                    point.low,
                    point.close,
                    point.volume,
                    True  # is_complete
                ))

            # Prepare statement
            stmt = f"""
            INSERT INTO {table} (
                timestamp, instrument, open, high, low, close, volume, is_complete
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (timestamp, instrument) DO UPDATE
            SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                is_complete = EXCLUDED.is_complete,
                updated_at = NOW()
            """

            # Execute batch insert
            async with self.pool.acquire() as conn:
                result = await conn.executemany(stmt, values)

            # Return count of affected rows
            return len(values)

        except Exception as e:
            logger.error(f"Error inserting OHLCV data: {e}")
            raise

    @track_query_performance(query_type="select", table="ohlcv")
    async def fetch_bulk_ohlcv(
        self,
        instruments: List[str],
        start_time: datetime,
        end_time: datetime,
        timeframe: TimeframeEnum,
        include_incomplete: bool = False
    ) -> Dict[str, List[OHLCVData]]:
        """
        Fetch historical OHLCV data for multiple instruments in a single optimized query.

        This method uses the optimized connection pool for better performance.

        Args:
            instruments: List of trading instrument identifiers
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            timeframe: Candle timeframe
            include_incomplete: Whether to include incomplete candles

        Returns:
            Dictionary mapping instrument to list of OHLCV data points
        """
        if not instruments:
            return {}

        try:
            # Determine appropriate table/hypertable based on timeframe
            table_name = self._get_table_for_timeframe(timeframe)

            # For frequent timeframes, check if continuous aggregates exist
            continuous_aggregate = self._get_continuous_aggregate(timeframe)
            if continuous_aggregate and (end_time - start_time).days > 7:
                # Use continuous aggregate for long time periods
                query_table = continuous_aggregate
            else:
                query_table = table_name

            # Build query with proper time bucket alignment
            query = f"""
            SELECT
                time_bucket('{timeframe.value}'::interval, timestamp) AS time,
                instrument,
                first(open, timestamp) AS open,
                max(high) AS high,
                min(low) AS low,
                last(close, timestamp) AS close,
                sum(volume) AS volume
            FROM {query_table}
            WHERE
                instrument = ANY($1)
                AND timestamp >= $2
                AND timestamp < $3
                {'' if include_incomplete else "AND is_complete = TRUE"}
            GROUP BY time, instrument
            ORDER BY instrument, time ASC
            """

            # Optimize the query
            from data_pipeline_service.optimization import optimize_query
            optimized_query, _ = optimize_query(query)

            # Use optimized connection pool for better performance
            async with get_optimized_asyncpg_connection() as conn:
                result = await conn.fetch(
                    optimized_query,
                    instruments,
                    start_time,
                    end_time
                )

            # Group results by instrument
            grouped_results: Dict[str, List[OHLCVData]] = {}
            for record in result:
                instrument = record["instrument"]
                if instrument not in grouped_results:
                    grouped_results[instrument] = []

                grouped_results[instrument].append(OHLCVData.from_record(record))

            return grouped_results

        except Exception as e:
            logger.error(f"Error fetching bulk OHLCV data: {e}")
            raise