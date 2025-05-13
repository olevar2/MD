"""
Service for OHLCV data operations.
"""
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

from redis import Redis
from asyncpg.pool import Pool

import pandas as pd
from core_foundations.exceptions.base_exceptions import DataValidationError
from core_foundations.utils.logger import get_logger
from models.schemas import OHLCVData, PaginatedResponse, TimeFrame, TimeframeEnum
from repositories.ohlcv_repository import OHLCVRepository
from data_pipeline_service.validation import get_validation_engine

# Initialize logger
logger = get_logger("data-pipeline-service")


class OHLCVCache:
    """Cache layer for frequently accessed OHLCV data."""

    def __init__(self, redis_client: Redis, ttl_seconds: int = 3600):
    """
      init  .
    
    Args:
        redis_client: Description of redis_client
        ttl_seconds: Description of ttl_seconds
    
    """

        self.redis = redis_client
        self.ttl = ttl_seconds

    async def get_cached_data(
        self,
        instrument: str,
        timeframe: TimeframeEnum,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[List[OHLCVData]]:
        """Retrieve data from cache if available."""
        cache_key = self._build_cache_key(instrument, timeframe, start_time, end_time)
        cached = await self.redis.get(cache_key)

        if cached:
            return pickle.loads(cached)
        return None

    async def cache_data(
        self,
        instrument: str,
        timeframe: TimeframeEnum,
        start_time: datetime,
        end_time: datetime,
        data: List[OHLCVData]
    ):
        """Store OHLCV data in cache."""
        cache_key = self._build_cache_key(instrument, timeframe, start_time, end_time)
        serialized = pickle.dumps(data)
        await self.redis.set(cache_key, serialized, ex=self.ttl)

    def _build_cache_key(
        self,
        instrument: str,
        timeframe: TimeframeEnum,
        start_time: datetime,
        end_time: datetime
    ) -> str:
        """Build consistent cache key."""
        start_str = start_time.isoformat()
        end_str = end_time.isoformat()
        return f"ohlcv:{instrument}:{timeframe.value}:{start_str}:{end_str}"

    async def invalidate_instrument_cache(self, instrument: str):
        """
        Invalidate all cached data for an instrument.
        Useful when new data is loaded that might affect multiple timeframes.
        """
        pattern = f"ohlcv:{instrument}:*"
        keys = await self.redis.keys(pattern)

        if keys:
            await self.redis.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries for instrument {instrument}")


class OHLCVService:
    """
    Service for OHLCV data operations.
    Handles business logic and caching for OHLCV data.
    """

    def __init__(
        self,
        pool: Pool,
        redis_client: Optional[Redis] = None,
        cache_ttl: int = 3600
    ):
        """
        Initialize service with database pool and optional Redis client.

        Args:
            pool: Database connection pool
            redis_client: Optional Redis client for caching
            cache_ttl: Cache TTL in seconds
        """
        self.repository = OHLCVRepository(pool)
        self.cache = OHLCVCache(redis_client, cache_ttl) if redis_client else None

    async def get_historical_ohlcv(
        self,
        instrument: str,
        start_time: datetime,
        end_time: datetime,
        timeframe: TimeframeEnum,
        include_incomplete: bool = False
    ) -> List[OHLCVData]:
        """
        Get historical OHLCV data with caching.

        Args:
            instrument: Trading instrument identifier
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            timeframe: Candle timeframe
            include_incomplete: Whether to include incomplete candles

        Returns:
            List of OHLCV data points
        """
        # Try to get from cache if available
        cached_data = None
        if self.cache and not include_incomplete:  # Only cache complete candles
            cached_data = await self.cache.get_cached_data(
                instrument, timeframe, start_time, end_time
            )

        if cached_data:
            logger.debug(f"Cache hit for {instrument} {timeframe.value} data")
            return cached_data

        # Get from database if not in cache
        data = await self.repository.fetch_historical_ohlcv(
            instrument=instrument,
            start_time=start_time,
            end_time=end_time,
            timeframe=timeframe,
            include_incomplete=include_incomplete,
            use_optimized_pool=True  # Use optimized connection pool for better performance
        )

        # Cache the data if caching is enabled and we have data
        if self.cache and data and not include_incomplete:
            await self.cache.cache_data(
                instrument, timeframe, start_time, end_time, data
            )

        return data

    async def get_multi_instrument_ohlcv(
        self,
        instruments: List[str],
        start_time: datetime,
        end_time: datetime,
        timeframe: TimeframeEnum
    ) -> Dict[str, List[OHLCVData]]:
        """
        Get historical OHLCV data for multiple instruments.

        Args:
            instruments: List of trading instrument identifiers
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            timeframe: Candle timeframe

        Returns:
            Dictionary mapping instrument to list of OHLCV data points
        """
        if not instruments:
            return {}

        # Check if we have a small number of instruments
        if len(instruments) <= 3:
            # For a small number of instruments, use individual queries with caching
            results = {}
            for instrument in instruments:
                data = await self.get_historical_ohlcv(
                    instrument=instrument,
                    start_time=start_time,
                    end_time=end_time,
                    timeframe=timeframe
                )
                results[instrument] = data
            return results

        # For larger numbers of instruments, use the bulk fetch method
        # which is more efficient for multiple instruments
        return await self.repository.fetch_bulk_ohlcv(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            timeframe=timeframe
        )