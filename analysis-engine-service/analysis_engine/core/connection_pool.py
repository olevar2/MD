"""
Connection Pool Manager

This module provides connection pooling functionality for database and Redis connections.
"""
import asyncio
from typing import Dict, Optional, Any
import aioredis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool
from prometheus_client import Counter, Gauge, Histogram
from analysis_engine.config import get_settings
from analysis_engine.core.logging import get_logger
logger = get_logger(__name__)
DB_CONNECTION_GAUGE = Gauge('db_connections_active',
    'Number of active database connections', ['type'])
DB_CONNECTION_TIME = Histogram('db_connection_time_seconds',
    'Time spent waiting for database connections', ['type'])
REDIS_CONNECTION_GAUGE = Gauge('redis_connections_active',
    'Number of active Redis connections')
REDIS_CONNECTION_TIME = Histogram('redis_connection_time_seconds',
    'Time spent waiting for Redis connections')
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ConnectionPoolManager:
    """Manages database and Redis connection pools."""

    def __init__(self):
        """Initialize the connection pool manager."""
        self._settings = get_settings()
        self._db_engine: Optional[AsyncEngine] = None
        self._redis_pool: Optional[aioredis.Redis] = None
        self._db_session_factory = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @async_with_exception_handling
    async def initialize(self) ->None:
        """Initialize connection pools."""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            try:
                self._db_engine = create_async_engine(self._settings.
                    database_url, poolclass=AsyncAdaptedQueuePool,
                    pool_size=20, max_overflow=10, pool_timeout=30,
                    pool_recycle=1800, echo=False)
                self._db_session_factory = sessionmaker(self._db_engine,
                    class_=AsyncSession, expire_on_commit=False)
                self._redis_pool = await aioredis.from_url(self._settings.
                    redis_url, max_connections=50, encoding='utf-8',
                    decode_responses=True)
                self._initialized = True
                logger.info('Connection pools initialized successfully')
            except Exception as e:
                logger.error(f'Failed to initialize connection pools: {e}')
                raise

    @with_database_resilience('get_db_session')
    async def get_db_session(self) ->AsyncSession:
        """
        Get a database session from the pool.

        Returns:
            AsyncSession: Database session
        """
        if not self._initialized:
            await self.initialize()
        with DB_CONNECTION_TIME.labels(type='database').time():
            session = self._db_session_factory()
            DB_CONNECTION_GAUGE.labels(type='database').inc()
            return session

    @with_resilience('get_redis_connection')
    async def get_redis_connection(self) ->aioredis.Redis:
        """
        Get a Redis connection from the pool.

        Returns:
            aioredis.Redis: Redis connection
        """
        if not self._initialized:
            await self.initialize()
        with REDIS_CONNECTION_TIME.time():
            REDIS_CONNECTION_GAUGE.inc()
            return self._redis_pool

    @async_with_exception_handling
    async def cleanup(self) ->None:
        """Clean up connection pools."""
        if not self._initialized:
            return
        try:
            if self._db_engine:
                await self._db_engine.dispose()
            if self._redis_pool:
                await self._redis_pool.close()
            self._initialized = False
            logger.info('Connection pools cleaned up successfully')
        except Exception as e:
            logger.error(f'Error cleaning up connection pools: {e}')
            raise


_pool_manager: Optional[ConnectionPoolManager] = None


async def get_pool_manager() ->ConnectionPoolManager:
    """
    Get the global connection pool manager instance.

    Returns:
        ConnectionPoolManager: The global connection pool manager instance
    """
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
        await _pool_manager.initialize()
    return _pool_manager
