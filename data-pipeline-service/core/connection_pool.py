"""
Optimized Connection Pool Module.

This module provides an optimized connection pool for database access,
particularly for time series data in TimescaleDB.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import asyncpg
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import QueuePool, NullPool
from config.settings import get_settings
logger = logging.getLogger(__name__)


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class OptimizedConnectionPool:
    """
    Optimized connection pool for database access.
    
    This class provides optimized connection pools for different types of database access,
    including SQLAlchemy and asyncpg direct access.
    """

    def __init__(self):
        """Initialize the optimized connection pool."""
        self.settings = get_settings()
        self._sa_engine: Optional[AsyncEngine] = None
        self._asyncpg_pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @async_with_exception_handling
    async def initialize(self):
        """Initialize the connection pools."""
        async with self._lock:
            if self._initialized:
                return
            try:
                self._sa_engine = await self._create_sqlalchemy_engine()
                self._asyncpg_pool = await self._create_asyncpg_pool()
                self._initialized = True
                logger.info(
                    'Optimized connection pools initialized successfully')
            except Exception as e:
                logger.error(
                    f'Failed to initialize optimized connection pools: {e}')
                raise

    async def _create_sqlalchemy_engine(self) ->AsyncEngine:
        """
        Create an optimized SQLAlchemy engine.
        
        Returns:
            AsyncEngine: SQLAlchemy async engine
        """
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        optimal_pool_size = min(2 * cpu_count + 1, 20)
        pool_size = min(optimal_pool_size, self.settings.db_pool_size or 10)
        max_overflow = min(pool_size * 2, 30)
        engine = create_async_engine(self.settings.DATABASE_URL, pool_size=
            pool_size, max_overflow=max_overflow, pool_timeout=10,
            pool_recycle=1800, pool_pre_ping=True, echo=self.settings.debug,
            connect_args={'statement_cache_size': 0,
            'prepared_statement_cache_size': 256, 'server_settings': {
            'application_name': 'data_pipeline_service',
            'statement_timeout': '60000',
            'idle_in_transaction_session_timeout': '300000',
            'effective_io_concurrency': '8', 'work_mem': '64MB',
            'maintenance_work_mem': '128MB'}})
        logger.info(
            f'Created SQLAlchemy engine with pool_size={pool_size}, max_overflow={max_overflow}'
            )
        return engine

    async def _create_asyncpg_pool(self) ->asyncpg.Pool:
        """
        Create an optimized asyncpg pool for direct access.
        
        Returns:
            asyncpg.Pool: asyncpg connection pool
        """
        from urllib.parse import urlparse
        url = urlparse(self.settings.DATABASE_URL.replace(
            'postgresql+asyncpg://', 'postgresql://'))
        user_pass, host_port = url.netloc.split('@')
        user, password = user_pass.split(':')
        host, port = host_port.split(':')
        database = url.path.lstrip('/')
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        optimal_pool_size = min(2 * cpu_count + 1, 20)
        pool = await asyncpg.create_pool(host=host, port=port, user=user,
            password=password, database=database, min_size=2, max_size=
            optimal_pool_size, command_timeout=60.0,
            max_inactive_connection_lifetime=1800.0, setup=self.
            _setup_asyncpg_connection)
        logger.info(
            f'Created asyncpg pool with min_size=2, max_size={optimal_pool_size}'
            )
        return pool

    async def _setup_asyncpg_connection(self, connection: asyncpg.Connection):
        """
        Set up an asyncpg connection with optimized settings.
        
        Args:
            connection: asyncpg connection
        """
        await connection.execute(
            "SET application_name = 'data_pipeline_service'")
        await connection.execute("SET statement_timeout = '60000'")
        await connection.execute(
            "SET idle_in_transaction_session_timeout = '300000'")
        await connection.execute("SET effective_io_concurrency = '8'")
        await connection.execute("SET work_mem = '64MB'")
        await connection.execute("SET maintenance_work_mem = '128MB'")

    @asynccontextmanager
    @async_with_exception_handling
    async def get_sa_session(self) ->AsyncGenerator[AsyncSession, None]:
        """
        Get a SQLAlchemy session from the optimized pool.
        
        Yields:
            AsyncSession: SQLAlchemy async session
        """
        if not self._initialized:
            await self.initialize()
        from sqlalchemy.ext.asyncio import async_sessionmaker
        session_factory = async_sessionmaker(bind=self._sa_engine,
            expire_on_commit=False, class_=AsyncSession)
        session = session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    @async_with_exception_handling
    async def get_asyncpg_connection(self) ->AsyncGenerator[asyncpg.
        Connection, None]:
        """
        Get an asyncpg connection from the optimized pool.
        
        Yields:
            asyncpg.Connection: asyncpg connection
        """
        if not self._initialized:
            await self.initialize()
        connection = await self._asyncpg_pool.acquire()
        try:
            yield connection
        finally:
            await self._asyncpg_pool.release(connection)

    async def close(self):
        """Close all connection pools."""
        if self._sa_engine:
            await self._sa_engine.dispose()
            self._sa_engine = None
        if self._asyncpg_pool:
            await self._asyncpg_pool.close()
            self._asyncpg_pool = None
        self._initialized = False
        logger.info('Optimized connection pools closed')


optimized_pool = OptimizedConnectionPool()


@asynccontextmanager
async def get_optimized_sa_session() ->AsyncGenerator[AsyncSession, None]:
    """
    Get a SQLAlchemy session from the optimized pool.
    
    Yields:
        AsyncSession: SQLAlchemy async session
    """
    async with optimized_pool.get_sa_session() as session:
        yield session


@asynccontextmanager
async def get_optimized_asyncpg_connection() ->AsyncGenerator[asyncpg.
    Connection, None]:
    """
    Get an asyncpg connection from the optimized pool.
    
    Yields:
        asyncpg.Connection: asyncpg connection
    """
    async with optimized_pool.get_asyncpg_connection() as connection:
        yield connection


async def initialize_optimized_pool():
    """Initialize the optimized connection pool."""
    await optimized_pool.initialize()


async def close_optimized_pool():
    """Close the optimized connection pool."""
    await optimized_pool.close()
