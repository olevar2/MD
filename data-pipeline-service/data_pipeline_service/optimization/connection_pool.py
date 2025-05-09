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

from data_pipeline_service.config.settings import get_settings

logger = logging.getLogger(__name__)


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
    
    async def initialize(self):
        """Initialize the connection pools."""
        async with self._lock:
            if self._initialized:
                return
            
            try:
                # Create SQLAlchemy engine with optimized settings
                self._sa_engine = await self._create_sqlalchemy_engine()
                
                # Create asyncpg pool for direct access
                self._asyncpg_pool = await self._create_asyncpg_pool()
                
                self._initialized = True
                logger.info("Optimized connection pools initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize optimized connection pools: {e}")
                raise
    
    async def _create_sqlalchemy_engine(self) -> AsyncEngine:
        """
        Create an optimized SQLAlchemy engine.
        
        Returns:
            AsyncEngine: SQLAlchemy async engine
        """
        # Determine optimal pool size based on available CPU cores
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        # Calculate optimal pool size: 2 * CPU cores + 1
        # This is a common formula for I/O-bound workloads
        optimal_pool_size = min(2 * cpu_count + 1, 20)  # Cap at 20 to avoid too many connections
        
        # Use the calculated pool size or the configured one, whichever is smaller
        pool_size = min(optimal_pool_size, self.settings.db_pool_size or 10)
        
        # Calculate max overflow based on pool size
        max_overflow = min(pool_size * 2, 30)  # Cap at 30 to avoid too many connections
        
        # Create engine with optimized settings
        engine = create_async_engine(
            self.settings.DATABASE_URL,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=10,  # Shorter timeout to fail fast
            pool_recycle=1800,  # Recycle connections every 30 minutes
            pool_pre_ping=True,  # Check connection health before use
            echo=self.settings.debug,
            # TimescaleDB specific optimizations
            connect_args={
                "statement_cache_size": 0,  # Disable statement caching for pgbouncer compatibility
                "prepared_statement_cache_size": 256,  # Limit prepared statement cache size
                "server_settings": {
                    "application_name": "data_pipeline_service",
                    "statement_timeout": "60000",  # 60 seconds timeout for statements
                    "idle_in_transaction_session_timeout": "300000",  # 5 minutes timeout for idle transactions
                    "effective_io_concurrency": "8",  # Optimize for SSD
                    "work_mem": "64MB",  # Memory for sorting operations
                    "maintenance_work_mem": "128MB",  # Memory for maintenance operations
                }
            }
        )
        
        logger.info(f"Created SQLAlchemy engine with pool_size={pool_size}, max_overflow={max_overflow}")
        return engine
    
    async def _create_asyncpg_pool(self) -> asyncpg.Pool:
        """
        Create an optimized asyncpg pool for direct access.
        
        Returns:
            asyncpg.Pool: asyncpg connection pool
        """
        # Parse database URL to extract components
        from urllib.parse import urlparse
        
        # Extract database connection parameters from URL
        url = urlparse(self.settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://"))
        user_pass, host_port = url.netloc.split("@")
        user, password = user_pass.split(":")
        host, port = host_port.split(":")
        database = url.path.lstrip("/")
        
        # Determine optimal pool size
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        optimal_pool_size = min(2 * cpu_count + 1, 20)
        
        # Create asyncpg pool with optimized settings
        pool = await asyncpg.create_pool(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            min_size=2,  # Minimum connections to keep in the pool
            max_size=optimal_pool_size,  # Maximum connections in the pool
            command_timeout=60.0,  # Command timeout in seconds
            max_inactive_connection_lifetime=1800.0,  # 30 minutes
            setup=self._setup_asyncpg_connection,  # Setup function for each connection
        )
        
        logger.info(f"Created asyncpg pool with min_size=2, max_size={optimal_pool_size}")
        return pool
    
    async def _setup_asyncpg_connection(self, connection: asyncpg.Connection):
        """
        Set up an asyncpg connection with optimized settings.
        
        Args:
            connection: asyncpg connection
        """
        # Set application name
        await connection.execute("SET application_name = 'data_pipeline_service'")
        
        # Set statement timeout
        await connection.execute("SET statement_timeout = '60000'")  # 60 seconds
        
        # Set idle in transaction timeout
        await connection.execute("SET idle_in_transaction_session_timeout = '300000'")  # 5 minutes
        
        # Set TimescaleDB specific settings
        await connection.execute("SET effective_io_concurrency = '8'")  # Optimize for SSD
        await connection.execute("SET work_mem = '64MB'")  # Memory for sorting operations
        await connection.execute("SET maintenance_work_mem = '128MB'")  # Memory for maintenance operations
    
    @asynccontextmanager
    async def get_sa_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a SQLAlchemy session from the optimized pool.
        
        Yields:
            AsyncSession: SQLAlchemy async session
        """
        if not self._initialized:
            await self.initialize()
        
        from sqlalchemy.ext.asyncio import async_sessionmaker
        
        # Create session factory
        session_factory = async_sessionmaker(
            bind=self._sa_engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
        
        # Create session
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
    async def get_asyncpg_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Get an asyncpg connection from the optimized pool.
        
        Yields:
            asyncpg.Connection: asyncpg connection
        """
        if not self._initialized:
            await self.initialize()
        
        # Get connection from pool
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
        logger.info("Optimized connection pools closed")


# Singleton instance
optimized_pool = OptimizedConnectionPool()


@asynccontextmanager
async def get_optimized_sa_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a SQLAlchemy session from the optimized pool.
    
    Yields:
        AsyncSession: SQLAlchemy async session
    """
    async with optimized_pool.get_sa_session() as session:
        yield session


@asynccontextmanager
async def get_optimized_asyncpg_connection() -> AsyncGenerator[asyncpg.Connection, None]:
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
