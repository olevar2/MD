"""
Database Core Module.

Provides centralized database functionality for the Feature Store Service
using the common-lib database utilities.
"""
from typing import Dict, Any, Optional, AsyncGenerator
import os
from contextlib import asynccontextmanager
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
import asyncpg
from common_lib.db import create_async_db_engine, configure_session_maker, get_session as get_common_session
from core_foundations.utils.logger import get_logger
logger = get_logger('feature-store-service.db')
_engine: Optional[AsyncEngine] = None
Base = declarative_base()
DB_USER = os.getenv('FEATURE_STORE_DB_USER', 'postgres')
DB_PASSWORD = os.getenv('FEATURE_STORE_DB_PASSWORD', 'postgres')
DB_HOST = os.getenv('FEATURE_STORE_DB_HOST', 'localhost')
DB_PORT = os.getenv('FEATURE_STORE_DB_PORT', '5432')
DB_NAME = os.getenv('FEATURE_STORE_DB_NAME', 'forex_platform')
DB_URL = (
    f'postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )
if DB_PASSWORD == 'postgres':
    logger.warning(
        "Using default database password 'postgres'. Set FEATURE_STORE_DB_PASSWORD environment variable for production."
        )
DB_ECHO = os.getenv('FEATURE_STORE_DB_ECHO', 'false').lower() in ('true', '1')
RAW_DB_URL = DB_URL.replace('+asyncpg', '')


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@async_with_exception_handling
async def initialize_database() ->None:
    """
    Initialize the database engine and session factory.
    Should be called during application startup.
    """
    global _engine
    if _engine is not None:
        logger.warning('Database engine is already initialized')
        return
    try:
        logger.info(f'Initializing database engine with URL: {DB_URL}')
        _engine = create_async_db_engine(database_url=DB_URL, echo=DB_ECHO)
        configure_session_maker(_engine)
        async with _engine.begin() as conn:
            await conn.execute('SELECT 1')
        logger.info('Database engine initialized successfully')
    except Exception as e:
        logger.error(f'Failed to initialize database: {e}')
        raise


@async_with_exception_handling
async def dispose_database() ->None:
    """
    Dispose of the database engine.
    Should be called during application shutdown.
    """
    global _engine
    if _engine is None:
        logger.warning('No database engine to dispose')
        return
    try:
        await _engine.dispose()
        _engine = None
        logger.info('Database engine disposed')
    except Exception as e:
        logger.error(f'Error disposing database engine: {e}')
        raise


def get_engine() ->AsyncEngine:
    """
    Get the initialized database engine.

    Returns:
        AsyncEngine: Initialized SQLAlchemy async engine.

    Raises:
        RuntimeError: If the engine hasn't been initialized.
    """
    if _engine is None:
        raise RuntimeError(
            'Database engine is not initialized. Call initialize_database() first'
            )
    return _engine


@asynccontextmanager
async def get_db_session() ->AsyncGenerator[AsyncSession, None]:
    """
    Get a database session using the common-lib session manager.

    Yields:
        AsyncSession: Database session.
    """
    if _engine is None:
        raise RuntimeError(
            'Database engine is not initialized. Call initialize_database() first'
            )
    async with get_common_session() as session:
        yield session


@async_with_exception_handling
async def create_asyncpg_pool(**pool_options) ->asyncpg.Pool:
    """
    Create an asyncpg connection pool for direct access.
    Used for optimized TimescaleDB queries that bypass SQLAlchemy.

    Args:
        **pool_options: Options for the pool creation.

    Returns:
        asyncpg.Pool: Connection pool.
    """
    default_options = {'min_size': 5, 'max_size': 20}
    options = {**default_options, **pool_options}
    try:
        pool = await asyncpg.create_pool(RAW_DB_URL, **options)
        logger.info('Created asyncpg connection pool')
        return pool
    except Exception as e:
        logger.error(f'Failed to create asyncpg pool: {e}')
        raise


@with_exception_handling
def check_connection() ->bool:
    """
    Check if the database connection is working.
    This is a synchronous method suitable for health checks.

    Returns:
        bool: True if connection works, False otherwise.
    """
    try:
        from sqlalchemy import create_engine, text
        sync_engine = create_engine(DB_URL.replace('+asyncpg', ''))
        with sync_engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        return True
    except Exception as e:
        logger.error(f'Database connection check failed: {e}')
        return False
