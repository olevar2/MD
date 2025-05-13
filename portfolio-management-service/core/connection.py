"""
Database Connection Module (Refactored for common-lib async usage).

Provides functionality for connecting to the database using the centralized common-lib.
"""
import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from common_lib.db import create_async_db_engine, configure_session_maker, get_session as get_common_session, Base as CommonBase
from core_foundations.utils.logger import get_logger
logger = get_logger('portfolio-db-connection')
Base = declarative_base()
_engine: Optional[AsyncEngine] = None
DB_HOST = os.getenv('PORTFOLIO_DB_HOST', 'localhost')
DB_PORT = os.getenv('PORTFOLIO_DB_PORT', '5432')
DB_USER = os.getenv('PORTFOLIO_DB_USER', 'forex_user')
DB_PASSWORD = os.getenv('PORTFOLIO_DB_PASSWORD', 'forex_password')
DB_NAME = os.getenv('PORTFOLIO_DB_NAME', 'forex_db')
DATABASE_URL = (
    f'postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )
DB_ECHO = os.getenv('PORTFOLIO_DB_ECHO', 'False').lower() in ('true', '1', 't')


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@async_with_exception_handling
async def initialize_database():
    """
    Initializes the database engine and configures the session maker using common-lib.
    Should be called once during application startup.
    """
    global _engine
    if _engine is not None:
        logger.warning('Database already initialized.')
        return
    try:
        logger.info(
            f'Initializing database engine for {DB_HOST}:{DB_PORT}/{DB_NAME}')
        _engine = create_async_db_engine(database_url=DATABASE_URL, echo=
            DB_ECHO)
        configure_session_maker(_engine)
        logger.info('Database engine initialized and session maker configured.'
            )
    except Exception as e:
        logger.error(f'Failed to initialize database: {e}', exc_info=True)
        raise


async def dispose_database():
    """
    Disposes of the database engine connections.
    Should be called once during application shutdown.
    """
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info('Database engine disposed.')
    else:
        logger.warning('Database engine not initialized or already disposed.')


def get_engine() ->AsyncEngine:
    """
    Get the initialized database engine.

    Returns:
        AsyncEngine: SQLAlchemy async engine instance.

    Raises:
        RuntimeError: If the engine has not been initialized.
    """
    if _engine is None:
        raise RuntimeError(
            'Database engine has not been initialized. Call initialize_database() first.'
            )
    return _engine


@asynccontextmanager
async def get_db_session() ->AsyncGenerator[AsyncSession, None]:
    """
    Provides an asynchronous session context manager using the common session factory.

    Yields:
        An AsyncSession instance managed by the common_lib session context manager.

    Raises:
        RuntimeError: If the database is not initialized.
    """
    if _engine is None:
        raise RuntimeError(
            'Database not initialized. Call initialize_database() first.')
    async with get_common_session() as session:
        yield session
