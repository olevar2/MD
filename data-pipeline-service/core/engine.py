"""
Database engine and session management using common_lib.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from common_lib.db import create_async_db_engine, configure_session_maker, get_session as get_common_session, init_db as common_init_db, Base as CommonBase
from core_foundations.utils.logger import get_logger
from config.settings import get_settings
logger = get_logger('data-pipeline-service')
_engine: Optional[AsyncEngine] = None


from data_pipeline_service.error.exceptions_bridge import (
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
    Initializes the database engine and configures the session maker.
    Should be called once during application startup.
    """
    global _engine
    if _engine is not None:
        logger.warning('Database already initialized.')
        return
    settings = get_settings()
    try:
        _engine = create_async_db_engine(database_url=settings.DATABASE_URL,
            echo=settings.debug)
        configure_session_maker(_engine)
        logger.info(
            f'Database engine initialized and session maker configured for {settings.db_host}:{settings.db_port}/{settings.db_name}'
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
    """
    async with get_common_session() as session:
        yield session
