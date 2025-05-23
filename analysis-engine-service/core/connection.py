"""
Database Connection Module (Refactored for common-lib async usage).

Provides functionality for connecting to the database using the centralized common-lib.
"""
import os
import logging
from typing import Optional, AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from common_lib.db import create_async_db_engine, configure_session_maker, get_session as get_common_session, Base as CommonBase
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DB_HOST = os.environ.get('ANALYSIS_ENGINE_DB_HOST', os.environ.get(
    'DB_HOST', 'localhost'))
DB_PORT = os.environ.get('ANALYSIS_ENGINE_DB_PORT', os.environ.get(
    'DB_PORT', '5432'))
DB_NAME = os.environ.get('ANALYSIS_ENGINE_DB_NAME', os.environ.get(
    'DB_NAME', 'forex_trading'))
DB_USER = os.environ.get('ANALYSIS_ENGINE_DB_USER', os.environ.get(
    'DB_USER', 'postgres'))
DB_PASSWORD = os.environ.get('ANALYSIS_ENGINE_DB_PASSWORD', os.environ.get(
    'DB_PASSWORD', 'postgres'))
SYNC_DATABASE_URL = (
    f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
ASYNC_DATABASE_URL = (
    f'postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )
_sync_engine = None
_async_engine = None
Base = declarative_base()
_sync_session_factory = None
_async_session_factory = None


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@with_exception_handling
def initialize_database() ->None:
    """
    Initialize the synchronous database engine and session factory.
    Should be called once during application startup.
    """
    global _sync_engine, _sync_session_factory
    if _sync_engine is not None:
        logger.warning('Synchronous database engine is already initialized')
        return
    try:
        logger.info(
            f'Initializing synchronous database engine for {DB_HOST}:{DB_PORT}/{DB_NAME}'
            )
        _sync_engine = create_engine(SYNC_DATABASE_URL, pool_pre_ping=True,
            pool_size=10, max_overflow=20, pool_recycle=3600, connect_args=
            {'application_name': 'analysis-engine-service'})
        _sync_session_factory = sessionmaker(autocommit=False, autoflush=
            False, bind=_sync_engine)
        with _sync_engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        logger.info('Synchronous database engine initialized successfully')
    except Exception as e:
        logger.error(f'Failed to initialize synchronous database: {e}',
            exc_info=True)
        raise


@async_with_exception_handling
async def initialize_async_database() ->None:
    """
    Initialize the asynchronous database engine and session factory.
    Should be called once during application startup.
    """
    global _async_engine
    if _async_engine is not None:
        logger.warning('Asynchronous database engine is already initialized')
        return
    try:
        logger.info(
            f'Initializing asynchronous database engine for {DB_HOST}:{DB_PORT}/{DB_NAME}'
            )
        _async_engine = create_async_db_engine(database_url=
            ASYNC_DATABASE_URL, echo=False)
        configure_session_maker(_async_engine)
        async with _async_engine.connect() as conn:
            await conn.execute(text('SELECT 1'))
        logger.info('Asynchronous database engine initialized successfully')
    except Exception as e:
        logger.error(f'Failed to initialize asynchronous database: {e}',
            exc_info=True)
        raise


@with_exception_handling
def dispose_database() ->None:
    """
    Dispose of the synchronous database engine.
    Should be called once during application shutdown.
    """
    global _sync_engine, _sync_session_factory
    if _sync_engine is None:
        logger.warning('No synchronous database engine to dispose')
        return
    try:
        _sync_engine.dispose()
        _sync_engine = None
        _sync_session_factory = None
        logger.info('Synchronous database engine disposed')
    except Exception as e:
        logger.error(f'Error disposing synchronous database engine: {e}',
            exc_info=True)
        raise


@async_with_exception_handling
async def dispose_async_database() ->None:
    """
    Dispose of the asynchronous database engine.
    Should be called once during application shutdown.
    """
    global _async_engine
    if _async_engine is None:
        logger.warning('No asynchronous database engine to dispose')
        return
    try:
        await _async_engine.dispose()
        _async_engine = None
        logger.info('Asynchronous database engine disposed')
    except Exception as e:
        logger.error(f'Error disposing asynchronous database engine: {e}',
            exc_info=True)
        raise


@contextmanager
@with_exception_handling
def get_db() ->Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of operations.
    Usage:
        with get_db() as db:
            db.query(...)
    """
    if _sync_engine is None:
        initialize_database()
    session = _sync_session_factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f'Database error: {str(e)}')
        session.rollback()
        raise
    finally:
        session.close()


@with_exception_handling
def get_db_session():
    """
    Dependency function for FastAPI endpoints.
    Usage:
        def my_endpoint(db: Session = Depends(get_db_session)):
            ...
    """
    if _sync_engine is None:
        initialize_database()
    db = _sync_session_factory()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def get_async_db_session() ->AsyncGenerator[AsyncSession, None]:
    """
    Provides an asynchronous session context manager using the common session factory.
    Usage:
        async with get_async_db_session() as db:
            result = await db.execute(...)

    Yields:
        An AsyncSession instance managed by the common_lib session context manager.

    Raises:
        RuntimeError: If the database is not initialized.
    """
    if _async_engine is None:
        await initialize_async_database()
    async with get_common_session() as session:
        yield session


async def get_async_db():
    """
    Dependency function for FastAPI endpoints using async sessions.
    Usage:
        async def my_endpoint(db: AsyncSession = Depends(get_async_db)):
            ...
    """
    if _async_engine is None:
        await initialize_async_database()
    async with get_common_session() as session:
        yield session


@with_exception_handling
def check_db_connection() ->bool:
    """
    Check if the synchronous database connection is working.

    Returns:
        bool: True if connection works, False otherwise
    """
    try:
        if _sync_engine is None:
            initialize_database()
        with _sync_engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        return True
    except Exception as e:
        logger.error(f'Synchronous database connection check failed: {e}')
        return False


@async_with_exception_handling
async def check_async_db_connection() ->bool:
    """
    Check if the asynchronous database connection is working.

    Returns:
        bool: True if connection works, False otherwise
    """
    try:
        if _async_engine is None:
            await initialize_async_database()
        async with _async_engine.connect() as conn:
            await conn.execute(text('SELECT 1'))
        return True
    except Exception as e:
        logger.error(f'Asynchronous database connection check failed: {e}')
        return False


@with_exception_handling
def init_db() ->None:
    """
    Initialize the database tables using the synchronous engine.

    This should be called during application startup if tables need to be created.
    """
    try:
        if _sync_engine is None:
            initialize_database()
        Base.metadata.create_all(bind=_sync_engine)
        logger.info('Initialized database tables using synchronous engine')
    except Exception as e:
        logger.error(f'Failed to initialize database tables: {e}')
        raise


@async_with_exception_handling
async def init_async_db() ->None:
    """
    Initialize the database tables using the asynchronous engine.

    This should be called during application startup if tables need to be created.
    """
    try:
        if _async_engine is None:
            await initialize_async_database()
        async with _async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info('Initialized database tables using asynchronous engine')
    except Exception as e:
        logger.error(f'Failed to initialize database tables: {e}')
        raise
