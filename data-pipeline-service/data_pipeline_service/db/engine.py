"""
Database engine and session management using common_lib.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

# Use centralized db utilities
from common_lib.db import (
    create_async_db_engine,
    configure_session_maker,
    get_session as get_common_session, # Rename to avoid potential local conflicts if any
    init_db as common_init_db,
    Base as CommonBase # Use the Base from common_lib if models will be defined there or shared
)
from core_foundations.utils.logger import get_logger # Keep logger if needed
from data_pipeline_service.config.settings import get_settings

# Initialize logger
logger = get_logger("data-pipeline-service")

# Global engine variable
_engine: Optional[AsyncEngine] = None

async def initialize_database():
    """
    Initializes the database engine and configures the session maker.
    Should be called once during application startup.
    """
    global _engine
    if _engine is not None:
        logger.warning("Database already initialized.")
        return

    settings = get_settings()
    try:
        # Create engine using the URL from settings
        _engine = create_async_db_engine(database_url=settings.DATABASE_URL, echo=settings.debug)
        # Configure the session maker provided by common_lib
        configure_session_maker(_engine)
        logger.info(f"Database engine initialized and session maker configured for {settings.db_host}:{settings.db_port}/{settings.db_name}")

        # Optional: Initialize database tables (e.g., create tables)
        # Consider if this should be run on startup or managed via migrations
        # await common_init_db(_engine)
        # logger.info("Database tables initialized (if applicable).")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        # Depending on the application, you might want to raise the exception
        # or handle it to prevent the app from starting in a broken state.
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
        logger.info("Database engine disposed.")
    else:
        logger.warning("Database engine not initialized or already disposed.")


def get_engine() -> AsyncEngine:
    """
    Get the initialized database engine.

    Returns:
        AsyncEngine: SQLAlchemy async engine instance.

    Raises:
        RuntimeError: If the engine has not been initialized.
    """
    if _engine is None:
        raise RuntimeError("Database engine has not been initialized. Call initialize_database() first.")
    return _engine


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provides an asynchronous session context manager using the common session factory.

    Yields:
        An AsyncSession instance managed by the common_lib session context manager.
    """
    # Use the session context manager from common_lib
    async with get_common_session() as session:
        yield session

# If you have service-specific models, they should inherit from CommonBase
# Example:
# from sqlalchemy import Column, Integer, String
# class ServiceSpecificModel(CommonBase):
#     __tablename__ = 'service_specific_table'
#     id = Column(Integer, primary_key=True)
#     data = Column(String)

# Note: Removed create_db_config, create_db_engine, dispose_db_engine (replaced by initialize/dispose),
# create_session_factory (handled by common_lib.configure_session_maker).
# The core logic now relies on initialize_database() and dispose_database() being called
# appropriately in the application lifecycle (e.g., FastAPI startup/shutdown events).