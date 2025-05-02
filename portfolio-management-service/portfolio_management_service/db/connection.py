"""
Database Connection Module (Refactored for common-lib async usage).

Provides functionality for connecting to the database using the centralized common-lib.
"""
import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

# Keep Base for local models for now
from sqlalchemy.ext.declarative import declarative_base
# Use async components from common_lib
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from common_lib.db import (
    create_async_db_engine,
    configure_session_maker,
    get_session as get_common_session,
    Base as CommonBase # Import common base if needed later
)

from core_foundations.utils.logger import get_logger

# Initialize logger
logger = get_logger("portfolio-db-connection")

# Base class for local SQLAlchemy models (consider inheriting from CommonBase later)
Base = declarative_base()

# Global engine variable
_engine: Optional[AsyncEngine] = None

# Database connection settings from environment variables
DB_HOST = os.getenv("PORTFOLIO_DB_HOST", "localhost")
DB_PORT = os.getenv("PORTFOLIO_DB_PORT", "5432")
DB_USER = os.getenv("PORTFOLIO_DB_USER", "forex_user")
DB_PASSWORD = os.getenv("PORTFOLIO_DB_PASSWORD", "forex_password")
DB_NAME = os.getenv("PORTFOLIO_DB_NAME", "forex_db")
# Construct DATABASE_URL for asyncpg
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# Optional: Get debug setting from env var if needed for engine echo
DB_ECHO = os.getenv("PORTFOLIO_DB_ECHO", "False").lower() in ("true", "1", "t")


async def initialize_database():
    """
    Initializes the database engine and configures the session maker using common-lib.
    Should be called once during application startup.
    """
    global _engine
    if _engine is not None:
        logger.warning("Database already initialized.")
        return

    try:
        logger.info(f"Initializing database engine for {DB_HOST}:{DB_PORT}/{DB_NAME}")
        # Create engine using the URL constructed from env vars
        _engine = create_async_db_engine(database_url=DATABASE_URL, echo=DB_ECHO)
        # Configure the session maker provided by common_lib
        configure_session_maker(_engine)
        logger.info("Database engine initialized and session maker configured.")

        # Optional: Initialize database tables defined using the local Base
        # This assumes models inheriting from the local Base exist in db/models.py etc.
        # async with _engine.begin() as conn:
        #     await conn.run_sync(Base.metadata.create_all)
        # logger.info("Initialized local database tables (if any).")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
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

    Raises:
        RuntimeError: If the database is not initialized.
    """
    if _engine is None:
         # Ensure engine is initialized before attempting to get a session
         raise RuntimeError("Database not initialized. Call initialize_database() first.")
    async with get_common_session() as session:
        yield session

# --- Old Synchronous Functions (Commented out / To be removed) ---
# def get_connection_string() -> str:
#     """Get the database connection string."""
#     return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# def get_engine(): # Old sync version
#     global _engine
#     if _engine is None:
#         connection_string = get_connection_string()
#         logger.info(f"Creating database engine for host {DB_HOST}")
#         _engine = create_engine(...) # Old sync create_engine call
#     return _engine

# def get_session_factory(): # Old sync version
#     global _SessionFactory
#     if _SessionFactory is None:
#         engine = get_engine() # Old sync engine
#         _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#     return _SessionFactory

# def get_db_session() -> Session: # Old sync version
#     return get_session_factory()()

# def init_db() -> None: # Old sync version
#     engine = get_engine() # Old sync engine
#     Base.metadata.create_all(bind=engine)
#     logger.info("Initialized database tables")

# def check_db_connection() -> Dict[str, Any]: # Old sync version, needs rework for async and health check
#     """Check database connection status."""
#     # ... implementation needs to be async or adapted for health check ...
#     pass