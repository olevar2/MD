"""
Database connection management using SQLAlchemy and connection pooling.
Handles both synchronous and asynchronous connections.
"""

import asyncio
import logging
import os
from typing import AsyncGenerator, Generator, Optional

from sqlalchemy import create_engine, exc as sqlalchemy_exc
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import Session, sessionmaker

from common_lib.config import AppSettings  # Assuming settings are loaded elsewhere
from common_lib.exceptions import DatabaseConnectionError
from core_foundations.resilience.retry_policy import (
    register_common_retryable_exceptions,
    retry_with_policy,
)

logger = logging.getLogger(__name__)

# Register common SQLAlchemy exceptions for retry
register_common_retryable_exceptions(
    [
        sqlalchemy_exc.TimeoutError,
        sqlalchemy_exc.OperationalError,
        # Add other relevant SQLAlchemy or DB driver exceptions if needed
    ]
)

# --- Global Session Factories (Initialize lazily or on app startup) ---
sync_session_factory: Optional[sessionmaker[Session]] = None
async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


# --- Initialization ---
@retry_with_policy(stop_max_attempt=5, wait_fixed_seconds=2)
def initialize_sync_database(settings: AppSettings):
    """Initializes the synchronous database engine and session factory."""
    global sync_session_factory
    if sync_session_factory:
        logger.info("Synchronous database already initialized.")
        return

    try:
        logger.info(
            f"Initializing synchronous database connection to {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        # Construct sync URL (adjust driver if needed, e.g., psycopg2)
        password = settings.DB_PASSWORD.get_secret_value()
        sync_db_url = (
            f"postgresql+psycopg2://{settings.DB_USER}:{password}@"
            f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )

        engine = create_engine(
            sync_db_url,
            pool_size=settings.DB_POOL_SIZE,
            pool_recycle=3600,  # Recycle connections every hour
            pool_pre_ping=True,  # Check connection health before use
            connect_args={"sslmode": "require"} if settings.DB_SSL_REQUIRED else {},
        )        # Test connection
        with engine.connect():
            logger.info("Synchronous database connection successful.")
        sync_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Synchronous database initialized successfully.")
    except sqlalchemy_exc.OperationalError as e:
        logger.error(f"Failed to connect to synchronous database: {e}", exc_info=True)
        raise DatabaseConnectionError(f"Failed to connect to synchronous database: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during synchronous database initialization: {e}", exc_info=True)
        raise DatabaseConnectionError(f"Unexpected error initializing sync database: {e}") from e


@retry_with_policy(stop_max_attempt=5, wait_fixed_seconds=2)
async def initialize_async_database(settings: AppSettings):
    """Initializes the asynchronous database engine and session factory."""
    global async_session_factory
    if async_session_factory:
        logger.info("Asynchronous database already initialized.")
        return

    try:
        logger.info(
            f"Initializing asynchronous database connection to {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        # Use the computed DATABASE_URL which defaults to asyncpg
        async_engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=settings.DB_POOL_SIZE,
            pool_recycle=3600,
            pool_pre_ping=True,
            connect_args={"ssl": "require"} if settings.DB_SSL_REQUIRED else {},  # asyncpg uses 'ssl'
        )        # Test connection
        async with async_engine.connect():
            logger.info("Asynchronous database connection successful.")

        async_session_factory = async_sessionmaker(
            bind=async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,  # Important for async usage
        )
        logger.info("Asynchronous database initialized successfully.")
    except (sqlalchemy_exc.OperationalError, OSError) as e:  # OSError can occur with connection issues
        logger.error(f"Failed to connect to asynchronous database: {e}", exc_info=True)
        raise DatabaseConnectionError(f"Failed to connect to asynchronous database: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during asynchronous database initialization: {e}", exc_info=True)
        raise DatabaseConnectionError(f"Unexpected error initializing async database: {e}") from e


# --- Session Dependency Functions ---
def get_db_session() -> Generator[Session, None, None]:
    """Provides a synchronous database session."""
    if not sync_session_factory:
        logger.error("Synchronous database not initialized. Call initialize_sync_database first.")
        raise DatabaseConnectionError("Synchronous database session factory not initialized.")

    db: Optional[Session] = None
    try:
        db = sync_session_factory()
        yield db
    except sqlalchemy_exc.OperationalError as e:
        logger.error(f"Database connection error during session usage: {e}", exc_info=True)
        raise DatabaseConnectionError("Failed to get database session due to connection error.") from e
    except Exception as e:
        logger.error(f"Error during synchronous database session: {e}", exc_info=True)
        # Consider if specific exceptions should be raised or handled differently
        raise
    finally:
        if db is not None:
            try:
                db.close()
            except Exception as e:
                logger.warning(f"Error closing synchronous database session: {e}", exc_info=True)


async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provides an asynchronous database session."""
    if not async_session_factory:
        logger.error("Asynchronous database not initialized. Call initialize_async_database first.")
        raise DatabaseConnectionError("Asynchronous database session factory not initialized.")

    session: Optional[AsyncSession] = None
    try:
        session = async_session_factory()
        yield session
        # Note: Commit/rollback logic should be handled by the calling code
    except (sqlalchemy_exc.OperationalError, OSError) as e:  # OSError can occur with connection issues
        logger.error(f"Database connection error during async session usage: {e}", exc_info=True)
        # Optionally rollback if session was created before error
        if session:
            try:
                await session.rollback()
            except Exception as rb_exc:
                logger.warning(f"Error during rollback after connection error: {rb_exc}", exc_info=True)
        raise DatabaseConnectionError("Failed to get async database session due to connection error.") from e
    except Exception as e:
        logger.error(f"Error during asynchronous database session: {e}", exc_info=True)
        if session:
            try:
                await session.rollback()  # Rollback on any other exception during yield
            except Exception as rb_exc:
                logger.warning(f"Error during rollback after general exception: {rb_exc}", exc_info=True)
        raise
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception as e:
                logger.warning(f"Error closing asynchronous database session: {e}", exc_info=True)


# --- Cleanup ---
async def close_async_database():
    """Closes the asynchronous database engine."""
    global async_session_factory
    if async_session_factory:
        engine = async_session_factory.kw["bind"]  # Access engine from sessionmaker kwargs
        if engine:
            logger.info("Closing asynchronous database engine.")
            await engine.dispose()
            async_session_factory = None  # Reset factory
            logger.info("Asynchronous database engine closed.")


def close_sync_database():
    """Closes the synchronous database engine."""
    global sync_session_factory
    if sync_session_factory:
        engine = sync_session_factory.kw["bind"]  # Access engine from sessionmaker kwargs
        if engine:
            logger.info("Closing synchronous database engine.")
            engine.dispose()
            sync_session_factory = None  # Reset factory
            logger.info("Synchronous database engine closed.")


# Example Usage (typically called during application startup/shutdown)
# async def main():
    """
    Main.
    
    """

#     # Load settings (replace with your actual settings loading)
#     settings = AppSettings()
#     await initialize_async_database(settings)
#
#     # Use the session
#     async for session in get_async_db_session():
#         # Perform database operations
#         print("Got async session")
#         # Example: result = await session.execute(select(MyModel))
#         # ...
#         await session.commit() # Or rollback
#
#     await close_async_database()
#
# if __name__ == "__main__":
#     # Example for sync
#     settings = AppSettings()
#     initialize_sync_database(settings)
#     for session in get_db_session():
#         print("Got sync session")
#         # ...
#         session.commit() # Or rollback
#     close_sync_database()
#
#     # Example for async
#     # asyncio.run(main())
