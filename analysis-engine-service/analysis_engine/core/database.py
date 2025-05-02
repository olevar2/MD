"""
Database session management for the analysis engine.
"""

from typing import Generator, Optional
from contextlib import contextmanager
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool

from analysis_engine.config import settings
from analysis_engine.core.errors import ConfigurationError

# Create logger
logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()

# Create engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    echo=settings.DB_ECHO
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Get a database session with automatic cleanup.

    Yields:
        Session: Database session

    Raises:
        ConfigurationError: If database configuration is invalid
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        session.close()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI endpoints to get database session.

    Yields:
        Session: Database session
    """
    with get_db_session() as session:
        yield session

def init_db() -> None:
    """Initialize database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {str(e)}")
        raise ConfigurationError("Database initialization failed")

def check_db_connection() -> bool:
    """
    Check database connection.

    Returns:
        bool: True if connection is successful
    """
    try:
        with get_db_session() as session:
            session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False

class DatabaseManager:
    """Manages database operations and connection state."""

    def __init__(self):
        """Initialize database manager."""
        self._engine = engine
        self._session_factory = SessionLocal
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize database and create tables."""
        if not self._is_initialized:
            init_db()
            self._is_initialized = True

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            Session: Database session
        """
        return self._session_factory()

    def close(self) -> None:
        """Close database connections."""
        self._engine.dispose()

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._is_initialized

    @property
    def engine(self):
        """Get database engine."""
        return self._engine

# Create global database manager instance
db_manager = DatabaseManager()