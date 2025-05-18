"""
Standardized database connection pool manager.

This module provides a standardized way to manage database connection pools
across all services in the forex trading platform.
"""
import os
import logging
import multiprocessing
from typing import Dict, Optional, Any, Union, Callable
from contextlib import asynccontextmanager, contextmanager
import asyncio
import unittest.mock as mock

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, AsyncAdaptedQueuePool
import asyncpg

from common_lib.resilience.decorators import (
    with_resilience,
    async_with_resilience,
    with_exception_handling,
    async_with_exception_handling
)
from common_lib.monitoring.metrics import get_metrics_manager
from common_lib.database.config import USE_MOCKS

# Initialize metrics
metrics_manager = get_metrics_manager("database_pool")
DB_POOL_SIZE_GAUGE = metrics_manager.create_gauge(
    name="db_pool_size",
    description="Database connection pool size",
    labels=["service", "pool_type"]
)
DB_POOL_USAGE_GAUGE = metrics_manager.create_gauge(
    name="db_pool_usage",
    description="Database connection pool usage",
    labels=["service", "pool_type"]
)
DB_CONNECTION_TIME = metrics_manager.create_histogram(
    name="db_connection_time_seconds",
    description="Time spent acquiring a database connection",
    labels=["service", "pool_type"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

logger = logging.getLogger(__name__)


def register_database_retryable_exceptions():
    """Register database-specific exceptions that should be retried."""
    from sqlalchemy.exc import OperationalError, TimeoutError
    return [OperationalError, TimeoutError, asyncpg.exceptions.PostgresError, 
            asyncpg.exceptions.ConnectionDoesNotExistError]


def with_database_resilience(operation_name: str):
    """
    Decorator that applies resilience patterns for database operations.

    Args:
        operation_name: Name of the operation

    Returns:
        Decorated function with resilience patterns applied
    """
    return with_resilience(
        service_name="database_pool",
        operation_name=operation_name,
        service_type="database",
        exceptions=register_database_retryable_exceptions()
    )


def async_with_database_resilience(operation_name: str):
    """
    Decorator that applies resilience patterns for async database operations.

    Args:
        operation_name: Name of the operation

    Returns:
        Decorated function with resilience patterns applied
    """
    return async_with_resilience(
        service_name="database_pool",
        operation_name=operation_name,
        service_type="database",
        exceptions=register_database_retryable_exceptions()
    )


class DatabaseConnectionPool:
    """
    Standardized database connection pool manager.
    
    This class provides a unified interface for managing database connections
    across all services in the forex trading platform.
    """
    
    _instances: Dict[str, "DatabaseConnectionPool"] = {}
    
    @classmethod
    def get_instance(cls, service_name: str, **config) -> "DatabaseConnectionPool":
        """
        Get or create a connection pool instance for a service.
        
        Args:
            service_name: Name of the service
            **config: Configuration options for the pool
            
        Returns:
            DatabaseConnectionPool instance
        """
        if service_name not in cls._instances:
            cls._instances[service_name] = cls(service_name, **config)
        return cls._instances[service_name]
    
    def __init__(self, service_name: str, **config):
        """
        Initialize the connection pool manager.
        
        Args:
            service_name: Name of the service
            **config: Configuration options for the pool
        """
        self.service_name = service_name
        self.config = self._get_config(**config)
        self._sync_engine = None
        self._async_engine = None
        self._sync_session_factory = None
        self._async_session_factory = None
        self._asyncpg_pool = None
        self._initialized = False
        self._lock = asyncio.Lock()
        
        # Initialize metrics
        DB_POOL_SIZE_GAUGE.labels(service=service_name, pool_type="sync").set(self.config["pool_size"])
        DB_POOL_SIZE_GAUGE.labels(service=service_name, pool_type="async").set(self.config["pool_size"])
        
        logger.info(f"Initialized database connection pool for {service_name}")
    
    def _get_config(self, **config) -> Dict[str, Any]:
        """
        Get configuration with defaults.
        
        Args:
            **config: User-provided configuration
            
        Returns:
            Complete configuration with defaults
        """
        # Calculate optimal pool size based on CPU count
        cpu_count = multiprocessing.cpu_count()
        optimal_pool_size = min(2 * cpu_count + 1, 20)
        
        # Default configuration
        default_config = {
            "database_url": os.getenv(f"{self.service_name.upper()}_DATABASE_URL", 
                                     os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/forex_trading")),
            "pool_size": int(os.getenv(f"{self.service_name.upper()}_DB_POOL_SIZE", 
                                      os.getenv("DB_POOL_SIZE", str(optimal_pool_size)))),
            "max_overflow": int(os.getenv(f"{self.service_name.upper()}_DB_MAX_OVERFLOW", 
                                         os.getenv("DB_MAX_OVERFLOW", str(optimal_pool_size * 2)))),
            "pool_timeout": int(os.getenv(f"{self.service_name.upper()}_DB_POOL_TIMEOUT", 
                                         os.getenv("DB_POOL_TIMEOUT", "30"))),
            "pool_recycle": int(os.getenv(f"{self.service_name.upper()}_DB_POOL_RECYCLE", 
                                         os.getenv("DB_POOL_RECYCLE", "1800"))),
            "echo": os.getenv(f"{self.service_name.upper()}_DB_ECHO", 
                             os.getenv("DB_ECHO", "false")).lower() in ("true", "1"),
            "prepared_statement_cache_size": int(os.getenv(f"{self.service_name.upper()}_DB_PREPARED_STATEMENT_CACHE_SIZE", 
                                                          os.getenv("DB_PREPARED_STATEMENT_CACHE_SIZE", "256"))),
        }
        
        # Override defaults with user-provided config
        result = {**default_config, **config}
        
        # Ensure we have both sync and async database URLs
        if "database_url" in result and not "async_database_url" in result:
            db_url = result["database_url"]
            if db_url.startswith("postgresql://") and not db_url.startswith("postgresql+asyncpg://"):
                result["async_database_url"] = db_url.replace("postgresql://", "postgresql+asyncpg://")
            else:
                result["async_database_url"] = db_url
        
        return result
        
    @with_database_resilience("initialize_sync")
    @with_exception_handling
    def initialize_sync(self) -> None:
        """Initialize synchronous database engine and session factory."""
        # Check if we're using mocks
        if USE_MOCKS:
            # Create mock engine and session factory
            self._sync_engine = mock.MagicMock()
            self._sync_session_factory = mock.MagicMock()
            logger.info(f"Created mock sync engine for {self.service_name}")
            return
            
        if self._sync_engine is not None:
            return
        
        logger.info(f"Initializing synchronous database engine for {self.service_name}")
        
        # Create engine with optimized settings
        self._sync_engine = create_engine(
            self.config["database_url"],
            poolclass=QueuePool,
            pool_size=self.config["pool_size"],
            max_overflow=self.config["max_overflow"],
            pool_timeout=self.config["pool_timeout"],
            pool_recycle=self.config["pool_recycle"],
            pool_pre_ping=True,
            echo=self.config["echo"],
            connect_args={
                "application_name": f"{self.service_name}"
            }
        )
        
        # Create session factory
        self._sync_session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self._sync_engine
        )
        
        # Test connection
        with self._sync_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        logger.info(f"Synchronous database engine initialized for {self.service_name}")
    
    @async_with_database_resilience("initialize_async")
    @async_with_exception_handling
    async def initialize_async(self) -> None:
        """Initialize asynchronous database engine and session factory."""
        # Check if we're using mocks
        if USE_MOCKS:
            # Create mock engine and session factory
            self._async_engine = mock.MagicMock()
            self._async_session_factory = mock.MagicMock()
            logger.info(f"Created mock async engine for {self.service_name}")
            return
            
        if self._async_engine is not None:
            return
        
        async with self._lock:
            if self._async_engine is not None:
                return
            
            logger.info(f"Initializing asynchronous database engine for {self.service_name}")
            
            # Create async engine with optimized settings
            self._async_engine = create_async_engine(
                self.config["async_database_url"],
                poolclass=AsyncAdaptedQueuePool,
                pool_size=self.config["pool_size"],
                max_overflow=self.config["max_overflow"],
                pool_timeout=self.config["pool_timeout"],
                pool_recycle=self.config["pool_recycle"],
                pool_pre_ping=True,
                echo=self.config["echo"],
                connect_args={
                    "statement_cache_size": 0,  # Disable statement cache for pgbouncer compatibility
                    "prepared_statement_cache_size": self.config["prepared_statement_cache_size"],
                    "server_settings": {
                        "application_name": f"{self.service_name}",
                        "statement_timeout": "60000",  # 60 seconds
                        "idle_in_transaction_session_timeout": "300000",  # 5 minutes
                    }
                }
            )
            
            # Create async session factory
            self._async_session_factory = sessionmaker(
                class_=AsyncSession,
                expire_on_commit=False,
                bind=self._async_engine
            )
            
            # Test connection
            async with self._async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info(f"Asynchronous database engine initialized for {self.service_name}")
    
    @async_with_database_resilience("initialize_asyncpg")
    @async_with_exception_handling
    async def initialize_asyncpg(self) -> None:
        """Initialize asyncpg connection pool for direct access."""
        # Check if we're using mocks
        if USE_MOCKS:
            # Create a mock asyncpg pool
            self._asyncpg_pool = mock.MagicMock()
            logger.info(f"Created mock asyncpg pool for {self.service_name}")
            return
            
        if self._asyncpg_pool is not None:
            return
        
        async with self._lock:
            if self._asyncpg_pool is not None:
                return
            
            logger.info(f"Initializing asyncpg connection pool for {self.service_name}")
            
            # Parse database URL to get connection parameters
            from urllib.parse import urlparse
            url = urlparse(self.config["database_url"].replace("postgresql+asyncpg://", "postgresql://"))
            user_pass, host_port = url.netloc.split("@")
            user, password = user_pass.split(":")
            host, port = host_port.split(":")
            database = url.path.lstrip("/")
            
            # Create asyncpg pool with optimized settings
            self._asyncpg_pool = await asyncpg.create_pool(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                min_size=2,
                max_size=self.config["pool_size"],
                command_timeout=60.0,
                max_inactive_connection_lifetime=self.config["pool_recycle"],
                setup=self._setup_asyncpg_connection
            )
            
            logger.info(f"Asyncpg connection pool initialized for {self.service_name}")
    
    @staticmethod
    async def _setup_asyncpg_connection(conn: asyncpg.Connection) -> None:
        """
        Set up an asyncpg connection with optimized settings.
        
        Args:
            conn: asyncpg connection to set up
        """
        # Set application name
        await conn.execute("SET application_name = 'forex_trading_platform'")
        
        # Set statement timeout (60 seconds)
        await conn.execute("SET statement_timeout = '60000'")
        
        # Set idle in transaction timeout (5 minutes)
        await conn.execute("SET idle_in_transaction_session_timeout = '300000'")
    
    @contextmanager
    @with_database_resilience("get_sync_session")
    @with_exception_handling
    def get_sync_session(self) -> Session:
        """
        Get a synchronous database session.
        
        Yields:
            SQLAlchemy Session
        """
        if self._sync_engine is None:
            self.initialize_sync()
        
        with DB_CONNECTION_TIME.labels(service=self.service_name, pool_type="sync").time():
            session = self._sync_session_factory()
            DB_POOL_USAGE_GAUGE.labels(service=self.service_name, pool_type="sync").inc()
        
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            DB_POOL_USAGE_GAUGE.labels(service=self.service_name, pool_type="sync").dec()
    
    @asynccontextmanager
    @async_with_database_resilience("get_async_session")
    @async_with_exception_handling
    async def get_async_session(self) -> AsyncSession:
        """
        Get an asynchronous database session.
        
        Yields:
            SQLAlchemy AsyncSession
        """
        # Check if we're using mocks
        if USE_MOCKS:
            # Create a mock session
            session = mock.MagicMock(spec=AsyncSession)
            
            # Create mock result for session.execute
            result_mock = mock.MagicMock()
            result_mock.fetchall = mock.MagicMock(return_value=[{"id": 1, "name": "test"}])
            result_mock.fetchone = mock.MagicMock(return_value={"id": 1, "name": "test"})
            result_mock.scalar_one = mock.MagicMock(return_value=1)
            result_mock.scalar_one_or_none = mock.MagicMock(return_value=1)
            result_mock.scalars = mock.MagicMock(return_value=result_mock)
            result_mock.first = mock.MagicMock(return_value={"id": 1, "name": "test"})
            result_mock.all = mock.MagicMock(return_value=[{"id": 1, "name": "test"}])
            
            # Set up session methods
            session.execute = mock.AsyncMock(return_value=result_mock)
            session.commit = mock.AsyncMock()
            session.rollback = mock.AsyncMock()
            session.close = mock.AsyncMock()
            
            # Create an async context manager for the session
            class AsyncSessionContextManager:
                def __init__(self, session):
                    self.session = session
                    
                async def __aenter__(self):
                    return self.session
                    
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            
            # Return the async context manager
            yield session
            return
            
        if self._async_engine is None:
            await self.initialize_async()
        
        with DB_CONNECTION_TIME.labels(service=self.service_name, pool_type="async").time():
            session = self._async_session_factory()
            DB_POOL_USAGE_GAUGE.labels(service=self.service_name, pool_type="async").inc()
        
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
            DB_POOL_USAGE_GAUGE.labels(service=self.service_name, pool_type="async").dec()
    
    @asynccontextmanager
    @async_with_database_resilience("get_asyncpg_connection")
    @async_with_exception_handling
    async def get_asyncpg_connection(self) -> asyncpg.Connection:
        """
        Get a direct asyncpg connection for optimized queries.
        
        Yields:
            asyncpg.Connection
        """
        # Check if we're using mocks
        if USE_MOCKS:
            # Create a mock connection
            conn = mock.MagicMock()
            conn.execute = mock.AsyncMock()
            conn.fetchval = mock.AsyncMock(return_value=1)
            conn.fetch = mock.AsyncMock(return_value=[{"id": 1, "name": "test"}])
            conn.fetchrow = mock.AsyncMock(return_value={"id": 1, "name": "test"})
            yield conn
            return
            
        if self._asyncpg_pool is None:
            await self.initialize_asyncpg()
        
        with DB_CONNECTION_TIME.labels(service=self.service_name, pool_type="asyncpg").time():
            conn = await self._asyncpg_pool.acquire()
            DB_POOL_USAGE_GAUGE.labels(service=self.service_name, pool_type="asyncpg").inc()
        
        try:
            yield conn
        finally:
            await self._asyncpg_pool.release(conn)
            DB_POOL_USAGE_GAUGE.labels(service=self.service_name, pool_type="asyncpg").dec()
    
    async def close(self) -> None:
        """Close all database connections and pools."""
        logger.info(f"Closing database connections for {self.service_name}")
        
        if self._sync_engine is not None:
            self._sync_engine.dispose()
            self._sync_engine = None
        
        if self._async_engine is not None:
            await self._async_engine.dispose()
            self._async_engine = None
        
        if self._asyncpg_pool is not None:
            await self._asyncpg_pool.close()
            self._asyncpg_pool = None
        
        logger.info(f"Database connections closed for {self.service_name}")


# Convenience functions for getting database sessions
def get_connection_pool(service_name: str, **config) -> DatabaseConnectionPool:
    """
    Get a database connection pool for a service.
    
    Args:
        service_name: Name of the service
        **config: Configuration options for the pool
        
    Returns:
        DatabaseConnectionPool instance
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import get_mock_connection_pool
        return get_mock_connection_pool(service_name, **config)
    else:
        return DatabaseConnectionPool.get_instance(service_name, **config)


@contextmanager
def get_sync_db_session(service_name: str, **config) -> Session:
    """
    Get a synchronous database session for a service.
    
    Args:
        service_name: Name of the service
        **config: Configuration options for the pool
        
    Yields:
        SQLAlchemy Session
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import get_mock_sync_db_session
        with get_mock_sync_db_session(service_name) as session:
            yield session
    else:
        pool = get_connection_pool(service_name, **config)
        with pool.get_sync_session() as session:
            yield session


@asynccontextmanager
async def get_async_db_session(service_name: str, **config) -> AsyncSession:
    """
    Get an asynchronous database session for a service.
    
    Args:
        service_name: Name of the service
        **config: Configuration options for the pool
        
    Yields:
        SQLAlchemy AsyncSession
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import get_mock_async_db_session
        async with get_mock_async_db_session(service_name) as session:
            yield session
    else:
        pool = get_connection_pool(service_name, **config)
        async with pool.get_async_session() as session:
            yield session


@asynccontextmanager
async def get_asyncpg_connection(service_name: str, **config) -> asyncpg.Connection:
    """
    Get a direct asyncpg connection for a service.
    
    Args:
        service_name: Name of the service
        **config: Configuration options for the pool
        
    Yields:
        asyncpg.Connection
    """
    if USE_MOCKS:
        # Create a mock connection
        conn = mock.MagicMock()
        conn.execute = mock.AsyncMock()
        conn.fetchval = mock.AsyncMock(return_value=1)
        conn.fetch = mock.AsyncMock(return_value=[{"id": 1, "name": "test"}])
        conn.fetchrow = mock.AsyncMock(return_value={"id": 1, "name": "test"})
        yield conn
    else:
        pool = get_connection_pool(service_name, **config)
        async with pool.get_asyncpg_connection() as conn:
            yield conn