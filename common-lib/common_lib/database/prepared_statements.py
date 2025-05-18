"""
Utilities for working with prepared statements.

This module provides utilities for working with prepared statements in both
SQLAlchemy and asyncpg contexts.
"""
import logging
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast
import asyncio

# Import USE_MOCKS flag
from common_lib.database import USE_MOCKS

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
import asyncpg

from common_lib.monitoring.metrics import get_metrics_manager
from common_lib.resilience.decorators import (
    with_exception_handling,
    async_with_exception_handling
)

# Initialize metrics
metrics_manager = get_metrics_manager("database_queries")
PREPARED_STMT_CACHE_SIZE_GAUGE = metrics_manager.create_gauge(
    name="prepared_stmt_cache_size",
    description="Size of the prepared statement cache",
    labels=["service"]
)
PREPARED_STMT_EXECUTION_TIME = metrics_manager.create_histogram(
    name="prepared_stmt_execution_time_seconds",
    description="Execution time of prepared statements",
    labels=["service", "operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)
PREPARED_STMT_EXECUTIONS_TOTAL = metrics_manager.create_counter(
    name="prepared_stmt_executions_total",
    description="Total number of prepared statement executions",
    labels=["service", "operation", "status"]
)

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')


class PreparedStatementCache:
    """
    Cache for prepared statements.
    
    This class provides a simple cache for prepared statements to avoid
    preparing the same statement multiple times.
    """
    
    _instances: Dict[str, "PreparedStatementCache"] = {}
    
    @classmethod
    def get_instance(cls, service_name: str) -> "PreparedStatementCache":
        """
        Get or create a prepared statement cache for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            PreparedStatementCache instance
        """
        if service_name not in cls._instances:
            cls._instances[service_name] = cls(service_name)
        return cls._instances[service_name]
    
    def __init__(self, service_name: str, max_size: int = 1000):
        """
        Initialize the prepared statement cache.
        
        Args:
            service_name: Name of the service
            max_size: Maximum number of statements to cache
        """
        self.service_name = service_name
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        
        # Initialize metrics
        PREPARED_STMT_CACHE_SIZE_GAUGE.labels(service=service_name).set(0)
        
        logger.info(f"Initialized prepared statement cache for {service_name}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a prepared statement from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Prepared statement or None if not found
        """
        return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """
        Add a prepared statement to the cache.
        
        Args:
            key: Cache key
            value: Prepared statement
        """
        # If cache is full, remove oldest item
        if len(self._cache) >= self.max_size:
            # Remove a random item (simple approach)
            if self._cache:
                self._cache.pop(next(iter(self._cache)))
        
        self._cache[key] = value
        PREPARED_STMT_CACHE_SIZE_GAUGE.labels(service=self.service_name).set(len(self._cache))
    
    async def get_or_set(self, key: str, create_func: Callable[[], Any]) -> Any:
        """
        Get a prepared statement from the cache or create it if not found.
        
        Args:
            key: Cache key
            create_func: Function to create the prepared statement
            
        Returns:
            Prepared statement
        """
        # Fast path: check if statement is already in cache
        stmt = self.get(key)
        if stmt is not None:
            return stmt
        
        # Slow path: acquire lock and check again
        async with self._lock:
            stmt = self.get(key)
            if stmt is not None:
                return stmt
            
            # Create and cache the statement
            stmt = create_func()
            self.set(key, stmt)
            return stmt
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        PREPARED_STMT_CACHE_SIZE_GAUGE.labels(service=self.service_name).set(0)


def get_prepared_statement_cache(service_name: str) -> PreparedStatementCache:
    """
    Get a prepared statement cache for a service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        PreparedStatementCache instance
    """
    return PreparedStatementCache.get_instance(service_name)


@with_exception_handling
def prepare_statement(
    session: Session,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    service_name: str = "default"
) -> Any:
    """
    Prepare a SQL statement for execution with SQLAlchemy.
    
    Args:
        session: SQLAlchemy session
        query: SQL query
        params: Query parameters
        service_name: Name of the service
        
    Returns:
        Prepared statement
    """
    cache = get_prepared_statement_cache(service_name)
    
    # Create a cache key from the query
    cache_key = f"sqlalchemy:{query}"
    
    # Get or create the prepared statement
    stmt = cache.get(cache_key)
    if stmt is None:
        stmt = text(query)
        cache.set(cache_key, stmt)
    
    return stmt


@async_with_exception_handling
async def prepare_asyncpg_statement(
    conn: asyncpg.Connection,
    query: str,
    service_name: str = "default"
) -> str:
    """
    Prepare a SQL statement for execution with asyncpg.
    
    Args:
        conn: asyncpg connection
        query: SQL query
        service_name: Name of the service
        
    Returns:
        Statement name
    """
    cache = get_prepared_statement_cache(service_name)
    
    # Create a cache key and statement name from the query
    import hashlib
    query_hash = hashlib.md5(query.encode()).hexdigest()
    stmt_name = f"stmt_{query_hash}"
    cache_key = f"asyncpg:{stmt_name}"
    
    # Check if the statement is already prepared
    if cache.get(cache_key) is None:
        # Prepare the statement
        await conn.prepare(query, name=stmt_name)
        cache.set(cache_key, stmt_name)
    
    return stmt_name


def with_prepared_statement(
    service_name: str,
    operation_name: str
):
    """
    Decorator for executing a function with a prepared statement.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        
    Returns:
        Decorated function
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_with_prepared_statement
        return mock_with_prepared_statement(service_name, operation_name)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                PREPARED_STMT_EXECUTION_TIME.labels(
                    service=service_name,
                    operation=operation_name
                ).observe(execution_time)
                PREPARED_STMT_EXECUTIONS_TOTAL.labels(
                    service=service_name,
                    operation=operation_name,
                    status="success"
                ).inc()
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                PREPARED_STMT_EXECUTION_TIME.labels(
                    service=service_name,
                    operation=operation_name
                ).observe(execution_time)
                PREPARED_STMT_EXECUTIONS_TOTAL.labels(
                    service=service_name,
                    operation=operation_name,
                    status="error"
                ).inc()
                logger.error(f"Error executing prepared statement: {str(e)}")
                raise
        return wrapper
    return decorator


def async_with_prepared_statement(
    service_name: str,
    operation_name: str
):
    """
    Decorator for executing an async function with a prepared statement.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                PREPARED_STMT_EXECUTION_TIME.labels(
                    service=service_name,
                    operation=operation_name
                ).observe(execution_time)
                PREPARED_STMT_EXECUTIONS_TOTAL.labels(
                    service=service_name,
                    operation=operation_name,
                    status="success"
                ).inc()
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                PREPARED_STMT_EXECUTION_TIME.labels(
                    service=service_name,
                    operation=operation_name
                ).observe(execution_time)
                PREPARED_STMT_EXECUTIONS_TOTAL.labels(
                    service=service_name,
                    operation=operation_name,
                    status="error"
                ).inc()
                logger.error(f"Error executing prepared statement: {str(e)}")
                raise
        return wrapper
    return decorator


@with_exception_handling
def execute_prepared_statement(
    session: Session,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    service_name: str = "default",
    operation_name: str = "execute"
) -> Any:
    """
    Execute a prepared SQL statement with SQLAlchemy.
    
    Args:
        session: SQLAlchemy session
        query: SQL query
        params: Query parameters
        service_name: Name of the service
        operation_name: Name of the operation
        
    Returns:
        Query result
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_execute_prepared_statement
        return mock_execute_prepared_statement(session, query, params or {}, service_name, operation_name)
    
    # Prepare the statement
    stmt = prepare_statement(session, query, params, service_name)
    
    # Execute the statement with metrics
    start_time = time.time()
    try:
        result = session.execute(stmt, params or {})
        execution_time = time.time() - start_time
        PREPARED_STMT_EXECUTION_TIME.labels(
            service=service_name,
            operation=operation_name
        ).observe(execution_time)
        PREPARED_STMT_EXECUTIONS_TOTAL.labels(
            service=service_name,
            operation=operation_name,
            status="success"
        ).inc()
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        PREPARED_STMT_EXECUTION_TIME.labels(
            service=service_name,
            operation=operation_name
        ).observe(execution_time)
        PREPARED_STMT_EXECUTIONS_TOTAL.labels(
            service=service_name,
            operation=operation_name,
            status="error"
        ).inc()
        logger.error(f"Error executing prepared statement: {str(e)}")
        raise


@async_with_exception_handling
async def execute_prepared_statement_async(
    session: AsyncSession,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    service_name: str = "default",
    operation_name: str = "execute_async"
) -> Any:
    """
    Execute a prepared SQL statement with SQLAlchemy async.
    
    Args:
        session: SQLAlchemy async session
        query: SQL query
        params: Query parameters
        service_name: Name of the service
        operation_name: Name of the operation
        
    Returns:
        Query result
    """
    # Prepare the statement
    stmt = prepare_statement(cast(Session, session), query, params, service_name)
    
    # Execute the statement with metrics
    start_time = time.time()
    try:
        result = await session.execute(stmt, params or {})
        execution_time = time.time() - start_time
        PREPARED_STMT_EXECUTION_TIME.labels(
            service=service_name,
            operation=operation_name
        ).observe(execution_time)
        PREPARED_STMT_EXECUTIONS_TOTAL.labels(
            service=service_name,
            operation=operation_name,
            status="success"
        ).inc()
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        PREPARED_STMT_EXECUTION_TIME.labels(
            service=service_name,
            operation=operation_name
        ).observe(execution_time)
        PREPARED_STMT_EXECUTIONS_TOTAL.labels(
            service=service_name,
            operation=operation_name,
            status="error"
        ).inc()
        logger.error(f"Error executing prepared statement: {str(e)}")
        raise


@async_with_exception_handling
async def execute_prepared_statement_asyncpg(
    conn: asyncpg.Connection,
    query: str,
    params: Optional[List[Any]] = None,
    service_name: str = "default",
    operation_name: str = "execute_asyncpg"
) -> Any:
    """
    Execute a prepared SQL statement with asyncpg.
    
    Args:
        conn: asyncpg connection
        query: SQL query
        params: Query parameters
        service_name: Name of the service
        operation_name: Name of the operation
        
    Returns:
        Query result
    """
    # Prepare the statement
    stmt_name = await prepare_asyncpg_statement(conn, query, service_name)
    
    # Execute the statement with metrics
    start_time = time.time()
    try:
        result = await conn.execute(stmt_name, *(params or []))
        execution_time = time.time() - start_time
        PREPARED_STMT_EXECUTION_TIME.labels(
            service=service_name,
            operation=operation_name
        ).observe(execution_time)
        PREPARED_STMT_EXECUTIONS_TOTAL.labels(
            service=service_name,
            operation=operation_name,
            status="success"
        ).inc()
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        PREPARED_STMT_EXECUTION_TIME.labels(
            service=service_name,
            operation=operation_name
        ).observe(execution_time)
        PREPARED_STMT_EXECUTIONS_TOTAL.labels(
            service=service_name,
            operation=operation_name,
            status="error"
        ).inc()
        logger.error(f"Error executing prepared statement: {str(e)}")
        raise


@async_with_exception_handling
async def fetch_prepared_statement_asyncpg(
    conn: asyncpg.Connection,
    query: str,
    params: Optional[List[Any]] = None,
    service_name: str = "default",
    operation_name: str = "fetch_asyncpg"
) -> List[asyncpg.Record]:
    """
    Fetch results from a prepared SQL statement with asyncpg.
    
    Args:
        conn: asyncpg connection
        query: SQL query
        params: Query parameters
        service_name: Name of the service
        operation_name: Name of the operation
        
    Returns:
        Query results
    """
    # Prepare the statement
    stmt_name = await prepare_asyncpg_statement(conn, query, service_name)
    
    # Execute the statement with metrics
    start_time = time.time()
    try:
        result = await conn.fetch(stmt_name, *(params or []))
        execution_time = time.time() - start_time
        PREPARED_STMT_EXECUTION_TIME.labels(
            service=service_name,
            operation=operation_name
        ).observe(execution_time)
        PREPARED_STMT_EXECUTIONS_TOTAL.labels(
            service=service_name,
            operation=operation_name,
            status="success"
        ).inc()
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        PREPARED_STMT_EXECUTION_TIME.labels(
            service=service_name,
            operation=operation_name
        ).observe(execution_time)
        PREPARED_STMT_EXECUTIONS_TOTAL.labels(
            service=service_name,
            operation=operation_name,
            status="error"
        ).inc()
        logger.error(f"Error executing prepared statement: {str(e)}")
        raise