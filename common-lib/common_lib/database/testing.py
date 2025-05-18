"""
Testing utilities for database operations.

This module provides mock implementations of database utilities for testing.
"""
import logging
import unittest.mock as mock
import functools
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable, Union
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDatabaseConnectionPool:
    """Mock implementation of DatabaseConnectionPool for testing."""
    
    def __init__(self, service_name: str, **kwargs):
        """Initialize the mock connection pool."""
        self.service_name = service_name
        self.kwargs = kwargs
        self.initialized_async = False
        self.initialized_sync = False
        self.initialized_asyncpg = False
        self.closed_async = False
        self.closed_sync = False
        
        # Create mock engines and pools
        self._async_engine = mock.MagicMock()
        self._sync_engine = mock.MagicMock()
        self._asyncpg_pool = mock.MagicMock()
        
        # Create mock result for session.execute
        result_mock = mock.MagicMock()
        result_mock.fetchall = mock.MagicMock(return_value=[{"id": 1, "name": "test"}])
        result_mock.fetchone = mock.MagicMock(return_value={"id": 1, "name": "test"})
        result_mock.scalar_one = mock.MagicMock(return_value=1)
        result_mock.scalar_one_or_none = mock.MagicMock(return_value=1)
        result_mock.scalars = mock.MagicMock(return_value=result_mock)
        result_mock.first = mock.MagicMock(return_value={"id": 1, "name": "test"})
        result_mock.all = mock.MagicMock(return_value=[{"id": 1, "name": "test"}])
        
        # Create mock session
        self.session = mock.MagicMock(spec=AsyncSession)
        self.session.execute = mock.AsyncMock(return_value=result_mock)
        self.session.commit = mock.AsyncMock()
        self.session.rollback = mock.AsyncMock()
        self.session.close = mock.AsyncMock()
        
        # Create mock prepared statement cache
        self.prepared_statement_cache = {}
        
        # Create mock connection for asyncpg
        self.connection = mock.MagicMock()
        self.connection.execute = mock.AsyncMock(return_value=result_mock)
        self.connection.fetchval = mock.AsyncMock(return_value=1)
        self.connection.fetch = mock.AsyncMock(return_value=[{"id": 1, "name": "test"}])
        self.connection.fetchrow = mock.AsyncMock(return_value={"id": 1, "name": "test"})
        self.connection.prepare = mock.AsyncMock()
        
        logger.info(f"Created mock connection pool for {service_name}")
    
    async def initialize_async(self):
        """Initialize the async connection pool."""
        self.initialized_async = True
        logger.info(f"Initialized async connection pool for {self.service_name}")
        return self
    
    async def initialize_sync(self):
        """Initialize the sync connection pool."""
        self.initialized_sync = True
        logger.info(f"Initialized sync connection pool for {self.service_name}")
        return self
    
    async def initialize_asyncpg(self):
        """Initialize the asyncpg connection pool."""
        self.initialized_asyncpg = True
        logger.info(f"Initialized asyncpg connection pool for {self.service_name}")
        return self
    
    async def close_async(self):
        """Close the async connection pool."""
        self.closed_async = True
        logger.info(f"Closed async connection pool for {self.service_name}")
    
    def close_sync(self):
        """Close the sync connection pool."""
        self.closed_sync = True
        logger.info(f"Closed sync connection pool for {self.service_name}")
    
    # Define an async context manager class for get_async_session
    class AsyncSessionContextManager:
        def __init__(self, session):
            self.session = session
            
        async def __aenter__(self):
            return self.session
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    def get_async_session(self):
        """Get an async session from the pool."""
        return self.AsyncSessionContextManager(self.session)
    
    def get_sync_session(self):
        """Get a sync session from the pool."""
        return self.session
    
    def get_prepared_statement_cache(self):
        """Get the prepared statement cache."""
        return self.prepared_statement_cache
    
    # Define an async context manager class for get_asyncpg_connection
    class AsyncpgConnectionContextManager:
        def __init__(self, connection):
            self.connection = connection
            
        async def __aenter__(self):
            return self.connection
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    def get_asyncpg_connection(self):
        """Get an asyncpg connection from the pool."""
        return self.AsyncpgConnectionContextManager(self.connection)


# Dictionary to store connection pools
_connection_pools: Dict[str, MockDatabaseConnectionPool] = {}


def get_mock_connection_pool(
    service_name: str,
    **kwargs
) -> MockDatabaseConnectionPool:
    """
    Get a mock connection pool for a service.
    
    Args:
        service_name: Name of the service
        **kwargs: Additional arguments
        
    Returns:
        A mock connection pool
    """
    if service_name not in _connection_pools:
        _connection_pools[service_name] = MockDatabaseConnectionPool(service_name, **kwargs)
    
    return _connection_pools[service_name]


class MockAsyncDBSessionContextManager:
    """Async context manager for mock database sessions."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.pool = get_mock_connection_pool(service_name)
        
    async def __aenter__(self):
        return self.pool.session
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def get_mock_async_db_session(service_name: str) -> MockAsyncDBSessionContextManager:
    """
    Get a mock async database session.
    
    Args:
        service_name: Name of the service
        
    Returns:
        A mock async database session context manager
    """
    return MockAsyncDBSessionContextManager(service_name)


def get_mock_sync_db_session(service_name: str):
    """
    Get a mock sync database session.
    
    Args:
        service_name: Name of the service
        
    Returns:
        A mock sync database session
    """
    pool = get_mock_connection_pool(service_name)
    return pool.get_sync_session()


def mock_with_prepared_statement(service_name: str, operation_name: str):
    """
    Mock decorator for prepared statements.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        
    Returns:
        A decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger.info(f"Executing prepared statement {operation_name} for {service_name}")
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


async def mock_execute_prepared_statement(
    session: AsyncSession,
    query: str,
    params: Dict[str, Any],
    service_name: str,
    operation_name: str,
):
    """
    Mock execution of a prepared statement.
    
    Args:
        session: Database session
        query: SQL query
        params: Query parameters
        service_name: Name of the service
        operation_name: Name of the operation
        
    Returns:
        A mock result
    """
    # Create a mock result with more realistic behavior
    result = mock.MagicMock()
    
    # Default row and multiple rows for different result types
    default_row = {"id": 1, "name": "test", "value": 100}
    multiple_rows = [
        {"id": i, "name": f"test_{i}", "value": i * 10} 
        for i in range(1, 11)  # Return 10 rows by default
    ]
    
    # Customize result based on query type
    if "SELECT" in query.upper():
        if "COUNT" in query.upper():
            result.fetchall = mock.MagicMock(return_value=[{"count": 10}])
            result.fetchone = mock.MagicMock(return_value={"count": 10})
            result.scalar_one = mock.MagicMock(return_value=10)
            result.scalar_one_or_none = mock.MagicMock(return_value=10)
            result.rowcount = 1
        else:
            result.fetchall = mock.MagicMock(return_value=multiple_rows)
            result.fetchone = mock.MagicMock(return_value=default_row)
            result.scalar_one = mock.MagicMock(return_value=default_row["id"])
            result.scalar_one_or_none = mock.MagicMock(return_value=default_row["id"])
            result.rowcount = len(multiple_rows)
    elif "INSERT" in query.upper():
        result.fetchall = mock.MagicMock(return_value=[{"id": 1}])
        result.fetchone = mock.MagicMock(return_value={"id": 1})
        result.scalar_one = mock.MagicMock(return_value=1)
        result.scalar_one_or_none = mock.MagicMock(return_value=1)
        result.rowcount = 1
    elif "UPDATE" in query.upper():
        result.rowcount = 10  # Return 10 rows updated by default
    elif "DELETE" in query.upper():
        result.rowcount = 10  # Return 10 rows deleted by default
    
    # Add additional result methods
    result.scalars = mock.MagicMock(return_value=result)
    result.first = mock.MagicMock(return_value=default_row)
    result.all = mock.MagicMock(return_value=multiple_rows)
    
    logger.info(f"Executed prepared statement {operation_name} for {service_name}")
    return result


async def mock_bulk_insert(
    session: AsyncSession,
    table: Any,
    data: List[Dict[str, Any]],
    service_name: str,
    chunk_size: int = 1000,
    return_defaults: bool = False,
):
    """
    Mock bulk insert operation.
    
    Args:
        session: Database session
        table: Table to insert into
        data: Data to insert
        service_name: Name of the service
        chunk_size: Size of each chunk
        return_defaults: Whether to return inserted rows with default values
        
    Returns:
        Number of rows inserted or list of inserted rows
    """
    logger.info(f"Bulk inserting {len(data)} rows into {getattr(table, 'name', 'unknown')} for {service_name}")
    
    if return_defaults:
        # Return the data with added default values
        result = []
        for item in data:
            row = item.copy()
            if "id" not in row:
                row["id"] = len(result) + 1
            if "created_at" not in row and hasattr(table, "created_at"):
                from datetime import datetime
                row["created_at"] = datetime.now()
            result.append(row)
        return result
    
    return len(data)


async def mock_bulk_update(
    session: AsyncSession,
    table: Any,
    data: List[Dict[str, Any]],
    primary_key: str,
    service_name: str,
    chunk_size: int = 1000,
):
    """
    Mock bulk update operation.
    
    Args:
        session: Database session
        table: Table to update
        data: Data to update
        primary_key: Primary key column
        service_name: Name of the service
        chunk_size: Size of each chunk
        
    Returns:
        Number of rows updated
    """
    logger.info(f"Bulk updating {len(data)} rows in {getattr(table, 'name', 'unknown')} for {service_name}")
    return len(data)


async def mock_bulk_delete(
    session: AsyncSession,
    table: Any,
    primary_keys: List[Any],
    primary_key_column: str,
    service_name: str,
    chunk_size: int = 1000,
):
    """
    Mock bulk delete operation.
    
    Args:
        session: Database session
        table: Table to delete from
        primary_keys: Primary keys to delete
        primary_key_column: Primary key column
        service_name: Name of the service
        chunk_size: Size of each chunk
        
    Returns:
        Number of rows deleted
    """
    logger.info(f"Bulk deleting {len(primary_keys)} rows from {getattr(table, 'name', 'unknown')} for {service_name}")
    return len(primary_keys)


def mock_track_query_performance(operation_name: str, table_name: str, service_name: str):
    """
    Mock decorator for tracking query performance.
    
    Args:
        operation_name: Name of the operation
        table_name: Name of the table
        service_name: Name of the service
        
    Returns:
        A decorator function
    """
    def decorator(func):
        # Check if the function is async or not
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger.info(f"Tracking performance for {operation_name} on {table_name} for {service_name}")
                start_time = __import__('time').time()
                result = await func(*args, **kwargs)
                elapsed_time = __import__('time').time() - start_time
                logger.info(f"Operation {operation_name} completed in {elapsed_time:.4f} seconds")
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                logger.info(f"Tracking performance for {operation_name} on {table_name} for {service_name}")
                start_time = __import__('time').time()
                result = func(*args, **kwargs)
                elapsed_time = __import__('time').time() - start_time
                logger.info(f"Operation {operation_name} completed in {elapsed_time:.4f} seconds")
                return result
            return sync_wrapper
    return decorator


def mock_track_transaction(operation_name: str, service_name: str):
    """
    Mock decorator for tracking transactions.
    
    Args:
        operation_name: Name of the operation
        service_name: Name of the service
        
    Returns:
        A decorator function or async context manager
    """
    def decorator(func=None):
        # If func is None, we're being used as a context manager
        if func is None:
            # Return an async context manager
            class AsyncTransactionContextManager:
                async def __aenter__(self):
                    logger.info(f"Starting transaction for {operation_name} for {service_name}")
                    self.start_time = __import__('time').time()
                    return None
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    elapsed_time = __import__('time').time() - self.start_time
                    if exc_type is not None:
                        logger.error(f"Transaction {operation_name} failed after {elapsed_time:.4f} seconds: {exc_val}")
                    else:
                        logger.info(f"Transaction {operation_name} completed in {elapsed_time:.4f} seconds")
                    return False  # Don't suppress exceptions
            
            return AsyncTransactionContextManager()
        
        # Otherwise, we're being used as a decorator
        # Check if the function is async or not
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger.info(f"Tracking transaction for {operation_name} for {service_name}")
                start_time = __import__('time').time()
                result = await func(*args, **kwargs)
                elapsed_time = __import__('time').time() - start_time
                logger.info(f"Transaction {operation_name} completed in {elapsed_time:.4f} seconds")
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                logger.info(f"Tracking transaction for {operation_name} for {service_name}")
                start_time = __import__('time').time()
                result = func(*args, **kwargs)
                elapsed_time = __import__('time').time() - start_time
                logger.info(f"Transaction {operation_name} completed in {elapsed_time:.4f} seconds")
                return result
            return sync_wrapper
    return decorator


async def mock_analyze_query(
    session: AsyncSession,
    query: str,
    params: Dict[str, Any],
    service_name: str,
):
    """
    Mock query analysis.
    
    Args:
        session: Database session
        query: SQL query
        params: Query parameters
        service_name: Name of the service
        
    Returns:
        Query analysis results
    """
    logger.info(f"Analyzing query for {service_name}: {query}")
    
    # Create a more detailed analysis result
    return {
        "planning_time": 0.1,
        "execution_time": 0.2,
        "total_time": 0.3,
        "plan": {
            "Node Type": "Seq Scan",
            "Relation Name": "test_table",
            "Alias": "test_table",
            "Startup Cost": 0.0,
            "Total Cost": 10.0,
            "Plan Rows": 100,
            "Plan Width": 100,
            "Filter": "id > 0",
            "Rows Removed by Filter": 0,
        },
        "settings": {
            "work_mem": "4MB",
            "effective_io_concurrency": "1",
            "random_page_cost": "4.0",
        },
    }


async def mock_check_database_health(service_name: str = "default"):
    """
    Mock database health check.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Health check results
    """
    logger.info(f"Checking database health for {service_name}")
    
    # Create a more detailed health check result
    return {
        "status": "healthy",
        "connection_pool": {
            "active_connections": 2,
            "idle_connections": 8,
            "total_connections": 10,
            "max_connections": 20,
            "connection_wait_time_ms": 0.5,
        },
        "database": {
            "version": "PostgreSQL 15.0",
            "uptime": "1 day, 2:34:56",
            "size": "100 MB",
            "active_queries": 5,
            "locks": 0,
            "deadlocks": 0,
            "cache_hit_ratio": 0.98,
        },
        "tables": {
            "total_count": 25,
            "largest_tables": [
                {"name": "test_table", "size": "20 MB", "rows": 10000},
                {"name": "another_table", "size": "15 MB", "rows": 8000},
            ],
        },
    }


async def mock_prepare_asyncpg_statement(
    conn: Any,
    query: str,
    service_name: str = "default"
) -> str:
    """
    Mock preparation of a SQL statement for execution with asyncpg.
    
    Args:
        conn: asyncpg connection
        query: SQL query
        service_name: Name of the service
        
    Returns:
        Statement name
    """
    # Create a statement name from the query
    import hashlib
    query_hash = hashlib.md5(query.encode()).hexdigest()
    stmt_name = f"stmt_{query_hash}"
    
    logger.info(f"Prepared asyncpg statement for {service_name}: {stmt_name}")
    
    # Mock the prepare method
    await conn.prepare(query, name=stmt_name)
    
    return stmt_name


async def mock_fetch_prepared_statement_asyncpg(
    conn: Any,
    query: str,
    params: Optional[List[Any]] = None,
    service_name: str = "default",
    operation_name: str = "fetch_asyncpg"
) -> List[Any]:
    """
    Mock fetching results from a prepared SQL statement with asyncpg.
    
    Args:
        conn: asyncpg connection
        query: SQL query or statement name
        params: Query parameters
        service_name: Name of the service
        operation_name: Name of the operation
        
    Returns:
        Query results
    """
    # If query is a statement name, use it directly
    if query.startswith("stmt_"):
        stmt_name = query
    else:
        # Otherwise, prepare the statement
        stmt_name = await mock_prepare_asyncpg_statement(conn, query, service_name)
    
    logger.info(f"Fetching from prepared asyncpg statement {stmt_name} for {service_name}")
    
    # Create mock results
    multiple_rows = [
        {"id": i, "name": f"test_{i}", "value": i * 10} 
        for i in range(1, 11)  # Return 10 rows by default
    ]
    
    # Mock the fetch method
    return await conn.fetch(stmt_name, *(params or []))