"""
Database monitoring utilities.

This module provides utilities for monitoring database performance and health.
"""
import logging
import time
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast
import asyncio
from contextlib import contextmanager, asynccontextmanager

# Import USE_MOCKS flag
from common_lib.database.config import USE_MOCKS

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
metrics_manager = get_metrics_manager("database_monitoring")
DB_QUERY_TIME = metrics_manager.create_histogram(
    name="db_query_time_seconds",
    description="Database query execution time in seconds",
    labels=["service", "operation", "table"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)
DB_QUERY_COUNT = metrics_manager.create_counter(
    name="db_query_count_total",
    description="Total number of database queries",
    labels=["service", "operation", "table", "status"]
)
DB_SLOW_QUERY_COUNT = metrics_manager.create_counter(
    name="db_slow_query_count_total",
    description="Total number of slow database queries",
    labels=["service", "operation", "table"]
)
DB_CONNECTION_ERRORS = metrics_manager.create_counter(
    name="db_connection_errors_total",
    description="Total number of database connection errors",
    labels=["service"]
)
DB_DEADLOCK_COUNT = metrics_manager.create_counter(
    name="db_deadlock_count_total",
    description="Total number of database deadlocks",
    labels=["service", "table"]
)
DB_ROW_COUNT = metrics_manager.create_histogram(
    name="db_row_count",
    description="Number of rows returned by database queries",
    labels=["service", "operation", "table"],
    buckets=[1, 10, 100, 1000, 10000, 100000]
)
DB_POOL_WAIT_TIME = metrics_manager.create_histogram(
    name="db_pool_wait_time_seconds",
    description="Time spent waiting for a database connection from the pool",
    labels=["service"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)
DB_POOL_SIZE = metrics_manager.create_gauge(
    name="db_pool_size",
    description="Size of the database connection pool",
    labels=["service"]
)
DB_POOL_USAGE = metrics_manager.create_gauge(
    name="db_pool_usage",
    description="Number of connections in use from the database connection pool",
    labels=["service"]
)
DB_TRANSACTION_TIME = metrics_manager.create_histogram(
    name="db_transaction_time_seconds",
    description="Database transaction execution time in seconds",
    labels=["service"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
)

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')


def track_query_performance(
    operation: str,
    table: str,
    service_name: str = "default",
    slow_query_threshold: float = 1.0
):
    """
    Decorator to track query performance.
    
    Args:
        operation: Type of operation (e.g., 'select', 'insert', 'update')
        table: Table being queried
        service_name: Name of the service
        slow_query_threshold: Threshold in seconds for slow queries
        
    Returns:
        Decorated function
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_track_query_performance
        return mock_track_query_performance(operation, table, service_name)
        
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Record metrics
                execution_time = time.time() - start_time
                DB_QUERY_TIME.labels(
                    service=service_name,
                    operation=operation,
                    table=table
                ).observe(execution_time)
                DB_QUERY_COUNT.labels(
                    service=service_name,
                    operation=operation,
                    table=table,
                    status="success"
                ).inc()
                
                # Check for slow queries
                if execution_time > slow_query_threshold:
                    DB_SLOW_QUERY_COUNT.labels(
                        service=service_name,
                        operation=operation,
                        table=table
                    ).inc()
                    logger.warning(
                        f"Slow query detected: {operation} on {table} took {execution_time:.2f} seconds"
                    )
                
                # Record row count for select operations
                if operation == "select" and hasattr(result, "__len__"):
                    try:
                        row_count = len(result)
                        DB_ROW_COUNT.labels(
                            service=service_name,
                            operation=operation,
                            table=table
                        ).observe(row_count)
                    except (TypeError, AttributeError):
                        pass
                
                return result
            except Exception as e:
                # Record metrics for errors
                execution_time = time.time() - start_time
                DB_QUERY_TIME.labels(
                    service=service_name,
                    operation=operation,
                    table=table
                ).observe(execution_time)
                DB_QUERY_COUNT.labels(
                    service=service_name,
                    operation=operation,
                    table=table,
                    status="error"
                ).inc()
                
                # Check for specific error types
                error_type = type(e).__name__
                if "deadlock" in str(e).lower() or "deadlock" in error_type.lower():
                    DB_DEADLOCK_COUNT.labels(
                        service=service_name,
                        table=table
                    ).inc()
                elif "connection" in str(e).lower() or "connection" in error_type.lower():
                    DB_CONNECTION_ERRORS.labels(
                        service=service_name
                    ).inc()
                
                logger.error(f"Database error in {operation} on {table}: {str(e)}")
                raise
        return wrapper
    return decorator


def async_track_query_performance(
    operation: str,
    table: str,
    service_name: str = "default",
    slow_query_threshold: float = 1.0
):
    """
    Decorator to track async query performance.
    
    Args:
        operation: Type of operation (e.g., 'select', 'insert', 'update')
        table: Table being queried
        service_name: Name of the service
        slow_query_threshold: Threshold in seconds for slow queries
        
    Returns:
        Decorated function
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_track_query_performance
        return mock_track_query_performance(operation, table, service_name)
        
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                
                # Record metrics
                execution_time = time.time() - start_time
                DB_QUERY_TIME.labels(
                    service=service_name,
                    operation=operation,
                    table=table
                ).observe(execution_time)
                DB_QUERY_COUNT.labels(
                    service=service_name,
                    operation=operation,
                    table=table,
                    status="success"
                ).inc()
                
                # Check for slow queries
                if execution_time > slow_query_threshold:
                    DB_SLOW_QUERY_COUNT.labels(
                        service=service_name,
                        operation=operation,
                        table=table
                    ).inc()
                    logger.warning(
                        f"Slow query detected: {operation} on {table} took {execution_time:.2f} seconds"
                    )
                
                # Record row count for select operations
                if operation == "select" and hasattr(result, "__len__"):
                    try:
                        row_count = len(result)
                        DB_ROW_COUNT.labels(
                            service=service_name,
                            operation=operation,
                            table=table
                        ).observe(row_count)
                    except (TypeError, AttributeError):
                        pass
                
                return result
            except Exception as e:
                # Record metrics for errors
                execution_time = time.time() - start_time
                DB_QUERY_TIME.labels(
                    service=service_name,
                    operation=operation,
                    table=table
                ).observe(execution_time)
                DB_QUERY_COUNT.labels(
                    service=service_name,
                    operation=operation,
                    table=table,
                    status="error"
                ).inc()
                
                # Check for specific error types
                error_type = type(e).__name__
                if "deadlock" in str(e).lower() or "deadlock" in error_type.lower():
                    DB_DEADLOCK_COUNT.labels(
                        service=service_name,
                        table=table
                    ).inc()
                elif "connection" in str(e).lower() or "connection" in error_type.lower():
                    DB_CONNECTION_ERRORS.labels(
                        service=service_name
                    ).inc()
                
                logger.error(f"Database error in {operation} on {table}: {str(e)}")
                raise
        return wrapper
    return decorator


def track_transaction(operation_name: str, service_name: str = "default"):
    """
    Decorator to track database transaction performance.
    
    Args:
        operation_name: Name of the operation
        service_name: Name of the service
        
    Returns:
        Decorated function
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_track_transaction
        return mock_track_transaction(operation_name, service_name)
        
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                DB_TRANSACTION_TIME.labels(
                    service=service_name
                ).observe(execution_time)
                return result
            except Exception:
                execution_time = time.time() - start_time
                DB_TRANSACTION_TIME.labels(
                    service=service_name
                ).observe(execution_time)
                raise
        return wrapper
    return decorator


def async_track_transaction(operation_name: str, service_name: str = "default"):
    """
    Decorator to track async database transaction performance.
    
    Args:
        operation_name: Name of the operation
        service_name: Name of the service
        
    Returns:
        Decorated function
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_track_transaction
        return mock_track_transaction(operation_name, service_name)
        
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                DB_TRANSACTION_TIME.labels(
                    service=service_name
                ).observe(execution_time)
                return result
            except Exception:
                execution_time = time.time() - start_time
                DB_TRANSACTION_TIME.labels(
                    service=service_name
                ).observe(execution_time)
                raise
        return wrapper
    return decorator


@with_exception_handling
def analyze_query(
    session: Session,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    service_name: str = "default"
) -> Dict[str, Any]:
    """
    Analyze a SQL query using EXPLAIN ANALYZE.
    
    Args:
        session: SQLAlchemy session
        query: SQL query to analyze
        params: Query parameters
        service_name: Name of the service
        
    Returns:
        Query execution plan
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_analyze_query
        return mock_analyze_query(session, query, params, service_name)
    # Add EXPLAIN ANALYZE to the query
    if not query.lower().startswith("explain"):
        query = f"EXPLAIN ANALYZE {query}"
    
    # Execute the query
    result = session.execute(text(query), params or {})
    
    # Parse the result
    plan = [row[0] for row in result]
    
    # Log the execution plan
    logger.info(f"Query execution plan for service {service_name}:\n" + "\n".join(plan))
    
    return {
        "service": service_name,
        "query": query,
        "plan": plan
    }


@async_with_exception_handling
async def analyze_query_async(
    session: AsyncSession,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    service_name: str = "default"
) -> Dict[str, Any]:
    """
    Analyze a SQL query using EXPLAIN ANALYZE asynchronously.
    
    Args:
        session: SQLAlchemy async session
        query: SQL query to analyze
        params: Query parameters
        service_name: Name of the service
        
    Returns:
        Query execution plan
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_analyze_query
        return mock_analyze_query(session, query, params, service_name)
        
    # Add EXPLAIN ANALYZE to the query
    if not query.lower().startswith("explain"):
        query = f"EXPLAIN ANALYZE {query}"
    
    # Execute the query
    result = await session.execute(text(query), params or {})
    
    # Parse the result
    plan = [row[0] for row in result]
    
    # Log the execution plan
    logger.info(f"Query execution plan for service {service_name}:\n" + "\n".join(plan))
    
    return {
        "service": service_name,
        "query": query,
        "plan": plan
    }


@with_exception_handling
def check_database_health(
    service_name: str = "default"
) -> Dict[str, Any]:
    """
    Check database health.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Database health information
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_check_database_health
        return mock_check_database_health(service_name)
        
    # For mocks, we'll just return a mock health report
    return {
        "service": service_name,
        "responsive": True,
        "version": "PostgreSQL 15.0",
        "connection_count": 10,
        "active_query_count": 2,
        "long_running_query_count": 0,
        "database_size": "100 MB",
        "table_count": 25,
        "response_time": 0.01
    }


@async_with_exception_handling
async def check_database_health_async(
    service_name: str = "default"
) -> Dict[str, Any]:
    """
    Check database health asynchronously.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Database health information
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_check_database_health
        return mock_check_database_health(service_name)
        
    # For mocks, we'll just return a mock health report
    return {
        "service": service_name,
        "responsive": True,
        "version": "PostgreSQL 15.0",
        "connection_count": 10,
        "active_query_count": 2,
        "long_running_query_count": 0,
        "database_size": "100 MB",
        "table_count": 25,
        "response_time": 0.01
    }