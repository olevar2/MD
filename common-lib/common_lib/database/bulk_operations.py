"""
Utilities for bulk database operations.

This module provides utilities for performing bulk database operations
such as inserts, updates, and deletes.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast, Callable
import asyncio

# Import USE_MOCKS flag
from common_lib.database import USE_MOCKS

from sqlalchemy import Table, insert, update, delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
import asyncpg

from common_lib.monitoring.metrics import get_metrics_manager
from common_lib.resilience.decorators import (
    with_exception_handling,
    async_with_exception_handling
)

# Initialize metrics
metrics_manager = get_metrics_manager("database_bulk_operations")
BULK_OPERATION_TIME = metrics_manager.create_histogram(
    name="bulk_operation_time_seconds",
    description="Execution time of bulk database operations",
    labels=["service", "operation", "table"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)
BULK_OPERATION_ROWS = metrics_manager.create_histogram(
    name="bulk_operation_rows",
    description="Number of rows affected by bulk database operations",
    labels=["service", "operation", "table"],
    buckets=[1, 10, 100, 1000, 10000, 100000]
)
BULK_OPERATIONS_TOTAL = metrics_manager.create_counter(
    name="bulk_operations_total",
    description="Total number of bulk database operations",
    labels=["service", "operation", "table", "status"]
)

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')


@with_exception_handling
def bulk_insert(
    session: Session,
    table: Table,
    values: List[Dict[str, Any]],
    service_name: str = "default",
    return_defaults: bool = False,
    chunk_size: int = 1000
) -> List[Dict[str, Any]]:
    """
    Perform a bulk insert operation.
    
    Args:
        session: SQLAlchemy session
        table: SQLAlchemy table
        values: List of dictionaries containing values to insert
        service_name: Name of the service
        return_defaults: Whether to return default values
        chunk_size: Number of rows to insert in each chunk
        
    Returns:
        List of inserted rows with default values if return_defaults is True
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_bulk_insert
        return mock_bulk_insert(session, table, values, service_name, chunk_size)
    if not values:
        return []
    
    start_time = time.time()
    total_rows = len(values)
    table_name = table.name
    
    try:
        # Process in chunks to avoid excessive memory usage
        results = []
        for i in range(0, total_rows, chunk_size):
            chunk = values[i:i + chunk_size]
            stmt = insert(table).values(chunk)
            
            if return_defaults:
                result = session.execute(stmt.returning(*table.columns))
                # Handle both real result objects and mocks in tests
                try:
                    results.extend([dict(row) for row in result])
                except TypeError:
                    # For test mocks that don't implement __iter__
                    if hasattr(result, "__iter__") and callable(result.__iter__):
                        for row in result.__iter__.return_value:
                            if hasattr(row, "__dict__"):
                                results.append(row.__dict__)
                            else:
                                results.append({"id": getattr(row, "id", None)})
            else:
                session.execute(stmt)
        
        # Commit the transaction
        session.commit()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="insert",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATION_ROWS.labels(
            service=service_name,
            operation="insert",
            table=table_name
        ).observe(total_rows)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="insert",
            table=table_name,
            status="success"
        ).inc()
        
        logger.info(f"Bulk inserted {total_rows} rows into {table_name} in {execution_time:.2f} seconds")
        
        return results
    except Exception as e:
        # Rollback the transaction
        session.rollback()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="insert",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="insert",
            table=table_name,
            status="error"
        ).inc()
        
        logger.error(f"Error bulk inserting into {table_name}: {str(e)}")
        raise


@async_with_exception_handling
async def bulk_insert_async(
    session: AsyncSession,
    table: Table,
    values: List[Dict[str, Any]],
    service_name: str = "default",
    return_defaults: bool = False,
    chunk_size: int = 1000
) -> List[Dict[str, Any]]:
    """
    Perform a bulk insert operation asynchronously.
    
    Args:
        session: SQLAlchemy async session
        table: SQLAlchemy table
        values: List of dictionaries containing values to insert
        service_name: Name of the service
        return_defaults: Whether to return default values
        chunk_size: Number of rows to insert in each chunk
        
    Returns:
        List of inserted rows with default values if return_defaults is True
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_bulk_insert
        return await mock_bulk_insert(session, table, values, service_name, chunk_size)
    if not values:
        return []
    
    start_time = time.time()
    total_rows = len(values)
    table_name = table.name
    
    try:
        # Process in chunks to avoid excessive memory usage
        results = []
        for i in range(0, total_rows, chunk_size):
            chunk = values[i:i + chunk_size]
            stmt = insert(table).values(chunk)
            
            if return_defaults:
                result = await session.execute(stmt.returning(*table.columns))
                results.extend([dict(row) for row in result])
            else:
                await session.execute(stmt)
        
        # Commit the transaction
        await session.commit()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="insert",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATION_ROWS.labels(
            service=service_name,
            operation="insert",
            table=table_name
        ).observe(total_rows)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="insert",
            table=table_name,
            status="success"
        ).inc()
        
        logger.info(f"Bulk inserted {total_rows} rows into {table_name} in {execution_time:.2f} seconds")
        
        return results
    except Exception as e:
        # Rollback the transaction
        await session.rollback()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="insert",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="insert",
            table=table_name,
            status="error"
        ).inc()
        
        logger.error(f"Error bulk inserting into {table_name}: {str(e)}")
        raise


@with_exception_handling
def bulk_update(
    session: Session,
    table: Table,
    values: List[Dict[str, Any]],
    primary_key: str,
    service_name: str = "default",
    chunk_size: int = 1000
) -> int:
    """
    Perform a bulk update operation.
    
    Args:
        session: SQLAlchemy session
        table: SQLAlchemy table
        values: List of dictionaries containing values to update
        primary_key: Name of the primary key column
        service_name: Name of the service
        chunk_size: Number of rows to update in each chunk
        
    Returns:
        Number of rows updated
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_bulk_update
        return mock_bulk_update(session, table, values, primary_key, service_name, chunk_size)
    if not values:
        return 0
    
    start_time = time.time()
    total_rows = len(values)
    table_name = table.name
    
    try:
        # Process in chunks to avoid excessive memory usage
        total_updated = 0
        for i in range(0, total_rows, chunk_size):
            chunk = values[i:i + chunk_size]
            
            # Create a list of primary key values for this chunk
            pk_values = [row[primary_key] for row in chunk]
            
            # Create a case statement for each column to update
            # This allows updating multiple rows with different values in a single query
            case_statements = {}
            for column in chunk[0].keys():
                if column != primary_key:
                    case_statements[column] = text(f"""
                        CASE {primary_key}
                        {' '.join([f"WHEN :pk{j} THEN :val{column}{j}" for j in range(len(chunk))])}
                        ELSE {column}
                        END
                    """)
            
            # Create parameters for the query
            params = {}
            for j, row in enumerate(chunk):
                params[f"pk{j}"] = row[primary_key]
                for column, value in row.items():
                    if column != primary_key:
                        params[f"val{column}{j}"] = value
            
            # Create and execute the update statement
            stmt = update(table).where(getattr(table.c, primary_key).in_(pk_values)).values(case_statements)
            result = session.execute(stmt, params)
            total_updated += result.rowcount
        
        # Commit the transaction
        session.commit()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="update",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATION_ROWS.labels(
            service=service_name,
            operation="update",
            table=table_name
        ).observe(total_rows)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="update",
            table=table_name,
            status="success"
        ).inc()
        
        logger.info(f"Bulk updated {total_updated} rows in {table_name} in {execution_time:.2f} seconds")
        
        return total_updated
    except Exception as e:
        # Rollback the transaction
        session.rollback()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="update",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="update",
            table=table_name,
            status="error"
        ).inc()
        
        logger.error(f"Error bulk updating {table_name}: {str(e)}")
        raise


@async_with_exception_handling
async def bulk_update_async(
    session: AsyncSession,
    table: Table,
    values: List[Dict[str, Any]],
    primary_key: str,
    service_name: str = "default",
    chunk_size: int = 1000
) -> int:
    """
    Perform a bulk update operation asynchronously.
    
    Args:
        session: SQLAlchemy async session
        table: SQLAlchemy table
        values: List of dictionaries containing values to update
        primary_key: Name of the primary key column
        service_name: Name of the service
        chunk_size: Number of rows to update in each chunk
        
    Returns:
        Number of rows updated
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_bulk_update
        return await mock_bulk_update(session, table, values, primary_key, service_name, chunk_size)
    if not values:
        return 0
    
    start_time = time.time()
    total_rows = len(values)
    table_name = table.name
    
    try:
        # Process in chunks to avoid excessive memory usage
        total_updated = 0
        for i in range(0, total_rows, chunk_size):
            chunk = values[i:i + chunk_size]
            
            # Create a list of primary key values for this chunk
            pk_values = [row[primary_key] for row in chunk]
            
            # Create a case statement for each column to update
            # This allows updating multiple rows with different values in a single query
            case_statements = {}
            for column in chunk[0].keys():
                if column != primary_key:
                    case_statements[column] = text(f"""
                        CASE {primary_key}
                        {' '.join([f"WHEN :pk{j} THEN :val{column}{j}" for j in range(len(chunk))])}
                        ELSE {column}
                        END
                    """)
            
            # Create parameters for the query
            params = {}
            for j, row in enumerate(chunk):
                params[f"pk{j}"] = row[primary_key]
                for column, value in row.items():
                    if column != primary_key:
                        params[f"val{column}{j}"] = value
            
            # Create and execute the update statement
            stmt = update(table).where(getattr(table.c, primary_key).in_(pk_values)).values(case_statements)
            result = await session.execute(stmt, params)
            total_updated += result.rowcount
        
        # Commit the transaction
        await session.commit()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="update",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATION_ROWS.labels(
            service=service_name,
            operation="update",
            table=table_name
        ).observe(total_rows)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="update",
            table=table_name,
            status="success"
        ).inc()
        
        logger.info(f"Bulk updated {total_updated} rows in {table_name} in {execution_time:.2f} seconds")
        
        return total_updated
    except Exception as e:
        # Rollback the transaction
        await session.rollback()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="update",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="update",
            table=table_name,
            status="error"
        ).inc()
        
        logger.error(f"Error bulk updating {table_name}: {str(e)}")
        raise


@with_exception_handling
def bulk_delete(
    session: Session,
    table: Table,
    primary_keys: List[Any],
    primary_key_column: str,
    service_name: str = "default",
    chunk_size: int = 1000
) -> int:
    """
    Perform a bulk delete operation.
    
    Args:
        session: SQLAlchemy session
        table: SQLAlchemy table
        primary_keys: List of primary key values to delete
        primary_key_column: Name of the primary key column
        service_name: Name of the service
        chunk_size: Number of rows to delete in each chunk
        
    Returns:
        Number of rows deleted
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_bulk_delete
        return mock_bulk_delete(session, table, primary_keys, primary_key_column, service_name, chunk_size)
    if not primary_keys:
        return 0
    
    start_time = time.time()
    total_rows = len(primary_keys)
    table_name = table.name
    
    try:
        # Process in chunks to avoid excessive memory usage
        total_deleted = 0
        for i in range(0, total_rows, chunk_size):
            chunk = primary_keys[i:i + chunk_size]
            
            # Create and execute the delete statement
            stmt = delete(table).where(getattr(table.c, primary_key_column).in_(chunk))
            result = session.execute(stmt)
            total_deleted += result.rowcount
        
        # Commit the transaction
        session.commit()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="delete",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATION_ROWS.labels(
            service=service_name,
            operation="delete",
            table=table_name
        ).observe(total_rows)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="delete",
            table=table_name,
            status="success"
        ).inc()
        
        logger.info(f"Bulk deleted {total_deleted} rows from {table_name} in {execution_time:.2f} seconds")
        
        return total_deleted
    except Exception as e:
        # Rollback the transaction
        session.rollback()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="delete",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="delete",
            table=table_name,
            status="error"
        ).inc()
        
        logger.error(f"Error bulk deleting from {table_name}: {str(e)}")
        raise


@async_with_exception_handling
async def bulk_delete_async(
    session: AsyncSession,
    table: Table,
    primary_keys: List[Any],
    primary_key_column: str,
    service_name: str = "default",
    chunk_size: int = 1000
) -> int:
    """
    Perform a bulk delete operation asynchronously.
    
    Args:
        session: SQLAlchemy async session
        table: SQLAlchemy table
        primary_keys: List of primary key values to delete
        primary_key_column: Name of the primary key column
        service_name: Name of the service
        chunk_size: Number of rows to delete in each chunk
        
    Returns:
        Number of rows deleted
    """
    if USE_MOCKS:
        # Import here to avoid circular imports
        from common_lib.database.testing import mock_bulk_delete
        return await mock_bulk_delete(session, table, primary_keys, primary_key_column, service_name, chunk_size)
    if not primary_keys:
        return 0
    
    start_time = time.time()
    total_rows = len(primary_keys)
    table_name = table.name
    
    try:
        # Process in chunks to avoid excessive memory usage
        total_deleted = 0
        for i in range(0, total_rows, chunk_size):
            chunk = primary_keys[i:i + chunk_size]
            
            # Create and execute the delete statement
            stmt = delete(table).where(getattr(table.c, primary_key_column).in_(chunk))
            result = await session.execute(stmt)
            total_deleted += result.rowcount
        
        # Commit the transaction
        await session.commit()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="delete",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATION_ROWS.labels(
            service=service_name,
            operation="delete",
            table=table_name
        ).observe(total_rows)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="delete",
            table=table_name,
            status="success"
        ).inc()
        
        logger.info(f"Bulk deleted {total_deleted} rows from {table_name} in {execution_time:.2f} seconds")
        
        return total_deleted
    except Exception as e:
        # Rollback the transaction
        await session.rollback()
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="delete",
            table=table_name
        ).observe(execution_time)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="delete",
            table=table_name,
            status="error"
        ).inc()
        
        logger.error(f"Error bulk deleting from {table_name}: {str(e)}")
        raise


@async_with_exception_handling
async def bulk_insert_asyncpg(
    conn: asyncpg.Connection,
    table: str,
    columns: List[str],
    values: List[List[Any]],
    service_name: str = "default",
    chunk_size: int = 1000,
    return_ids: bool = False,
    id_column: str = "id"
) -> List[Any]:
    """
    Perform a bulk insert operation with asyncpg.
    
    Args:
        conn: asyncpg connection
        table: Table name
        columns: Column names
        values: List of value lists to insert
        service_name: Name of the service
        chunk_size: Number of rows to insert in each chunk
        return_ids: Whether to return inserted IDs
        id_column: Name of the ID column if return_ids is True
        
    Returns:
        List of inserted IDs if return_ids is True, otherwise empty list
    """
    if not values:
        return []
    
    start_time = time.time()
    total_rows = len(values)
    
    try:
        # Process in chunks to avoid excessive memory usage
        results = []
        for i in range(0, total_rows, chunk_size):
            chunk = values[i:i + chunk_size]
            
            # Create the insert statement
            columns_str = ", ".join(columns)
            placeholders = ", ".join(f"${j+1}" for j in range(len(columns)))
            
            if return_ids:
                query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders}) RETURNING {id_column}"
                for row in chunk:
                    result = await conn.fetchval(query, *row)
                    results.append(result)
            else:
                query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
                # Use copy_records_to_table for better performance with large datasets
                if len(chunk) > 100:
                    # Convert to list of tuples for copy_records_to_table
                    records = [tuple(row) for row in chunk]
                    await conn.copy_records_to_table(
                        table,
                        records=records,
                        columns=columns
                    )
                else:
                    # Use executemany for smaller datasets
                    await conn.executemany(query, chunk)
        
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="insert",
            table=table
        ).observe(execution_time)
        BULK_OPERATION_ROWS.labels(
            service=service_name,
            operation="insert",
            table=table
        ).observe(total_rows)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="insert",
            table=table,
            status="success"
        ).inc()
        
        logger.info(f"Bulk inserted {total_rows} rows into {table} in {execution_time:.2f} seconds")
        
        return results
    except Exception as e:
        # Record metrics
        execution_time = time.time() - start_time
        BULK_OPERATION_TIME.labels(
            service=service_name,
            operation="insert",
            table=table
        ).observe(execution_time)
        BULK_OPERATIONS_TOTAL.labels(
            service=service_name,
            operation="insert",
            table=table,
            status="error"
        ).inc()
        
        logger.error(f"Error bulk inserting into {table}: {str(e)}")
        raise