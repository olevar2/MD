# Database Utilities Guide

This guide provides comprehensive documentation for the database utilities in the forex trading platform.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Connection Pool](#connection-pool)
4. [Prepared Statements](#prepared-statements)
5. [Bulk Operations](#bulk-operations)
6. [Monitoring](#monitoring)
7. [Resilience Patterns](#resilience-patterns)
8. [Configuration](#configuration)
9. [Best Practices](#best-practices)
10. [Performance Considerations](#performance-considerations)
11. [Integration with Services](#integration-with-services)
12. [Troubleshooting](#troubleshooting)
13. [Examples](#examples)

## Introduction

The database utilities in the common-lib package provide standardized database access, optimization, and monitoring capabilities for all services in the forex trading platform. These utilities are designed to improve performance, reliability, and observability of database operations.

### Key Features

- **Connection Pooling**: Standardized connection pool management for both synchronous and asynchronous database connections.
- **Prepared Statements**: Utilities for working with prepared statements to improve performance and security.
- **Bulk Operations**: Utilities for efficient bulk insert, update, and delete operations.
- **Monitoring**: Comprehensive database monitoring and metrics.
- **Resilience Patterns**: Circuit breakers, retries, and timeouts for database operations.

## Architecture

The database utilities are organized into four main components:

1. **Connection Pool**: Manages database connections for services.
2. **Prepared Statements**: Provides utilities for working with prepared statements.
3. **Bulk Operations**: Provides utilities for efficient bulk data operations.
4. **Monitoring**: Provides utilities for monitoring database performance.

For detailed architecture diagrams, see the [architecture documentation](architecture/database/database_utilities_architecture.md).

## Connection Pool

The connection pool component manages database connections for services. It provides optimized connection pooling for both synchronous and asynchronous database operations.

### Key Features

- Dynamic pool sizing based on CPU count
- Connection recycling and timeout handling
- Support for both SQLAlchemy and asyncpg
- Metrics for pool size, usage, and connection time

### Usage

```python
from common_lib.database import get_connection_pool, get_sync_db_session, get_async_db_session

# Get a connection pool
pool = get_connection_pool("my_service")

# Use a synchronous session
with get_sync_db_session("my_service") as session:
    # Execute a query
    result = session.execute("SELECT 1")
    print(result.scalar())

# Use an asynchronous session
async with get_async_db_session("my_service") as session:
    # Execute a query
    result = await session.execute("SELECT 1")
    print(result.scalar())
```

### Configuration

The connection pool can be configured through environment variables or by passing configuration options directly to the `get_connection_pool` function:

```python
pool = get_connection_pool(
    "my_service",
    database_url="postgresql://user:password@localhost:5432/mydatabase",
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    echo=False,
    prepared_statement_cache_size=256,
)
```

## Prepared Statements

The prepared statements component provides utilities for working with prepared statements. It provides caching for prepared statements to improve performance and security.

### Key Features

- Caching for prepared statements to improve performance
- Support for both SQLAlchemy ORM and direct asyncpg connections
- Metrics for prepared statement execution time and cache size

### Usage

```python
from common_lib.database import with_prepared_statement, execute_prepared_statement

# Use a prepared statement
@with_prepared_statement("my_service", "get_user")
def get_user(session, user_id):
    # Execute a prepared statement
    result = execute_prepared_statement(
        session,
        "SELECT id, name, email FROM users WHERE id = :user_id",
        {"user_id": user_id},
        "my_service",
        "get_user",
    )
    
    # Get the first row
    row = result.fetchone()
    
    # Return user data
    if row:
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
        }
    
    return None
```

### Asynchronous Usage

```python
from common_lib.database import async_with_prepared_statement, execute_prepared_statement_async

# Use a prepared statement
@async_with_prepared_statement("my_service", "get_user")
async def get_user(session, user_id):
    # Execute a prepared statement
    result = await execute_prepared_statement_async(
        session,
        "SELECT id, name, email FROM users WHERE id = :user_id",
        {"user_id": user_id},
        "my_service",
        "get_user",
    )
    
    # Get the first row
    row = result.fetchone()
    
    # Return user data
    if row:
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
        }
    
    return None
```

### Direct asyncpg Usage

```python
from common_lib.database import get_asyncpg_connection, prepare_asyncpg_statement, fetch_prepared_statement_asyncpg

async def get_user(user_id):
    async with get_asyncpg_connection("my_service") as conn:
        # Prepare the statement
        stmt_name = await prepare_asyncpg_statement(
            conn,
            "SELECT id, name, email FROM users WHERE id = $1",
            "my_service",
            "get_user",
        )
        
        # Execute the prepared statement
        rows = await fetch_prepared_statement_asyncpg(
            conn,
            stmt_name,
            [user_id],
            "my_service",
            "get_user",
        )
        
        # Return user data
        if rows:
            return {
                "id": rows[0]["id"],
                "name": rows[0]["name"],
                "email": rows[0]["email"],
            }
        
        return None
```

## Bulk Operations

The bulk operations component provides utilities for efficient bulk data operations. It provides optimized methods for inserting, updating, and deleting multiple rows at once.

### Key Features

- Bulk insert, update, and delete operations
- Support for both SQLAlchemy ORM and direct asyncpg connections
- Chunking for large datasets
- Metrics for bulk operation performance

### Usage

```python
from common_lib.database import bulk_insert, bulk_update, bulk_delete

# Bulk insert
bulk_insert(
    session,
    users_table,
    [
        {"name": "User 1", "email": "user1@example.com"},
        {"name": "User 2", "email": "user2@example.com"},
    ],
    "my_service",
)

# Bulk update
bulk_update(
    session,
    users_table,
    [
        {"id": 1, "name": "Updated User 1"},
        {"id": 2, "name": "Updated User 2"},
    ],
    "id",
    "my_service",
)

# Bulk delete
bulk_delete(
    session,
    users_table,
    [1, 2],
    "id",
    "my_service",
)
```

### Asynchronous Usage

```python
from common_lib.database import bulk_insert_async, bulk_update_async, bulk_delete_async

# Bulk insert
await bulk_insert_async(
    session,
    users_table,
    [
        {"name": "User 1", "email": "user1@example.com"},
        {"name": "User 2", "email": "user2@example.com"},
    ],
    "my_service",
)

# Bulk update
await bulk_update_async(
    session,
    users_table,
    [
        {"id": 1, "name": "Updated User 1"},
        {"id": 2, "name": "Updated User 2"},
    ],
    "id",
    "my_service",
)

# Bulk delete
await bulk_delete_async(
    session,
    users_table,
    [1, 2],
    "id",
    "my_service",
)
```

### Direct asyncpg Usage

```python
from common_lib.database import get_asyncpg_connection, bulk_insert_asyncpg

async def insert_users(users):
    async with get_asyncpg_connection("my_service") as conn:
        # Bulk insert with asyncpg
        await bulk_insert_asyncpg(
            conn,
            "users",
            ["name", "email"],
            [
                ["User 1", "user1@example.com"],
                ["User 2", "user2@example.com"],
            ],
            "my_service",
        )
```

## Monitoring

The monitoring component provides utilities for monitoring database performance. It provides comprehensive monitoring and metrics for database operations.

### Key Features

- Query performance tracking with metrics collection
- Transaction tracking
- Query analysis with execution plan inspection
- Database health checks

### Usage

```python
from common_lib.database import track_query_performance, track_transaction, analyze_query, check_database_health

# Track query performance
@track_query_performance("select", "users", "my_service")
def get_all_users(session):
    # Execute a query
    result = session.execute("SELECT id, name, email FROM users")
    
    # Return users
    return [
        {
            "id": row[0],
            "name": row[1],
            "email": row[2],
        }
        for row in result
    ]

# Track transaction performance
with track_transaction("my_service"):
    # Execute multiple queries
    users = get_all_users(session)
    
    # Analyze a query
    plan = analyze_query(
        session,
        "SELECT * FROM users WHERE id = :user_id",
        {"user_id": 1},
        "my_service",
    )
    
    # Check database health
    health = check_database_health(
        session,
        "my_service",
    )
```

### Asynchronous Usage

```python
from common_lib.database import async_track_query_performance, async_track_transaction, analyze_query_async, check_database_health_async

# Track query performance
@async_track_query_performance("select", "users", "my_service")
async def get_all_users(session):
    # Execute a query
    result = await session.execute("SELECT id, name, email FROM users")
    
    # Return users
    return [
        {
            "id": row[0],
            "name": row[1],
            "email": row[2],
        }
        for row in result
    ]

# Track transaction performance
async with async_track_transaction("my_service"):
    # Execute multiple queries
    users = await get_all_users(session)
    
    # Analyze a query
    plan = await analyze_query_async(
        session,
        "SELECT * FROM users WHERE id = :user_id",
        {"user_id": 1},
        "my_service",
    )
    
    # Check database health
    health = await check_database_health_async(
        session,
        "my_service",
    )
```

## Resilience Patterns

The database utilities include several resilience patterns to improve reliability:

### Circuit Breaker

The circuit breaker pattern prevents cascading failures by failing fast when the database is unavailable. It is automatically applied to all database operations.

### Retry with Backoff

The retry with backoff pattern automatically retries failed operations with exponential backoff. It is automatically applied to all database operations.

### Timeout Handling

The timeout handling pattern sets appropriate timeouts for database operations. It is automatically applied to all database operations.

### Connection Recycling

The connection recycling pattern recycles connections to prevent stale connections. It is automatically applied to all database operations.

## Configuration

The database utilities can be configured through environment variables or by passing configuration options directly to the functions:

### Environment Variables

```
# Database URL
DATABASE_URL=postgresql://user:password@localhost:5432/mydatabase

# Connection pool configuration
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=1800
DB_ECHO=false
DB_PREPARED_STATEMENT_CACHE_SIZE=256

# Resilience configuration
DB_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
DB_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60
DB_RETRY_MAX_ATTEMPTS=3
DB_RETRY_BACKOFF_FACTOR=2
DB_RETRY_MAX_BACKOFF=30
```

### Direct Configuration

```python
pool = get_connection_pool(
    "my_service",
    database_url="postgresql://user:password@localhost:5432/mydatabase",
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    echo=False,
    prepared_statement_cache_size=256,
)
```

## Best Practices

### Connection Pooling

- Use the connection pool for all database access
- Configure the pool size based on the number of CPU cores
- Set appropriate timeouts and recycling intervals
- Use the same service name for all operations within a service

### Prepared Statements

- Use prepared statements for all queries
- Use the `with_prepared_statement` decorator for functions that execute queries
- Use the `execute_prepared_statement` function to execute prepared statements
- Use the `prepare_asyncpg_statement` and `fetch_prepared_statement_asyncpg` functions for direct asyncpg access

### Bulk Operations

- Use bulk operations for inserting, updating, or deleting multiple rows
- Use appropriate chunk sizes for large datasets
- Use the `bulk_insert_asyncpg` function for high-performance bulk inserts with asyncpg

### Monitoring

- Use the `track_query_performance` decorator for functions that execute queries
- Use the `track_transaction` context manager for transactions
- Use the `analyze_query` function to analyze query execution plans
- Use the `check_database_health` function to check database health

## Performance Considerations

### Connection Pool

- The connection pool is optimized for performance with dynamic pool sizing based on CPU count
- Connection recycling and timeout handling prevent stale connections
- The pool is shared across all operations within a service

### Prepared Statements

- Prepared statements are cached for reuse, reducing parsing overhead
- The cache size is configurable to balance memory usage and performance
- Direct asyncpg access provides the highest performance for prepared statements

### Bulk Operations

- Bulk operations reduce network round-trips and transaction overhead
- Chunking large datasets prevents memory issues
- Direct asyncpg access provides the highest performance for bulk inserts

### Monitoring

- Monitoring adds minimal overhead to database operations
- Metrics are collected asynchronously to minimize impact on performance
- Query analysis and health checks can be run on demand

## Integration with Services

The database utilities are integrated with services through the following patterns:

### Direct Import

Services import the database utilities directly from common-lib:

```python
from common_lib.database import get_connection_pool, get_sync_db_session, get_async_db_session
```

### Service Template

The service template includes the database utilities by default:

```python
from common_lib.templates.service_template.database import Database
```

### Automatic Application

The `apply_database_optimization.py` script can automatically apply the database utilities to existing services:

```bash
python apply_database_optimization.py --service <service_name>
```

## Troubleshooting

### Connection Issues

- Check that the database URL is correct
- Verify that the database is running and accessible
- Check that the connection pool is properly configured
- Look for connection errors in the logs

### Performance Issues

- Check that prepared statements are being used
- Verify that bulk operations are being used for multiple rows
- Look for slow queries in the logs
- Use query analysis to identify performance bottlenecks

### Resilience Issues

- Check that the circuit breaker is properly configured
- Verify that retries are being attempted
- Look for timeout errors in the logs
- Check that connection recycling is working properly

## Examples

### Basic Usage

```python
from common_lib.database import get_connection_pool, get_sync_db_session, get_async_db_session

# Get a connection pool
pool = get_connection_pool("my_service")

# Use a synchronous session
with get_sync_db_session("my_service") as session:
    # Execute a query
    result = session.execute("SELECT 1")
    print(result.scalar())

# Use an asynchronous session
async with get_async_db_session("my_service") as session:
    # Execute a query
    result = await session.execute("SELECT 1")
    print(result.scalar())
```

### Complete Example

```python
from common_lib.database import (
    get_connection_pool,
    get_sync_db_session,
    get_async_db_session,
    with_prepared_statement,
    execute_prepared_statement,
    bulk_insert,
    track_query_performance,
    track_transaction,
    analyze_query,
    check_database_health,
)

# Get a connection pool
pool = get_connection_pool("my_service")

# Define a function that uses prepared statements
@with_prepared_statement("my_service", "get_user")
def get_user(session, user_id):
    # Execute a prepared statement
    result = execute_prepared_statement(
        session,
        "SELECT id, name, email FROM users WHERE id = :user_id",
        {"user_id": user_id},
        "my_service",
        "get_user",
    )
    
    # Get the first row
    row = result.fetchone()
    
    # Return user data
    if row:
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
        }
    
    return None

# Define a function that tracks query performance
@track_query_performance("select", "users", "my_service")
def get_all_users(session):
    # Execute a query
    result = session.execute("SELECT id, name, email FROM users")
    
    # Return users
    return [
        {
            "id": row[0],
            "name": row[1],
            "email": row[2],
        }
        for row in result
    ]

# Use the functions
with get_sync_db_session("my_service") as session:
    # Track transaction performance
    with track_transaction("my_service"):
        # Get a user
        user = get_user(session, 1)
        
        # Get all users
        users = get_all_users(session)
        
        # Analyze a query
        plan = analyze_query(
            session,
            "SELECT * FROM users WHERE id = :user_id",
            {"user_id": 1},
            "my_service",
        )
        
        # Check database health
        health = check_database_health(
            session,
            "my_service",
        )
        
        # Bulk insert
        bulk_insert(
            session,
            users_table,
            [
                {"name": "User 1", "email": "user1@example.com"},
                {"name": "User 2", "email": "user2@example.com"},
            ],
            "my_service",
        )
```

For more examples, see the [example usage](common-lib/examples/database/example_usage.py) file.