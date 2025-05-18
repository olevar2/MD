# Database Utilities

This package provides standardized database utilities for the forex trading platform, including connection pooling, prepared statements, bulk operations, and monitoring.

## Features

- **Connection Pooling**: Standardized connection pool management for both synchronous and asynchronous database connections.
- **Prepared Statements**: Utilities for working with prepared statements to improve performance and security.
- **Bulk Operations**: Efficient bulk insert, update, and delete operations.
- **Monitoring**: Comprehensive database monitoring and metrics.

## Usage

### Connection Pooling

```python
from common_lib.database import get_connection_pool, get_sync_db_session, get_async_db_session

# Get a connection pool for a service
pool = get_connection_pool("my_service")

# Use a synchronous session
with get_sync_db_session("my_service") as session:
    # Execute a query
    result = session.execute("SELECT 1")
    print(f"Query result: {result.scalar()}")

# Use an asynchronous session
async with get_async_db_session("my_service") as session:
    # Execute a query
    result = await session.execute("SELECT 1")
    print(f"Query result: {result.scalar()}")
```

### Prepared Statements

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

### Bulk Operations

```python
from common_lib.database import bulk_insert, bulk_update, bulk_delete

# Define a table
from sqlalchemy import Table, Column, Integer, String, MetaData
metadata = MetaData()
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String),
    Column("email", String),
)

# Bulk insert
users = [
    {"id": 1, "name": "User 1", "email": "user1@example.com"},
    {"id": 2, "name": "User 2", "email": "user2@example.com"},
]

bulk_insert(
    session,
    users_table,
    users,
    "my_service",
)

# Bulk update
updated_users = [
    {"id": 1, "name": "Updated User 1", "email": "updated1@example.com"},
    {"id": 2, "name": "Updated User 2", "email": "updated2@example.com"},
]

bulk_update(
    session,
    users_table,
    updated_users,
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

### Monitoring

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

## Configuration

The database utilities can be configured using environment variables or by passing configuration options directly to the functions.

### Environment Variables

- `{SERVICE_NAME}_DATABASE_URL`: Database URL for the service.
- `{SERVICE_NAME}_DB_POOL_SIZE`: Connection pool size for the service.
- `{SERVICE_NAME}_DB_MAX_OVERFLOW`: Maximum number of connections that can be created beyond the pool size.
- `{SERVICE_NAME}_DB_POOL_TIMEOUT`: Number of seconds to wait before timing out when getting a connection from the pool.
- `{SERVICE_NAME}_DB_POOL_RECYCLE`: Number of seconds after which a connection is recycled.
- `{SERVICE_NAME}_DB_ECHO`: Whether to echo SQL statements to the console.
- `{SERVICE_NAME}_DB_PREPARED_STATEMENT_CACHE_SIZE`: Size of the prepared statement cache.

### Direct Configuration

```python
# Get a connection pool with custom configuration
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

## Metrics

The database utilities collect the following metrics:

- **Connection Pool**:
  - `db_pool_size`: Size of the database connection pool.
  - `db_pool_usage`: Number of connections in use from the pool.
  - `db_connection_time_seconds`: Time spent acquiring a database connection.

- **Prepared Statements**:
  - `prepared_stmt_cache_size`: Size of the prepared statement cache.
  - `prepared_stmt_execution_time_seconds`: Execution time of prepared statements.
  - `prepared_stmt_executions_total`: Total number of prepared statement executions.

- **Bulk Operations**:
  - `bulk_operation_time_seconds`: Execution time of bulk database operations.
  - `bulk_operation_rows`: Number of rows affected by bulk database operations.
  - `bulk_operations_total`: Total number of bulk database operations.

- **Monitoring**:
  - `db_query_time_seconds`: Database query execution time.
  - `db_query_count_total`: Total number of database queries.
  - `db_slow_query_count_total`: Total number of slow database queries.
  - `db_connection_errors_total`: Total number of database connection errors.
  - `db_deadlock_count_total`: Total number of database deadlocks.
  - `db_row_count`: Number of rows returned by database queries.
  - `db_pool_wait_time_seconds`: Time spent waiting for a database connection from the pool.
  - `db_transaction_time_seconds`: Database transaction execution time.

## Examples

See the `examples/database` directory for complete examples of how to use the database utilities.