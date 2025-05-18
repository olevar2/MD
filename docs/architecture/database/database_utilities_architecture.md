# Database Utilities Architecture

This document provides a comprehensive overview of the database utilities architecture in the forex trading platform.

## Overview

The database utilities in the common-lib package provide standardized database access, optimization, and monitoring capabilities for all services in the forex trading platform. These utilities are designed to improve performance, reliability, and observability of database operations.

## Key Components

The database utilities consist of four main components:

1. **Connection Pool**: Manages database connections for services
2. **Prepared Statements**: Provides utilities for working with prepared statements
3. **Bulk Operations**: Provides utilities for efficient bulk data operations
4. **Monitoring**: Provides utilities for monitoring database performance

## Component Architecture

The component architecture diagram shows the relationships between the different components of the database utilities:

![Component Diagram](database_component_diagram.md)

### Connection Pool

The connection pool component manages database connections for services. It provides:

- Optimized connection pooling for both synchronous and asynchronous database operations
- Dynamic pool sizing based on CPU count
- Connection recycling and timeout handling
- Metrics for pool size, usage, and connection time

The sequence diagram for connection acquisition and release:

![Connection Sequence Diagram](connection_sequence_diagram.md)

### Prepared Statements

The prepared statements component provides utilities for working with prepared statements. It provides:

- Caching for prepared statements to improve performance
- Support for both SQLAlchemy ORM and direct asyncpg connections
- Metrics for prepared statement execution time and cache size

The sequence diagram for prepared statement execution:

![Prepared Statement Sequence Diagram](prepared_statement_sequence_diagram.md)

### Bulk Operations

The bulk operations component provides utilities for efficient bulk data operations. It provides:

- Bulk insert, update, and delete operations
- Support for both SQLAlchemy ORM and direct asyncpg connections
- Chunking for large datasets
- Metrics for bulk operation performance

The sequence diagram for bulk operations:

![Bulk Operations Sequence Diagram](bulk_operations_sequence_diagram.md)

### Monitoring

The monitoring component provides utilities for monitoring database performance. It provides:

- Query performance tracking with metrics collection
- Transaction tracking
- Query analysis with execution plan inspection
- Database health checks

The sequence diagram for monitoring:

![Monitoring Sequence Diagram](monitoring_sequence_diagram.md)

## Integration with Services

The database utilities are integrated with services through the following patterns:

1. **Direct Import**: Services import the database utilities directly from common-lib
2. **Service Template**: The service template includes the database utilities by default
3. **Automatic Application**: The `apply_database_optimization.py` script can automatically apply the database utilities to existing services

## Performance Considerations

The database utilities are designed with performance in mind:

1. **Connection Pooling**: Optimized connection pooling reduces connection overhead
2. **Prepared Statements**: Caching prepared statements reduces parsing overhead
3. **Bulk Operations**: Bulk operations reduce network round-trips and transaction overhead
4. **Monitoring**: Performance monitoring helps identify and fix bottlenecks

## Resilience Patterns

The database utilities include several resilience patterns:

1. **Circuit Breaker**: Prevents cascading failures by failing fast when the database is unavailable
2. **Retry with Backoff**: Automatically retries failed operations with exponential backoff
3. **Timeout Handling**: Sets appropriate timeouts for database operations
4. **Connection Recycling**: Recycles connections to prevent stale connections

## Metrics

The database utilities collect the following metrics:

1. **Connection Pool**:
   - `db_pool_size`: Size of the database connection pool
   - `db_pool_usage`: Number of connections in use from the pool
   - `db_connection_time_seconds`: Time spent acquiring a database connection

2. **Prepared Statements**:
   - `prepared_stmt_cache_size`: Size of the prepared statement cache
   - `prepared_stmt_execution_time_seconds`: Time spent executing prepared statements
   - `prepared_stmt_executions_total`: Total number of prepared statement executions

3. **Bulk Operations**:
   - `bulk_operation_time_seconds`: Time spent on bulk operations
   - `bulk_operation_rows_total`: Total number of rows processed by bulk operations
   - `bulk_operation_executions_total`: Total number of bulk operations

4. **Query Performance**:
   - `db_query_time_seconds`: Time spent executing database queries
   - `db_query_count`: Number of database queries executed
   - `db_slow_query_count`: Number of slow database queries
   - `db_deadlock_count`: Number of deadlocks encountered
   - `db_connection_errors`: Number of connection errors

## Configuration

The database utilities can be configured through environment variables or by passing configuration options directly to the functions:

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

## Best Practices

1. **Use Connection Pooling**: Always use the connection pool for database access
2. **Use Prepared Statements**: Use prepared statements for frequently executed queries
3. **Use Bulk Operations**: Use bulk operations for inserting, updating, or deleting multiple rows
4. **Monitor Performance**: Use the monitoring utilities to track database performance
5. **Handle Errors**: Handle database errors appropriately using the provided resilience patterns
6. **Configure Appropriately**: Configure the database utilities appropriately for your service's needs

## Conclusion

The database utilities in the common-lib package provide a comprehensive set of tools for optimizing database access, improving reliability, and monitoring performance. By using these utilities, services can achieve better performance, reliability, and observability for their database operations.