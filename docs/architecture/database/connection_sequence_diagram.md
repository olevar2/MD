# Database Connection Sequence Diagram

This diagram shows the sequence of operations for acquiring and releasing a database connection.

```mermaid
sequenceDiagram
    participant Client as Client Code
    participant Pool as DatabaseConnectionPool
    participant Engine as SQLAlchemy Engine
    participant Session as Database Session
    participant Metrics as Connection Metrics
    participant Resilience as Resilience Patterns

    Client->>Pool: get_sync_db_session() / get_async_db_session()
    activate Pool
    
    Pool->>Resilience: Apply resilience patterns
    activate Resilience
    
    Resilience->>Pool: Wrapped operation
    deactivate Resilience
    
    Pool->>Pool: Check if initialized
    
    alt Not initialized
        Pool->>Pool: initialize_sync() / initialize_async()
        Pool->>Engine: Create engine with optimized settings
    end
    
    Pool->>Metrics: Start timing connection acquisition
    activate Metrics
    
    Pool->>Engine: Create session
    activate Engine
    
    Engine->>Session: Create new session
    activate Session
    
    Engine-->>Pool: Return session
    deactivate Engine
    
    Pool->>Metrics: Record connection acquisition time
    deactivate Metrics
    
    Pool->>Metrics: Update pool usage metrics
    
    Pool-->>Client: Return session (context manager)
    deactivate Pool
    
    Note over Client, Session: Client uses session for database operations
    
    Client->>Session: Exit context manager
    
    Session->>Session: Commit or rollback transaction
    
    Session->>Engine: Return session to pool
    deactivate Session
    
    Engine->>Pool: Update pool metrics
    
    Pool->>Metrics: Update pool usage metrics
```

## Sequence Description

1. **Client Requests Session**: The client code requests a database session using `get_sync_db_session()` or `get_async_db_session()`.

2. **Apply Resilience Patterns**: The operation is wrapped with resilience patterns like retries and circuit breakers.

3. **Check Initialization**: The connection pool checks if it has been initialized.

4. **Initialize if Needed**: If not initialized, the pool creates the database engine with optimized settings.

5. **Acquire Connection**: The pool acquires a connection from the engine, measuring the time it takes.

6. **Create Session**: The engine creates a new session and returns it to the pool.

7. **Return Session**: The pool returns the session to the client as a context manager.

8. **Client Uses Session**: The client uses the session for database operations.

9. **Session Cleanup**: When the client exits the context manager, the session is committed or rolled back.

10. **Return to Pool**: The session is returned to the connection pool.

11. **Update Metrics**: Connection pool metrics are updated to reflect the current state.