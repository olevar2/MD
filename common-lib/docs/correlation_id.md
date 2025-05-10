# Correlation ID Utility

This document provides a comprehensive guide to using the correlation ID utility in the Forex Trading Platform.

## Overview

Correlation IDs are unique identifiers that are propagated across service boundaries to track requests and events through the system. They are essential for:

1. **Distributed Tracing**: Tracking requests across multiple services
2. **Error Correlation**: Linking errors across services to a single request
3. **Log Correlation**: Connecting log entries from different services
4. **Debugging**: Identifying the flow of a request through the system

The correlation ID utility provides a standardized implementation for correlation ID generation, propagation, and retrieval across different communication patterns.

## Key Features

1. **Consistent Correlation ID Generation**: Generate UUIDs for correlation IDs
2. **Thread-local and Async-context Storage**: Store correlation IDs in both synchronous and asynchronous code
3. **Automatic Propagation**: Propagate correlation IDs between services
4. **Support for Multiple Communication Patterns**: HTTP, messaging, and event-based communication
5. **Middleware Integration**: FastAPI middleware for automatic correlation ID handling
6. **Service Client Integration**: BaseServiceClient integration for correlation ID propagation
7. **Event Correlation**: Support for correlation IDs in event-based communication
8. **Logging Integration**: Add correlation IDs to log messages

## Installation

The correlation ID utility is part of the `common-lib` package and is available to all services in the Forex Trading Platform.

## Usage

### Basic Usage

```python
from common_lib.correlation import (
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id
)

# Generate a new correlation ID
correlation_id = generate_correlation_id()

# Set the correlation ID in the current context
set_correlation_id(correlation_id)

# Get the correlation ID from the current context
current_id = get_correlation_id()

# Clear the correlation ID from the current context
clear_correlation_id()
```

### Context Managers

```python
from common_lib.correlation import (
    correlation_id_context,
    async_correlation_id_context
)

# Synchronous context manager
with correlation_id_context("my-correlation-id"):
    # Code in this block has access to the correlation ID
    current_id = get_correlation_id()  # "my-correlation-id"

# Asynchronous context manager
async with async_correlation_id_context("my-correlation-id"):
    # Code in this block has access to the correlation ID
    current_id = get_correlation_id()  # "my-correlation-id"
```

### Decorators

```python
from common_lib.correlation import (
    with_correlation_id,
    with_async_correlation_id
)

# Synchronous decorator
@with_correlation_id
def process_data(data):
    # This function will have a correlation ID available
    correlation_id = get_correlation_id()
    # Process data...

# Asynchronous decorator
@with_async_correlation_id
async def process_data_async(data):
    # This function will have a correlation ID available
    correlation_id = get_correlation_id()
    # Process data...
```

### FastAPI Middleware

```python
from fastapi import FastAPI, Request
from common_lib.correlation import FastAPICorrelationIdMiddleware

app = FastAPI()
app.add_middleware(FastAPICorrelationIdMiddleware)

@app.get("/api/resources/{resource_id}")
async def get_resource(resource_id: str, request: Request):
    # The correlation ID is automatically set by the middleware
    # and available in the request state
    correlation_id = request.state.correlation_id
    
    # It's also available in the current context
    correlation_id = get_correlation_id()
    
    # Process the request...
```

### Service Client Integration

```python
from common_lib.clients import BaseServiceClient, ClientConfig
from common_lib.correlation import get_correlation_id

# Create a client
client = BaseServiceClient(
    ClientConfig(
        base_url="https://example.com/api",
        service_name="example-service"
    )
)

# The client automatically adds correlation ID to headers
result = await client.get("resources/123")

# Create a client with a specific correlation ID
correlation_id = "my-correlation-id"
client_with_correlation = client.with_correlation_id(correlation_id)
result = await client_with_correlation.get("resources/123")
```

### Event Correlation

```python
from common_lib.correlation import (
    add_correlation_to_event_metadata,
    extract_correlation_id_from_event,
    with_event_correlation,
    with_async_event_correlation
)

# Create an event with correlation ID
metadata = add_correlation_to_event_metadata({})
event = {
    "event_type": "example_event",
    "data": {"key": "value"},
    "metadata": metadata
}

# Extract correlation ID from event
correlation_id = extract_correlation_id_from_event(event)

# Handle events with correlation ID
@with_event_correlation
def handle_event(event):
    # The correlation ID is extracted from the event
    # and set in the current context
    correlation_id = get_correlation_id()
    # Handle the event...

# Handle events asynchronously with correlation ID
@with_async_event_correlation
async def handle_event_async(event):
    # The correlation ID is extracted from the event
    # and set in the current context
    correlation_id = get_correlation_id()
    # Handle the event...
```

### Logging Integration

```python
import logging
from common_lib.correlation import get_correlation_id

# Create a logging filter that adds correlation ID to log records
class CorrelationFilter(logging.Filter):
    def filter(self, record):
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id or "no-correlation-id"
        return True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] %(message)s'
)

# Create a logger and add the correlation filter
logger = logging.getLogger("my-logger")
logger.addFilter(CorrelationFilter())

# Log messages with correlation ID
logger.info("This log message includes the correlation ID")
```

## Best Practices

1. **Always Propagate Correlation IDs**: Ensure correlation IDs are propagated across all service boundaries
2. **Use Middleware**: Use the FastAPI middleware to automatically handle correlation IDs in HTTP requests
3. **Use Service Clients**: Use the BaseServiceClient to automatically propagate correlation IDs in service-to-service communication
4. **Include in Logs**: Always include correlation IDs in log messages
5. **Use in Events**: Always include correlation IDs in event metadata
6. **Use Context Managers**: Use context managers to ensure correlation IDs are properly managed
7. **Use Decorators**: Use decorators to ensure correlation IDs are available in functions

## Reference

### Core Functions

- `generate_correlation_id()`: Generate a new correlation ID
- `get_correlation_id()`: Get the correlation ID from the current context
- `set_correlation_id(correlation_id)`: Set the correlation ID in the current context
- `clear_correlation_id()`: Clear the correlation ID from the current context

### Context Managers

- `correlation_id_context(correlation_id=None)`: Synchronous context manager for correlation IDs
- `async_correlation_id_context(correlation_id=None)`: Asynchronous context manager for correlation IDs

### Decorators

- `with_correlation_id`: Decorator for synchronous functions
- `with_async_correlation_id`: Decorator for asynchronous functions
- `with_event_correlation`: Decorator for event handlers
- `with_async_event_correlation`: Decorator for asynchronous event handlers

### Middleware

- `FastAPICorrelationIdMiddleware`: FastAPI middleware for correlation IDs
- `create_correlation_id_middleware(framework)`: Factory function for creating middleware

### Service Client Integration

- `add_correlation_id_to_headers(headers, correlation_id=None)`: Add correlation ID to headers
- `with_correlation_headers`: Decorator for client methods
- `with_async_correlation_headers`: Decorator for async client methods
- `ClientCorrelationMixin`: Mixin for adding correlation ID support to clients

### Event Correlation

- `add_correlation_to_event_metadata(metadata, correlation_id=None, causation_id=None)`: Add correlation ID to event metadata
- `extract_correlation_id_from_event(event)`: Extract correlation ID from event
- `with_event_correlation`: Decorator for event handlers
- `with_async_event_correlation`: Decorator for async event handlers

### Constants

- `CORRELATION_ID_HEADER`: The header name for correlation IDs (default: "X-Correlation-ID")
