# Correlation ID Implementation Guide

This document provides a guide for implementing correlation ID propagation in services of the Forex Trading Platform.

## Overview

Correlation IDs are unique identifiers that are propagated across service boundaries to track requests and events through the system. They are essential for:

1. **Distributed Tracing**: Tracking requests across multiple services
2. **Error Correlation**: Linking errors across services to a single request
3. **Log Correlation**: Connecting log entries from different services
4. **Debugging**: Identifying the flow of a request through the system

## Implementation Status

The following services have been updated to use standardized correlation ID propagation:

| Service | Status | Notes |
|---------|--------|-------|
| analysis-engine-service | ⚠️ | Main file not found, logging configuration updated |
| portfolio-management-service | ✅ | Fully implemented with middleware and logging |
| risk-management-service | ✅ | Fully implemented with middleware and logging |
| monitoring-alerting-service | ✅ | Fully implemented with middleware and logging |
| ml-integration-service | ✅ | Fully implemented with middleware and logging |
| ml-workbench-service | ✅ | Fully implemented with middleware and logging |
| data-pipeline-service | ✅ | Fully implemented with middleware and logging |
| feature-store-service | ✅ | Fully implemented with middleware and logging |
| trading-gateway-service | ✅ | Fully implemented with middleware and logging |

All services now have standardized correlation ID propagation implemented, with the exception of the analysis-engine-service which requires further investigation to locate the main application file.

## Implementation Details

### 1. HTTP Requests

All services use the `FastAPICorrelationIdMiddleware` to handle correlation IDs in HTTP requests:

```python
from fastapi import FastAPI
from common_lib.correlation import FastAPICorrelationIdMiddleware

app = FastAPI()
app.add_middleware(FastAPICorrelationIdMiddleware)
```

The middleware:
1. Extracts the correlation ID from the `X-Correlation-ID` header if present
2. Generates a new correlation ID if not present
3. Sets the correlation ID in the request state
4. Sets the correlation ID in thread-local and async context
5. Adds the correlation ID to response headers
6. Clears the correlation ID after the request is processed

### 2. Service-to-Service Communication

All service clients use the `BaseServiceClient` with correlation ID support:

```python
from common_lib.clients import BaseServiceClient, ClientConfig

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

### 3. Event-Based Communication

All event producers and consumers use the correlation ID utilities:

```python
from common_lib.correlation import (
    add_correlation_to_event_metadata,
    extract_correlation_id_from_event,
    with_event_correlation,
    with_async_event_correlation
)

# Event producer
def publish_event(event_data):
    # Add correlation ID to metadata
    metadata = add_correlation_to_event_metadata({})

    # Create the event
    event = {
        "event_type": "example.event",
        "data": event_data,
        "metadata": metadata
    }

    # Publish the event
    event_bus.publish(event)

# Event consumer
@with_event_correlation
def handle_event(event):
    # The correlation ID is extracted from the event
    # and set in the current context
    correlation_id = get_correlation_id()

    # Handle the event
    # ...
```

### 4. Logging

All services use a standardized logging configuration that includes correlation IDs:

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
```

## Usage Examples

### 1. HTTP Request Handling

```python
from fastapi import FastAPI, Request
from common_lib.correlation import get_correlation_id

app = FastAPI()
app.add_middleware(FastAPICorrelationIdMiddleware)

@app.get("/api/resources/{resource_id}")
async def get_resource(resource_id: str, request: Request):
    # Get the correlation ID from the request
    correlation_id = request.state.correlation_id

    # Or get it from the current context
    correlation_id = get_correlation_id()

    # Use the correlation ID in logging
    logger.info(f"Processing request for resource {resource_id}")

    # Return the resource
    return {
        "id": resource_id,
        "name": f"Resource {resource_id}",
        "correlation_id": correlation_id
    }
```

### 2. Service-to-Service Communication

```python
from fastapi import FastAPI, Request
from common_lib.correlation import get_correlation_id
from common_lib.clients import BaseServiceClient, ClientConfig

app = FastAPI()
app.add_middleware(FastAPICorrelationIdMiddleware)

# Create a client for another service
other_service_client = BaseServiceClient(
    ClientConfig(
        base_url="https://other-service/api",
        service_name="other-service"
    )
)

@app.get("/api/proxy/{resource_id}")
async def proxy_resource(resource_id: str, request: Request):
    # Get the correlation ID from the request
    correlation_id = request.state.correlation_id

    # Create a client with the correlation ID
    client = other_service_client.with_correlation_id(correlation_id)

    # Call the other service
    result = await client.get(f"resources/{resource_id}")

    # Return the result
    return {
        "proxied_resource": result,
        "correlation_id": correlation_id
    }
```

### 3. Event-Based Communication

```python
from common_lib.correlation import (
    get_correlation_id,
    add_correlation_to_event_metadata,
    with_event_correlation
)

# Event producer
def create_resource(resource_data):
    # Create a resource
    resource_id = str(uuid.uuid4())
    resource = {
        "id": resource_id,
        "name": resource_data.get("name", f"Resource {resource_id}")
    }

    # Add correlation ID to metadata
    metadata = add_correlation_to_event_metadata({})

    # Create the event
    event = {
        "event_type": "resource.created",
        "data": resource,
        "metadata": metadata
    }

    # Publish the event
    event_bus.publish(event)

    return resource

# Event consumer
@with_event_correlation
def handle_resource_created(event):
    # The correlation ID is extracted from the event
    # and set in the current context
    correlation_id = get_correlation_id()

    # Log the event
    logger.info(f"Handling resource.created event: {event['data']['id']}")

    # Process the event
    # ...
```

## Testing

### Unit Tests

Comprehensive unit tests have been added to verify all aspects of the correlation ID implementation:

1. `common-lib/tests/correlation/test_correlation_id.py`: Tests core correlation ID functionality
2. `common-lib/tests/correlation/test_middleware.py`: Tests middleware behavior
3. `common-lib/tests/correlation/test_client_correlation.py`: Tests client-side propagation
4. `common-lib/tests/correlation/test_event_correlation.py`: Tests event-based propagation

To run the unit tests:

```bash
cd common-lib
pytest tests/correlation/
```

### Integration Tests

Integration tests have been added to verify correlation ID propagation across service boundaries:

1. `common-lib/tests/integration/test_correlation_id_propagation.py`: Tests correlation ID propagation in HTTP requests
2. `common-lib/tests/integration/test_event_correlation.py`: Tests correlation ID propagation in event-based communication

To run the integration tests (requires external dependencies like Kafka):

```bash
cd common-lib
pytest tests/integration/test_correlation_id_propagation.py
pytest tests/integration/test_event_correlation.py
```

## Troubleshooting

### Missing Correlation ID in Logs

If correlation IDs are missing in logs, check:

1. The logging configuration includes the correlation ID format: `[%(correlation_id)s]`
2. The logger has the `CorrelationFilter` added
3. The correlation ID is set in the current context

### Missing Correlation ID in Service-to-Service Communication

If correlation IDs are not propagated in service-to-service communication, check:

1. The service client is created with `BaseServiceClient`
2. The service client is used with `with_correlation_id` or the correlation ID is set in the current context
3. The service being called has the `FastAPICorrelationIdMiddleware` added

### Missing Correlation ID in Event-Based Communication

If correlation IDs are not propagated in event-based communication, check:

1. The event producer adds correlation ID to metadata with `add_correlation_to_event_metadata`
2. The event consumer uses the `with_event_correlation` or `with_async_event_correlation` decorator
3. The correlation ID is extracted from the event with `extract_correlation_id_from_event`
