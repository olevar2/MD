# Service Communication Guidelines

## Overview

This document outlines the standards and best practices for service-to-service communication in the Forex Trading Platform.

## Core Principles

1. **Interface-Based Communication**
   - Define interfaces in common-lib
   - Implement adapters in services
   - Use dependency injection
   - Follow interface evolution rules

2. **Event-Driven Communication**
   - Use event bus for async operations
   - Define clear event schemas
   - Implement proper error handling
   - Maintain event versioning

3. **Isolation and Dependencies**
   - No direct service dependencies
   - Minimize synchronous communication
   - Use circuit breakers
   - Implement fallbacks

## Implementation Patterns

### 1. Interface Definitions
```python
# In common-lib
from abc import ABC, abstractmethod
from typing import Optional

class IServiceAdapter(ABC):
    """Base interface for service communication"""
    
    @abstractmethod
    async def process_request(self, request: Request) -> Response:
        """Process a service request"""
        pass
        
    @abstractmethod
    async def health_check(self) -> bool:
        """Check service health"""
        pass
```

### 2. Event Schemas
```python
# In common-lib
from pydantic import BaseModel
from datetime import datetime

class ServiceEvent(BaseModel):
    """Base class for service events"""
    event_id: str
    event_type: str
    timestamp: datetime
    version: str
    payload: dict
```

### 3. Error Handling
```python
# In common-lib
class ServiceError(Exception):
    """Base class for service errors"""
    def __init__(self, message: str, code: str, retryable: bool = False):
        self.code = code
        self.retryable = retryable
        super().__init__(message)
```

## Communication Patterns

### 1. Synchronous Communication
- Use for immediate responses
- Implement timeouts
- Handle failures gracefully
- Use circuit breakers

```python
async def call_service(request: Request) -> Response:
    with CircuitBreaker(name="service_call"):
        try:
            return await service.process_request(request)
        except ServiceError as e:
            handle_error(e)
        except Exception as e:
            handle_unexpected_error(e)
```

### 2. Asynchronous Communication
- Use for non-immediate needs
- Implement retry logic
- Handle out-of-order events
- Maintain idempotency

```python
async def publish_event(event: ServiceEvent) -> None:
    try:
        await event_bus.publish(
            topic="service.events",
            event=event,
            retry_policy=RetryPolicy.DEFAULT
        )
    except PublishError as e:
        handle_publish_error(e)
```

### 3. Event Handling
- Validate event schema
- Handle version compatibility
- Process idempotently
- Track event flow

```python
async def handle_event(event: ServiceEvent) -> None:
    if not is_valid_version(event.version):
        handle_version_mismatch(event)
        return
        
    if await is_duplicate_event(event.event_id):
        logger.info(f"Duplicate event {event.event_id}")
        return
        
    try:
        await process_event(event)
        await mark_event_processed(event.event_id)
    except EventProcessingError as e:
        handle_processing_error(e)
```

## Best Practices

### 1. Interface Evolution
- Maintain backward compatibility
- Use interface versioning
- Provide migration paths
- Document breaking changes

### 2. Error Handling
- Define error categories
- Implement retry strategies
- Use circuit breakers
- Log appropriately

### 3. Monitoring
- Track latency metrics
- Monitor error rates
- Trace request flow
- Alert on issues

### 4. Testing
- Test interface compliance
- Verify error handling
- Check event processing
- Validate retries

## Anti-Patterns to Avoid

1. **Direct Dependencies**
```python
# BAD: Direct import from another service
from other_service.module import function

# GOOD: Use interface from common-lib
from common_lib.interfaces import IService
```

2. **Tight Coupling**
```python
# BAD: Direct knowledge of other service
if other_service.is_available():
    result = other_service.process()

# GOOD: Use interface and handle unavailability
try:
    result = await service.process()
except ServiceUnavailableError:
    result = fallback_process()
```

3. **Missing Error Handling**
```python
# BAD: No error handling
result = service.call()

# GOOD: Proper error handling
try:
    result = await service.call()
except ServiceError as e:
    handle_error(e)
except Exception as e:
    handle_unexpected_error(e)
```

## Monitoring Requirements

### 1. Metrics
- Request latency
- Error rates
- Circuit breaker status
- Event processing times

### 2. Logging
- Request/response pairs
- Error details
- Event flow
- Performance issues

### 3. Tracing
- Request context
- Service hops
- Error propagation
- Timing breakdown

## Security Considerations

### 1. Authentication
- Use service accounts
- Implement mTLS
- Rotate credentials
- Audit access

### 2. Authorization
- Define service roles
- Check permissions
- Log access attempts
- Review regularly

### 3. Data Protection
- Encrypt in transit
- Validate input
- Sanitize output
- Handle sensitive data
