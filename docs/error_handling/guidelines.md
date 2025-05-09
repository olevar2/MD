# Forex Trading Platform Error Handling Guidelines

## Introduction

Proper error handling is critical for the Forex Trading Platform's stability, reliability, and user experience. As a financial system handling real-time trading operations, the impact of unhandled or improperly handled errors can be severe:

- **Financial Impact**: Trading errors can lead to financial losses for users and the platform
- **Regulatory Compliance**: Financial systems must maintain audit trails of errors for regulatory compliance
- **User Experience**: Poor error handling leads to frustrated users and reduced platform adoption
- **System Stability**: Unhandled errors can cascade into system-wide failures

This document provides comprehensive guidelines for implementing error handling across all services in the Forex Trading Platform. By following these guidelines, developers will ensure:

- **Consistency**: Uniform error handling across all services
- **Actionability**: Error messages that help users and developers take appropriate action
- **Observability**: Proper logging and monitoring of errors for debugging and analysis
- **Resilience**: Systems that can recover from failures gracefully

## Error Handling Principles

### 1. Domain-Specific Exceptions

Use exceptions that align with business concepts rather than generic technical exceptions. This makes errors more meaningful and actionable.

**Good Example**:
```python
raise InsufficientBalanceError(
    message="Insufficient balance to execute trade",
    account_id="ACC123",
    required_amount=1000.00,
    available_balance=500.00
)
```

**Bad Example**:
```python
raise ValueError("Not enough money")
```

### 2. Fail Fast

Detect and report errors as early as possible to prevent cascading failures and data corruption.

**Good Example**:
```python
def execute_trade(trade_request: TradeRequest):
    # Validate request before processing
    if not trade_request.is_valid():
        raise OrderValidationError(
            message="Invalid trade request",
            details=trade_request.validation_errors()
        )
    # Proceed with valid request...
```

### 3. Graceful Degradation

Design systems to continue functioning with reduced capabilities when components fail.

**Good Example**:
```python
try:
    real_time_data = market_data_service.get_real_time_data(symbol)
    return real_time_data
except ServiceUnavailableError:
    # Fall back to slightly delayed data if real-time service is down
    logger.warning("Real-time data service unavailable, using delayed data")
    return delayed_data_service.get_delayed_data(symbol)
```

### 4. Actionable Error Messages

Provide error messages that are helpful for both users and developers.

**User-Facing Example**:
```json
{
  "error": {
    "code": "INSUFFICIENT_BALANCE",
    "message": "Your account balance is insufficient for this trade",
    "details": {
      "required_amount": 1000.00,
      "available_balance": 500.00
    }
  }
}
```

**Developer-Facing Log**:
```
ERROR [portfolio-service] InsufficientBalanceError: Account ACC123 has insufficient balance for trade TR456
  correlation_id: 7f8d9e10-a1b2-c3d4-e5f6
  error_code: INSUFFICIENT_BALANCE
  details: {
    "account_id": "ACC123",
    "trade_id": "TR456",
    "required_amount": 1000.00,
    "available_balance": 500.00,
    "currency": "USD"
  }
```

### 5. Consistent Error Handling

Apply consistent error handling patterns across all services.

- Use the same exception hierarchy
- Format error responses consistently
- Log errors with the same structure
- Propagate correlation IDs across service boundaries

### 6. Proper Error Propagation

Propagate errors to the appropriate layer for handling.

**Good Example**:
```python
@async_with_exception_handling
async def get_portfolio(self, portfolio_id: str) -> Portfolio:
    try:
        return await self.repository.get_portfolio(portfolio_id)
    except DataFetchError as e:
        # Translate to domain-specific exception
        raise PortfolioNotFoundError(
            message=f"Portfolio {portfolio_id} not found",
            portfolio_id=portfolio_id,
            details={"original_error": str(e)}
        )
```

### 7. Error Correlation

Use correlation IDs to track errors across service boundaries.

**Good Example**:
```python
async def call_downstream_service(self, request_data, correlation_id=None):
    correlation_id = correlation_id or generate_correlation_id()
    try:
        return await self.client.make_request(
            request_data,
            headers={"X-Correlation-ID": correlation_id}
        )
    except Exception as e:
        logger.error(
            f"Error calling downstream service: {str(e)}",
            extra={"correlation_id": correlation_id}
        )
        raise
```

## Exception Hierarchy

The Forex Trading Platform uses a standardized exception hierarchy with `ForexTradingPlatformError` as the base class. All custom exceptions should inherit from this class or its subclasses.

```
ForexTradingPlatformError (Base Exception)
├── ConfigurationError
│   ├── ConfigNotFoundError
│   └── ConfigValidationError
├── DataError
│   ├── DataValidationError
│   ├── DataFetchError
│   ├── DataStorageError
│   └── DataTransformationError
├── ServiceError
│   ├── ServiceUnavailableError
│   ├── ServiceTimeoutError
│   └── ServiceAuthenticationError
├── TradingError
│   ├── OrderExecutionError
│   ├── PositionError
│   └── RiskLimitError
├── ModelError
│   ├── ModelTrainingError
│   ├── ModelPredictionError
│   └── ModelLoadError
├── SecurityError
│   ├── AuthenticationError
│   └── AuthorizationError
└── ResilienceError
    ├── CircuitBreakerOpenError
    ├── RetryExhaustedError
    ├── TimeoutError
    └── BulkheadFullError
```

### Service-Specific Exceptions

Each service should define its own domain-specific exceptions that extend the common-lib exceptions. For example:

**Portfolio Management Service**:
```
PortfolioManagementError (extends ForexTradingPlatformError)
├── PortfolioNotFoundError
├── PositionNotFoundError
├── InsufficientBalanceError
├── PortfolioOperationError
├── AccountReconciliationError
└── TaxCalculationError
```

**Trading Gateway Service**:
```
TradingGatewayError (extends ForexTradingPlatformError)
├── BrokerConnectionError
├── OrderValidationError
└── MarketDataError
```

### Creating New Exception Types

When creating new exception types:

1. Extend the appropriate base class from the hierarchy
2. Use a descriptive name that reflects the error domain
3. Include appropriate constructor parameters for context
4. Provide a default error message and error code

**Example**:
```python
class PortfolioNotFoundError(PortfolioManagementError):
    """Exception raised when a portfolio is not found."""
    
    def __init__(
        self,
        message: str = "Portfolio not found",
        portfolio_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if portfolio_id:
            details["portfolio_id"] = portfolio_id
            
        super().__init__(message, "PORTFOLIO_NOT_FOUND_ERROR", details)
```

## Error Response Structure

All error responses should follow a standardized format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "key1": "value1",
      "key2": "value2"
    },
    "correlation_id": "unique-correlation-id",
    "timestamp": "2023-05-22T12:34:56.789Z",
    "service": "service-name"
  }
}
```

### Required Fields

- **code**: A unique identifier for the error type (e.g., "INSUFFICIENT_BALANCE")
- **message**: A human-readable description of the error
- **details**: Additional context about the error (optional)
- **correlation_id**: A unique identifier for tracking the error across services
- **timestamp**: The time when the error occurred (ISO 8601 format)
- **service**: The name of the service that generated the error

### HTTP Status Code Mapping

Exceptions should be mapped to appropriate HTTP status codes:

| HTTP Status | Exception Types |
|-------------|-----------------|
| 400 Bad Request | DataValidationError, ConfigValidationError, OrderValidationError |
| 401 Unauthorized | AuthenticationError, ServiceAuthenticationError |
| 403 Forbidden | AuthorizationError, InsufficientBalanceError |
| 404 Not Found | PortfolioNotFoundError, PositionNotFoundError |
| 408 Request Timeout | TimeoutError |
| 422 Unprocessable Entity | AccountReconciliationError, TaxCalculationError |
| 429 Too Many Requests | BulkheadFullError |
| 503 Service Unavailable | ServiceUnavailableError, CircuitBreakerOpenError, ServiceTimeoutError |
| 500 Internal Server Error | All other exceptions |

## Error Logging

Proper error logging is essential for debugging and monitoring. All error logs should include:

### Required Log Fields

- **Error Type**: The class name of the exception
- **Error Message**: The human-readable error message
- **Error Code**: The unique error code
- **Correlation ID**: The unique identifier for tracking the error
- **Service Name**: The name of the service logging the error
- **Timestamp**: When the error occurred
- **Stack Trace**: For internal errors (not in production user-facing logs)
- **Context**: Additional information relevant to the error

### Log Level Guidelines

- **ERROR**: Use for exceptions that indicate a failure requiring attention
- **WARNING**: Use for handled exceptions that don't cause service failure
- **INFO**: Use for expected exceptions that are part of normal operation
- **DEBUG**: Use for detailed debugging information

### Example Error Log

```
ERROR [portfolio-service] InsufficientBalanceError: Account ACC123 has insufficient balance for trade TR456
  correlation_id: 7f8d9e10-a1b2-c3d4-e5f6
  error_code: INSUFFICIENT_BALANCE
  details: {
    "account_id": "ACC123",
    "trade_id": "TR456",
    "required_amount": 1000.00,
    "available_balance": 500.00,
    "currency": "USD"
  }
  timestamp: 2023-05-22T12:34:56.789Z
```

## Resilience Patterns

The Forex Trading Platform uses several resilience patterns to handle failures gracefully:

### 1. Circuit Breaker

The Circuit Breaker pattern prevents cascading failures by stopping calls to failing services.

**When to Use**:
- When calling external services that might fail
- When calling internal services that are critical but might be unstable

**Example**:
```python
from common_lib.resilience import CircuitBreaker, CircuitBreakerConfig

# Create a circuit breaker
cb = CircuitBreaker(
    service_name="portfolio-service",
    resource_name="trading-gateway",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=30
    )
)

# Use the circuit breaker
async def execute_trade(trade_request):
    try:
        return await cb.execute(lambda: trading_gateway_client.execute_trade(trade_request))
    except CircuitBreakerOpenError:
        # Handle circuit breaker open
        logger.warning("Circuit breaker open for trading-gateway")
        raise ServiceUnavailableError(
            message="Trading service is currently unavailable",
            details={"retry_after_seconds": 30}
        )
```

### 2. Retry Policy

The Retry Policy pattern automatically retries temporary failures using configurable strategies.

**When to Use**:
- For transient failures like network glitches
- For operations that are idempotent (can be safely retried)

**Example**:
```python
from common_lib.resilience import retry_with_policy

@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    jitter=True,
    exceptions=[ConnectionError, TimeoutError]
)
async def fetch_market_data(symbol):
    return await market_data_client.get_price(symbol)
```

### 3. Timeout Handler

The Timeout Handler pattern ensures operations complete within specific time constraints.

**When to Use**:
- For operations that might hang indefinitely
- For user-facing operations with response time requirements

**Example**:
```python
from common_lib.resilience import timeout_handler

@timeout_handler(timeout_seconds=5.0)
async def get_real_time_price(symbol):
    return await market_data_client.get_real_time_price(symbol)
```

### 4. Bulkhead Pattern

The Bulkhead Pattern isolates failures by partitioning resources.

**When to Use**:
- To prevent one operation from consuming all resources
- To isolate critical operations from non-critical ones

**Example**:
```python
from common_lib.resilience import bulkhead

# Create a bulkhead for critical operations
critical_bulkhead = bulkhead(
    name="critical-operations",
    max_concurrent=10,
    max_queue_size=5
)

# Use the bulkhead
@critical_bulkhead
async def execute_critical_trade(trade_request):
    return await trading_gateway_client.execute_trade(trade_request)
```

## Language-Specific Guidelines

### Python Implementation (FastAPI Services)

#### 1. Using Exception Decorators

Use the `with_exception_handling` and `async_with_exception_handling` decorators to standardize exception handling:

```python
from service_name.error import async_with_exception_handling

@async_with_exception_handling
async def get_portfolio(self, portfolio_id: str) -> Portfolio:
    # Implementation that might raise exceptions
    return await self.repository.get_portfolio(portfolio_id)
```

#### 2. Registering Exception Handlers

Register exception handlers in your FastAPI application:

```python
from fastapi import FastAPI
from service_name.error import register_exception_handlers

app = FastAPI()

# Register all exception handlers
register_exception_handlers(app)
```

#### 3. Converting Exceptions to HTTP Responses

Use the `convert_to_http_exception` function in API endpoints:

```python
from fastapi import APIRouter, Depends, HTTPException
from service_name.error import convert_to_http_exception, PortfolioNotFoundError

router = APIRouter()

@router.get("/portfolios/{portfolio_id}")
async def get_portfolio(portfolio_id: str):
    try:
        return await portfolio_service.get_portfolio(portfolio_id)
    except PortfolioNotFoundError as e:
        raise convert_to_http_exception(e)
```

### JavaScript Implementation

#### 1. Error Middleware

Use error middleware to handle errors in Express applications:

```javascript
const { errorMiddleware } = require('common-js-lib');

// Add error middleware to Express app
app.use(errorMiddleware);
```

#### 2. Error Mapping

Map standard JavaScript errors to platform-specific errors:

```javascript
const { errors, mapError } = require('common-js-lib');

try {
  // Code that might throw an error
} catch (error) {
  // Map to platform-specific error
  const mappedError = error instanceof errors.ForexTradingPlatformError ? 
    error : 
    mapError(error);
  
  // Handle the mapped error
  handleError(mappedError);
}
```

#### 3. Client-Side Error Handling

Handle errors in API clients:

```javascript
const { errors, handleApiError } = require('common-js-lib');

async function fetchPortfolio(portfolioId) {
  try {
    const response = await axios.get(`/api/portfolios/${portfolioId}`);
    return response.data;
  } catch (error) {
    // Handle API error
    const handledError = handleApiError(error);
    
    // Show user-friendly message
    showErrorMessage(handledError.message);
    
    // Re-throw for upstream handling if needed
    throw handledError;
  }
}
```

## Testing Error Handling

### Unit Testing Exception Handling

Test that your code raises the expected exceptions:

```python
import pytest
from service_name.error import PortfolioNotFoundError

def test_get_portfolio_not_found():
    # Arrange
    portfolio_service = PortfolioService(MockRepository())
    portfolio_id = "non-existent-id"
    
    # Act & Assert
    with pytest.raises(PortfolioNotFoundError) as exc_info:
        portfolio_service.get_portfolio(portfolio_id)
    
    # Verify exception details
    assert exc_info.value.error_code == "PORTFOLIO_NOT_FOUND_ERROR"
    assert portfolio_id in exc_info.value.message
    assert exc_info.value.details["portfolio_id"] == portfolio_id
```

### Integration Testing with Simulated Failures

Test how your system handles failures in dependencies:

```python
def test_portfolio_service_handles_repository_failure():
    # Arrange
    failing_repository = MockRepository(should_fail=True)
    portfolio_service = PortfolioService(failing_repository)
    
    # Act & Assert
    with pytest.raises(ServiceError) as exc_info:
        portfolio_service.get_all_portfolios()
    
    # Verify exception details
    assert exc_info.value.error_code == "SERVICE_ERROR"
    assert "repository" in exc_info.value.message.lower()
```

### Chaos Testing to Verify Resilience

Test how your system handles unexpected failures:

```python
def test_circuit_breaker_opens_after_failures():
    # Arrange
    failing_client = MockClient(fail_count=6)  # Fail 6 times
    service = ServiceWithCircuitBreaker(failing_client)
    
    # Act - Call service multiple times
    for i in range(5):
        with pytest.raises(ServiceError):
            service.call_downstream()
    
    # Assert - Circuit breaker should be open now
    with pytest.raises(CircuitBreakerOpenError):
        service.call_downstream()
```

### Test Coverage Requirements

- Aim for at least 80% test coverage of error handling code
- Test both happy path and error scenarios
- Test all custom exception types
- Test resilience patterns under failure conditions

## Monitoring and Alerting

### Error Rate Metrics

Monitor the following metrics:

- **Error Rate by Service**: The rate of errors per service
- **Error Rate by Type**: The distribution of errors by type
- **Error Rate by Endpoint**: The rate of errors per API endpoint
- **Circuit Breaker Status**: The state of circuit breakers (open/closed)
- **Retry Attempts**: The number of retry attempts for operations

### Setting Up Alerts

Set up alerts for:

- **High Error Rate**: Alert when error rate exceeds threshold
- **Circuit Breaker Open**: Alert when circuit breakers open
- **Repeated Retries**: Alert when operations require multiple retries
- **Timeout Frequency**: Alert when timeouts occur frequently
- **Bulkhead Rejection**: Alert when bulkheads reject requests

### Error Dashboards

Create dashboards to visualize:

- **Error Trends**: How error rates change over time
- **Error Distribution**: Which services have the most errors
- **Error Types**: Which types of errors are most common
- **Correlation**: Correlation between errors and other metrics

### Service Health Assessment

Use error metrics to assess service health:

- **Error Budget**: Define an acceptable error rate
- **SLO Compliance**: Track compliance with Service Level Objectives
- **Error Trends**: Identify services with increasing error rates
- **Error Impact**: Assess the impact of errors on user experience

## Anti-patterns to Avoid

### 1. Generic Exception Handling

**Bad Example**:
```python
try:
    # Complex operation
except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
```

**Good Example**:
```python
try:
    # Complex operation
except DataFetchError as e:
    logger.error(f"Failed to fetch data: {str(e)}")
    raise
except ValidationError as e:
    logger.error(f"Validation failed: {str(e)}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    raise ServiceError(f"Unexpected error: {str(e)}")
```

### 2. Silent Failures

**Bad Example**:
```python
try:
    result = api_client.get_data()
except Exception:
    result = None  # Silently ignore the error
```

**Good Example**:
```python
try:
    result = api_client.get_data()
except Exception as e:
    logger.error(f"Failed to get data: {str(e)}")
    raise DataFetchError(f"Failed to get data: {str(e)}")
```

### 3. Inconsistent Error Formats

**Bad Example**:
```python
# Service 1
return {"error": "Something went wrong"}

# Service 2
return {"message": "Error occurred", "code": 500}
```

**Good Example**:
```python
# All services use the same format
return {
    "error": {
        "code": "ERROR_CODE",
        "message": "Human-readable error message",
        "details": {...},
        "correlation_id": "unique-id",
        "timestamp": "2023-05-22T12:34:56.789Z",
        "service": "service-name"
    }
}
```

### 4. Missing Context in Logs

**Bad Example**:
```python
logger.error("Database query failed")
```

**Good Example**:
```python
logger.error(
    "Database query failed",
    extra={
        "query_id": query_id,
        "parameters": parameters,
        "database": database_name,
        "error_code": error_code
    }
)
```

### 5. Overuse of Resilience Patterns

**Bad Example**:
```python
# Adding circuit breaker to a local function call
@circuit_breaker
def calculate_sum(a, b):
    return a + b
```

**Good Example**:
```python
# Only use circuit breakers for remote calls
@circuit_breaker
async def call_external_service(request_data):
    return await external_service_client.make_request(request_data)
```

### 6. Exposing Implementation Details

**Bad Example**:
```python
# User-facing error
return {
    "error": {
        "message": "java.sql.SQLException: ORA-00001: unique constraint violated",
        "stack_trace": "..."
    }
}
```

**Good Example**:
```python
# User-facing error
return {
    "error": {
        "code": "DUPLICATE_RECORD",
        "message": "A record with this ID already exists"
    }
}

# Internal log
logger.error(
    "Database constraint violation",
    extra={
        "exception": str(e),
        "stack_trace": traceback.format_exc(),
        "sql_error_code": "ORA-00001"
    }
)
```

### 7. Duplicate Error Handling

**Bad Example**:
```python
# In repository layer
try:
    result = db.query(...)
except Exception as e:
    logger.error(f"Database error: {str(e)}")
    raise

# In service layer
try:
    result = repository.get_data()
except Exception as e:
    logger.error(f"Repository error: {str(e)}")
    raise

# In API layer
try:
    result = service.get_data()
except Exception as e:
    logger.error(f"Service error: {str(e)}")
    return {"error": str(e)}
```

**Good Example**:
```python
# In repository layer - Catch specific DB exceptions and convert to domain exceptions
try:
    result = db.query(...)
except DBError as e:
    raise DataFetchError(f"Database error: {str(e)}")

# In service layer - Use exception decorator for consistent handling
@async_with_exception_handling
async def get_data(self):
    return await self.repository.get_data()

# In API layer - Register exception handlers for automatic conversion
app.add_exception_handler(DataFetchError, data_fetch_exception_handler)
```

## References

- [Common-Lib Exceptions Documentation](../common-lib/exceptions.md)
- [Resilience Patterns Documentation](../common-lib/resilience.md)
- [Error Monitoring Dashboard](../monitoring-alerting-service/dashboards/error_monitoring/README.md)
- [Error Handling Best Practices](https://martinfowler.com/articles/resilience-patterns.html)
