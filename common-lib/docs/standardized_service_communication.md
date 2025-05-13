# Standardized Service Communication

## Overview

This document provides a comprehensive guide to the standardized service communication approach implemented in the Forex Trading Platform. The implementation follows a structured approach to ensure resilient, consistent, and observable service-to-service communication across the platform.

## Key Components

The standardized service communication implementation consists of the following key components:

1. **Client Abstractions**: Base client class and service-specific implementations
2. **Client Configuration**: Centralized configuration management
3. **Error Handling**: Standardized exceptions and error mapping
4. **Resilience Patterns**: Circuit breakers, retry policies, timeouts, and bulkheads
5. **Metrics and Logging**: Performance tracking and structured logging

## 1. Client Abstractions

### BaseServiceClient

The `BaseServiceClient` class in `common_lib.clients.base_client` serves as the foundation for all service clients. It provides:

- Standard HTTP methods (GET, POST, PUT, DELETE) with both async and sync versions
- Built-in resilience patterns
- Error handling and mapping
- Metrics collection
- Structured logging

```python
class BaseServiceClient:
    """Base class for service clients with built-in resilience patterns."""
    
    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """Initialize the base service client."""
        # Configuration and initialization
        
    async def get(self, endpoint: str, params: Optional[Dict] = None, ...):
        """Make a GET request."""
        
    async def post(self, endpoint: str, data: Optional[Dict] = None, ...):
        """Make a POST request."""
        
    async def put(self, endpoint: str, data: Optional[Dict] = None, ...):
        """Make a PUT request."""
        
    async def delete(self, endpoint: str, params: Optional[Dict] = None, ...):
        """Make a DELETE request."""
```

### Service-Specific Clients

Service-specific clients inherit from `BaseServiceClient` and implement methods specific to their service:

```python
class AnalysisEngineClient(BaseServiceClient):
    """Client for interacting with the Analysis Engine Service."""
    
    async def get_technical_indicators(self, symbol: str, timeframe: str, ...):
        """Get technical indicators for a symbol and timeframe."""
        
    async def detect_market_regime(self, symbol: str, timeframe: str, ...):
        """Detect the current market regime."""
```

### Client Factory

The client factory pattern centralizes client creation and configuration:

```python
def get_analysis_engine_client(config_override: Optional[Dict[str, Any]] = None):
    """Get a configured Analysis Engine client."""
    return get_client(
        client_class=AnalysisEngineClient,
        service_name="analysis-engine-service",
        config_override=config_override
    )
```

## 2. Client Configuration

### ClientConfig Model

The `ClientConfig` model in `common_lib.clients.base_client` defines the configuration parameters for service clients:

```python
class ClientConfig(BaseModel):
    """Configuration for service clients."""
    
    # Service connection
    base_url: str
    timeout_seconds: float = 10.0
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    
    # Resilience settings
    max_retries: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 30.0
    retry_backoff_factor: float = 2.0
    retry_jitter: bool = True
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_timeout_seconds: int = 60
    
    # Bulkhead settings
    bulkhead_max_concurrent: int = 10
    bulkhead_max_waiting: int = 20
    
    # Metrics and logging
    enable_metrics: bool = True
    enable_request_logging: bool = True
    log_request_body: bool = False
    log_response_body: bool = False
    
    # Service name for metrics and logging
    service_name: str = "unknown-service"
```

### Centralized Configuration

Client configurations are centralized in `common_lib.clients.config`:

```python
# Standard service configurations
SERVICE_CONFIGS = {
    # Analysis Engine Service
    "analysis-engine-service": {
        "base_url": os.environ.get("ANALYSIS_ENGINE_API_URL", "http://analysis-engine-service:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("ANALYSIS_ENGINE_TIMEOUT", "30.0")),
        "max_retries": int(os.environ.get("ANALYSIS_ENGINE_MAX_RETRIES", "3")),
        # ... other settings
    },
    
    # ML Workbench Service
    "ml-workbench-service": {
        "base_url": os.environ.get("ML_WORKBENCH_API_URL", "http://ml-workbench-service:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("ML_WORKBENCH_TIMEOUT", "30.0")),
        "max_retries": int(os.environ.get("ML_WORKBENCH_MAX_RETRIES", "3")),
        # ... other settings
    },
    
    # ... other services
}
```

### Configuration Registration

Configurations are registered during service startup:

```python
def initialize_clients():
    """Initialize all clients with proper configuration."""
    # Import registration function
    from common_lib.clients import register_client_config
    
    # Configure Analysis Engine client
    analysis_engine_config = {
        "base_url": settings.ANALYSIS_ENGINE_API_URL,
        "service_name": "analysis-engine-service",
        "api_key": settings.ANALYSIS_ENGINE_API_KEY.get_secret_value() if settings.ANALYSIS_ENGINE_API_KEY else None,
        "timeout_seconds": 30.0,
        # ... other settings
    }
    
    # Register client configurations
    register_client_config("analysis-engine-service", ClientConfig(**analysis_engine_config))
```

## 3. Error Handling

### Client Exceptions

The `common_lib.clients.exceptions` module defines standardized exceptions for client errors:

```python
class ClientError(ServiceError):
    """Base exception for client errors."""
    
class ClientConnectionError(ServiceUnavailableError):
    """Exception raised when a client cannot connect to a service."""
    
class ClientTimeoutError(ServiceTimeoutError):
    """Exception raised when a client request times out."""
    
class ClientValidationError(DataValidationError):
    """Exception raised when client data validation fails."""
    
class ClientAuthenticationError(AuthenticationError):
    """Exception raised when client authentication fails."""
    
class CircuitBreakerOpenError(ServiceUnavailableError):
    """Exception raised when a circuit breaker is open."""
    
class BulkheadFullError(ServiceUnavailableError):
    """Exception raised when a bulkhead is full."""
    
class RetryExhaustedError(ServiceUnavailableError):
    """Exception raised when retries are exhausted."""
```

### Error Mapping

The `BaseServiceClient._map_exception` method maps HTTP exceptions to service-specific exceptions:

```python
def _map_exception(self, exception: Exception) -> Exception:
    """Map HTTP exceptions to service-specific exceptions."""
    if isinstance(exception, aiohttp.ClientResponseError):
        status = exception.status
        message = str(exception)
        
        if status == 401 or status == 403:
            return AuthenticationError(f"Authentication failed: {message}")
        elif status == 404:
            return ServiceError(f"Resource not found: {message}")
        # ... other mappings
```

### Error Handling in Service Clients

Service clients handle errors and provide meaningful error messages:

```python
async def get_technical_indicators(self, symbol: str, timeframe: str, ...):
    """Get technical indicators for a symbol and timeframe."""
    try:
        response = await self.post("indicators/calculate", data=data)
        return response
    except Exception as e:
        logger.error(f"Error getting technical indicators: {str(e)}")
        raise ClientError(
            f"Failed to get technical indicators for {symbol} {timeframe}",
            service_name=self.config.service_name
        ) from e
```

## 4. Resilience Patterns

### Circuit Breaker

The circuit breaker pattern prevents cascading failures by stopping calls to failing services:

```python
# Initialize circuit breaker
self.circuit_breaker = create_circuit_breaker(
    service_name=self.config.service_name,
    resource_name="http",
    config=CircuitBreakerConfig(
        failure_threshold=self.config.circuit_breaker_failure_threshold,
        reset_timeout_seconds=self.config.circuit_breaker_reset_timeout_seconds
    )
)

# Use circuit breaker
result = await self.circuit_breaker.execute(execute_request)
```

### Retry Policy

The retry policy automatically retries temporary failures with exponential backoff:

```python
@retry_with_policy(
    exceptions=[
        aiohttp.ClientError,
        asyncio.TimeoutError,
        ConnectionError,
        TimeoutError
    ]
)
async def _make_request(self, method: str, endpoint: str, ...):
    """Make an HTTP request with resilience patterns."""
    # Request implementation
```

### Timeout Handler

The timeout handler ensures operations complete within specific time constraints:

```python
@timeout_handler(timeout_seconds=10.0)  # Default timeout, will be overridden by instance timeout
async def _make_request(self, method: str, endpoint: str, ...):
    """Make an HTTP request with resilience patterns."""
    # Request implementation
```

### Bulkhead Pattern

The bulkhead pattern isolates failures by partitioning resources:

```python
@bulkhead(name="service_client", max_concurrent=10, max_waiting=20)  # Default values, will be overridden
async def _make_request(self, method: str, endpoint: str, ...):
    """Make an HTTP request with resilience patterns."""
    # Request implementation
```

## 5. Metrics and Logging

### Request Logging

The `BaseServiceClient` logs request and response details:

```python
def _log_request(self, method: str, url: str, params: Optional[Dict] = None, data: Optional[Dict] = None):
    """Log request details."""
    if not self.config.enable_request_logging:
        return
    
    log_data = {
        "method": method,
        "url": url,
        "service": self.config.service_name,
    }
    
    if params:
        log_data["params"] = params
        
    if data and self.config.log_request_body:
        log_data["body"] = data
        
    logger.info(f"Service request: {json.dumps(log_data)}")
```

### Metrics Collection

The `BaseServiceClient` records metrics for requests:

```python
def _record_metrics(self, method: str, endpoint: str, status: int, response_time: float):
    """Record metrics for the request."""
    if not self.config.enable_metrics:
        return
    
    # This is a placeholder for actual metrics implementation
    logger.debug(
        f"METRIC: service_request "
        f"method={method} "
        f"endpoint={endpoint} "
        f"service={self.config.service_name} "
        f"status={status} "
        f"response_time_ms={response_time:.2f}"
    )
```

## Implementation Details

### File Structure

```
common-lib/
├── common_lib/
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── base_client.py
│   │   ├── client_factory.py
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── README.md
│   └── resilience/
│       ├── __init__.py
│       ├── circuit_breaker.py
│       ├── retry_policy.py
│       ├── timeout_handler.py
│       └── bulkhead.py
└── docs/
    ├── service_communication.md
    └── standardized_service_communication.md

ml-integration-service/
├── ml_integration_service/
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── analysis_engine_client.py
│   │   ├── ml_workbench_client.py
│   │   ├── client_factory.py
│   │   └── README.md
│   ├── examples/
│   │   └── client_usage_example.py
│   └── main.py
```

### Client Initialization

Clients are initialized during service startup in the `main.py` file:

```python
@app.on_event("startup")
async def startup_event():
    """Initialize service components on startup"""
    logger.info("Starting ML Integration Service")

    # Initialize service clients
    initialize_clients()
    logger.info("Service clients initialized successfully")
    
    # ... other initialization
```

### Client Cleanup

Clients are closed during service shutdown:

```python
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down ML Integration Service")
    
    # Close service clients
    from ml_integration_service.clients.client_factory import get_analysis_engine_client, get_ml_workbench_client
    
    # Close the Analysis Engine client
    try:
        analysis_engine_client = get_analysis_engine_client()
        await analysis_engine_client.close()
        logger.info("Analysis Engine client closed successfully")
    except Exception as e:
        logger.error(f"Error closing Analysis Engine client: {str(e)}")
    
    # ... close other clients
```

## Usage Examples

### Basic Usage

```python
from ml_integration_service.clients.client_factory import get_analysis_engine_client

# Get the client
client = get_analysis_engine_client()

# Make a request
result = await client.get_technical_indicators(
    symbol="EURUSD",
    timeframe="1h",
    indicators=[{"name": "SMA", "params": {"period": 20}}]
)
```

### Error Handling

```python
from common_lib.clients.exceptions import ClientError, ClientTimeoutError

try:
    result = await client.get_technical_indicators(...)
except ClientTimeoutError as e:
    # Handle timeout
    logger.error(f"Request timed out: {str(e)}")
except ClientError as e:
    # Handle other client errors
    logger.error(f"Client error: {str(e)}")
```

### Custom Configuration

```python
# Get a client with custom configuration
client = get_analysis_engine_client(
    config_override={
        "timeout_seconds": 60.0,
        "max_retries": 5,
        "circuit_breaker_failure_threshold": 10
    }
)
```

## Benefits

The standardized service communication approach provides several benefits:

1. **Improved Resilience**: Built-in resilience patterns prevent cascading failures and handle transient errors gracefully.

2. **Consistent Error Handling**: Standardized exceptions and error mapping provide a consistent approach to error handling across all services.

3. **Better Observability**: Metrics collection and structured logging enable comprehensive monitoring and troubleshooting.

4. **Simplified Development**: Easy-to-use client abstractions reduce boilerplate code and ensure consistent implementation.

5. **Centralized Configuration**: Consistent configuration across all services makes it easier to manage and update client settings.

6. **Enhanced Security**: Standardized API key handling and authentication improve security across service communications.

7. **Performance Monitoring**: Built-in metrics collection enables performance monitoring and optimization.

8. **Reduced Code Duplication**: Common functionality is centralized in the base client, reducing duplication across services.

## Best Practices

1. **Use the Base Client**: Always inherit from `BaseServiceClient` for new clients to ensure consistent implementation.

2. **Centralize Configuration**: Use the client factory and configuration module to manage client settings.

3. **Handle Errors Properly**: Map exceptions to domain-specific errors and provide meaningful error messages.

4. **Add Proper Logging**: Use structured logging for requests and responses to aid in troubleshooting.

5. **Include Metrics**: Enable metrics collection for monitoring and performance optimization.

6. **Close Clients**: Close clients during service shutdown to release resources properly.

7. **Use Dependency Injection**: Provide clients to components through dependency injection for better testability.

8. **Test with Mocks**: Use mocks for testing client interactions to avoid actual network calls during tests.

9. **Configure Timeouts Appropriately**: Set appropriate timeouts for different types of operations to prevent hanging requests.

10. **Monitor Circuit Breakers**: Monitor circuit breaker states to detect service health issues early.

## Future Enhancements

1. **Metrics Integration**: Integrate with a metrics system like Prometheus for comprehensive monitoring.

2. **Distributed Tracing**: Add distributed tracing with OpenTelemetry for end-to-end request tracking.

3. **Dynamic Configuration**: Implement dynamic configuration updates without service restarts.

4. **Client-Side Load Balancing**: Add client-side load balancing for services with multiple instances.

5. **Service Discovery**: Integrate with service discovery mechanisms for dynamic service resolution.

6. **Enhanced Security**: Add support for more authentication methods and token-based authentication.

7. **Request Validation**: Add client-side request validation to catch errors before sending requests.

8. **Response Caching**: Implement response caching for frequently accessed, rarely changing data.

9. **Rate Limiting**: Add client-side rate limiting to prevent overwhelming services.

10. **Graceful Degradation**: Implement more sophisticated fallback mechanisms for degraded service operation.

## Conclusion

The standardized service communication approach provides a robust, consistent, and observable foundation for service-to-service communication in the Forex Trading Platform. By centralizing common functionality and implementing best practices for resilience, error handling, and observability, the platform can achieve higher reliability, better performance, and improved maintainability.
