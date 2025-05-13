# Service Template

This template provides a standardized structure for creating new services in the forex trading platform.

## Features

- **Configuration Management**: Standardized configuration loading from files and environment variables
- **Logging Setup**: Consistent logging setup across services
- **Service Clients**: Resilient service clients for communicating with other services
- **Database Connectivity**: Standardized database connectivity and operations
- **Error Handling**: Consistent error handling and reporting

## Usage

1. Copy the template files to your new service
2. Customize the `ServiceConfig` class in `config.py` to include your service-specific configuration parameters
3. Create a `config.yaml` file with your service configuration
4. Use the provided modules in your service implementation

## Configuration

The template uses the configuration management system from the common-lib package. Configuration can be loaded from YAML or JSON files and overridden with environment variables.

Example configuration file:

```yaml
app:
  environment: development
  debug: true
  testing: false

database:
  host: localhost
  port: 5432
  username: postgres
  password: password
  database: forex_platform

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/service.log

service:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 60

service_clients:
  market_data_service:
    base_url: http://market-data-service:8001
    timeout: 30.0
    retry:
      max_retries: 3
      initial_backoff: 1.0
      max_backoff: 60.0
      backoff_factor: 2.0
    circuit_breaker:
      failure_threshold: 5
      recovery_timeout: 60.0
      expected_exceptions:
        - ConnectionError
        - Timeout

service_specific:
  max_workers: 8
  cache_size: 2000
```

## Environment Variables

Configuration can be overridden with environment variables using the following format:

```
APP_<SECTION>__<KEY>=<VALUE>
```

For example:

```
APP_APP__ENVIRONMENT=production
APP_DATABASE__HOST=db.example.com
APP_SERVICE__PORT=9000
APP_SERVICE_SPECIFIC__MAX_WORKERS=16
```

## Modules

### Config

The `config.py` module provides configuration management for the service. It defines a `ServiceConfig` class for service-specific configuration and provides helper functions for accessing configuration values.

```python
from service_template import get_service_config, is_development

# Get service-specific configuration
service_config = get_service_config()
max_workers = service_config.max_workers

# Check environment
if is_development():
    # Development-specific code
    pass
```

### Logging Setup

The `logging_setup.py` module provides logging setup for the service. It sets up console and file logging based on the configuration.

```python
from service_template import setup_logging

# Set up logging
logger = setup_logging("my-service")
logger.info("Service started")
```

### Service Clients

The `service_clients.py` module provides service clients for communicating with other services. It creates resilient service clients based on the configuration.

```python
from service_template import service_clients

# Get a service client
market_data_client = service_clients.get_market_data_client()

# Make a request
data = await market_data_client.get("/api/v1/market-data/EURUSD")
```

### Database

The `database.py` module provides database connectivity and operations for the service. It creates a connection pool based on the configuration.

```python
from service_template import database

# Connect to the database
await database.connect()

# Execute a query
rows = await database.fetch("SELECT * FROM users WHERE id = $1", user_id)

# Close the connection
await database.close()
```

### Error Handling

The `error_handling.py` module provides error handling functionality for the service. It includes decorators for handling exceptions in functions and methods.

```python
from service_template import handle_async_exception

@handle_async_exception(operation="get_user")
async def get_user(user_id):
    # Function implementation
    pass
```

## Best Practices

- Use the provided modules for configuration, logging, service clients, database connectivity, and error handling
- Customize the `ServiceConfig` class to include your service-specific configuration parameters
- Use environment variables for configuration in different environments
- Handle errors consistently using the provided error handling functionality
- Use the resilient service clients for communicating with other services
- Close database connections and service clients when they are no longer needed
