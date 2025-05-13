# ML Integration Service Migration Guide

This guide provides instructions for migrating the ML Integration Service to the standardized service template.

## Overview

The migration involves updating the following components:

1. **Configuration Management**: Migrating to the standardized configuration management system
2. **Logging Setup**: Migrating to the standardized logging configuration
3. **Service Clients**: Migrating to the standardized service client system
4. **Error Handling**: Migrating to the standardized error handling

## Migration Steps

### 1. Configuration Management

#### Current Implementation

The current implementation uses a custom configuration system with the following files:

- `ml_integration_service/config/settings.py`
- `ml_integration_service/config/enhanced_settings.py`

#### New Implementation

The new implementation uses the standardized configuration management system from common-lib:

- `ml_integration_service/config/standardized_config.py`

#### Migration Steps

1. Review the new configuration file and ensure all service-specific settings are included
2. Update imports in other modules to use the new configuration:

```python
# Before
from ml_integration_service.config.settings import settings

# After
from ml_integration_service.config.standardized_config import settings
```

3. Update configuration access:

```python
# Before
value = settings.some_setting

# After
value = settings.SOME_SETTING
```

### 2. Logging Setup

#### Current Implementation

The current implementation uses a custom logging setup or relies on the framework's default logging.

#### New Implementation

The new implementation uses the standardized logging configuration from common-lib:

- `ml_integration_service/logging_setup_standardized.py`

#### Migration Steps

1. Review the new logging setup and ensure all service-specific logging requirements are met
2. Update imports in other modules to use the new logging setup:

```python
# Before
import logging
logger = logging.getLogger(__name__)

# After
from ml_integration_service.logging_setup_standardized import get_service_logger
logger = get_service_logger(__name__)
```

3. Update logging calls to use the new logging functions:

```python
# Before
logger.info("Message")

# After
logger.info("Message")
# Or for structured logging with context
from ml_integration_service.logging_setup_standardized import log_with_context
log_with_context(logger, "INFO", "Message", context={"key": "value"})
```

4. For model-related logging, use the specialized logging function:

```python
from ml_integration_service.logging_setup_standardized import log_model_operation
log_model_operation(
    logger=logger,
    operation="training",
    model_name="my_model",
    model_version="1.0.0",
    duration=10.5,
    status="success",
    metrics={"accuracy": 0.95}
)
```

### 3. Service Clients

#### Current Implementation

The current implementation uses a custom service client system or direct HTTP requests.

#### New Implementation

The new implementation uses the standardized service client system from common-lib:

- `ml_integration_service/service_clients_standardized.py`

#### Migration Steps

1. Review the new service clients and ensure all service-specific client requirements are met
2. Update imports in other modules to use the new service clients:

```python
# Before
from ml_integration_service.clients.ml_workbench_client import ml_workbench_client

# After
from ml_integration_service.service_clients_standardized import service_clients
ml_workbench_client = service_clients.get_ml_workbench_client()
```

3. Update service client usage:

```python
# Before
response = await ml_workbench_client.get_model("my_model")

# After
response = await service_clients.get_ml_workbench_client().get("/models/my_model")
```

### 4. Error Handling

#### Current Implementation

The current implementation uses a custom error handling system or relies on the framework's default error handling.

#### New Implementation

The new implementation uses the standardized error handling from common-lib:

- `ml_integration_service/error_handling_standardized.py`

#### Migration Steps

1. Review the new error handling module and ensure all service-specific error types are included
2. Update imports in other modules to use the new error handling:

```python
# Before
from ml_integration_service.error_handlers import handle_error

# After
from ml_integration_service.error_handling_standardized import handle_error, handle_exception
```

3. Update error handling usage:

```python
# Before
try:
    # Code that might raise an exception
except Exception as e:
    handle_error(e)

# After
@handle_exception(operation="operation_name")
def my_function():
    # Code that might raise an exception
```

4. Use the new error types:

```python
# Before
from ml_integration_service.error_handlers import ModelNotFoundError

# After
from ml_integration_service.error_handling_standardized import ModelNotFoundError
```

## Testing

After completing the migration, thoroughly test the service to ensure all functionality works as expected:

1. **Unit Tests**: Run all unit tests to ensure the core functionality works
2. **Integration Tests**: Run integration tests to ensure the service integrates correctly with other services
3. **End-to-End Tests**: Run end-to-end tests to ensure the service works correctly in the full system

## Rollback Plan

If issues are encountered during the migration, follow these steps to roll back:

1. Revert the changes to the original implementation
2. Run tests to ensure the service works correctly with the original implementation
3. Document the issues encountered for future migration attempts

## Timeline

The migration should be completed in the following phases:

1. **Phase 1**: Implement and test the new configuration management (1 day)
2. **Phase 2**: Implement and test the new logging setup (1 day)
3. **Phase 3**: Implement and test the new service clients (1 day)
4. **Phase 4**: Implement and test the new error handling (1 day)
5. **Phase 5**: Final testing and deployment (1 day)

Total estimated time: 5 days
