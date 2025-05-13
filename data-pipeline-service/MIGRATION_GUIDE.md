# Data Pipeline Service Migration Guide

This guide provides instructions for migrating the Data Pipeline Service to the standardized service template.

## Overview

The migration involves updating the following components:

1. **Configuration Management**: Migrating to the standardized configuration management system
2. **Logging Setup**: Migrating to the standardized logging configuration
3. **Service Clients**: Migrating to the standardized service client system
4. **Database Connectivity**: Migrating to the standardized database connectivity
5. **Error Handling**: Migrating to the standardized error handling

## Migration Steps

### 1. Configuration Management

#### Current Implementation

The current implementation uses a custom configuration system with the following files:

- `data_pipeline_service/config/config.py`

#### New Implementation

The new implementation uses the standardized configuration management system from common-lib:

- `data_pipeline_service/config/standardized_config.py`

#### Migration Steps

1. Review the new configuration file and ensure all service-specific settings are included
2. Update imports in other modules to use the new configuration:

```python
# Before
from data_pipeline_service.config import get_config

# After
from data_pipeline_service.config.standardized_config import settings
```

3. Update configuration access:

```python
# Before
config = get_config()
value = config.some_setting

# After
value = settings.SOME_SETTING
```

### 2. Logging Setup

#### Current Implementation

The current implementation uses a custom logging setup:

- `data_pipeline_service/logging_setup.py`

#### New Implementation

The new implementation uses the standardized logging configuration from common-lib:

- `data_pipeline_service/logging_setup_standardized.py`

#### Migration Steps

1. Review the new logging setup and ensure all service-specific logging requirements are met
2. Update imports in other modules to use the new logging setup:

```python
# Before
from data_pipeline_service.logging_setup import get_logger

# After
from data_pipeline_service.logging_setup_standardized import get_service_logger
```

3. Update logger creation:

```python
# Before
logger = get_logger(__name__)

# After
logger = get_service_logger(__name__)
```

### 3. Service Clients

#### Current Implementation

The current implementation uses a custom service client system:

- `data_pipeline_service/service_clients.py`

#### New Implementation

The new implementation uses the standardized service client system from common-lib:

- `data_pipeline_service/service_clients_standardized.py`

#### Migration Steps

1. Review the new service clients and ensure all service-specific client requirements are met
2. Update imports in other modules to use the new service clients:

```python
# Before
from data_pipeline_service.service_clients import service_clients

# After
from data_pipeline_service.service_clients_standardized import service_clients
```

3. Update service client usage:

```python
# Before
client = service_clients.get_client("market_data_service")
response = await client.get("/endpoint")

# After
client = service_clients.get_client("market_data_service")
response = await client.get("/endpoint")
```

### 4. Database Connectivity

#### Current Implementation

The current implementation uses a custom database connectivity system:

- `data_pipeline_service/database.py`

#### New Implementation

The new implementation uses the standardized database connectivity from common-lib:

- `data_pipeline_service/database_standardized.py`

#### Migration Steps

1. Review the new database module and ensure all service-specific database requirements are met
2. Update imports in other modules to use the new database module:

```python
# Before
from data_pipeline_service.database import database

# After
from data_pipeline_service.database_standardized import database
```

3. Update database usage:

```python
# Before
result = await database.fetch("SELECT * FROM table")

# After
result = await database.fetch_all("SELECT * FROM table")
```

### 5. Error Handling

#### Current Implementation

The current implementation uses a custom error handling system:

- `data_pipeline_service/error_handling.py`

#### New Implementation

The new implementation uses the standardized error handling from common-lib:

- `data_pipeline_service/error_handling_standardized.py`

#### Migration Steps

1. Review the new error handling module and ensure all service-specific error types are included
2. Update imports in other modules to use the new error handling:

```python
# Before
from data_pipeline_service.error_handling import handle_error, handle_exception

# After
from data_pipeline_service.error_handling_standardized import handle_error, handle_exception
```

3. Update error handling usage:

```python
# Before
@handle_exception(operation="get_data")
def get_data():
    # Function implementation

# After
@handle_exception(operation="get_data")
def get_data():
    # Function implementation
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
4. **Phase 4**: Implement and test the new database connectivity (1 day)
5. **Phase 5**: Implement and test the new error handling (1 day)
6. **Phase 6**: Final testing and deployment (1 day)

Total estimated time: 6 days
