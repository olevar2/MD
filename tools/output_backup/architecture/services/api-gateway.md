# Api Gateway

*Generated on 2025-05-13 05:58:22*

## Description

Service for api gateway functionality.

## Dependencies

This service has no dependencies on other services.

## Dependents

No other services depend on this service.

## Interfaces

This service provides the following interfaces:

### ApiGatewayInterface

Interface for api-gateway service.

#### Methods

- **get_status() -> Dict**: Get the status of the service.

Returns:
    Service status information
- **get_info(resource_id: str) -> Dict**: Get information from the service.

Args:
    resource_id: Resource identifier
Returns:
    Resource information
- **list_resources(filter_params: Optional, limit: int, offset: int) -> Dict**: List available resources.

Args:
    filter_params: Filter parameters
Args:
    limit: Maximum number of results
Args:
    offset: Result offset
Returns:
    Dictionary with resources and pagination information

## Directory Structure

The service follows the standardized directory structure:

- **api**: API routes and controllers
- **config**: Configuration files
- **core**: Core business logic
- **models**: Data models and schemas
- **repositories**: Data access layer
- **services**: Service implementations
- **utils**: Utility functions
- **adapters**: Adapters for external services
- **interfaces**: Interface definitions
- **tests**: Unit and integration tests
