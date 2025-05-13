# Model Registry Service

*Generated on 2025-05-13 05:58:22*

## Description

A dedicated service for managing machine learning model versioning, deployment, and lifecycle in the Forex Trading Platform.

## Dependencies

This service has no dependencies on other services.

## Dependents

No other services depend on this service.

## Interfaces

This service provides the following interfaces:

### ModelRegistryServiceInterface

Interface for model-registry-service service.

#### Methods

- **get_status() -> Dict**: Get the status of the service.

Returns:
    Service status information
- **get_model(model_id: str) -> Dict**: Get a model from the registry.

Args:
    model_id: Model identifier
Returns:
    Model information
- **list_models(tags: Optional, limit: int, offset: int) -> Dict**: List available models.

Args:
    tags: Filter by tags
Args:
    limit: Maximum number of results
Args:
    offset: Result offset
Returns:
    Dictionary with models and pagination information

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
