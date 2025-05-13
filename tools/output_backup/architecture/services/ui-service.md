# UI Service

*Generated on 2025-05-13 05:58:22*

## Description

## Overview
The UI Service provides the frontend interface for the Forex Trading Platform. It offers a modern, responsive web application for traders, analysts, and administrators to interact with the platform's features, monitor market data, execute trades, and analyze performance.

## Dependencies

This service depends on the following services:

- **analysis-engine**: Service for analysis engine functionality.
- **feature-store-service**: ## Overview
The Feature Store Service is a centralized repository for storing, managing, and serving features used in the Forex Trading Platform. It acts as the canonical source for all indicator implementations, providing consistent data access patterns for machine learning models and analysis components.

## Dependents

No other services depend on this service.

## Interfaces

This service provides the following interfaces:

### UiServiceInterface

Interface for ui-service service.

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
