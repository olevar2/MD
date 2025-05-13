# PortfolioManagementServiceInterface

*Generated on 2025-05-13 05:58:22*

## Description

Interface for portfolio-management-service service.

## File

`portfolio_management_service_interface.py`

## Methods

### get_status() -> Dict

Get the status of the service.

Returns:
    Service status information

#### Returns

- Dict

### get_info(resource_id: str) -> Dict

Get information from the service.

Args:
    resource_id: Resource identifier
Returns:
    Resource information

#### Parameters

- **resource_id** (str)

#### Returns

- Dict

### list_resources(filter_params: Optional, limit: int, offset: int) -> Dict

List available resources.

Args:
    filter_params: Filter parameters
Args:
    limit: Maximum number of results
Args:
    offset: Result offset
Returns:
    Dictionary with resources and pagination information

#### Parameters

- **filter_params** (Optional)
- **limit** (int)
- **offset** (int)

#### Returns

- Dict

